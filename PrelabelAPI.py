#checking ci/cd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body, Query
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
import os
import json
import yaml
import requests
import time
import logging
import sys
from dotenv import load_dotenv
from video_description_generator import VideoDescriptionGenerator
from video_analysis_prompt import VIDEO_ANALYSIS_PROMPT
from fastapi.responses import PlainTextResponse
from datetime import datetime , timedelta
import numpy as np
from sewar.full_ref import mse, psnr, ssim
from video_analysis_prompt_ads import VIDEO_ANALYSIS_PROMPT_ADS

# Configure logging to ensure output goes to stdout/stderr for systemd
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('config.env')

app = FastAPI()

# Pydantic model for request validation
class PrelabelRequest(BaseModel):
    task_id: str
    project_id: str
    user_id: str

class InstructionRequest(BaseModel):
    instructions: str 

# MongoDB connection
MONGODB_URL = os.getenv('MONGODB_URI')
if not MONGODB_URL:
    raise ValueError("MONGO_URI environment variable is not set")

client = MongoClient(MONGODB_URL)
db = client["AkaiDb0"]
datapoints_collection = db["datapoints"]
users_collection = db["users"]


# Initialize the generator
TRANSNET_MODEL_DIR = os.getenv('TRANSNET_MODEL_DIR', './TransNetV2_Keyframe_Detection/transnetv2-weights/')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

generator = VideoDescriptionGenerator(TRANSNET_MODEL_DIR, OPENAI_API_KEY)


@app.post("/process-instructions", response_class=PlainTextResponse)
async def process_instructions(
    instructions: str = Body(..., media_type="text/plain"),
    platform: str = Query(None, description="Platform type to determine which prompt to use (e.g., 'ads')")
):
    
    try:
        # Determine which prompt to use based on platform parameter
        if platform and platform.lower() == "ads":
            base_prompt = VIDEO_ANALYSIS_PROMPT_ADS
            print("🎯 Using ADS prompt for process-instructions based on platform parameter")
        else:
            base_prompt = VIDEO_ANALYSIS_PROMPT
            print("🎯 Using default prompt for process-instructions")
        
        # Use instructions directly instead of parsing JSON
        prompt = f"""You are an AI assistant tasked with modifying a default prompt for analyzing video keyframes based on a user-provided paragraph that contains details about the dataset, special labeling instructions, and objects to focus on. Your goal is to update the default aspects list and example questions to align with the specific requirements of the dataset, ensuring high-quality labeling output.
Here is the default prompt you will modify:
kindly make sure to keep the same format as the default prompt given in string not any type of json: {base_prompt}

Instructions for Modification:
When you receive the user's paragraph, follow these steps:
Analyze the Paragraph: Carefully read the user-provided paragraph to identify key details about the dataset, special labeling instructions, and objects or elements to focus on. Conduct a deep analysis to understand the context and specific requirements.
Update the Aspects List:
Start with the default aspects list provided above.
Keep aspects that remain relevant to the dataset described in the paragraph.
Modify existing aspects if the paragraph suggests a different focus or specificity (e.g., changing "Number of participants" to "Number of vehicles" for a traffic dataset).
Remove aspects that are irrelevant based on the paragraph.
Add new aspects if the paragraph highlights unique elements not covered in the default list (e.g., "Types of animals" for a wildlife dataset).
Generate Example Questions:
Based on the updated aspects list, create 5 new example questions that reflect the specific dataset and instructions from the paragraph.
Ensure each question combines an aspect with a potential answer, tailored to the context (e.g., "What types of vehicles are present and what are their actions?" for a traffic dataset).
The questions should cover the most critical aspects identified in the paragraph.
Preserve Output Structure:
The modified prompt will be used to analyze keyframes and produce a JSON output with the following structure:
"questions": List of 5 question-answer pairs.
"keywords": List of relevant keywords.
"map_placement": Selected option (Town, Village, Water body, Mountains, Snow, Road).
"summary": Detailed description of the video.
Output the Modified Prompt:
Your response should be the full text of the modified prompt, incorporating the updated aspects list and example questions.
Maintain the structure of the default prompt, adjusting only the aspects list and example questions based on the paragraph.
Do not analyze keyframes or produce the JSON output here—your task is to generate the modified prompt text that can later be used for keyframe analysis.
Response Format:
Provide the modified prompt as plain text, starting with "You are an AI assistant..." and including the updated aspects list and example questions.
Do not include the JSON output or keyframe analysis in your response—only the modified prompt text.
Example Workflow:
If the user's paragraph is: "This dataset consists of videos from urban traffic cameras. Focus on identifying types of vehicles, traffic flow, and any incidents."
Update aspects to include "Types of vehicles," "Traffic flow," "Incidents," while keeping relevant defaults like "Location" and "Timing."
Remove irrelevant aspects like "Emotions/Expressions."
Generate example questions like "What types of vehicles are visible and how are they moving?" and "Is there any incident affecting the traffic flow?"
Paragraph Input: {instructions}
"""
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
        )
        
        if response.status_code == 200:
            openai_response = response.json()['choices'][0]['message']['content']
            # Replace escaped newlines with actual newlines for clean formatting
            clean_response = openai_response.replace('\\n', '\n')
            return PlainTextResponse(content=clean_response)
        else:
            print(f"OpenAI API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {response.status_code}")
            
    except Exception as e:
        print(f"Error processing instructions: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing instructions: {str(e)}")


@app.post("/prelabel")
async def prelabel_videos(request: PrelabelRequest, background_tasks: BackgroundTasks):
    """Endpoint to initiate prelabeling of videos for a given task_id and project_id."""
    try:
        task_id = ObjectId(request.task_id)
        project_id = ObjectId(request.project_id)
        user_id = ObjectId(request.user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid task_id or project_id")

    # Fetch datapoints from MongoDB with status "created"
    query = {
        "task_id": task_id, 
        "project_id": project_id,
        "processingStatus": "created"
    }
    datapoints = list(datapoints_collection.find(query)) 

    if not datapoints:
        raise HTTPException(status_code=404, detail="No datapoints found with 'created' status for the given task_id and project_id")


    # Fetch custom prompt from users collection if available
    custom_prompt = None
    is_ads_project = False
    try:
        print(f"🔍 Looking for project_id: {project_id}")
        # Find user document where one of the projects has _id == project_id
        user = users_collection.find_one({"projects._id": project_id})
        print(f"🔍 Found user: {user is not None}")

        if user and "projects" in user:
            print(f"🔍 User has {len(user['projects'])} projects")
            for project in user["projects"]:
                if project.get("_id") == project_id:
                    print(f"🔍 Found matching project with labeledBy: {project.get('labeledBy')}")

                    # Check if labeledBy is ads or not
                    if project.get("labeledBy") == "ads":
                        is_ads_project = True
                        print("🎯 Labeled by ads")
                        
                        # Get PreLabelPrompt if it exists inside instruction
                        prelabel_prompt = project.get("instruction", {}).get("preLabelPrompt")
                        print(f"🔍 PreLabelPrompt exists: {prelabel_prompt is not None}")
                        if prelabel_prompt:
                            custom_prompt = prelabel_prompt
                            print(f"✅ Using custom preLabelPrompt for ads project: {custom_prompt[:200]}...")
                        else:
                            custom_prompt = VIDEO_ANALYSIS_PROMPT_ADS
                            print("✅ Using ADS default prompt")
                    else:
                        is_ads_project = False
                        print(f"🎯 Not ads project (labeledBy: {project.get('labeledBy')})")
                        # For non-ads projects, get PreLabelPrompt if it exists
                        prelabel_prompt = project.get("instruction", {}).get("preLabelPrompt")
                        if prelabel_prompt:
                            custom_prompt = prelabel_prompt
                            print(f"✅ Found custom prompt for non-ads project: {custom_prompt[:200]}...")
                        else:
                            custom_prompt = VIDEO_ANALYSIS_PROMPT
                            print("✅ Using default VIDEO_ANALYSIS_PROMPT for non-ads project")
                    break
        else:
            print("❌ No user found or user has no projects")
                    
    except Exception as e:
        print(f"❌ Error fetching custom prompt: {e}")

    # Update status to "pre-label" for all selected datapoints
    datapoint_ids = [dp["_id"] for dp in datapoints]
    datapoints_collection.update_many(
        {"_id": {"$in": datapoint_ids}},
        {"$set": {"processingStatus": "pre-label"}}
    )

    # Schedule processing in the background
    background_tasks.add_task(process_datapoints, datapoints, custom_prompt, user_id=user_id, is_ads_project=is_ads_project)
    return {"message": "Prelabeling started in the background"}

def update_pre_label_list(user_id: ObjectId, project_id: ObjectId):
    try:
        user = users_collection.find_one({"_id": user_id})
        projects = user.get("projects", [])
        
        for idx, ds in enumerate(projects):
            if ds["_id"] == project_id:
                project = ds
                break
        else:
            print("Project not found in user's projects.")
            return
        
        current_time = datetime.utcnow()
        interval = timedelta(hours=12)
        window_end_time = project["preLabelWindow"]

        time_difference = current_time - window_end_time
        interval_count = int(time_difference.total_seconds() // interval.total_seconds())
        interval_count = max(0, interval_count)

        if "preLabelList" not in project:
            project["preLabelList"] = []

        if time_difference.total_seconds() < 0:
            # Within window
            if not project["preLabelList"]:
                project["preLabelList"].append(1)
            else:
                project["preLabelList"][-1] += 1
        else:
            # Outside window
            project["preLabelList"].extend([0] * interval_count)
            project["preLabelList"].append(1)
            project["preLabelWindow"] = window_end_time + (interval * (interval_count + 1))

        # Update project in list
        projects[idx] = project

        # Update back in DB
        users_collection.update_one({"_id": user_id}, {"$set": {"projects": projects}})

    except Exception as e:
        print("Error in update_pre_label_list:", str(e))

def process_datapoints(datapoints, custom_prompt=None, user_id=None, is_ads_project=False):
    """Process datapoints and update their preLabel field in the database."""
    logger.info(f"Starting to process {len(datapoints)} datapoints")
    logger.info(f"🏷️ Project type: {'ADS' if is_ads_project else 'NON-ADS'}")
    
    for i, datapoint in enumerate(datapoints):
        video_path = datapoint["mediaUrl"]
        try:

            # Process video and get description using custom prompt if available
            logger.info(f"Processing video {i+1}/{len(datapoints)}: {video_path}")
            results = generator.process_videos([video_path], custom_prompt)
            description = results.get(video_path)
            if description:
                logger.info(f"Description received for video: {video_path}")

                try:
                    # Parse the JSON description
                    desc_json = json.loads(description)
                    
                    if is_ads_project:
                        # For ADS projects, handle the different response format
                        questions = []
                        
                        # Add dummy_question as the first question if it exists
                        if "dummy_question" in desc_json:
                            questions.append({
                                "q": desc_json["dummy_question"]["q"],
                                "a": desc_json["dummy_question"]["a"],
                                "textAnswers": [],
                                "mcqAnswers": []
                            })
                        
                        # Add the rest of the questions
                        if "questions" in desc_json:
                            for q in desc_json["questions"]:
                                questions.append({
                                    "q": q["q"],
                                    "a": q["a"],
                                    "textAnswers": [],
                                    "mcqAnswers": []
                                })
                        
                        prelabel = {
                            "questions": questions,
                            "keywords": desc_json.get("keywords", []),
                            "map_placement": desc_json.get("map_placement", {}).get("value", "") if isinstance(desc_json.get("map_placement", {}), dict) else desc_json.get("map_placement", ""),
                            "summary": desc_json.get("summary", "")
                        }
                    else:
                        # For non-ADS projects, use the original format
                        prelabel = {
                            "questions": [
                                {
                                    "q": q["q"],
                                    "a": q["a"],
                                    "textAnswers": [],
                                    "mcqAnswers": []
                                } for q in desc_json["questions"]
                            ],
                            "keywords": desc_json["keywords"],
                            "map_placement": desc_json["map_placement"],
                            "summary": desc_json["summary"]
                        }



                    # Update the datapoint's preLabel field and status
                    datapoints_collection.update_one(
                        {"_id": datapoint["_id"]},
                        {
                            "$set": {
                                "preLabel": prelabel,
                                "processingStatus": "live-label-mcq"
                            }
                        }
                    )
                    logger.info(f"✅ Successfully saved preLabel data to MongoDB for project_id: {datapoint['project_id']}")
                    update_pre_label_list(user_id,datapoint["project_id"])
                    
                    # Clean up temporary video file
                    if os.path.exists(video_path):
                        try:
                            os.remove(video_path)
                            print(f"🧹 Cleaned up temporary video file: {video_path}")
                        except Exception as cleanup_error:
                            print(f"Warning: Could not clean up temporary file {video_path}: {cleanup_error}")
                    
                    # Add cooling time after successful processing to prevent CPU overheating
                    print(f"💤 Cooling down CPU for 10 seconds...")
                    time.sleep(10)
                    
                except json.JSONDecodeError as e:
                    print(f"JSON Decode Error for {video_path}: {e}")
                    print(f"Response that failed to parse: {description}")
                    # Update status back to created if processing failed
                    datapoints_collection.update_one(
                        {"_id": datapoint["_id"]},
                        {"$set": {"processingStatus": "created"}}
                    )
                    # Clean up temporary video file
                    if os.path.exists(video_path):
                        try:
                            os.remove(video_path)
                            print(f"🧹 Cleaned up temporary video file: {video_path}")
                        except Exception as cleanup_error:
                            print(f"Warning: Could not clean up temporary file {video_path}: {cleanup_error}")
                except KeyError as e:
                    print(f"Missing key in JSON response for {video_path}: {e}")
                    print(f"Available keys: {list(desc_json.keys()) if 'desc_json' in locals() else 'Unable to parse JSON'}")
                    # Update status back to created if processing failed
                    datapoints_collection.update_one(
                        {"_id": datapoint["_id"]},
                        {"$set": {"processingStatus": "created"}}
                    )
                    # Clean up temporary video file
                    if os.path.exists(video_path):
                        try:
                            os.remove(video_path)
                            print(f"🧹 Cleaned up temporary video file: {video_path}")
                        except Exception as cleanup_error:
                            print(f"Warning: Could not clean up temporary file {video_path}: {cleanup_error}")
            else:
                print(f"No description generated for {video_path}")
                # Update status back to created if no description was generated
                datapoints_collection.update_one(
                    {"_id": datapoint["_id"]},
                    {"$set": {"processingStatus": "created"}}
                )
                # Clean up temporary video file
                if os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                        print(f"🧹 Cleaned up temporary video file: {video_path}")
                    except Exception as cleanup_error:
                        print(f"Warning: Could not clean up temporary file {video_path}: {cleanup_error}")
        except Exception as e:
            # Update status back to created if an error occurred
            datapoints_collection.update_one(
                {"_id": datapoint["_id"]},
                {"$set": {"processingStatus": "created"}}
            )
            # Clean up video_path if it was assigned before the error
            if 'video_path' in locals() and video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    print(f"🧹 Cleaned up temporary video file: {video_path}")
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temporary file {video_path}: {cleanup_error}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)