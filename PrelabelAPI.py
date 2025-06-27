from fastapi import FastAPI, HTTPException, BackgroundTasks , Body
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
import os
import json
import yaml
import requests
from dotenv import load_dotenv
from video_description_generator import VideoDescriptionGenerator
from video_analysis_prompt import VIDEO_ANALYSIS_PROMPT

# Load environment variables
load_dotenv('config.env')

app = FastAPI()

# Pydantic model for request validation
class PrelabelRequest(BaseModel):
    task_id: str
    project_id: str

class InstructionRequest(BaseModel):
    instructions: str 

# MongoDB connection
MONGODB_URL = os.getenv('MONGO_URI')
if not MONGODB_URL:
    raise ValueError("MONGO_URI environment variable is not set")

client = MongoClient(MONGODB_URL)
db = client["AkaiDb0"]
datapoints_collection = db["datapoints"]


# Initialize the generator
TRANSNET_MODEL_DIR = os.getenv('TRANSNET_MODEL_DIR', './TransNetV2_Keyframe_Detection/transnetv2-weights/')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

generator = VideoDescriptionGenerator(TRANSNET_MODEL_DIR, OPENAI_API_KEY)


@app.post("/process-instructions")
async def process_instructions(instructions_yaml: str = Body(..., media_type="application/x-yaml")):
    
    try:
        instructions_data = yaml.safe_load(instructions_yaml)

        instructions = instructions_data.get("instructions", "")
        if not instructions:
            raise HTTPException(status_code=400, detail="Missing 'instructions' field in YAML.")

        # Create a prompt that includes the user's instructions
        prompt = f"""You are an AI assistant tasked with modifying a default prompt for analyzing video keyframes based on a user-provided paragraph that contains details about the dataset, special labeling instructions, and objects to focus on. Your goal is to update the default aspects list and example questions to align with the specific requirements of the dataset, ensuring high-quality labeling output.
Here is the default prompt you will modify:
kindly make sure to keep the same format as the default prompt given in string not any type of json: {VIDEO_ANALYSIS_PROMPT}

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
Paragraph Input :  {instructions}
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
            return openai_response
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

    # Update status to "pre-label" for all selected datapoints
    datapoint_ids = [dp["_id"] for dp in datapoints]
    datapoints_collection.update_many(
        {"_id": {"$in": datapoint_ids}},
        {"$set": {"processingStatus": "pre-label"}}
    )

    # Schedule processing in the background
    background_tasks.add_task(process_datapoints, datapoints)
    return {"message": "Prelabeling started in the background"}



def process_datapoints(datapoints):
    """Process datapoints and update their preLabel field in the database."""
    for datapoint in datapoints:
        video_path = datapoint["mediaUrl"]
        try:
            # Process video and get description
            results = generator.process_videos([video_path])
            description = results.get(video_path)
            if description:
                try:
                    # Parse the JSON description
                    desc_json = json.loads(description)
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
                except json.JSONDecodeError:
                    print(f"Failed to parse description for {video_path}")
                    # Update status back to created if processing failed
                    datapoints_collection.update_one(
                        {"_id": datapoint["_id"]},
                        {"$set": {"processingStatus": "created"}}
                    )
            else:
                print(f"No description generated for {video_path}")
                # Update status back to created if no description was generated
                datapoints_collection.update_one(
                    {"_id": datapoint["_id"]},
                    {"$set": {"processingStatus": "created"}}
                )
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            # Update status back to created if an error occurred
            datapoints_collection.update_one(
                {"_id": datapoint["_id"]},
                {"$set": {"processingStatus": "created"}}
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)