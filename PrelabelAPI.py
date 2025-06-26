from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
import os
import json
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
async def process_instructions(request: InstructionRequest):
    print("Received instructions:", request.instructions)
    
    try:
        # Create a prompt that includes the user's instructions
        prompt = f"""You are an AI assistant helping to process video labeling instructions.

The user has provided the following instructions for their video labeling project:

Instructions: {request.instructions}

Please analyze these instructions and provide guidance on how to implement them effectively for video data labeling. create and answer example question like prompt structure below:
{VIDEO_ANALYSIS_PROMPT}

Provide a structured response that can help guide the video labeling process."""
        
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