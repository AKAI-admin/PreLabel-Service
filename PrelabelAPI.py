from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
import os
import json
import cv2
import numpy as np
import requests
import base64
from transnetv2 import TransNetV2
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config.env')

app = FastAPI()

# Pydantic model for request validation
class PrelabelRequest(BaseModel):
    task_id: str
    project_id: str

# MongoDB connection
MONGODB_URL = os.getenv('MONGODB_URL')
if not MONGODB_URL:
    raise ValueError("MONGODB_URL environment variable is not set")

client = MongoClient(MONGODB_URL)
db = client["AkaiDb0"]
datapoints_collection = db["datapoints"]


# VideoDescriptionGenerator class
class VideoDescriptionGenerator:
    def __init__(self, transnet_model_dir, gpt_api_key):
        """Initialize the class with TransNetV2 model and GPT API key."""
        self.transnet_model = TransNetV2(model_dir=transnet_model_dir)
        self.gpt_api_key = gpt_api_key

    def extract_keyframes(self, video_path):
        """Extract keyframes from a video using TransNetV2."""
        frames = []
        vidcap = cv2.VideoCapture(video_path)
        if not vidcap.isOpened():
            print(f"Failed to open video: {video_path}")
            return None
        while True:
            success, image = vidcap.read()
            if not success:
                break
            frames.append(image)
        vidcap.release()

        if not frames:
            print(f"No frames extracted from: {video_path}")
            return None

        try:
            resized_frames = np.array([cv2.resize(frame, (48, 27)) for frame in frames], dtype=np.uint8)
            _, scene_predictions = self.transnet_model.predict_frames(resized_frames)
            scenes = TransNetV2.predictions_to_scenes(scene_predictions)
            key_frames = [frames[scene_start] for scene_start, _ in scenes]
            return key_frames
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return None

    def generate_description(self, keyframes):
        """Generate a description for a set of keyframes using the gpt-4o-mini API."""
        try:
            # Convert keyframes to base64
            image_contents = []
            for keyframe in keyframes:
                _, buffer = cv2.imencode('.jpg', keyframe)
                img_str = base64.b64encode(buffer.tobytes()).decode('utf-8')
                image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}})

            # Construct the API request
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.gpt_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": """
                                 
You are an AI assistant that is to assist in data labeling of videos. Please analyze these video keyframes and identify the following aspects:
1. Location (Where?)
2. Number of participants (Who?)
3. Event description (What?)
4. Timing (When?)
5. Objects present
6. Actions/Activities
7. Interactions
8. Scene type
9. Lighting conditions
10. Weather (if applicable)
11. Time of day
12. Camera perspective
13. Emotions/Expressions
14. Text in scene
15. Anomalies/Events
16. Background and Foreground
17. Occlusion
18. Scale and Size
19. Colors
20. Additional notes: Provide any other observations or details that might be relevant.
                                 
Please provide detailed descriptions for each aspect based on the keyframes.
For all of these aspect:
- Generate 5 relevant questions that include a combination of the aspect and answers
- Flag any uncertain information

Additional requirements:
- Generate relevant keywords describing the main elements and themes
- Create a detailed description of the video in as many words as possible that includes all aspects of the video
- Determine map placement from these options: Town, Village, Water body, Mountains, Snow, Road

Example:
For these keyframes:
- Frame 1: People sitting in a classroom setting
- Frame 2: A teacher writing on a blackboard
                                 
Instructions:
- Output ONLY raw JSON (no markdown, no ```json blocks, no extra text)
- The response must begin with `{` and end with `}`
- Do not add any explanation, introduction, or comments                           

Please provide output in exctly this format do not deviate from this json format:
                                 

{
    "questions": [
        {"q": "What is the setting and time of day for this educational activity?", "a": "An indoor classroom during daytime hours with natural lighting"},
        {"q": "How many participants are present and what are their roles in this learning environment?", "a": "One teacher and approximately 20 students engaged in a lecture format"},
        {"q": "What objects and equipment are visible in the classroom and how are they being used?", "a": "A whiteboard for instruction, desks and chairs arranged in rows, and a projector for visual aids"},
        {"q": "What is the camera perspective and how does it capture the classroom dynamics?", "a": "Static camera positioned at the front of the class, providing a clear view of both teacher and student interactions"},
        {"q": "What teaching methods and student interactions are visible in this educational setting?", "a": "Teacher-led instruction using board work, followed by student group discussions in a collaborative learning environment"}
    ],
    "keywords": ["education", "classroom", "teaching", "students", "group work", "learning"],
    "map_placement": {
        "value": "Town"
    },
    "summary": "The video depicts a daytime classroom scene with a teacher instructing approximately 20 students, followed by students engaging in group discussions. The classroom is equipped with desks, chairs, a whiteboard, and a projector. The lighting is well-lit with natural light, and the camera is static from the front of the class."
}

Please analyze the provided all keyframes and provide only one output combining all information in the same JSON structure
"""}
                            ] + image_contents
                        }
                    ]
                }
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error generating description: {e}")
            return None

    def process_videos(self, video_paths):
        """Process a batch of videos, extracting keyframes and generating one description per video."""
        results = {}
        for video_path in video_paths:
            keyframes = self.extract_keyframes(video_path)
            if keyframes is None:
                continue
            description = self.generate_description(keyframes)
            if description:
                results[video_path] = description
        return results

# Initialize the generator
TRANSNET_MODEL_DIR = os.getenv('TRANSNET_MODEL_DIR', './TransNetV2_Keyframe_Detection/transnetv2-weights/')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

generator = VideoDescriptionGenerator(TRANSNET_MODEL_DIR, OPENAI_API_KEY)

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