from fastapi import FastAPI, HTTPException, BackgroundTasks , Body
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
import os
import json
import yaml
import requests
import time
from dotenv import load_dotenv
from video_description_generator import VideoDescriptionGenerator
from video_analysis_prompt import VIDEO_ANALYSIS_PROMPT
from fastapi.responses import PlainTextResponse
from datetime import datetime , timedelta
import subprocess
import uuid
import re
import ffmpeg
import cv2
import numpy as np
import tempfile
import shutil
from sewar.full_ref import mse, psnr, ssim



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
async def process_instructions(instructions: str = Body(..., media_type="text/plain")):
    
    try:
        # Use instructions directly instead of parsing JSON
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
    try:
        # Find user document where one of the projects has _id == project_id
        user = users_collection.find_one({"projects._id": project_id})
        
        if user and "projects" in user:
            for project in user["projects"]:
                if project.get("_id") == project_id:
                    # Get PreLabelPrompt if it exists inside instruction
                    custom_prompt = project.get("instruction", {}).get("preLabelPrompt")
                    if custom_prompt:
                        print(f"Found custom prompt: {custom_prompt[:200]}...")
                    else:
                        print("No custom prompt found, will use default")
                    break
                    
    except Exception as e:
        print(f"Error fetching custom prompt: {e}")

    # Update status to "pre-label" for all selected datapoints
    datapoint_ids = [dp["_id"] for dp in datapoints]
    datapoints_collection.update_many(
        {"_id": {"$in": datapoint_ids}},
        {"$set": {"processingStatus": "pre-label"}}
    )

    # Schedule processing in the background
    background_tasks.add_task(process_datapoints, datapoints, custom_prompt , user_id=user_id)
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

# def calculate_video_quality_metrics(original_path: str, compressed_path: str) -> dict:
#     """Calculate video quality metrics and file info using sewar library - comprehensive version."""
#     try:
#         # Get file sizes
#         original_size = os.path.getsize(original_path) / 1048576  # MB
#         compressed_size = os.path.getsize(compressed_path) / 1048576  # MB
        
#         cap_orig, cap_comp = cv2.VideoCapture(original_path), cv2.VideoCapture(compressed_path)
#         if not (cap_orig.isOpened() and cap_comp.isOpened()):
#             return {"error": "Could not open video files"}
        
#         # Sample 5 frames for quick analysis
#         frames_data = []
#         for _ in range(5):
#             ret_orig, frame_orig = cap_orig.read()
#             ret_comp, frame_comp = cap_comp.read()
#             if not (ret_orig and ret_comp): break
            
#             gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
#             gray_comp = cv2.cvtColor(frame_comp, cv2.COLOR_BGR2GRAY)
#             if gray_orig.shape != gray_comp.shape:
#                 gray_comp = cv2.resize(gray_comp, (gray_orig.shape[1], gray_orig.shape[0]))
            
#             frames_data.append({
#                 'psnr': psnr(gray_orig, gray_comp),
#                 'ssim': ssim(gray_orig, gray_comp),
#                 'mse': mse(gray_orig, gray_comp)
#             })
        
#         cap_orig.release()
#         cap_comp.release()
        
#         if not frames_data:
#             return {"error": "No frames processed"}
        
#         avg_metrics = {k: np.mean([f[k] for f in frames_data]) for k in ['psnr', 'ssim', 'mse']}
#         size_reduction = ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0
        
#         return {
#             "avg_psnr": round(avg_metrics['psnr'], 2),
#             "avg_ssim": round(avg_metrics['ssim'], 4),
#             "avg_mse": round(avg_metrics['mse'], 2),
#             "original_size_mb": round(original_size, 2),
#             "compressed_size_mb": round(compressed_size, 2),
#             "size_reduction_percent": round(size_reduction, 2),
#             "compression_ratio": round(original_size / compressed_size, 2) if compressed_size > 0 else 0,
#             "frames_analyzed": len(frames_data)
#         }
#     except Exception as e:
#         return {"error": f"Error: {str(e)}"}

def download_and_compress_video(video_url: str) -> str:
    """Download and compress video with cross-platform compatibility."""
    try:
        # Check if ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise Exception("ffmpeg is not installed or not in PATH")
        
        # Create cross-platform temporary files
        temp_dir = tempfile.gettempdir()
        local_filename = os.path.join(temp_dir, f"{uuid.uuid4()}.mp4")
        compressed_filename = os.path.join(temp_dir, f"compressed_{uuid.uuid4()}.mp4")

        # Download video with timeout and size limit
        print(f"Downloading video from: {video_url}")
        with requests.get(video_url, stream=True, timeout=300) as r:
            r.raise_for_status()
            
            # Check content length if available
            total_size = int(r.headers.get('content-length', 0))
            if total_size > 500 * 1024 * 1024:  # 500MB limit
                raise Exception(f"Video too large: {total_size / (1024*1024):.1f}MB (limit: 500MB)")
            
            with open(local_filename, 'wb') as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Safety check during download
                        if downloaded > 500 * 1024 * 1024:
                            raise Exception("Download size exceeded 500MB limit")
        
        print(f"Downloaded video to: {local_filename}")

        # Compress using ffmpeg with error handling
        print(f"Compressing video to: {compressed_filename}")
        cmd = [
            "ffmpeg", "-y",  # Overwrite output files
            "-i", local_filename,
            "-vcodec", "libx264", 
            "-crf", "28",
            "-preset", "fast",
            "-movflags", "+faststart",  # Optimize for streaming
            compressed_filename
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise Exception(f"ffmpeg compression failed: {result.stderr}")
        
        print(f"Compression completed.")

        # # Calculate comprehensive quality metrics
        # print("🔬 Analyzing video quality and compression efficiency...")
        # quality_metrics = calculate_video_quality_metrics(local_filename, compressed_filename)
        # if "error" not in quality_metrics:
        #     print(f"📊 Quality Analysis Results:")
        #     print(f"   • Original size: {quality_metrics['original_size_mb']} MB")
        #     print(f"   • Compressed size: {quality_metrics['compressed_size_mb']} MB")
        #     print(f"   • Size reduction: {quality_metrics['size_reduction_percent']}%")
        #     print(f"   • Compression ratio: {quality_metrics['compression_ratio']}:1")
        #     print(f"   • Average PSNR: {quality_metrics['avg_psnr']} dB")
        #     print(f"   • Average SSIM: {quality_metrics['avg_ssim']}")
        #     print(f"   • Average MSE: {quality_metrics['avg_mse']}")
        #     print(f"   • Frames analyzed: {quality_metrics['frames_analyzed']}")
        # else:
        #     print(f"⚠️ Could not calculate quality metrics: {quality_metrics['error']}")

        # Clean up original file
        try:
            os.remove(local_filename)
            print(f"🗑️ Cleaned up original file")
        except Exception as cleanup_error:
            print(f"⚠️ Could not clean up original file: {cleanup_error}")

        return compressed_filename
    except Exception as e:
        # Clean up any partial files
        for filepath in [locals().get('local_filename'), locals().get('compressed_filename')]:
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
        print(f"Error in downloading or compressing video: {e}")
        raise

def process_datapoints(datapoints, custom_prompt=None , user_id=None):
    """Process datapoints and update their preLabel field in the database."""
    print(f"Starting to process {len(datapoints)} datapoints")
    for i, datapoint in enumerate(datapoints):
        # video_path = datapoint["mediaUrl"]
        video_path_mongodb = datapoint["mediaUrl"] 
        try:
            print(f"Downloading and compressing video {i+1}/{len(datapoints)}: {video_path_mongodb}")
            video_path = download_and_compress_video(video_path_mongodb)

            # Process video and get description using custom prompt if available
            print(f"Processing video {i+1}/{len(datapoints)}: {video_path}")
            results = generator.process_videos([video_path], custom_prompt)
            print(f"Processing completed for video {i+1}")
            print(f"Results: {results}")
            description = results.get(video_path)
            if description:
                print(f"Description received: {description[:200]}...")  # Show first 200 chars
                try:
                    # Parse the JSON description
                    print(f"Attempting to parse JSON description...")
                    desc_json = json.loads(description)
                    print(f"Successfully parsed JSON with keys: {list(desc_json.keys())}")
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
                    print(f"✅ Successfully saved preLabel data to MongoDB for project_id: {datapoint['project_id']}")
                    update_pre_label_list(user_id,datapoint["project_id"])
                    
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
                except KeyError as e:
                    print(f"Missing key in JSON response for {video_path}: {e}")
                    print(f"Available keys: {list(desc_json.keys()) if 'desc_json' in locals() else 'Unable to parse JSON'}")
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