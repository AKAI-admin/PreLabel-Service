import cv2
import numpy as np
import requests
import base64
import subprocess
from transnetv2 import TransNetV2
from video_analysis_prompt import VIDEO_ANALYSIS_PROMPT

class VideoDescriptionGenerator:
    def __init__(self, transnet_model_dir, gpt_api_key):
        """Initialize the class with TransNetV2 model and GPT API key."""
        self.transnet_model = TransNetV2(model_dir=transnet_model_dir)
        self.gpt_api_key = gpt_api_key

    def extract_keyframes(self, video_path):
        """Extract keyframes from a video path using TransNetV2, streamed and downscaled via ffmpeg."""
        
        # TransNetV2 input resolution (can be changed if needed)
        target_width, target_height = 48, 27
        frame_size = target_width * target_height * 3  # 3 bytes per pixel for BGR

        # FFmpeg command to stream and resize video
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", video_path,             # input path/URL
            "-vf", f"scale={target_width}:{target_height}",  # scale to target size
            "-f", "rawvideo",            # raw video output
            "-pix_fmt", "bgr24",         # pixel format compatible with OpenCV
            "-vcodec", "rawvideo",       # raw output
            "-nostdin",                  # avoid ffmpeg waiting for input
            "-"                          # write to stdout
        ]

        try:
            # Start ffmpeg process
            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
            frames = []

            while True:
                raw_frame = pipe.stdout.read(frame_size)
                if not raw_frame:
                    break

                frame = np.frombuffer(raw_frame, np.uint8).reshape((target_height, target_width, 3))
                frames.append(frame)

            pipe.stdout.close()
            pipe.wait()

            if not frames:
                print(f"⚠️ No frames extracted from: {video_path}")
                return None

            # Predict scene changes
            resized_frames = np.array(frames, dtype=np.uint8)
            _, scene_predictions = self.transnet_model.predict_frames(resized_frames)
            scenes = self.transnet_model.predictions_to_scenes(scene_predictions)

            key_frames = [frames[scene_start] for scene_start, _ in scenes]
            print("✅ Keyframes extracted successfully")
            return key_frames

        except Exception as e:
            print(f"❌ Error extracting keyframes from stream: {e}")
            return None

    def generate_description(self, keyframes, custom_prompt=None):
        """Generate a description for a set of keyframes using the gpt-4o-mini API."""
        try:
            # Use custom prompt if provided, otherwise use default
            if custom_prompt:
                print(f"Using custom prompt")
                prompt_to_use = custom_prompt
            else:
                print(f"Using default prompt")
                prompt_to_use = VIDEO_ANALYSIS_PROMPT
            
            # Convert keyframes to base64
            image_contents = []
            for i, keyframe in enumerate(keyframes):
                _, buffer = cv2.imencode('.jpg', keyframe)
                img_str = base64.b64encode(buffer.tobytes()).decode('utf-8')
                image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}})
                print(f"Encoded keyframe {i+1}/{len(keyframes)}")

            print("Making API request to OpenAI...")
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
                                {"type": "text", "text": prompt_to_use}
                            ] + image_contents
                        }
                    ]
                }
            )
            print(f"API response status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content']
                print(f"Successfully got response from OpenAI: {result[:100]}...")
                return result
            else:
                print(f"API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error generating description: {e}")
            return None

    def process_videos(self, video_paths, custom_prompt=None):
        """Process a batch of videos, extracting keyframes and generating one description per video."""
        results = {}
        for video_path in video_paths:
            print(f"Processing video: {video_path}")
            keyframes = self.extract_keyframes(video_path)
            if keyframes is None:
                print(f"Failed to extract keyframes from {video_path}")
                continue
            print(f"Extracted {len(keyframes)} keyframes from {video_path}")
            description = self.generate_description(keyframes, custom_prompt)
            if description:
                results[video_path] = description
                print(f"Successfully generated description for {video_path}")
            else:
                print(f"Failed to generate description for {video_path}")
        return results
