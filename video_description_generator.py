import cv2
import numpy as np
import requests
import base64
from transnetv2 import TransNetV2
from video_analysis_prompt import VIDEO_ANALYSIS_PROMPT

class VideoDescriptionGenerator:
    def __init__(self, transnet_model_dir, gpt_api_key):
        """Initialize the class with TransNetV2 model and GPT API key."""
        self.transnet_model = TransNetV2(model_dir=transnet_model_dir)
        self.gpt_api_key = gpt_api_key

    def extract_keyframes(self, video_path, compression_quality=30, frame_skip=2, max_frames=1000):
        """Extract keyframes from a video path using OpenCV VideoCapture with aggressive compression for 4K videos.
        
        Args:
            video_path: URL or local path to video
            compression_quality: JPEG compression quality (1-100, lower = more compression) - default 30 for 4K
            frame_skip: Skip every N frames for temporal compression (2 = process every 3rd frame)
            max_frames: Maximum number of frames to process (prevents memory overflow on long 4K videos)
        """
        
        # TransNetV2 input resolution - very small for 4K compression
        target_width, target_height = 48, 27

        try:
            # Open video stream directly with OpenCV (works with URLs)
            cap = cv2.VideoCapture(video_path)
            
            # Optimize buffer settings for streaming large videos
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent memory buildup
            
            if not cap.isOpened():
                print(f"❌ Could not open video stream: {video_path}")
                return None

            # Get video properties for optimization
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Streaming video from: {video_path}")
            print(f"Video info: {total_frames} frames at {fps:.2f} FPS, Resolution: {width}x{height}")
            
            # Calculate compression settings - focus on quality reduction, not frame skipping
            is_4k = width >= 3840 or height >= 2160
            is_hd = width >= 1920 or height >= 1080
            
            if is_4k:
                print("4K video detected - applying aggressive quality compression")
                # For 4K: keep all frames but use very low quality
                actual_frame_skip = frame_skip  # No forced frame skipping
                actual_quality = min(compression_quality, 15)  # Very low quality for 4K
            elif is_hd:
                print("HD video detected - applying moderate quality compression")
                actual_frame_skip = frame_skip  # No forced frame skipping
                actual_quality = min(compression_quality, 25)  # Low quality for HD
            else:
                print("Standard resolution - using normal compression")
                actual_frame_skip = frame_skip
                actual_quality = compression_quality

            frames = []
            frame_count = 0
            processed_count = 0
            
            # Enhanced JPEG compression parameters for aggressive quality reduction
            if is_4k:
                # Very aggressive compression for 4K
                jpeg_params = [
                    cv2.IMWRITE_JPEG_QUALITY, actual_quality,
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 1,  # Progressive JPEG for better compression
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1      # Optimize Huffman tables
                ]
            else:
                # Standard JPEG compression
                jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, actual_quality]
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Temporal compression: Skip frames
                if frame_count % (actual_frame_skip + 1) != 0:
                    continue
                
                # Early exit for long videos to prevent memory overflow
                if processed_count >= max_frames:
                    print(f"⚠️ Reached maximum frames limit ({max_frames}), stopping processing")
                    break
                
                # Spatial compression: Resize frame
                resized_frame = cv2.resize(frame, (target_width, target_height))
                
                # Quality compression: Encode as JPEG and decode back
                # This applies lossy compression to reduce memory usage
                _, encoded_frame = cv2.imencode('.jpg', resized_frame, jpeg_params)
                compressed_frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
                
                frames.append(compressed_frame)
                processed_count += 1
                
                # Progress reporting
                if processed_count % 50 == 0:
                    compression_ratio = (frame_count / processed_count) if processed_count > 0 else 1
                    print(f"Processed {processed_count} frames (compression ratio: {compression_ratio:.1f}x)")
                
                # Memory management for 4K videos - more frequent since keeping all frames
                if is_4k and processed_count % 100 == 0:
                    import gc
                    gc.collect()  # More frequent garbage collection for 4K videos

            cap.release()

            if not frames:
                print(f"⚠️ No frames extracted from: {video_path}")
                return None

            print(f"✅ Extracted {len(frames)} frames from stream")

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
                print(f"Successfully got response from OpenAI")
                return result
            else:
                print(f"API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error generating description: {e}")
            return None

    def process_videos(self, video_paths, custom_prompt=None, compression_quality=30, frame_skip=2):
        """Process a batch of videos, extracting keyframes and generating one description per video."""
        results = {}
        for video_path in video_paths:
            print(f"Processing video: {video_path}")
            keyframes = self.extract_keyframes(video_path, compression_quality, frame_skip)
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
