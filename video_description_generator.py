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

    def extract_keyframes(self, video_path, compression_quality=30, frame_skip=1, max_frames=1500):
        """Extract keyframe indices from a video path using compressed frames for scene detection only.
        
        Args:
            video_path: URL or local path to video
            compression_quality: JPEG compression quality (1-100, lower = more compression) - default 30 for 4K
            frame_skip: Skip every N frames for temporal compression (1 = process every 2nd frame)
            max_frames: Maximum number of frames to process (prevents memory overflow on long 4K videos)
            
        Returns:
            list: keyframe_indices - original frame positions of detected keyframes
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
            
            # print(f"Streaming video from: {video_path}")
            print(f"Video info: {total_frames} frames at {fps:.2f} FPS, Resolution: {width}x{height}")
            
            # Calculate compression settings - focus on quality reduction, not frame skipping
            is_4k = width >= 3840 or height >= 2160
            is_hd = width >= 1920 or height >= 1080
            
            if is_4k:
                print("4K video detected - applying aggressive quality compression")
                # For 4K: keep all frames but use very low quality
                actual_frame_skip = frame_skip  # No forced frame skipping
                actual_quality = min(compression_quality, 20)  # Slightly higher quality for better analysis
            elif is_hd:
                print("HD video detected - applying moderate quality compression")
                actual_frame_skip = frame_skip  # No forced frame skipping
                actual_quality = min(compression_quality, 28)  # Low quality for HD
            else:
                print("Standard resolution - using normal compression")
                actual_frame_skip = frame_skip
                actual_quality = max(compression_quality, 35)  # Ensure reasonable quality for lower res

            frames = []
            frame_indices = []  # Track actual frame indices for high-quality extraction
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
                frame_indices.append(frame_count - 1)  # Store the actual frame index (0-based)
                processed_count += 1
                
                # Progress reporting
                if processed_count % 50 == 0:
                    compression_ratio = (frame_count / processed_count) if processed_count > 0 else 1
                    # print(f"Processed {processed_count} frames (compression ratio: {compression_ratio:.1f}x)")
                
                # Memory management for 4K videos - more frequent since keeping all frames
                if is_4k and processed_count % 100 == 0:
                    import gc
                    gc.collect()  # More frequent garbage collection for 4K videos

            cap.release()

            if not frames:
                print(f"⚠️ No frames extracted from: {video_path}")
                return None

            print(f"Extracted {len(frames)} frames from stream")

            # Predict scene changes
            resized_frames = np.array(frames, dtype=np.uint8)
            _, scene_predictions = self.transnet_model.predict_frames(resized_frames)
            scenes = self.transnet_model.predictions_to_scenes(scene_predictions)

            # Get keyframe indices only
            keyframe_indices = [frame_indices[scene_start] for scene_start, _ in scenes]
            
            print(f"Keyframes extracted successfully: {len(keyframe_indices)} keyframes at indices {keyframe_indices}")
            return keyframe_indices

        except Exception as e:
            print(f"❌ Error extracting keyframes from stream: {e}")
            return None

    def generate_description(self, keyframes, custom_prompt=None):
        """Generate a description for a set of keyframes using the gpt-4o-mini API."""
        try:
            # Convert keyframes to base64
            image_contents = []
            for i, keyframe in enumerate(keyframes):
                _, buffer = cv2.imencode('.jpg', keyframe)
                img_str = base64.b64encode(buffer.tobytes()).decode('utf-8')
                image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}})
                print(f"Encoded keyframe {i+1}/{len(keyframes)}")

            # print("Making API request to OpenAI...")
            # Construct the API request
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.gpt_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "o4-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": custom_prompt}
                            ] + image_contents
                        }
                    ]
                }
            )
            # print(f"API response status: {response.status_code}")
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
            
            # Extract keyframe indices using compressed frames for scene detection
            keyframe_indices = self.extract_keyframes(video_path, compression_quality, frame_skip)
            if keyframe_indices is None:
                print(f"Failed to extract keyframes from {video_path}")
                continue
            
            # print(f"Detected {len(keyframe_indices)} keyframe indices from {video_path}")
            
            # Get high-quality keyframes at the identified indices
            high_quality_keyframes = self.get_high_quality_keyframes(video_path, keyframe_indices)
            if high_quality_keyframes is None:
                print(f"Failed to extract high-quality keyframes from {video_path}")
                continue
            
            # Use high-quality keyframes for description generation
            description = self.generate_description(high_quality_keyframes, custom_prompt)
            # print(f"Generated description")
            if description:
                results[video_path] = description
                # print(f"Successfully generated description")
            else:
                print(f"Failed to generate description for {video_path}")
        return results

    def get_high_quality_keyframes(self, video_path, keyframe_indices, quality=95):
        """Extract high-quality keyframes at specific frame indices without loading the entire video.
        
        Args:
            video_path: URL or local path to video
            keyframe_indices: List of frame indices to extract
            quality: JPEG quality for encoding (1-100, higher = better quality)
            
        Returns:
            list: High-quality keyframe images
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"❌ Could not open video stream for high-quality extraction: {video_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Extracting {len(keyframe_indices)} high-quality keyframes from {width}x{height} video")
            
            high_quality_frames = []
            
            # Sort indices to minimize seeking
            sorted_indices = sorted(keyframe_indices)
            
            for i, frame_idx in enumerate(sorted_indices):
                # Validate frame index
                if frame_idx >= total_frames:
                    print(f"⚠️ Frame index {frame_idx} exceeds video length ({total_frames}), skipping")
                    continue
                
                # Seek to specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"⚠️ Failed to read frame at index {frame_idx}")
                    continue
                
                # Use high-quality JPEG encoding
                jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                _, encoded_frame = cv2.imencode('.jpg', frame, jpeg_params)
                high_quality_frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
                
                high_quality_frames.append(high_quality_frame)
                print(f"✅ Extracted high-quality frame {i+1}/{len(sorted_indices)} at index {frame_idx}")
            
            cap.release()
            
            print(f"✅ Successfully extracted {len(high_quality_frames)} high-quality keyframes")
            return high_quality_frames
            
        except Exception as e:
            print(f"❌ Error extracting high-quality keyframes: {e}")
            return None
