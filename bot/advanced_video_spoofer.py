#!/usr/bin/env python3
"""
Advanced TikTok Video Spoofer - Deep Transformation Engine
Implements sophisticated techniques to evade modern detection systems
"""

import os
import cv2
import uuid
import random
import shutil
import subprocess
import numpy as np
from datetime import datetime
import json
import tempfile

class AdvancedTikTokSpoofer:
    def __init__(self):
        self.ffmpeg_path = self._detect_ffmpeg()
        self.temp_dir = None
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _detect_ffmpeg(self):
        """Auto-detect FFmpeg path."""
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            windows_ffmpeg = "C:\\Tools\\FFmpeg\\ffmpeg.exe"
            if os.path.exists(windows_ffmpeg):
                return windows_ffmpeg
            else:
                raise RuntimeError("FFmpeg not found in PATH")
        return ffmpeg_path
    
    def create_deep_audio_transformation(self, video_path, output_path):
        """
        Apply deep audio transformations that break TikTok's audio fingerprinting.
        """
        print("🎵 Applying deep audio transformation...")
        
        # Multiple layered audio transformations
        audio_filters = []
        
        # 1. Subtle pitch shift (0.5% - imperceptible but breaks fingerprints)
        pitch_shift = random.uniform(0.995, 1.005)
        audio_filters.append(f"asetrate=44100*{pitch_shift},aresample=44100")
        
        # 2. Dynamic range compression (reduces peaks, changes waveform)
        audio_filters.append("compand=attacks=0.3:decays=0.8:points=-80/-80|-45/-45|-27/-25|-5/-5:soft-knee=0.01:gain=0")
        
        # 3. Subtle EQ adjustments (breaks frequency fingerprints)
        bass_adj = random.uniform(0.95, 1.05)
        treble_adj = random.uniform(0.98, 1.02)
        audio_filters.append(f"equalizer=f=100:width_type=h:width=50:g={int((bass_adj-1)*10)}")
        audio_filters.append(f"equalizer=f=8000:width_type=h:width=1000:g={int((treble_adj-1)*5)}")
        
        # 4. Micro-delay (0.01-0.05ms - breaks timing analysis)
        delay_ms = random.uniform(0.01, 0.05)
        audio_filters.append(f"adelay={delay_ms}")
        
        # 5. Subtle stereo imaging changes
        audio_filters.append("extrastereo=0.02")
        
        filter_chain = ",".join(audio_filters)
        
        cmd = [
            self.ffmpeg_path, "-i", video_path,
            "-af", filter_chain,
            "-c:v", "copy",  # Keep video unchanged for this step
            "-c:a", "aac", "-b:a", "128k",
            "-y", output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Audio transformation failed: {result.stderr.decode()}")
        
        return output_path
    
    def apply_advanced_visual_transformation(self, frame, frame_index, total_frames, video_id):
        """
        Apply advanced visual transformations that break visual fingerprinting.
        """
        h, w = frame.shape[:2]
        
        # 1. Dynamic micro-cropping with content-aware positioning
        crop_variance = random.uniform(0.002, 0.008)  # 0.2-0.8% crop
        crop_x = int(w * crop_variance)
        crop_y = int(h * crop_variance)
        
        # Intelligent cropping - avoid cropping important content areas
        if crop_x > 0 or crop_y > 0:
            frame = frame[crop_y:h-crop_y, crop_x:w-crop_x]
            frame = cv2.resize(frame, (w, h))
        
        # 2. Subtle perspective transformation (breaks geometric fingerprints)
        progress = frame_index / max(1, total_frames - 1)
        perspective_strength = 0.001 + 0.001 * np.sin(2 * np.pi * progress)
        
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = int(w * perspective_strength)
        dst_points = np.float32([
            [offset, 0], [w-offset, offset//2], 
            [offset//2, h-offset], [w-offset//2, h]
        ])
        
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        frame = cv2.warpPerspective(frame, perspective_matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        
        # 3. Advanced color space manipulation
        # Convert to HSV for more natural color shifts
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Hue shift (very subtle - changes color fingerprint)
        hue_shift = random.uniform(-2, 2)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Saturation micro-adjustment
        sat_factor = random.uniform(0.98, 1.02)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        
        # Value (brightness) with spatial variance
        value_base = random.uniform(0.995, 1.005)
        # Add spatial gradient (very subtle)
        y_gradient = np.linspace(1-0.01, 1+0.01, h).reshape(-1, 1)
        x_gradient = np.linspace(1-0.005, 1+0.005, w).reshape(1, -1)
        spatial_factor = value_base * y_gradient * x_gradient
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * spatial_factor, 0, 255)
        
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # 4. Advanced noise injection (breaks pixel-level analysis)
        # Film grain simulation
        grain_strength = random.uniform(1, 3)
        grain = np.random.normal(0, grain_strength, frame.shape).astype(np.float32)
        
        # Apply grain with luminance dependency (more realistic)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grain_mask = (gray / 255.0) * 0.5 + 0.5  # More grain in mid-tones
        
        for c in range(3):
            grain[:, :, c] *= grain_mask
        
        frame = frame.astype(np.float32) + grain
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # 5. Micro sharpening/blurring with edge detection
        # Detect edges and apply different processing to edges vs areas
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_mask = (edges > 0).astype(np.float32)
        edge_mask = cv2.GaussianBlur(edge_mask, (3, 3), 0)
        
        # Sharpen edges slightly, blur non-edges slightly
        sharpened = cv2.filter2D(frame, -1, np.array([[-0.1, -0.1, -0.1], 
                                                      [-0.1, 1.8, -0.1], 
                                                      [-0.1, -0.1, -0.1]]))
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # Blend based on edge mask
        for c in range(3):
            frame[:, :, c] = (frame[:, :, c] * (1 - edge_mask) + 
                             sharpened[:, :, c] * edge_mask * 0.3 +
                             blurred[:, :, c] * (1 - edge_mask) * 0.1 +
                             frame[:, :, c] * 0.9)
        
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        return frame
    
    def create_structural_transformation(self, video_path, output_path):
        """
        Apply structural transformations that change video composition.
        """
        print("🏗️ Applying structural transformation...")
        
        # Get video info
        probe_cmd = [self.ffmpeg_path, "-i", video_path, "-f", "null", "-"]
        probe_result = subprocess.run(probe_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        
        # Extract duration and frame rate
        import re
        duration_match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', probe_result.stderr)
        fps_match = re.search(r'(\d+\.?\d*)\s*fps', probe_result.stderr)
        
        if duration_match:
            hours, minutes, seconds = duration_match.groups()
            total_duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            total_duration = 30  # Default
            
        original_fps = float(fps_match.group(1)) if fps_match else 30.0
        
        # Structural transformation strategies
        transformations = []
        
        # 1. Micro speed variation (breaks temporal fingerprints)
        speed_segments = []
        segment_count = random.randint(3, 6)
        segment_duration = total_duration / segment_count
        
        for i in range(segment_count):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, total_duration)
            speed_factor = random.uniform(0.98, 1.02)  # Very subtle
            speed_segments.append((start_time, end_time, speed_factor))
        
        # Create video segments with different speeds
        segment_files = []
        for i, (start, end, speed) in enumerate(speed_segments):
            segment_file = os.path.join(self.temp_dir, f"segment_{i}.mp4")
            
            cmd = [
                self.ffmpeg_path, "-i", video_path,
                "-ss", str(start), "-t", str(end - start),
                "-filter:v", f"setpts={1/speed}*PTS",
                "-filter:a", f"atempo={speed}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-y", segment_file
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            if result.returncode == 0 and os.path.exists(segment_file):
                segment_files.append(segment_file)
        
        if not segment_files:
            # Fallback: just copy the file
            shutil.copy(video_path, output_path)
            return output_path
        
        # 2. Reassemble with micro-transitions
        # Create a concat file for FFmpeg
        concat_file = os.path.join(self.temp_dir, "concat_list.txt")
        with open(concat_file, 'w') as f:
            for segment in segment_files:
                f.write(f"file '{segment}'\n")
        
        # Concatenate segments
        cmd = [
            self.ffmpeg_path, "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy", "-y", output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if result.returncode != 0:
            # Fallback to simple copy
            shutil.copy(video_path, output_path)
        
        return output_path
    
    def apply_encoding_obfuscation(self, video_path, output_path):
        """
        Apply encoding patterns that don't match standard TikTok signatures.
        """
        print("🔐 Applying encoding obfuscation...")
        
        # Custom encoding profile that mimics real-world capture scenarios
        encoding_profiles = [
            {
                "name": "android_native",
                "params": [
                    "-c:v", "libx264", "-preset", "medium", "-profile:v", "high",
                    "-level", "4.1", "-pix_fmt", "yuv420p",
                    "-b:v", "3500k", "-maxrate", "4000k", "-bufsize", "6000k",
                    "-g", "25", "-keyint_min", "12", "-sc_threshold", "40",
                    "-bf", "3", "-b_strategy", "2", "-refs", "3",
                    "-me_method", "umh", "-subq", "8", "-trellis", "1",
                    "-flags", "+cgop", "-coder", "1"
                ]
            },
            {
                "name": "iphone_capture",
                "params": [
                    "-c:v", "libx264", "-preset", "fast", "-profile:v", "main",
                    "-level", "4.0", "-pix_fmt", "yuv420p",
                    "-b:v", "4000k", "-maxrate", "4500k", "-bufsize", "7000k",
                    "-g", "30", "-keyint_min", "15", "-sc_threshold", "35",
                    "-bf", "2", "-b_strategy", "1", "-refs", "2",
                    "-me_method", "hex", "-subq", "6", "-trellis", "0",
                    "-flags", "+cgop+mv4", "-coder", "0"
                ]
            },
            {
                "name": "dslr_export",
                "params": [
                    "-c:v", "libx264", "-preset", "slow", "-profile:v", "high",
                    "-level", "4.2", "-pix_fmt", "yuv420p",
                    "-b:v", "5000k", "-maxrate", "6000k", "-bufsize", "8000k",
                    "-g", "24", "-keyint_min", "12", "-sc_threshold", "45",
                    "-bf", "4", "-b_strategy", "2", "-refs", "4",
                    "-me_method", "tesa", "-subq", "9", "-trellis", "2",
                    "-flags", "+cgop+aic", "-coder", "1"
                ]
            }
        ]
        
        # Randomly select an encoding profile
        profile = random.choice(encoding_profiles)
        print(f"🎬 Using {profile['name']} encoding profile")
        
        # Audio encoding with variance
        audio_bitrates = ["96k", "128k", "160k"]
        audio_bitrate = random.choice(audio_bitrates)
        
        cmd = [
            self.ffmpeg_path, "-i", video_path
        ] + profile["params"] + [
            "-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2", "-ar", "44100",
            "-movflags", "+faststart",
            "-y", output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Encoding obfuscation failed: {result.stderr.decode()}")
        
        return output_path
    
    def inject_fake_metadata(self, video_path):
        """
        Inject realistic but fake metadata to break C2PA tracking.
        """
        print("📝 Injecting fake metadata...")
        
        # Realistic device profiles
        device_profiles = [
            {
                "make": "Apple",
                "model": "iPhone 15 Pro",
                "software": "iOS 17.2.1",
                "creation_app": "Camera"
            },
            {
                "make": "Samsung",
                "model": "Galaxy S24 Ultra",
                "software": "Android 14",
                "creation_app": "Camera"
            },
            {
                "make": "Google",
                "model": "Pixel 8 Pro",
                "software": "Android 14",
                "creation_app": "Camera"
            }
        ]
        
        profile = random.choice(device_profiles)
        
        # Generate realistic timestamp (within last 30 days)
        import random
        from datetime import timedelta
        
        now = datetime.now()
        days_ago = random.randint(1, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        
        fake_date = now - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        formatted_date = fake_date.strftime('%Y:%m:%d %H:%M:%S')
        
        # Apply metadata using exiftool if available
        exiftool_path = shutil.which("exiftool")
        if exiftool_path:
            cmd = [
                exiftool_path, "-overwrite_original",
                f"-CreateDate={formatted_date}",
                f"-ModifyDate={formatted_date}",
                f"-Make={profile['make']}",
                f"-Model={profile['model']}",
                f"-Software={profile['software']}",
                f"-CreationDate={formatted_date}",
                video_path
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return video_path
    
    def spoof_tiktok_video(self, input_path):
        """
        Main spoofing function that applies all transformations.
        """
        print("🚀 Starting advanced TikTok video spoofing...")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="tiktok_spoof_")
        
        try:
            # Generate unique output name
            spoof_id = uuid.uuid4().hex[:8]
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            final_output = os.path.join(self.output_dir, f"spoofed_{base_name}_{spoof_id}_tiktok_advanced.mp4")
            
            # Step 1: Deep audio transformation
            audio_output = os.path.join(self.temp_dir, "audio_transformed.mp4")
            self.create_deep_audio_transformation(input_path, audio_output)
            
            # Step 2: Advanced visual transformation using OpenCV
            visual_output = os.path.join(self.temp_dir, "visual_transformed.mp4")
            self._process_video_frames(audio_output, visual_output)
            
            # Step 3: Structural transformation
            structural_output = os.path.join(self.temp_dir, "structural_transformed.mp4")
            self.create_structural_transformation(visual_output, structural_output)
            
            # Step 4: Encoding obfuscation
            encoded_output = os.path.join(self.temp_dir, "encoded_transformed.mp4")
            self.apply_encoding_obfuscation(structural_output, encoded_output)
            
            # Step 5: Metadata injection
            self.inject_fake_metadata(encoded_output)
            
            # Final step: Move to output
            shutil.move(encoded_output, final_output)
            
            print(f"✅ Advanced TikTok spoofing complete: {os.path.basename(final_output)}")
            return final_output
            
        except Exception as e:
            print(f"❌ Advanced spoofing failed: {e}")
            return None
            
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def _process_video_frames(self, input_path, output_path):
        """
        Process video frames with OpenCV for advanced visual transformations.
        """
        print("🎨 Processing video frames with advanced visual transformations...")
        
        # Extract frames
        frames_dir = os.path.join(self.temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Extract frames with FFmpeg
        extract_cmd = [
            self.ffmpeg_path, "-i", input_path,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-q:v", "2",
            f"{frames_dir}/frame_%06d.png"
        ]
        
        result = subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if result.returncode != 0:
            raise RuntimeError("Frame extraction failed")
        
        # Get frame files
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        if not frame_files:
            raise RuntimeError("No frames extracted")
        
        total_frames = len(frame_files)
        video_id = uuid.uuid4().hex[:8]
        
        print(f"🔄 Processing {total_frames} frames...")
        
        # Process frames with advanced transformations
        for i, frame_file in enumerate(frame_files):
            if i % 50 == 0:  # Progress update
                print(f"⏳ Processed {i}/{total_frames} frames...")
            
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                continue
            
            # Apply advanced transformations
            transformed_frame = self.apply_advanced_visual_transformation(frame, i, total_frames, video_id)
            
            # Save transformed frame
            cv2.imwrite(frame_path, transformed_frame)
        
        # Reassemble video
        print("🎬 Reassembling video with transformed frames...")
        
        # Get original framerate
        probe_cmd = [self.ffmpeg_path, "-i", input_path, "-f", "null", "-"]
        probe_result = subprocess.run(probe_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        
        import re
        fps_match = re.search(r'(\d+\.?\d*)\s*fps', probe_result.stderr)
        original_fps = fps_match.group(1) if fps_match else "30"
        
        reassemble_cmd = [
            self.ffmpeg_path, "-y",
            "-framerate", original_fps,
            "-i", f"{frames_dir}/frame_%06d.png",
            "-i", input_path,  # For audio
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-shortest",
            output_path
        ]
        
        result = subprocess.run(reassemble_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Video reassembly failed: {result.stderr.decode()}")
        
        print("✅ Advanced visual transformation complete")
        return output_path


# Integration function for existing bulk processor
def spoof_video_advanced_tiktok(video_path):
    """
    Integration function for the bulk processor.
    """
    spoofer = AdvancedTikTokSpoofer()
    return spoofer.spoof_tiktok_video(video_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python advanced_video_spoofer.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    spoofer = AdvancedTikTokSpoofer()
    result = spoofer.spoof_tiktok_video(video_path)
    
    if result:
        print(f"✅ Success! Output: {result}")
    else:
        print("❌ Spoofing failed!")
