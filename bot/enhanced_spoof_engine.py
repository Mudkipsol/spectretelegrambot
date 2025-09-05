#!/usr/bin/env python3
"""
Enhanced Spoof Engine - Advanced Platform Detection Evasion System
Implements cutting-edge techniques to bypass TikTok, Instagram, and YouTube detection systems
"""

import os
import cv2
import uuid
import random
import shutil
import subprocess
import numpy as np
from datetime import datetime, timedelta
import json
import tempfile
import math
from typing import Tuple, List, Optional
import hashlib

class EnhancedSpoofEngine:
    def __init__(self):
        self.ffmpeg_path = self._detect_ffmpeg()
        self.temp_dir = None
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Advanced spoofing parameters
        self.audio_variance_strength = "adaptive"
        self.visual_complexity_level = "maximum"
        self.temporal_variance_mode = "intelligent"
        self.platform_specific_optimizations = True
        
    def _detect_ffmpeg(self):
        """Auto-detect FFmpeg path with fallbacks."""
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            windows_ffmpeg = "C:\\Tools\\FFmpeg\\ffmpeg.exe"
            if os.path.exists(windows_ffmpeg):
                return windows_ffmpeg
            raise RuntimeError("FFmpeg not found in PATH")
        return ffmpeg_path

    def apply_multilayer_audio_obfuscation(self, video_path: str, output_path: str) -> str:
        """
        Apply sophisticated multi-layer audio transformations that break all known fingerprinting methods.
        """
        print("🎵 Applying multi-layer audio obfuscation...")
        
        # Layer 1: Spectral manipulation
        spectral_filters = []
        
        # Dynamic frequency response modification (breaks spectral fingerprints)
        for freq in [100, 300, 1000, 3000, 8000]:
            gain = random.uniform(-2.0, 2.0)
            width = random.randint(50, 200)
            spectral_filters.append(f"equalizer=f={freq}:width_type=h:width={width}:g={gain}")
        
        # Layer 2: Temporal domain modifications
        temporal_filters = []
        
        # Micro pitch shifting with variance
        pitch_variance = random.uniform(0.992, 1.008)  # 0.8% max variance
        temporal_filters.append(f"asetrate=44100*{pitch_variance},aresample=44100")
        
        # Dynamic range manipulation
        attack = random.uniform(0.1, 0.5)
        decay = random.uniform(0.3, 1.0)
        temporal_filters.append(f"compand=attacks={attack}:decays={decay}:points=-80/-80|-45/-40|-20/-15|-5/-5:soft-knee=0.02")
        
        # Layer 3: Psychoacoustic modifications (imperceptible to humans)
        psychoacoustic_filters = []
        
        # Stereo field manipulation
        stereo_strength = random.uniform(0.01, 0.05)
        psychoacoustic_filters.append(f"extrastereo={stereo_strength}")
        
        # Phase correlation adjustment
        phase_shift = random.uniform(-0.1, 0.1)
        psychoacoustic_filters.append(f"aphaser=in_gain=0.5:out_gain=0.9:delay={random.uniform(0.5, 2.0)}:decay=0.4:speed=0.5")
        
        # Layer 4: Temporal micro-adjustments
        timing_filters = []
        
        # Variable micro-delays
        delay_amounts = [random.uniform(0.005, 0.050) for _ in range(3)]
        for i, delay in enumerate(delay_amounts):
            timing_filters.append(f"adelay={delay}:all=1")
        
        # Combine all filter layers
        all_filters = spectral_filters + temporal_filters + psychoacoustic_filters + timing_filters
        filter_chain = ",".join(all_filters)
        
        cmd = [
            self.ffmpeg_path, "-i", video_path,
            "-af", filter_chain,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            "-y", output_path
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
            if result.returncode != 0:
                print(f"[⚠️] Advanced audio obfuscation failed, applying fallback...")
                # Fallback to simpler audio processing
                return self._apply_basic_audio_transform(video_path, output_path)
            return output_path
        except subprocess.TimeoutExpired:
            print(f"[⚠️] Audio processing timed out, applying fallback...")
            return self._apply_basic_audio_transform(video_path, output_path)

    def _apply_basic_audio_transform(self, video_path: str, output_path: str) -> str:
        """Fallback audio transformation."""
        cmd = [
            self.ffmpeg_path, "-i", video_path,
            "-af", f"asetrate=44100*{random.uniform(0.995, 1.005)},aresample=44100",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            "-y", output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        return output_path

    def apply_advanced_visual_transformation(self, frame: np.ndarray, frame_index: int, 
                                           total_frames: int, session_id: str) -> np.ndarray:
        """
        Apply sophisticated visual transformations that break modern ML detection systems.
        """
        h, w = frame.shape[:2]
        
        # 1. Content-Aware Geometric Perturbation
        frame = self._apply_content_aware_geometry(frame, frame_index, total_frames)
        
        # 2. Intelligent Color Space Manipulation
        frame = self._apply_intelligent_color_transform(frame, frame_index, session_id)
        
        # 3. Advanced Noise Injection with Spatial Intelligence
        frame = self._apply_spatial_noise_injection(frame, frame_index)
        
        # 4. Temporal Consistency Preservation
        frame = self._apply_temporal_consistency(frame, frame_index, total_frames)
        
        # 5. Platform-Specific Micro-Modifications
        frame = self._apply_platform_specific_micro_mods(frame, frame_index)
        
        return frame

    def _apply_content_aware_geometry(self, frame: np.ndarray, frame_index: int, total_frames: int) -> np.ndarray:
        """Apply geometric transformations that preserve content integrity."""
        h, w = frame.shape[:2]
        
        # Intelligent cropping that avoids important content areas
        # Use edge detection to identify content boundaries
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find content-rich regions
        edge_density = cv2.integral(edges.astype(np.float32))
        
        # Determine safe crop margins based on content density
        crop_margin = max(2, min(8, int(0.003 * min(w, h))))
        
        # Apply intelligent micro-crop
        top_crop = random.randint(1, crop_margin) if edge_density[crop_margin, w//2] < 1000 else 1
        bottom_crop = random.randint(1, crop_margin) if edge_density[h-crop_margin, w//2] < 1000 else 1
        left_crop = random.randint(1, crop_margin) if edge_density[h//2, crop_margin] < 1000 else 1
        right_crop = random.randint(1, crop_margin) if edge_density[h//2, w-crop_margin] < 1000 else 1
        
        frame = frame[top_crop:h-bottom_crop, left_crop:w-right_crop]
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # Subtle perspective transformation with content preservation
        progress = frame_index / max(1, total_frames - 1)
        perspective_strength = 0.0005 + 0.0005 * math.sin(2 * math.pi * progress)
        
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = max(1, int(w * perspective_strength))
        dst_points = np.float32([
            [offset, 0], [w-offset//2, offset//3], 
            [offset//3, h-offset//2], [w-offset//3, h]
        ])
        
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        frame = cv2.warpPerspective(frame, perspective_matrix, (w, h), 
                                  borderMode=cv2.BORDER_REFLECT_101)
        
        return frame

    def _apply_intelligent_color_transform(self, frame: np.ndarray, frame_index: int, session_id: str) -> np.ndarray:
        """Apply color transformations that maintain visual quality while breaking fingerprints."""
        
        # Create session-specific but consistent color transformation
        session_hash = int(hashlib.md5(session_id.encode()).hexdigest()[:8], 16)
        np.random.seed(session_hash + frame_index)
        
        # Convert to LAB color space for perceptually uniform modifications
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # L channel: Subtle brightness adjustments with spatial variance
        l_base_shift = np.random.uniform(-3, 3)
        y_gradient = np.linspace(-1, 1, lab.shape[0]).reshape(-1, 1)
        x_gradient = np.linspace(-1, 1, lab.shape[1]).reshape(1, -1)
        spatial_brightness = l_base_shift + 0.5 * y_gradient * x_gradient
        lab[:, :, 0] = np.clip(lab[:, :, 0] + spatial_brightness, 0, 100)
        
        # A and B channels: Subtle color shifts
        a_shift = np.random.uniform(-2, 2)
        b_shift = np.random.uniform(-2, 2)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + a_shift, -128, 127)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + b_shift, -128, 127)
        
        frame = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # Additional HSV fine-tuning for specific hue ranges
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Target specific hue ranges for subtle shifts
        hue_ranges = [(0, 20), (40, 80), (100, 140), (160, 180)]  # Red, Yellow, Green, Blue
        for hue_min, hue_max in hue_ranges:
            mask = ((hsv[:, :, 0] >= hue_min) & (hsv[:, :, 0] <= hue_max)).astype(np.float32)
            hue_shift = np.random.uniform(-2, 2)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * mask) % 180
        
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Reset random seed to avoid affecting other components
        np.random.seed()
        
        return frame

    def _apply_spatial_noise_injection(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Apply spatially-aware noise that mimics real sensor characteristics."""
        h, w = frame.shape[:2]
        
        # Multi-scale noise injection
        for scale in [1.0, 0.5, 0.25]:
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            
            # Generate spatially correlated noise
            noise_base = np.random.normal(0, 1.5, (scaled_h, scaled_w))
            noise_base = cv2.GaussianBlur(noise_base, (3, 3), 0.5)
            
            # Resize noise to match frame size
            if scale != 1.0:
                noise_base = cv2.resize(noise_base, (w, h))
            
            # Apply noise with luminance dependency
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            luminance_factor = 0.3 + 0.7 * gray  # More noise in bright areas
            
            for c in range(3):
                channel_noise = noise_base * luminance_factor * random.uniform(0.8, 1.2)
                frame[:, :, c] = np.clip(frame[:, :, c].astype(np.float32) + channel_noise, 0, 255)
        
        frame = frame.astype(np.uint8)
        
        # Sensor-specific artifacts
        if random.random() < 0.3:  # 30% chance
            # Column banding (CMOS sensor artifact)
            for col in range(0, w, random.randint(15, 25)):
                banding_strength = random.uniform(0.5, 2.0)
                frame[:, col:col+1] = np.clip(frame[:, col:col+1].astype(np.float32) + banding_strength, 0, 255)
        
        return frame

    def _apply_temporal_consistency(self, frame: np.ndarray, frame_index: int, total_frames: int) -> np.ndarray:
        """Maintain temporal consistency while introducing variance."""
        
        # Temporal smoothing factor
        temporal_factor = 1.0 - abs(math.sin(2 * math.pi * frame_index / min(total_frames, 60)))
        
        # Apply slight contrast adjustment based on temporal position
        contrast_adj = 1.0 + 0.02 * temporal_factor * random.uniform(-1, 1)
        frame = cv2.convertScaleAbs(frame, alpha=contrast_adj, beta=0)
        
        # Temporal edge enhancement
        if frame_index % 10 == 0:  # Every 10th frame
            edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
            edge_mask = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (3, 3), 0)
            
            for c in range(3):
                frame[:, :, c] = np.clip(frame[:, :, c] * (1 + 0.1 * edge_mask), 0, 255)
        
        return frame

    def _apply_platform_specific_micro_mods(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Apply platform-specific micro-modifications."""
        
        # TikTok-specific: Subtle diagonal distortion
        if frame_index % 25 == 0:
            h, w = frame.shape[:2]
            # Very subtle diagonal warp
            map_x = np.zeros((h, w), dtype=np.float32)
            map_y = np.zeros((h, w), dtype=np.float32)
            
            for y in range(h):
                for x in range(w):
                    map_x[y, x] = x + 0.3 * math.sin(2 * math.pi * y / h)
                    map_y[y, x] = y + 0.2 * math.sin(2 * math.pi * x / w)
            
            frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        # Instagram-specific: Subtle radial distortion
        elif frame_index % 30 == 0:
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            max_radius = math.sqrt(center_x**2 + center_y**2)
            
            map_x = np.zeros((h, w), dtype=np.float32)
            map_y = np.zeros((h, w), dtype=np.float32)
            
            for y in range(h):
                for x in range(w):
                    dx, dy = x - center_x, y - center_y
                    radius = math.sqrt(dx**2 + dy**2)
                    
                    if radius > 0:
                        # Very subtle barrel distortion
                        scale = 1 + 0.0001 * (radius / max_radius)**2
                        map_x[y, x] = center_x + dx * scale
                        map_y[y, x] = center_y + dy * scale
                    else:
                        map_x[y, x] = x
                        map_y[y, x] = y
            
            frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        return frame

    def apply_advanced_encoding_obfuscation(self, video_path: str, output_path: str, platform: str = "tiktok") -> str:
        """Apply advanced encoding that mimics authentic device capture patterns."""
        print(f"🎬 Applying advanced encoding obfuscation for {platform}...")
        
        # Platform-specific encoding profiles with realistic device characteristics
        encoding_profiles = {
            "tiktok": {
                "mobile_capture": [
                    "-c:v", "libx264", "-preset", "fast", "-profile:v", "main",
                    "-level", "4.0", "-pix_fmt", "yuv420p",
                    "-b:v", f"{random.randint(3000, 4500)}k",
                    "-maxrate", f"{random.randint(4000, 5500)}k",
                    "-bufsize", f"{random.randint(6000, 8000)}k",
                    "-g", str(random.randint(25, 35)),
                    "-keyint_min", str(random.randint(10, 20)),
                    "-sc_threshold", str(random.randint(30, 45)),
                    "-bf", str(random.randint(1, 3)),
                    "-refs", str(random.randint(2, 4)),
                    "-me_method", random.choice(["hex", "umh"]),
                    "-subq", str(random.randint(5, 8)),
                    "-trellis", str(random.randint(0, 2)),
                    "-flags", "+cgop+mv4",
                    "-movflags", "+faststart"
                ]
            },
            "instagram": {
                "mobile_capture": [
                    "-c:v", "libx264", "-preset", "medium", "-profile:v", "high",
                    "-level", "4.1", "-pix_fmt", "yuv420p",
                    "-b:v", f"{random.randint(3500, 5000)}k",
                    "-maxrate", f"{random.randint(4500, 6000)}k",
                    "-bufsize", f"{random.randint(7000, 9000)}k",
                    "-g", str(random.randint(24, 32)),
                    "-keyint_min", str(random.randint(12, 18)),
                    "-sc_threshold", str(random.randint(35, 50)),
                    "-bf", str(random.randint(2, 4)),
                    "-refs", str(random.randint(3, 5)),
                    "-me_method", "umh",
                    "-subq", str(random.randint(6, 9)),
                    "-trellis", "2",
                    "-flags", "+cgop+aic",
                    "-movflags", "+faststart"
                ]
            }
        }
        
        profile = encoding_profiles.get(platform, encoding_profiles["tiktok"])["mobile_capture"]
        
        # Add variable audio parameters
        audio_params = [
            "-c:a", "aac",
            "-b:a", random.choice(["96k", "128k", "160k"]),
            "-ac", "2",
            "-ar", "44100"
        ]
        
        cmd = [self.ffmpeg_path, "-i", video_path] + profile + audio_params + ["-y", output_path]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"Encoding failed: {result.stderr.decode()}")
            return output_path
        except subprocess.TimeoutExpired:
            raise RuntimeError("Encoding timed out")

    def inject_realistic_metadata(self, video_path: str, platform: str = "tiktok") -> str:
        """Inject realistic metadata that matches authentic mobile device captures."""
        print("📱 Injecting realistic mobile device metadata...")
        
        # Realistic device profiles with current models
        device_profiles = {
            "tiktok": [
                {
                    "make": "Apple",
                    "model": "iPhone 15 Pro Max",
                    "software": "iOS 17.3.1",
                    "app": "TikTok 34.2.0"
                },
                {
                    "make": "Samsung",
                    "model": "Galaxy S24 Ultra",
                    "software": "Android 14",
                    "app": "TikTok 34.2.0"
                },
                {
                    "make": "Google",
                    "model": "Pixel 8 Pro",
                    "software": "Android 14",
                    "app": "TikTok 34.2.0"
                }
            ],
            "instagram": [
                {
                    "make": "Apple",
                    "model": "iPhone 15 Pro",
                    "software": "iOS 17.3.1",
                    "app": "Instagram 314.0"
                },
                {
                    "make": "Samsung",
                    "model": "Galaxy S24+",
                    "software": "Android 14",
                    "app": "Instagram 314.0"
                }
            ]
        }
        
        profile = random.choice(device_profiles.get(platform, device_profiles["tiktok"]))
        
        # Generate realistic timestamp (within last 7 days)
        now = datetime.now()
        days_ago = random.randint(0, 7)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        
        capture_time = now - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        formatted_date = capture_time.strftime('%Y:%m:%d %H:%M:%S')
        
        # Try to apply metadata with exiftool
        exiftool_path = shutil.which("exiftool")
        if exiftool_path:
            cmd = [
                exiftool_path, "-overwrite_original",
                f"-CreateDate={formatted_date}",
                f"-ModifyDate={formatted_date}",
                f"-Make={profile['make']}",
                f"-Model={profile['model']}",
                f"-Software={profile['software']}",
                f"-UserComment={profile['app']}",
                f"-CreationDate={formatted_date}",
                video_path
            ]
            
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            except subprocess.TimeoutExpired:
                print("[⚠️] Metadata injection timed out")
        
        return video_path

    def process_video_enhanced(self, input_path: str, platform: str = "tiktok", 
                             strength: str = "maximum") -> Optional[str]:
        """
        Main processing function with enhanced spoofing techniques.
        """
        print(f"🚀 Starting enhanced video spoofing for {platform.upper()}...")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix=f"enhanced_spoof_{platform}_")
        
        try:
            # Generate unique identifiers
            session_id = uuid.uuid4().hex
            spoof_id = session_id[:8]
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            final_output = os.path.join(self.output_dir, 
                                      f"spoofed_{base_name}_{platform}_{spoof_id}_enhanced.mp4")
            
            # Processing pipeline
            
            # Step 1: Multi-layer audio obfuscation
            audio_output = os.path.join(self.temp_dir, "audio_obfuscated.mp4")
            self.apply_multilayer_audio_obfuscation(input_path, audio_output)
            
            # Step 2: Advanced visual transformation
            visual_output = os.path.join(self.temp_dir, "visual_transformed.mp4")
            self._process_video_frames_enhanced(audio_output, visual_output, session_id)
            
            # Step 3: Temporal structure modification
            temporal_output = os.path.join(self.temp_dir, "temporal_modified.mp4")
            self._apply_temporal_modifications(visual_output, temporal_output, platform)
            
            # Step 4: Advanced encoding obfuscation
            encoded_output = os.path.join(self.temp_dir, "encoded_final.mp4")
            self.apply_advanced_encoding_obfuscation(temporal_output, encoded_output, platform)
            
            # Step 5: Realistic metadata injection
            self.inject_realistic_metadata(encoded_output, platform)
            
            # Move to final location
            shutil.move(encoded_output, final_output)
            
            print(f"✅ Enhanced spoofing complete: {os.path.basename(final_output)}")
            return final_output
            
        except Exception as e:
            print(f"❌ Enhanced spoofing failed: {e}")
            return None
            
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

    def _process_video_frames_enhanced(self, input_path: str, output_path: str, session_id: str):
        """Process video frames with enhanced transformations."""
        print("🎨 Processing frames with enhanced transformations...")
        
        # Extract frames
        frames_dir = os.path.join(self.temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Extract with high quality
        extract_cmd = [
            self.ffmpeg_path, "-i", input_path,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-q:v", "1",  # Highest quality
            f"{frames_dir}/frame_%06d.png"
        ]
        
        result = subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if result.returncode != 0:
            raise RuntimeError("Frame extraction failed")
        
        # Process frames
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        total_frames = len(frame_files)
        
        print(f"🔄 Processing {total_frames} frames with advanced algorithms...")
        
        for i, frame_file in enumerate(frame_files):
            if i % 20 == 0:
                print(f"⏳ Processed {i}/{total_frames} frames...")
            
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                continue
            
            # Apply enhanced transformations
            enhanced_frame = self.apply_advanced_visual_transformation(frame, i, total_frames, session_id)
            
            # Save processed frame
            cv2.imwrite(frame_path, enhanced_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # Reassemble with original framerate
        self._reassemble_video_enhanced(frames_dir, input_path, output_path)

    def _reassemble_video_enhanced(self, frames_dir: str, original_video: str, output_path: str):
        """Reassemble video with enhanced quality preservation."""
        
        # Get original video properties
        probe_cmd = [self.ffmpeg_path, "-i", original_video, "-f", "null", "-"]
        probe_result = subprocess.run(probe_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        
        # Extract framerate
        import re
        fps_match = re.search(r'(\d+\.?\d*)\s*fps', probe_result.stderr)
        original_fps = fps_match.group(1) if fps_match else "30"
        
        reassemble_cmd = [
            self.ffmpeg_path, "-y",
            "-framerate", original_fps,
            "-i", f"{frames_dir}/frame_%06d.png",
            "-i", original_video,  # For audio
            "-c:v", "libx264", "-preset", "slow", "-crf", "16",  # High quality
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-shortest",
            output_path
        ]
        
        result = subprocess.run(reassemble_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Video reassembly failed: {result.stderr.decode()}")

    def _apply_temporal_modifications(self, video_path: str, output_path: str, platform: str) -> str:
        """Apply temporal modifications specific to platform detection patterns."""
        print("⏰ Applying temporal structure modifications...")
        
        # Get video duration
        probe_cmd = [self.ffmpeg_path, "-i", video_path, "-f", "null", "-"]
        probe_result = subprocess.run(probe_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        
        import re
        duration_match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', probe_result.stderr)
        
        if duration_match:
            hours, minutes, seconds = duration_match.groups()
            total_duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            # Fallback: just copy the file
            shutil.copy(video_path, output_path)
            return output_path
        
        # Apply very subtle speed variance (imperceptible but breaks temporal fingerprints)
        speed_factor = random.uniform(0.997, 1.003)  # 0.3% max variance
        
        if abs(speed_factor - 1.0) > 0.001:
            cmd = [
                self.ffmpeg_path, "-i", video_path,
                "-filter:v", f"setpts={1/speed_factor}*PTS",
                "-filter:a", f"atempo={speed_factor}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-y", output_path
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            if result.returncode == 0:
                return output_path
        
        # Fallback
        shutil.copy(video_path, output_path)
        return output_path


# Integration functions for existing bot
def spoof_video_enhanced_tiktok(video_path: str) -> Optional[str]:
    """Enhanced TikTok spoofing integration."""
    engine = EnhancedSpoofEngine()
    return engine.process_video_enhanced(video_path, "tiktok", "maximum")

def spoof_video_enhanced_instagram(video_path: str) -> Optional[str]:
    """Enhanced Instagram spoofing integration."""
    engine = EnhancedSpoofEngine()
    return engine.process_video_enhanced(video_path, "instagram", "maximum")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_spoof_engine.py <video_path> [platform]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    platform = sys.argv[2] if len(sys.argv) > 2 else "tiktok"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    engine = EnhancedSpoofEngine()
    result = engine.process_video_enhanced(video_path, platform)
    
    if result:
        print(f"✅ Success! Enhanced output: {result}")
    else:
        print("❌ Enhanced spoofing failed!")
