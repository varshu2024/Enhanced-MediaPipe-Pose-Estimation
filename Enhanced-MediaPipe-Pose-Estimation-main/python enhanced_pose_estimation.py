import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import copy

# Configuration
INPUT_VIDEO = "/content/Input_Video(1).mp4"  # Your uploaded video
OUTPUT_VIDEO = "Output_Video_Enhanced_v1.mp4"  # Output file name

# Enhanced MediaPipe Pose Configuration
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Initialize pose with maximum accuracy settings
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # Maximum complexity for best accuracy
    smooth_landmarks=True,
    enable_segmentation=True,  # Enable segmentation for better pose detection
    smooth_segmentation=True,
    min_detection_confidence=0.7,  # Increased from 0.5
    min_tracking_confidence=0.7   # Increased from 0.5
)

class PoseSmoother:
    """Smooth pose landmarks over time to reduce jitter"""
    def __init__(self, window_size=5, alpha=0.7):
        self.window_size = window_size
        self.alpha = alpha  # For exponential smoothing
        self.history = deque(maxlen=window_size)
        self.smoothed_landmarks = None
    
    def smooth_landmarks(self, landmarks):
        """Apply temporal smoothing to landmarks"""
        if landmarks is None:
            return self.smoothed_landmarks  # Return last known good pose
        
        # Add current landmarks to history
        self.history.append(copy.deepcopy(landmarks))
        
        if len(self.history) < 2:
            self.smoothed_landmarks = landmarks
            return landmarks
        
        # Create smoothed landmarks
        smoothed = copy.deepcopy(landmarks)
        
        # Apply weighted average smoothing
        for i, landmark in enumerate(smoothed.landmark):
            x_vals = []
            y_vals = []
            z_vals = []
            visibility_vals = []
            
            # Collect values from history
            for hist_landmarks in self.history:
                x_vals.append(hist_landmarks.landmark[i].x)
                y_vals.append(hist_landmarks.landmark[i].y)
                z_vals.append(hist_landmarks.landmark[i].z)
                visibility_vals.append(hist_landmarks.landmark[i].visibility)
            
            # Apply exponential weighted average (recent frames have more weight)
            weights = np.array([self.alpha ** (len(x_vals) - 1 - j) for j in range(len(x_vals))])
            weights = weights / weights.sum()
            
            landmark.x = np.average(x_vals, weights=weights)
            landmark.y = np.average(y_vals, weights=weights)
            landmark.z = np.average(z_vals, weights=weights)
            landmark.visibility = np.average(visibility_vals, weights=weights)
        
        self.smoothed_landmarks = smoothed
        return smoothed

def preprocess_frame(frame):
    """Preprocess frame for better pose detection"""
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    enhanced_frame = cv2.merge([l, a, b])
    enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_LAB2BGR)
    
    # Apply bilateral filter to reduce noise while preserving edges
    enhanced_frame = cv2.bilateralFilter(enhanced_frame, 9, 75, 75)
    
    return enhanced_frame

def interpolate_missing_landmarks(current_landmarks, previous_landmarks, confidence_threshold=0.3):
    """Interpolate landmarks with low confidence using previous frame"""
    if current_landmarks is None:
        return previous_landmarks
    
    if previous_landmarks is None:
        return current_landmarks
    
    interpolated = copy.deepcopy(current_landmarks)
    
    for i, landmark in enumerate(interpolated.landmark):
        # If visibility is low, use previous frame's landmark with decay
        if landmark.visibility < confidence_threshold and previous_landmarks:
            prev_landmark = previous_landmarks.landmark[i]
            # Weighted average favoring previous frame for low confidence
            weight = landmark.visibility / confidence_threshold
            landmark.x = landmark.x * weight + prev_landmark.x * (1 - weight)
            landmark.y = landmark.y * weight + prev_landmark.y * (1 - weight)
            landmark.z = landmark.z * weight + prev_landmark.z * (1 - weight)
    
    return interpolated

def create_skeleton_frame(landmarks, frame_width, frame_height, confidence_threshold=0.3):
    """Create a black frame with enhanced skeleton visualization"""
    skeleton_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    if landmarks:
        # Draw skeleton with variable thickness based on confidence
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_landmark = landmarks.landmark[start_idx]
            end_landmark = landmarks.landmark[end_idx]
            
            # Only draw if both landmarks have sufficient confidence
            if start_landmark.visibility > confidence_threshold and end_landmark.visibility > confidence_threshold:
                start_point = (int(start_landmark.x * frame_width), int(start_landmark.y * frame_height))
                end_point = (int(end_landmark.x * frame_width), int(end_landmark.y * frame_height))
                
                # Vary thickness based on average confidence
                avg_confidence = (start_landmark.visibility + end_landmark.visibility) / 2
                thickness = int(2 + 3 * avg_confidence)  # 2-5 pixels based on confidence
                
                # Color gradient based on confidence (white to cyan)
                color = (int(255 * avg_confidence), 255, 255)
                
                cv2.line(skeleton_frame, start_point, end_point, color, thickness, cv2.LINE_AA)
        
        # Draw landmarks with size based on confidence
        for landmark in landmarks.landmark:
            if landmark.visibility > confidence_threshold:
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                radius = int(3 + 4 * landmark.visibility)  # 3-7 pixels based on confidence
                color = (int(255 * landmark.visibility), 255, 255)
                cv2.circle(skeleton_frame, (x, y), radius, color, -1, cv2.LINE_AA)
    
    return skeleton_frame

def add_pose_confidence_bar(frame, avg_confidence, x=10, y=60, width=200, height=20):
    """Add a confidence indicator bar to the frame"""
    # Draw background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    
    # Draw confidence bar
    conf_width = int(width * avg_confidence)
    color = (0, int(255 * avg_confidence), int(255 * (1 - avg_confidence)))  # Green to red
    cv2.rectangle(frame, (x, y), (x + conf_width, y + height), color, -1)
    
    # Add text
    cv2.putText(frame, f"Confidence: {avg_confidence:.1%}", (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def main():
    # Open input video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output will be double width for side-by-side display
    output_width = frame_width * 2
    
    # Setup output video writer with better codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (output_width, frame_height)
    )
    
    print(f"📹 Processing {frame_count} frames from {INPUT_VIDEO}...")
    print(f"🎯 Output will be saved as: {OUTPUT_VIDEO}")
    print("🎭 Creating enhanced dancer + mirror skeleton video...")
    print("⚡ Enhanced accuracy mode enabled with:")
    print("   - Temporal smoothing")
    print("   - Frame preprocessing")
    print("   - Confidence-based visualization")
    print("   - Landmark interpolation")
    
    # Initialize smoother and tracking variables
    smoother = PoseSmoother(window_size=5, alpha=0.7)
    previous_landmarks = None
    processed_frames = 0
    total_confidence = 0
    frames_with_pose = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame for better detection
        enhanced_frame = preprocess_frame(frame)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        
        # Process frame for pose detection
        results = pose.process(rgb_frame)
        
        # Process landmarks
        if results.pose_landmarks:
            # Interpolate low confidence landmarks
            interpolated_landmarks = interpolate_missing_landmarks(
                results.pose_landmarks, 
                previous_landmarks,
                confidence_threshold=0.3
            )
            
            # Apply temporal smoothing
            smoothed_landmarks = smoother.smooth_landmarks(interpolated_landmarks)
            
            # Create mirrored landmarks
            mirrored_landmarks = copy.deepcopy(smoothed_landmarks)
            for landmark in mirrored_landmarks.landmark:
                landmark.x = 1.0 - landmark.x  # Mirror horizontally
            
            # Calculate average confidence
            avg_confidence = np.mean([lm.visibility for lm in smoothed_landmarks.landmark])
            total_confidence += avg_confidence
            frames_with_pose += 1
            
            # Create skeleton frame
            skeleton_frame = create_skeleton_frame(
                mirrored_landmarks, 
                frame_width, 
                frame_height,
                confidence_threshold=0.3
            )
            
            # Add confidence indicator
            skeleton_frame = add_pose_confidence_bar(skeleton_frame, avg_confidence)
            
            # Store for next frame
            previous_landmarks = smoothed_landmarks
            
        else:
            # Use previous frame's pose if available
            if previous_landmarks:
                mirrored_landmarks = copy.deepcopy(previous_landmarks)
                for landmark in mirrored_landmarks.landmark:
                    landmark.x = 1.0 - landmark.x
                
                skeleton_frame = create_skeleton_frame(
                    mirrored_landmarks, 
                    frame_width, 
                    frame_height,
                    confidence_threshold=0.3
                )
                cv2.putText(skeleton_frame, "Using previous frame", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                skeleton_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                cv2.putText(skeleton_frame, "No pose detected", (frame_width//4, frame_height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add labels
        cv2.putText(frame, "Original Dancer", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(skeleton_frame, "Enhanced Mirror Skeleton", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Create side-by-side display
        combined_frame = np.hstack([frame, skeleton_frame])
        
        # Add separator line
        cv2.line(combined_frame, (frame_width, 0), (frame_width, frame_height), (255, 255, 255), 2)
        
        # Save processed frame
        out.write(combined_frame)
        
        processed_frames += 1
        if processed_frames % 30 == 0:  # Progress update every 30 frames
            progress = (processed_frames / frame_count) * 100
            print(f"Progress: {progress:.1f}% ({processed_frames}/{frame_count} frames)")
    
    # Cleanup
    cap.release()
    out.release()
    pose.close()
    
    # Print statistics
    print(f"\n✅ Enhanced MediaPipe Pose Skeleton video saved as: {OUTPUT_VIDEO}")
    print(f"🎉 Processed {processed_frames} frames successfully!")
    print(f"📊 Statistics:")
    print(f"   - Frames with detected pose: {frames_with_pose}/{processed_frames}")
    if frames_with_pose > 0:
        print(f"   - Average pose confidence: {(total_confidence/frames_with_pose):.1%}")
    print("📺 Video shows: Original Dancer | Enhanced MediaPipe Pose Skeleton")

if __name__ == "__main__":
    main()