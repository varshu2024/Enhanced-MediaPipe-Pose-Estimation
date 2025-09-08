# Enhanced MediaPipe Pose Estimation 🕺💃

An advanced pose estimation system that creates high-quality skeleton visualizations from dance videos using MediaPipe with enhanced accuracy and temporal smoothing.

## 🎬 Demo Video

**See it in action:** [Enhanced MediaPipe Pose Estimation Demo](https://youtu.be/tH-Nks6i70g)

Watch how the system transforms dance videos into smooth, accurate skeleton visualizations with real-time confidence indicators!

## ✨ Features

- **Enhanced Accuracy**: Maximum complexity MediaPipe settings with increased confidence thresholds (70% vs default 50%)
- **Temporal Smoothing**: Advanced pose smoothing algorithm to reduce jitter across frames using exponential weighted averaging
- **Smart Interpolation**: Intelligent landmark interpolation for low-confidence detections using previous frame data
- **Frame Preprocessing**: CLAHE enhancement and bilateral filtering for improved pose detection in challenging lighting
- **Confidence Visualization**: Real-time confidence indicators and adaptive skeleton rendering based on detection quality
- **Mirror Mode**: Creates mirrored skeleton visualization alongside original video for detailed analysis
- **Robust Tracking**: Fallback mechanisms for missed detections maintaining pose continuity

## 🎯 Results

**📹 [Watch the Demo Video](https://youtu.be/tH-Nks6i70g)** - See the enhanced pose estimation in action!

The system produces side-by-side videos showing:
- **Left**: Original dancer video
- **Right**: Enhanced skeleton visualization with confidence-based rendering

**Key improvements over standard MediaPipe:**
- ✅ Significantly reduced skeleton jitter through temporal smoothing
- ✅ Smart handling of occlusions and low-confidence detections  
- ✅ Confidence-based visual feedback for quality assessment
- ✅ Robust tracking that maintains pose continuity

## 🚀 Installation & Quick Start

### Prerequisites

```bash
pip install opencv-python mediapipe numpy
```

### Usage

1. **Clone the repository**:
   ```bash
   https://github.com/amit-chakdhare09/Enhanced-MediaPipe-Pose-Estimation.git
   cd enhanced-mediapipe-pose
   ```

2. **Update the input video path**:
   ```python
   INPUT_VIDEO = "path/to/your/video.mp4"
   ```

3. **Run the processing or Run on Colab**:
   ```bash
   python enhanced_pose_estimation.py
   ```

4. **Output**: Enhanced video will be saved as `Output_Video_Enhanced_v1.mp4`

## ⚙️ Configuration

### MediaPipe Settings
```python
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,              # Maximum accuracy (0-2)
    smooth_landmarks=True,           # Built-in smoothing
    enable_segmentation=True,        # Better pose detection
    smooth_segmentation=True,
    min_detection_confidence=0.7,    # Increased from default 0.5
    min_tracking_confidence=0.7      # Increased from default 0.5
)
```

### Temporal Smoothing Parameters
```python
smoother = PoseSmoother(
    window_size=5,                   # Frames for averaging (3-10 recommended)
    alpha=0.7                        # Exponential smoothing factor (0.5-0.9)
)
```

### Visualization Thresholds
```python
confidence_threshold=0.3             # Minimum visibility for rendering
```

## 🏗️ Technical Architecture

### Core Components

#### 1. PoseSmoother Class
- **Temporal Smoothing**: Exponential weighted averaging across multiple frames
- **Jitter Reduction**: Maintains pose history buffer for stable tracking
- **Confidence Weighting**: Recent frames have higher influence on final pose

#### 2. Frame Enhancement Pipeline
```
Input Frame → CLAHE Enhancement → Bilateral Filtering → 
MediaPipe Processing → Landmark Interpolation → 
Temporal Smoothing → Skeleton Visualization
```

#### 3. Visualization Engine
- **Adaptive Rendering**: Line thickness (2-5px) and colors based on confidence scores
- **Dynamic Joints**: Circle size (3-7px) varies with landmark visibility
- **Confidence Bar**: Real-time pose detection quality indicator
- **Fallback Display**: Uses previous frame data when current detection fails

### Advanced Features

#### Smart Interpolation Algorithm
```python
def interpolate_missing_landmarks(current, previous, threshold=0.3):
    # For landmarks below confidence threshold:
    # new_landmark = current * (confidence/threshold) + previous * (1 - confidence/threshold)
```

#### Exponential Weighted Smoothing
```python
# Recent frames get higher weights
weights = [alpha^(n-1-i) for i in range(n)]  # where n = window_size
smoothed_landmark = weighted_average(landmark_history, weights)
```

## 📊 Performance Metrics

The system tracks and displays:
- **Processing Progress**: Frame-by-frame completion status
- **Detection Success Rate**: Percentage of frames with successful pose detection
- **Average Confidence**: Mean confidence scores across all detected poses
- **Processing Speed**: Frames per second processing rate

Example output:
```
✅ Enhanced MediaPipe Pose Skeleton video saved as: Output_Video_Enhanced_v1.mp4
🎉 Processed 1247 frames successfully!
📊 Statistics:
   - Frames with detected pose: 1198/1247 (96.1%)
   - Average pose confidence: 84.3%
```

## 🎨 Customization Options

### Skeleton Appearance
```python
# Confidence-based styling
thickness = int(2 + 3 * avg_confidence)        # Line thickness (2-5px)
color = (int(255 * confidence), 255, 255)      # White to cyan gradient
radius = int(3 + 4 * landmark.visibility)      # Joint size (3-7px)
```

### Smoothing Behavior
```python
# More aggressive smoothing
window_size=7,      # More frames for averaging
alpha=0.8,          # Higher recent frame influence

# Lighter smoothing  
window_size=3,      # Fewer frames
alpha=0.5,          # More distributed weighting
```

### Output Format
```python
# Different video codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'H264', 'MJPG'

# Adjust output resolution
output_width = frame_width * 2  # Side-by-side
output_height = frame_height    # Same height
```

## 🎯 Use Cases & Applications

### Dance & Performance Analysis
- **Movement Quality**: Analyze smoothness and precision of dance movements
- **Technique Comparison**: Side-by-side analysis of different performances
- **Progress Tracking**: Monitor improvement in dance technique over time

### Sports & Fitness
- **Form Analysis**: Detailed breakdown of exercise form and technique
- **Injury Prevention**: Identify movement patterns that may lead to injury
- **Training Optimization**: Data-driven insights for athletic performance

### Research & Development
- **Motion Studies**: Human movement research and biomechanical analysis
- **Gesture Recognition**: Training data for gesture-based interfaces
- **Animation Reference**: High-quality pose data for character animation

## 🔧 Troubleshooting

### Common Issues

**Low detection rates:**
- Increase lighting in recording environment
- Ensure full body is visible in frame
- Lower `min_detection_confidence` to 0.5 or 0.6

**Too much smoothing:**
- Reduce `window_size` to 3-4 frames
- Lower `alpha` value to 0.5-0.6

**Jittery skeleton:**
- Increase `window_size` to 6-8 frames  
- Increase `alpha` value to 0.8-0.9
- Lower `confidence_threshold` to 0.2

## 📈 Performance Benchmarks

Tested on various video types:
- **Dance Videos**: 95%+ detection rate, 85%+ average confidence
- **Sports Activities**: 90%+ detection rate, 80%+ average confidence  
- **Casual Movement**: 98%+ detection rate, 90%+ average confidence

Processing speed: ~15-25 FPS on modern hardware (depends on video resolution and frame rate)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- **Multi-person pose tracking**
- **3D pose estimation integration**
- **Real-time processing optimization**
- **Additional smoothing algorithms**
- **Custom skeleton visualization themes**

## 📝 License

MIT License

Copyright (c) 2025 Amit Omprakash Chakdhare

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🙏 Acknowledgments

- **Google MediaPipe Team**: Excellent pose estimation framework
- **OpenCV Community**: Computer vision tools and libraries
- **Dance Community**: Inspiration and test content for development

##
**⭐ If this project helped you, please give it a star!!**

**Made with ❤️**
