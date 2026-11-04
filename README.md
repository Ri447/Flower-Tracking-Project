# Flower Tracking

An interactive, real-time flower painting application built using Python, OpenCV, and MediaPipe. The application uses hand gestures to plant and control animated flowers on a live webcam feed, along with face tracking for augmented reality crown effects.

## Features

### Hand Tracking
- Real-time hand detection and tracking.
- Uses the index finger of the right hand to place flowers.
- Flowers are placed along the movement of the finger.
- Random flower selection with a shuffle system to avoid immediate repetition.

### Gesture Controls
- **Right Hand – Index Finger:** Plant flowers on the screen.
- **Left Hand – Pinch Gesture:** Adjust flower size by changing the distance between the thumb and index finger.
- **Left Hand – Open Palm:** Clear the screen with a glowing, falling flower animation.

### Flower Animations
- Floating and swaying flower effects.
- Randomized flower rotation and movement.
- Glow effects when clearing flowers.
- Falling petals and flowers with wind-like movement.
- Smooth fading as flowers reach the bottom of the screen.

### Face Tracking – Crown Overlay
- Real-time face detection using MediaPipe Face Mesh.
- Crown overlay that follows forehead position.
- Automatic crown scaling based on face width.
- Rotation based on head tilt.
- Supports tracking up to three faces.
- Smoothing for more natural overlay movement.
- On-screen button to toggle the crown effect.

### Screenshot Capture
- Capture and save your flower artwork.
- Screenshots are stored in the `Captured Photos` folder.
- Images are saved in PNG format.

### Performance
- Real-time webcam processing.
- FPS counter displayed on the screen.
- Mirrored webcam preview.

## Controls

| Input | Action |
|---|---|
| Right-hand index finger | Plant flowers |
| Left-hand pinch | Adjust flower size |
| Left-hand open palm | Clear all flowers |
| Click Crown button | Toggle crown overlay |
| `S` | Save screenshot |
| `Q` | Quit application |

## Project Structure

```text
Flower Tracking/
│
├── Accessories/
│   └── crown.png
│
├── Captured Photos/
│
├── Flowers/
│   └── Flower PNG assets
│
├── Transformations/
│   ├── __init__.py
