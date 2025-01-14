# Snap Studio
<img width="959" alt="image" src="https://github.com/user-attachments/assets/ac03fab4-a66a-4939-9908-741eaa50609b" />


Snap Studio is a powerful, user-friendly desktop image editing application built with Python. It combines the image processing capabilities of OpenCV with a modern Tkinter-based interface to provide a comprehensive set of tools for both basic and advanced image editing.

## Features

### Core Functionality
- Open and save images in various formats
- Undo/redo support
- Copy and paste functionality
- Keyboard shortcuts for common operations
- Real-time preview of adjustments

### Image Adjustments
- Brightness
- Contrast
- Saturation
- Sharpness
- Color temperature
- Tint

### Filters
- Cartoon
- Sketch
- Oil Painting
- Watercolor
- Vintage
- Cyberpunk
- Noir
- Pop Art
- Minimal

### Effects
- Blur
- Sharpen
- Edge Detection
- Denoise
- Vignette
- Grain

### Tools
- Crop
- Draw with customizable brush size and color
- Text overlay
- Eraser
- Image rotation and flipping
- Zoom controls

### Presets
- Natural
- Vivid
- B&W (Black & White)
- Sepia
- Cool
- Warm

### Camera Integration
- Webcam capture support
- Video recording capabilities
- Real-time frame capture

## Requirements

```
python >= 3.6
opencv-python
numpy
Pillow
tkinter (usually comes with Python)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/snap-studio.git
cd snap-studio
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To launch Snap Studio:
```bash
python main.py
```

### Basic Operations

1. **Opening an Image**
   - Click the "Open" button in the toolbar
   - Use the keyboard shortcut Ctrl+O
   - File → Open from the menu

2. **Saving an Image**
   - Click the "Save" button in the toolbar
   - Use the keyboard shortcut Ctrl+S
   - File → Save from the menu
   - For saving with a new name, use Ctrl+Shift+S

3. **Adjusting Images**
   - Use the sliders in the left panel to adjust various parameters
   - Changes are previewed in real-time
   - Click "Reset" to revert to the original image

4. **Applying Filters**
   - Select a filter from the right panel
   - Effects are applied immediately with a preview
   - Use the undo function (Ctrl+Z) to revert changes

### Keyboard Shortcuts

- Ctrl+O: Open image
- Ctrl+S: Save image
- Ctrl+Shift+S: Save image as
- Ctrl+Z: Undo
- Ctrl+C: Copy
- Ctrl+V: Paste
- Delete: Clear selection
- Escape: Cancel current operation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For support, please open an issue in the GitHub repository.
