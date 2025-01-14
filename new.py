import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, colorchooser
from PIL import Image, ImageTk
import threading
import os
from datetime import datetime
import json

class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Snap Studio")
    
        self.root.configure(bg='#2b2b2b')
        self.setup_styles()
        
        self.original_image = None
        self.current_image = None
        self.display_image_copy = None
        self.webcam = None
        self.webcam_active = False
        self.recording = False
        self.video_writer = None
        self.cropping = False
        self.drawing = False
        self.crop_start = None
        self.undo_stack = []
        self.drawing_color = "#ffffff"
        self.drawing_size = 2
        self.last_used_directory = os.path.expanduser("~")
        self.load_settings()
        self.root.minsize(1200, 800)
        self.create_ui()
        self.bind_shortcuts()
        
    def setup_styles(self):
        """Configure ttk styles for the application"""
        self.style = ttk.Style()
        self.style.configure("TFrame", background='#2b2b2b')
        self.style.configure("TLabelframe", background='#2b2b2b', foreground='white')
        self.style.configure("TLabelframe.Label", background='#2b2b2b', foreground='white')
        self.style.configure("TButton", padding=5)
        self.style.configure("Accent.TButton", background='#4a90e2', padding=5)
        self.style.configure("TScale", background='#2b2b2b', troughcolor='#404040')
        self.style.configure("TRadiobutton", background='#2b2b2b', foreground='white')
        
    def create_ui(self):
        """Create the main user interface"""
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.create_menu()
        self.create_toolbar()
        self.create_panels()
        self.create_status_bar()
        
    def create_menu(self):
        """Create the application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_image, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As", command=self.save_image_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit_application)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_separator()
        edit_menu.add_command(label="Copy", command=self.copy_to_clipboard, accelerator="Ctrl+C")
        edit_menu.add_command(label="Paste", command=self.paste_from_clipboard, accelerator="Ctrl+V")
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Actual Size", command=self.zoom_actual)
        view_menu.add_command(label="Fit to Window", command=self.zoom_fit)
        
        # Image menu
        image_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Image", menu=image_menu)
        image_menu.add_command(label="Crop", command=self.toggle_crop)
       
        
        # Effects menu
        effects_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Effects", menu=effects_menu)
        effects_menu.add_command(label="Blur", command=lambda: self.apply_effect("blur"))
        effects_menu.add_command(label="Sharpen", command=lambda: self.apply_effect("sharpen"))
        effects_menu.add_command(label="Edge Detection", command=lambda: self.apply_effect("edge"))
        effects_menu.add_command(label="Denoise", command=lambda: self.apply_effect("denoise"))
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="About", command=self.show_about)

    def toggle_crop(self):
        """Toggle cropping mode"""
        self.cropping = not self.cropping
        if self.cropping:
            self.status_bar.config(text="Cropping mode activated")
        else:
            self.status_bar.config(text="Cropping mode deactivated")

    def zoom_actual(self):
        """Reset the image to its actual size"""
        if self.current_image is not None:
            self.display_image_copy = self.original_image.copy()
            self.display_image()

    def zoom_fit(self):
        """Fit the image to the window"""
        if self.current_image is not None:
            self.display_image_copy = self.current_image.copy()
            self.display_image()

    def create_toolbar(self):
        """Create the application toolbar"""
        toolbar = ttk.Frame(self.main_container)
        toolbar.pack(fill=tk.X, padx=5, pady=2)
        
        # Common tools
        ttk.Button(toolbar, text="Open", command=self.open_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save", command=self.save_image).pack(side=tk.LEFT, padx=2)
        
        # Undo/Redo
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Reset", command=self.reset_to_original).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Rotate Left", command=lambda: self.rotate_image(-90)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Rotate Right", command=lambda: self.rotate_image(90)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Flip Horizontal", command=lambda: self.flip_image(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Flip Vertical", command=lambda: self.flip_image(0)).pack(side=tk.LEFT, padx=2)
        
    def setup_toolbar(self):
        """Set up the toolbar with tools, color picker, and brush size slider."""
        # Tools
        self.tool_var = tk.StringVar(value="select")
        tools = [
            ("Crop", "crop"),
            ("Draw", "draw"),
            ("Text", "text"),
            ("Eraser", "eraser")
        ]
        
        for text, value in tools:
            ttk.Radiobutton(self.toolbar, text=text, value=value,
                            variable=self.tool_var).pack(side=tk.LEFT, padx=2)
        
        # Color picker
        self.color_button = ttk.Button(self.toolbar, text="Color", command=self.choose_color)
        self.color_button.pack(side=tk.LEFT, padx=2)
        
        # Brush size
        ttk.Label(self.toolbar, text="Size:").pack(side=tk.LEFT, padx=2)
        self.brush_size = ttk.Scale(self.toolbar, from_=1, to=20, orient=tk.HORIZONTAL,
                                    command=self.update_brush_size)
        self.brush_size.pack(side=tk.LEFT, padx=2)
        self.brush_size.set(2)
        
        # Force rendering of the toolbar
        self.toolbar.update_idletasks()


    def create_panels(self):
        """Create the main application panels"""
        self.paned_window = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.left_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_panel, weight=1)
        
        self.center_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.center_panel, weight=4)
        
        self.right_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_panel, weight=1)
        
        self.create_left_panel()
        self.create_center_panel()
        self.create_right_panel()
        
    def create_left_panel(self):
        """Create the left panel contents"""
        # Adjustments
        adjust_frame = ttk.LabelFrame(self.left_panel, text="Adjustments", padding=5)
        adjust_frame.pack(fill=tk.X, padx=5, pady=5)

        adjustments = [
            ("Brightness", -100, 100),
            ("Contrast", -100, 100),
            ("Saturation", -100, 100),
            ("Sharpness", 0, 100),
            ("Temperature", -100, 100),
            ("Tint", -100, 100)
        ]

        for name, min_val, max_val in adjustments:
            self.create_adjustment_slider(adjust_frame, name, min_val, max_val)

        # Presets
        preset_frame = ttk.LabelFrame(self.left_panel, text="Presets", padding=5)
        preset_frame.pack(fill=tk.X, padx=5, pady=5)

        presets = ["Natural", "Vivid", "B&W", "Sepia", "Cool", "Warm"]
        self.preset_var = tk.StringVar(value="None")

        for preset in presets:
            ttk.Radiobutton(preset_frame, text=preset, value=preset,
                            variable=self.preset_var, 
                            command=self.apply_preset).pack(fill=tk.X)

        # Tools
        tool_frame = ttk.LabelFrame(self.left_panel, text="Tools", padding=5)
        tool_frame.pack(fill=tk.X, padx=5, pady=5)

        self.tool_var = tk.StringVar(value="select")
        tools = [
            ("Crop", "crop"),
            ("Draw", "draw"),
            ("Text", "text"),
            ("Eraser", "eraser")
        ]

        for text, value in tools:
            ttk.Radiobutton(tool_frame, text=text, value=value,
                            variable=self.tool_var).pack(fill=tk.X)

        # Color picker
        color_picker_frame = ttk.LabelFrame(self.left_panel, text="Brush Options", padding=5)
        color_picker_frame.pack(fill=tk.X, padx=5, pady=5)

        self.color_button = ttk.Button(color_picker_frame, text="Color", command=self.choose_color)
        self.color_button.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(color_picker_frame, text="Brush Size:").pack(fill=tk.X, padx=5, pady=2)
        self.brush_size = ttk.Scale(color_picker_frame, from_=1, to=20, orient=tk.HORIZONTAL,
                                    command=self.update_brush_size)
        self.brush_size.pack(fill=tk.X, padx=5, pady=5)
        self.brush_size.set(2)
        self.root.after(100, self.load_heavy_operations)
        
        

    def reset_to_original(self):
        """Reset the image and adjustments to their original state"""
        if hasattr(self, 'original_image'):
            self.current_image = self.original_image.copy()  
            self.display_image_copy = self.original_image.copy()
            self.display_image()

        if hasattr(self, 'adjustment_sliders'):
            for slider in self.adjustment_sliders.values():
                slider.set(0)

        self.preset_var.set("None")
        self.tool_var.set("select")

    def load_image(self, path):
        """Load the image and store its original state"""
        self.current_image = cv2.imread(path)  # Load the image
        self.original_image = self.current_image.copy()  # Store original copy
        self.display_image_copy = self.current_image.copy()  # For displaying
        self.display_image()

    def create_center_panel(self):
        """Create the center panel contents"""
        canvas_frame = ttk.Frame(self.center_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg='#1e1e1e', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, 
                                  command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, 
                                  command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scrollbar.set, 
                            yscrollcommand=v_scrollbar.set)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        
    def on_mousewheel(self, event):
        """Handle mouse wheel events for zooming"""
        if event.delta > 0:
            self.zoom_actual()
        else:
            self.zoom_fit()

    def create_right_panel(self):
        """Create the right panel contents"""
        # Filters
        filter_frame = ttk.LabelFrame(self.right_panel, text="Filters", padding=5)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.filter_var = tk.StringVar(value="None")
        filters = [
            "None", "Cartoon", "Sketch", "Oil Painting", "Watercolor",
            "Vintage", "Cyberpunk", "Noir", "Pop Art", "Minimal"
        ]
        
        for filter_name in filters:
            ttk.Radiobutton(filter_frame, text=filter_name, value=filter_name,
                          variable=self.filter_var, 
                          command=self.apply_filter).pack(fill=tk.X)
                          
        # Effects
        effects_frame = ttk.LabelFrame(self.right_panel, text="Effects", padding=5)
        effects_frame.pack(fill=tk.X, padx=5, pady=5)
        
        effects = [
            ("Blur", "blur"),
            ("Sharpen", "sharpen"),
            ("Edge Detect", "edge"),
            ("Denoise", "denoise"),
            ("Vignette", "vignette"),
            ("Grain", "grain")
        ]
        
        for text, effect in effects:
            ttk.Button(effects_frame, text=text,
                      command=lambda e=effect: self.apply_effect(e)).pack(fill=tk.X, pady=2)
                      
        # Webcam controls
        webcam_frame = ttk.LabelFrame(self.right_panel, text="Webcam", padding=5)
        webcam_frame.pack(fill=tk.X , padx=5, pady=5)
        
        ttk.Button(webcam_frame, text="Start Webcam",
                  command=self.start_webcam).pack(fill=tk.X, pady=2)
        ttk.Button(webcam_frame, text="Stop Webcam",
                  command=self.stop_webcam).pack(fill=tk.X, pady=2)
        ttk.Button(webcam_frame, text="Capture Frame",
                  command=self.capture_frame).pack(fill=tk.X, pady=2)
        ttk.Button(webcam_frame, text="Record Video",
                  command=self.toggle_recording).pack(fill=tk.X, pady=2)
    
    def load_heavy_operations(self):
        """Perform heavy operations like image loading"""
        if hasattr(self, "current_image_path"):
            self.load_image(self.current_image_path)

    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, padding=5)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_adjustment_slider(self, parent, name, min_val, max_val):
        """Create a slider for image adjustments"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=name).pack(side=tk.LEFT)
        slider = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                         command=lambda x: self.update_adjustments())
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        slider.set(0)
        setattr(self, f"{name.lower()}_slider", slider)

    def bind_shortcuts(self):
        """Bind keyboard shortcuts"""
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_image())
        self.root.bind("<Control-S>", lambda e: self.save_image_as())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-c>", lambda e: self.copy_to_clipboard())
        self.root.bind("<Control-v>", lambda e: self.paste_from_clipboard())
        self.root.bind("<Delete>", lambda e: self.clear_selection())
        self.root.bind("<Escape>", lambda e: self.cancel_operation())
    
    def rotate_image(self, angle):
        if self.current_image is not None:
            height, width = self.current_image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            self.current_image = cv2.warpAffine(self.current_image, rotation_matrix, (width, height))
            self.display_image_copy = self.current_image.copy()
            self.display_image()

    def flip_image(self, direction):
        if self.current_image is not None:
            self.current_image = cv2.flip(self.current_image, direction)
            self.display_image_copy = self.current_image.copy()
            self.display_image()

    def update_adjustments(self):
        """Apply current adjustment settings to the image"""
        if self.current_image is None:
            return

        img = self.original_image.copy()
        brightness = self.brightness_slider.get()
        contrast = self.contrast_slider.get()
        saturation = self.saturation_slider.get()
        sharpness = self.sharpness_slider.get()
        temperature = self.temperature_slider.get()
        tint = self.tint_slider.get()
        
        try:
            # Apply brightness and contrast
            alpha = (contrast + 100) / 100.0
            beta = brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            # Apply saturation
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,1] = hsv[:,:,1] * (1 + saturation/100)
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
            # Apply sharpness
            if sharpness > 0:
                kernel = np.array([[-1,-1,-1],
                                 [-1, 9,-1],
                                 [-1,-1,-1]]) * (sharpness/100)
                img = cv2.filter2D(img, -1, kernel)
            
            # Apply temperature
            if temperature != 0:
                b, g, r = cv2.split(img)
                if temperature > 0:
                    r = cv2.addWeighted(r, 1 + temperature/100, np.zeros_like(r), 0, 0)
                else:
                    b = cv2.addWeighted(b, 1 - temperature/100, np.zeros_like(b), 0, 0)
                img = cv2.merge([b, g, r])
            
            # Apply tint
            if tint != 0:
                b, g, r = cv2.split(img)
                if tint > 0:
                    g = cv2.addWeighted(g, 1 + tint/100, np.zeros_like(g), 0, 0)
                else:
                    g = cv2.addWeighted(g, 1 + tint/100, np.zeros_like(g), 0, 0)
                    r = cv2.addWeighted(r, 1 - tint/100, np.zeros_like(r),  0, 0)
                img = cv2.merge([b, g, r])
            
            self.current_image = img.copy()
            self.apply_filter()  
            
        except Exception as e:
            self.show_error(f"Error applying adjustments: {str(e)}")

    def apply_filter(self):
        """Apply the selected filter to the image"""
        if self.current_image is None:
            return
            
        img = self.current_image.copy()
        filter_name = self.filter_var.get()
        
        try:
            if filter_name == "Cartoon":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 9, 9)
                color = cv2.bilateralFilter(img, 9, 300, 300)
                img = cv2.bitwise_and(color, color, mask=edges)
                
            elif filter_name == "Sketch":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                inv = 255 - gray
                blur = cv2.GaussianBlur(inv, (21, 21), 0)
                inv_blur = 255 - blur
                img = cv2.divide(gray, inv_blur, scale=256.0)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
            elif filter_name == "Oil Painting":
                img = cv2.xphoto.oilPainting(img, 7, 1)
                
            elif filter_name == "Watercolor":
                img = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
                
            elif filter_name == "Vintage":
                kernel = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
                img = cv2.transform(img, kernel)
                
            elif filter_name == "Cyberpunk":
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv[:,:,1] = cv2.addWeighted(hsv[:,:,1], 1.5, np.zeros_like(hsv[:,:,1]), 0, 0)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                b, g, r = cv2.split(img)
                r = cv2.addWeighted(r, 1.2, np.zeros_like(r), 0, 0)
                b = cv2.addWeighted(b, 1.4, np.zeros_like(b), 0, 0)
                img = cv2.merge([b, g, r])
                
            elif filter_name == "Noir":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
            elif filter_name == "Pop Art":
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv[:,:,1] = cv2.addWeighted(hsv[:,:,1], 2.0, np.zeros_like(hsv[:,:,1]), 0, 0)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
            elif filter_name == "Minimal":
                Z = img.reshape((-1,3))
                Z = np.float32(Z)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 8
                ret,label,center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                center = np.uint8(center)
                res = center[label.flatten()]
                img = res.reshape((img.shape))
            
            self.display_image_copy = img
            self.display_image()
            
        except Exception as e:
            self.show_error(f"Error applying filter: {str(e)}")

    def apply_effect(self, effect):
        """Apply a specific effect to the image"""
        if self.current_image is None:
            return
            
        img = self.current_image.copy()
        
        try:
            if effect == "blur":
                img = cv2.GaussianBlur(img, (15, 15), 0)
                
            elif effect == "sharpen":
                kernel = np.array([[-1,-1,-1],
                                 [-1, 9,-1],
                                 [- 1,-1,-1]])
                img = cv2.filter2D(img, -1, kernel)
                
            elif effect == "edge":
                img = cv2.Canny(img, 100, 200)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
            elif effect == "denoise":
                img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
                
            elif effect == "vignette":
                rows, cols = img.shape[:2]
                kernel_x = cv2.getGaussianKernel(cols, cols/4)
                kernel_y = cv2.getGaussianKernel(rows, rows/4)
                kernel = kernel_y * kernel_x.T
                mask = kernel / kernel.max()
                mask = np.stack([mask]*3, axis=2)
                img = img * mask
                
            elif effect == "grain":
                noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)
            
            self.add_to_undo_stack()
            self.current_image = img
            self.display_image_copy = img.copy()
            self.display_image()
            
        except Exception as e:
            self.show_error(f"Error applying effect: {str(e)}")

    def apply_preset(self):
        """Apply a selected preset to the image"""
        preset = self.preset_var.get()
        self.reset_adjustments()
        
        if preset == "Natural":
            self.brightness_slider.set(5)
            self.contrast_slider.set(10)
            self.saturation_slider.set(5)
            
        elif preset == "Vivid":
            self.contrast_slider.set(20)
            self.saturation_slider.set(30)
            self.sharpness_slider.set(20)
            
        elif preset == "B&W":
            self.saturation_slider.set(-100)
            self.contrast_slider.set(20)
            
        elif preset == "Sepia":
            self.saturation_slider.set(-50)
            self.temperature_slider.set(40)
            self.tint_slider.set(-10)
            
        elif preset == "Cool":
            self.temperature_slider.set(-30)
            self.saturation_slider.set(10)
            
        elif preset == "Warm":
            self.temperature_slider.set(30)
            self.saturation_slider.set(10)
            
        self.update_adjustments()

    def update_brush_size(self, value):
        """Update the brush size for drawing"""
        self.drawing_size = int(float(value))

    def choose_color(self):
        """Open color picker dialog"""
        color = colorchooser.askcolor(title="Choose Color")[1]
        if color:
            self.drawing_color = color

    def on_canvas_press(self, event):
        """Handle mouse press events on canvas"""
        if self.current_image is None:
            return
            
        tool = self.tool_var.get()
        
        if tool == "crop":
            self.crop_start = (event.x, event.y)
            self.cropping = True
            
        elif tool == "draw":
            self.drawing = True
            self.last_x = event.x
            self.last_y = event.y
            
        elif tool == "text":
            self.add_text(event.x, event.y)

    def on_canvas_drag(self, event):
        """Handle mouse drag events on canvas"""
        if self.current_image is None:
            return
            
        tool = self.tool_var.get()
        
        if tool == "crop" and self.cropping:
            self.canvas.delete("crop_rect")
            self.canvas.create_rectangle(
                self.crop_start[0], self.crop_start[1],
                event.x, event.y,
                outline="white",
                tags="crop_rect"
            )
            
        elif tool == "draw" and self.drawing:
            x, y = event.x, event.y
            self.draw_line(self.last_x, self.last_y, x, y)
            self.last_x = x
            self.last_y = y

    def on_canvas_release(self, event):
        """Handle mouse release events on canvas"""
        if self.current_image is None:
            return
            
        tool = self.tool_var.get()
        
        if tool == "crop" and self.cropping:
            self.perform_crop(event.x, event.y)
            
        elif tool == "draw":
            self.drawing = False
            self.add_to_undo_stack()

    def draw_line(self, x1, y1, x2, y2):
        """Draw a line on the image"""
        if self.current_image is None:
            return
            
        # Convert canvas coordinates to image coordinates
        scale_x = self.current_image.shape[1] / self.canvas.winfo_width()
        scale_y = self.current_image.shape[0] / self.canvas.winfo_height()
        start = (int(x1 * scale_x), int(y1 * scale_y))
        end = (int(x2 * scale_x), int(y2 * scale_y))
        
        # Convert color from hex to BGR
        color = tuple(int(self.drawing_color[i:i+2], 16) for i in (5, 3, 1))
        cv2.line(self.current_image, start, end, color, self.drawing_size)
        self.display_image()

    def add_text(self, x, y):
        """Add text to the image"""
        if self.current_image is None:
            return
            
        text = tk.simpledialog.askstring("Add Text", "Enter text:")
        if text:
            # Convert canvas coordinates to image coordinates
            scale_x = self.current_image.shape[1] / self.canvas.winfo_width()
            scale_y = self.current_image.shape[0] / self.canvas.winfo_height()
            position = (int(x * scale_x), int(y * scale_y))
            color = tuple(int(self.drawing_color[i:i+2], 16) for i in (5, 3, 1))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.current_image, text, position, font, 1, color, 2)
            self.add_to_undo_stack()
            self.display_image()

    def perform_crop(self, end_x, end_y):
        """Perform the crop operation"""
        if not self.crop_start:
            return
            
        # Convert canvas coordinates to image coordinates
        scale_x = self.current_image.shape[1] / self.canvas.winfo_width()
        scale_y = self.current_image.shape[0] / self.canvas.winfo_height()
        
        start_x = int(min(self.crop_start[0], end_x) * scale_x)
        start_y = int(min(self.crop_start[1], end_y) * scale_y)
        end_x = int(max(self.crop_start[0], end_x) * scale_x)
        end_y = int(max(self.crop_start[1], end_y) * scale_y)
        
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(self.current_image.shape[1], end_x)
        end_y = min(self.current_image.shape[0], end_y)
        
        self.add_to_undo_stack()
        self.current_image = self.current_image[start_y:end_y, start_x:end_x]
        self.display_image()
        
        self.cropping = False
        self.crop_start = None
        self.canvas.delete("crop_rect")

    def start_webcam(self):
        """Start webcam capture"""
        if self.webcam is None:
            self.webcam = cv2.VideoCapture(0)
            if not self.webcam.isOpened():
                self.show_error("Could not open webcam")
                self.webcam = None
                return
                
            self.webcam_active = True
            threading.Thread(target=self.update_webcam, daemon=True).start()
            self.status_bar.config(text="Webcam active")

    def stop_webcam(self):
        """Stop webcam capture"""
        self.webcam_active = False
        if self.webcam:
            self.webcam.release()
            self.webcam = None
        self.status_bar.config(text="Webcam stopped")

    def update_webcam(self):
        """Update webcam feed"""
        while self.webcam_active:
            ret, frame = self.webcam.read()
            if ret:
                frame = cv2.flip(frame, 1) 
                self.current_image = frame
                self.display_image()

    def capture_frame(self):
        """Capture current webcam frame"""
        if self.webcam and self.current_image is not None:
            self.add_to_undo_stack()
            self.original_image = self.current_image.copy()
            self.status_bar.config(text="Frame captured")

    def toggle_recording(self):
        """Toggle video recording"""
        if not self.webcam:
            self.show_error("Webcam not active")
            return
            
        if not self.recording:
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, 
                                              (int(self.webcam.get(3)), int(self.webcam.get(4))))
            self.recording = True
            self.status_bar.config(text="Recording ...")
        else:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.status_bar.config(text="Recording saved")

    def add_to_undo_stack(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())


    def undo(self):
        """Undo last operation"""
        if len(self.undo_stack) > 0:
            self.current_image = self.undo_stack.pop()
            self.display_image()

    def reset_adjustments(self):
        """Reset all adjustment sliders to default values"""
        for name in ["brightness", "contrast", "saturation", "sharpness", "temperature", "tint"]:
            slider = getattr(self, f"{name}_slider")
            slider.set(0)

    def load_settings(self):
        """Load application settings from file"""
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", "r") as f:
                    settings = json.load(f)
                    self.last_used_directory = settings.get("last_directory", os.path.expanduser("~"))
        except Exception as e:
            self.show_error(f"Error loading settings: {str(e)}")

    def save_settings(self):
        """Save application settings to file"""
        try:
            settings = {
                "last_directory": self.last_used_directory
            }
            with open("settings.json", "w") as f:
                json.dump(settings, f)
        except Exception as e:
            self.show_error(f"Error saving settings: {str(e)}")

    def show_error(self, message):
        """Show error message"""
        messagebox.showerror("Error", message)

    def show_about(self):
        """Show about dialog"""
        about_text = """Snap Studio by Yogita Jha"""
        messagebox.showinfo("About", about_text)

    def show_shortcuts(self):
        """Show keyboard shortcuts dialog"""
        shortcuts_text = """Keyboard Shortcuts:
Ctrl+O: Open
Ctrl+S: Save
Ctrl+Shift+S: Save As
Ctrl+Z: Undo
Ctrl+C: Copy
Ctrl+V: Paste
Delete: Clear Selection
Esc: Cancel Operation"""
        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)

    def open_image(self):
        """Open an image file"""
        file_path = filedialog.askopenfilename(
            initialdir=self.last_used_directory,
            title="Open Image",
            filetypes=(
                ("All supported formats", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("GIF files", "*.gif"),
                ("All files", "*.*")
            )
        )

        if file_path:
            try:
                self.last_used_directory = os.path.dirname(file_path)
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("Could not load image")
                self.current_image = self.original_image.copy()
                self.undo_stack.clear()
                self.reset_adjustments()
                self.filter_var.set("None")
                self.display_image()
                self.status_bar.config(text=f"Opened: {os.path.basename(file_path)}")
            except Exception as e:
                self.show_error(f"Error opening image: {str(e)}")

    def save_image(self):
        """Save the current image"""
        if self.current_image is None:
            self.show_error("No image to save")
            return

        if hasattr(self, 'current_file_path'):
            self._save_to_path(self.current_file_path)
        else:
            self.save_image_as()

    def save_image_as(self):
        """Save the current image with a new filename"""
        if self.current_image is None:
            self.show_error("No image to save")
            return

        file_path = filedialog.asksaveasfilename(
            initialdir=self.last_used_directory,
            title="Save Image As",
            defaultextension=".png",
            filetypes=(
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            )
        )

        if file_path:
            self._save_to_path(file_path)

    def _save_to_path(self, file_path):
        """Helper method to save image to specified path"""
        try:
            self.last_used_directory = os.path.dirname(file_path)
            cv2.imwrite(file_path, self.current_image)
            self.current_file_path = file_path
            self.status_bar.config(text=f"Saved: {os.path.basename(file_path)}")
        except Exception as e:
            self.show_error(f"Error saving image: {str(e)}")

    def display_image(self):
        """Display the current image on the canvas"""
        if self.current_image is None:
            return
        display_image = self.display_image_copy if self.display_image_copy is not None else self.current_image.copy()
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_height, image_width = display_image.shape[:2]
        scale_w = canvas_width / image_width
        scale_h = canvas_height / image_height
        scale = min(scale_w, scale_h) 

        if scale < 1:
            new_width = int(image_width * scale)
            new_height = int(image_height * scale)
            display_image = cv2.resize(display_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        image = Image.fromarray(display_image)
        photo = ImageTk.PhotoImage(image)

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width//2,
            canvas_height//2,
            image=photo,
            anchor="center"
        )
        self.canvas.image = photo  

    def copy_to_clipboard(self):
        """Copy the current image to clipboard"""
        if self.current_image is not None:
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_image)
            image.thumbnail((1000, 1000))  # Limit size for clipboard
            self.root.clipboard_clear()
            self.root.clipboard_append(image)
            self.status_bar.config(text="Image copied to clipboard")

    def paste_from_clipboard(self):
        """Paste image from clipboard"""
        try:
            image = self.root.clipboard_get()
            if isinstance(image, Image.Image):
                rgb_image = np.array(image)
                self.current_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                self.original_image = self.current_image.copy()
                self.display_image()
                self.status_bar.config(text="Image pasted from clipboard")
        except:
            self.show_error("No valid image in clipboard")

    def clear_selection(self):
        """Clear the current selection"""
        if self.tool_var.get() == "crop":
            self.canvas.delete("crop_rect")
            self.cropping = False
            self.crop_start = None

    def cancel_operation(self):
        """Cancel the current operation"""
        self.clear_selection()
        self.drawing = False
        self.tool_var.set("select")
        self.status_bar.config(text="Operation cancelled")

    def quit_application(self):
        """Clean up and quit application"""
        self.save_settings()
        self.stop_webcam()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditor(root)
    root.mainloop()