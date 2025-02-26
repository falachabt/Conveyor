import cv2
import numpy as np
import tensorflow as tf
import json
import serial
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime
from collections import deque


class ArduinoCommunicator:
    """
    Handles all communication with the Arduino
    """

    def __init__(self, port='COM7', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False
        self.incoming_buffer = []
        self.callbacks = {
            'NEW_OBJECT_DETECTED': [],
            'connected': [],
            'disconnected': [],
            'message': []
        }
        self.reader_thread = None
        self.running = False

    def connect(self):
        """Connect to the Arduino"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            time.sleep(2)  # Let Arduino reset and stabilize
            self.connected = True
            self._trigger_callbacks('connected', None)
            self.log(f"Connected to Arduino on {self.port}")

            # Start reader thread
            self.running = True
            self.reader_thread = threading.Thread(target=self._read_serial)
            self.reader_thread.daemon = True
            self.reader_thread.start()

            return True
        except Exception as e:
            self.log(f"Error connecting to Arduino: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from the Arduino"""
        self.running = False
        if self.reader_thread:
            self.reader_thread.join(timeout=1.0)

        if self.serial and self.serial.is_open:
            self.serial.close()
            self.connected = False
            self._trigger_callbacks('disconnected', None)
            self.log("Disconnected from Arduino")

    def send_command(self, command):
        """Send a command to the Arduino"""
        if not self.connected or not self.serial or not self.serial.is_open:
            self.log("Cannot send command: Not connected to Arduino")
            return False

        try:
            # Add newline to command if needed
            if not command.endswith('\n'):
                command += '\n'

            self.log(f"Sending command to Arduino: '{command.strip()}'")
            bytes_sent = self.serial.write(command.encode())
            self.serial.flush()  # Ensure the command is sent immediately
            self.log(f"Sent {bytes_sent} bytes to Arduino")

            # Wait for and read response
            time.sleep(0.2)  # Give Arduino time to process and respond

            response = ""
            start_time = time.time()
            while time.time() - start_time < 1.0 and self.serial.in_waiting > 0:
                try:
                    line = self.serial.readline().decode('utf-8').strip()
                    if line:
                        response += line + "\n"
                        self.log(f"Arduino response: {line}")
                except Exception as e:
                    self.log(f"Error reading response: {e}")
                    break

            return True
        except Exception as e:
            self.log(f"Error sending command: {e}")
            return False

    def register_callback(self, event_type, callback):
        """Register a callback for specific events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            self.callbacks[event_type] = [callback]

    def _trigger_callbacks(self, event_type, data):
        """Trigger all callbacks for a specific event"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                callback(data)

    def _read_serial(self):
        """Read data from the serial port in a separate thread"""
        while self.running and self.serial and self.serial.is_open:
            try:
                if self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8').strip()

                    if line:
                        self.log(f"Arduino: {line}")
                        self.incoming_buffer.append(line)

                        # Check for specific messages
                        if line == "NEW_OBJECT_DETECTED":
                            self._trigger_callbacks('NEW_OBJECT_DETECTED', None)

                        # General message event
                        self._trigger_callbacks('message', line)
            except Exception as e:
                self.log(f"Error reading from serial: {e}")
                time.sleep(0.5)  # Add delay to avoid flooding logs on error

            time.sleep(0.01)  # Small delay to prevent CPU hogging

    def log(self, message):
        """Log a message (override this in subclasses if needed)"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {message}")


class ObjectDetector:
    """
    Handles the camera feed and ML model for object detection
    """

    def __init__(self, model_path='interactive_model.h5', class_names_path='class_names.json'):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names = []
        self.image_size = (224, 224)
        self.camera = None
        self.camera_index = 1  # Default to camera index 1
        self.running = False
        self.processor_thread = None
        self.callbacks = {
            'detection': []
        }
        self.last_detection = None
        self.last_frame = None
        self.detection_active = False
        self.available_cameras = self._get_available_cameras()

    def _get_available_cameras(self):
        """Check available camera indices"""
        available = []
        # Try indices 0 to 5 (most systems won't have more than that)
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def load_model(self):
        """Load the TensorFlow model and class names"""
        try:
            print("Loading model...")
            self.model = tf.keras.models.load_model(self.model_path)

            print("Loading class names...")
            with open(self.class_names_path, 'r') as f:
                self.class_names = json.load(f)

            print(f"Model loaded successfully!")
            print(f"Available classes: {', '.join(self.class_names)}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def start_camera(self, camera_index=None):
        """Start the camera capture"""
        if camera_index is not None:
            self.camera_index = camera_index

        print(f"Attempting to open camera with index {self.camera_index}")
        self.camera = cv2.VideoCapture(self.camera_index)

        if not self.camera.isOpened():
            print(f"Failed to open camera {self.camera_index}, trying alternatives...")

            # If camera 1, try camera 0
            if self.camera_index == 1:
                alt_index = 0
            else:
                alt_index = 1

            print(f"Trying camera index {alt_index}")
            self.camera = cv2.VideoCapture(alt_index)

            if self.camera.isOpened():
                self.camera_index = alt_index
                print(f"Successfully opened camera {alt_index}")
            else:
                print("Error: No camera available!")
                return False

        # Check camera resolution and set it higher if possible
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print(f"Camera started with index {self.camera_index}")
        print(f"Resolution: {self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        return True

    def stop_camera(self):
        """Stop the camera capture"""
        if self.camera:
            self.camera.release()
            self.camera = None
            print("Camera stopped")

    def analyze_frame(self, frame=None):
        """Analyze a frame and return the class prediction"""
        if self.model is None:
            print("No model loaded!")
            return None, None, None

        if frame is None:
            if self.camera is None or not self.camera.isOpened():
                print("No camera available!")
                return None, None, None

            ret, frame = self.camera.read()
            if not ret:
                print("Error reading from camera!")
                return None, None, None

        # Process frame for model
        processed = cv2.resize(frame, self.image_size)
        processed = np.expand_dims(processed, axis=0) / 255.0

        # Get prediction
        prediction = self.model.predict(processed, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]

        if class_idx < len(self.class_names):
            class_name = self.class_names[class_idx]
        else:
            class_name = f"Unknown ({class_idx})"

        return class_idx, confidence, class_name

    def register_callback(self, event_type, callback):
        """Register a callback for specific events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            self.callbacks[event_type] = [callback]

    def _trigger_callbacks(self, event_type, data):
        """Trigger all callbacks for a specific event"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                callback(data)

    def start_detection(self):
        """Start continuous detection"""
        if not self.camera or not self.camera.isOpened():
            if not self.start_camera(self.camera_index):
                print("Failed to start camera for detection")
                return False

        self.running = True
        self.detection_active = True
        self.processor_thread = threading.Thread(target=self._process_frames)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        print("Detection started")
        return True

    def stop_detection(self):
        """Stop continuous detection"""
        self.running = False
        self.detection_active = False
        if self.processor_thread:
            self.processor_thread.join(timeout=1.0)
            self.processor_thread = None
        print("Detection stopped")

    def _process_frames(self):
        """Process frames in a continuous loop"""
        while self.running and self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                print("Error reading from camera!")
                time.sleep(0.1)
                continue

            self.last_frame = frame.copy()

            if self.detection_active:
                class_idx, confidence, class_name = self.analyze_frame(frame)
                if class_idx is not None:
                    self.last_detection = {
                        'class_idx': class_idx,
                        'confidence': confidence,
                        'class_name': class_name,
                        'timestamp': time.time()
                    }

                    # Trigger detection callback
                    self._trigger_callbacks('detection', self.last_detection)

            time.sleep(0.01)  # Small delay to prevent CPU hogging


class ObjectQueue:
    """
    Manages the queue of detected objects
    """

    def __init__(self, max_size=10):
        self.max_size = max_size
        self.objects = deque(maxlen=max_size)
        self.callbacks = {
            'added': [],
            'removed': [],
            'updated': []
        }

    def add(self, class_idx, class_name, confidence):
        """Add an object to the queue"""
        obj = {
            'class_idx': class_idx,
            'class_name': class_name,
            'confidence': confidence,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'status': 'Queued'  # Queued, Processing, Completed
        }
        self.objects.append(obj)
        self._trigger_callbacks('added', obj)
        self._trigger_callbacks('updated', self.objects)
        return obj

    def remove(self, index=0):
        """Remove an object from the queue"""
        if len(self.objects) > index:
            obj = self.objects.popleft() if index == 0 else self.objects.pop(index)
            self._trigger_callbacks('removed', obj)
            self._trigger_callbacks('updated', self.objects)
            return obj
        return None

    def update_status(self, index, status):
        """Update the status of an object in the queue"""
        if 0 <= index < len(self.objects):
            self.objects[index]['status'] = status
            self._trigger_callbacks('updated', self.objects)

    def get_objects(self):
        """Get all objects in the queue"""
        return list(self.objects)

    def get_size(self):
        """Get the number of objects in the queue"""
        return len(self.objects)

    def clear(self):
        """Clear the queue"""
        self.objects.clear()
        self._trigger_callbacks('updated', self.objects)

    def register_callback(self, event_type, callback):
        """Register a callback for specific events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            self.callbacks[event_type] = [callback]

    def _trigger_callbacks(self, event_type, data):
        """Trigger all callbacks for a specific event"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                callback(data)


class ConveyorGUI:
    """
    GUI for the conveyor control system
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Conveyor Control System")
        self.root.geometry("1200x700")

        # Components
        self.arduino = ArduinoCommunicator()
        self.detector = ObjectDetector()
        self.object_queue = ObjectQueue()

        # Register Arduino message callback
        self.arduino.register_callback('message', self.on_arduino_message)
        self.arduino.register_callback('NEW_OBJECT_DETECTED', self.on_new_object)
        self.detector.register_callback('detection', self.on_detection)
        self.object_queue.register_callback('updated', self.on_queue_updated)

        # State variables
        self.is_running = False
        self.last_class_sent = None
        self.last_class_time = 0
        self.min_send_interval = 1.0  # Reduced minimum time between class sends in seconds

        # GUI setup
        self.setup_gui()

        # Initialize
        self.update_status("System initialized")

        # Log available cameras
        if self.detector.available_cameras:
            self.log(f"Available cameras: {', '.join(map(str, self.detector.available_cameras))}")
        else:
            self.log("No cameras detected")

    def setup_gui(self):
        """Set up the GUI components"""
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Top control frame
        control_frame = ttk.LabelFrame(self.main_frame, text="System Control")
        control_frame.pack(fill=tk.X, pady=5)

        # Connection frame
        conn_frame = ttk.Frame(control_frame)
        conn_frame.pack(fill=tk.X, pady=5)

        ttk.Label(conn_frame, text="Arduino Port:").grid(row=0, column=0, padx=5, pady=5)
        self.port_var = tk.StringVar(value="COM7")
        port_entry = ttk.Entry(conn_frame, textvariable=self.port_var, width=10)
        port_entry.grid(row=0, column=1, padx=5, pady=5)

        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=2, padx=5, pady=5)

        # Camera selection with dropdown
        ttk.Label(conn_frame, text="Camera:").grid(row=0, column=3, padx=5, pady=5)

        # Add camera options based on available cameras
        camera_options = ["Camera 0", "Camera 1"] if not self.detector.available_cameras else \
            [f"Camera {i}" for i in self.detector.available_cameras]

        self.camera_var = tk.StringVar(value="Camera 1")  # Default to Camera 1
        camera_dropdown = ttk.Combobox(conn_frame, textvariable=self.camera_var, values=camera_options, width=10,
                                       state="readonly")
        camera_dropdown.grid(row=0, column=4, padx=5, pady=5)

        self.camera_btn = ttk.Button(conn_frame, text="Start Camera", command=self.toggle_camera)
        self.camera_btn.grid(row=0, column=5, padx=5, pady=5)

        # Command buttons
        cmd_frame = ttk.Frame(control_frame)
        cmd_frame.pack(fill=tk.X, pady=5)

        ttk.Button(cmd_frame, text="STOP", command=lambda: self.arduino.send_command("STOP")).grid(row=0, column=0,
                                                                                                   padx=5, pady=5)
        ttk.Button(cmd_frame, text="SLOW MODE", command=lambda: self.arduino.send_command("SLOW")).grid(row=0, column=1,
                                                                                                        padx=5, pady=5)
        ttk.Button(cmd_frame, text="NORMAL MODE", command=lambda: self.arduino.send_command("NORMAL")).grid(row=0,
                                                                                                            column=2,
                                                                                                            padx=5,
                                                                                                            pady=5)
        ttk.Button(cmd_frame, text="RESET", command=self.reset_system).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(cmd_frame, text="STATUS", command=lambda: self.arduino.send_command("STATUS")).grid(row=0, column=4,
                                                                                                       padx=5, pady=5)

        # Direction buttons
        dir_frame = ttk.Frame(control_frame)
        dir_frame.pack(fill=tk.X, pady=5)

        ttk.Label(dir_frame, text="Conveyor 2 Direction:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(dir_frame, text="LEFT", command=lambda: self.arduino.send_command("DIR_LEFT")).grid(row=0, column=1,
                                                                                                       padx=5, pady=5)
        ttk.Button(dir_frame, text="RIGHT", command=lambda: self.arduino.send_command("DIR_RIGHT")).grid(row=0,
                                                                                                         column=2,
                                                                                                         padx=5, pady=5)

        # Test classify and send buttons
        test_frame = ttk.Frame(control_frame)
        test_frame.pack(fill=tk.X, pady=5)

        # Directly send class buttons
        ttk.Label(test_frame, text="Send Class:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(test_frame, text="BLUE (1)", command=lambda: self.direct_send_class(1, "BLUE")).grid(row=0, column=1,
                                                                                                        padx=5, pady=5)
        ttk.Button(test_frame, text="BLACK (2)", command=lambda: self.direct_send_class(2, "BLACK")).grid(row=0,
                                                                                                          column=2,
                                                                                                          padx=5,
                                                                                                          pady=5)
        ttk.Button(test_frame, text="RED (3)", command=lambda: self.direct_send_class(3, "RED")).grid(row=0, column=3,
                                                                                                      padx=5, pady=5)

        # Main content area - use a vertical split
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Left side panel
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Camera display
        camera_frame = ttk.LabelFrame(left_panel, text="Camera Feed")
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.camera_canvas = tk.Canvas(camera_frame, bg="black")
        self.camera_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Object Queue Display
        queue_frame = ttk.LabelFrame(left_panel, text="Object Queue")
        queue_frame.pack(fill=tk.X, pady=5)

        # Create a Treeview for the queue
        self.queue_tree = ttk.Treeview(queue_frame, columns=("Position", "Class", "Index", "Time", "Status"),
                                       show="headings", height=5)
        self.queue_tree.column("Position", width=70, anchor=tk.CENTER)
        self.queue_tree.column("Class", width=120, anchor=tk.CENTER)
        self.queue_tree.column("Index", width=70, anchor=tk.CENTER)
        self.queue_tree.column("Time", width=80, anchor=tk.CENTER)
        self.queue_tree.column("Status", width=100, anchor=tk.CENTER)

        self.queue_tree.heading("Position", text="Position")
        self.queue_tree.heading("Class", text="Object Class")
        self.queue_tree.heading("Index", text="Class Index")
        self.queue_tree.heading("Time", text="Detected At")
        self.queue_tree.heading("Status", text="Status")

        self.queue_tree.pack(fill=tk.X, padx=5, pady=5)

        # Add scrollbar to queue view
        queue_scrollbar = ttk.Scrollbar(queue_frame, orient="vertical", command=self.queue_tree.yview)
        queue_scrollbar.place(relx=1, rely=0, relheight=1, anchor='ne')
        self.queue_tree.configure(yscrollcommand=queue_scrollbar.set)

        # Right panel for current detection and log
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Current detection info
        detection_frame = ttk.LabelFrame(right_panel, text="Detection Status")
        detection_frame.pack(fill=tk.X, pady=(0, 5))

        detection_info = ttk.Frame(detection_frame)
        detection_info.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(detection_info, text="Last Detection:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.detection_var = tk.StringVar(value="None")
        ttk.Label(detection_info, textvariable=self.detection_var, font=("Arial", 12, "bold")).grid(row=0, column=1,
                                                                                                    padx=5, pady=5,
                                                                                                    sticky=tk.W)

        ttk.Label(detection_info, text="Last Sent:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.last_sent_var = tk.StringVar(value="None")
        ttk.Label(detection_info, textvariable=self.last_sent_var, font=("Arial", 12, "bold")).grid(row=1, column=1,
                                                                                                    padx=5, pady=5,
                                                                                                    sticky=tk.W)

        # Button to force classification
        ttk.Button(detection_info, text="Force Classification", command=self.force_classification).grid(row=2, column=0,
                                                                                                        columnspan=2,
                                                                                                        padx=5, pady=5,
                                                                                                        sticky=tk.W + tk.E)

        # Status display
        status_frame = ttk.LabelFrame(right_panel, text="System Status")
        status_frame.pack(fill=tk.X, pady=5)

        self.status_var = tk.StringVar(value="Not connected")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 12, "bold"))
        status_label.pack(padx=5, pady=5)

        # Log display
        log_frame = ttk.LabelFrame(right_panel, text="System Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Manual command entry
        cmd_entry_frame = ttk.Frame(right_panel)
        cmd_entry_frame.pack(fill=tk.X, pady=5)

        ttk.Label(cmd_entry_frame, text="Command:").pack(side=tk.LEFT, padx=5)
        self.cmd_var = tk.StringVar()
        cmd_entry = ttk.Entry(cmd_entry_frame, textvariable=self.cmd_var)
        cmd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        cmd_entry.bind("<Return>", self.send_manual_command)

        ttk.Button(cmd_entry_frame, text="Send", command=self.send_manual_command).pack(side=tk.LEFT, padx=5)

        # Start camera update loop
        self.update_camera()

    def toggle_connection(self):
        """Connect or disconnect from Arduino"""
        if not self.arduino.connected:
            port = self.port_var.get()
            self.arduino.port = port
            if self.arduino.connect():
                self.connect_btn.config(text="Disconnect")
                self.update_status(f"Connected to Arduino on {port}")
                # Send status request to verify connection
                self.arduino.send_command("STATUS")
        else:
            self.arduino.disconnect()
            self.connect_btn.config(text="Connect")
            self.update_status("Disconnected from Arduino")

    def toggle_camera(self):
        """Start or stop the camera and detection"""
        if not self.detector.running:
            # First load the model if needed
            if self.detector.model is None:
                if not self.detector.load_model():
                    self.log("Failed to load detection model")
                    return

            # Get camera index from dropdown
            camera_str = self.camera_var.get()
            camera_index = int(camera_str.split()[1])  # Parse "Camera X" to get X

            self.log(f"Starting camera with index {camera_index}")

            if self.detector.start_camera(camera_index) and self.detector.start_detection():
                self.camera_btn.config(text="Stop Camera")
                self.update_status(f"Camera running with index {camera_index}")
            else:
                self.log("Failed to start camera")
        else:
            self.detector.stop_detection()
            self.camera_btn.config(text="Start Camera")
            self.update_status("Camera stopped")

    def reset_system(self):
        """Reset the entire system"""
        if self.arduino.connected:
            self.arduino.send_command("RESET")

        # Clear the object queue
        self.object_queue.clear()

        # Update queue display
        self.update_queue_display()

        self.update_status("System reset")

    def on_arduino_message(self, message):
        """Process messages from Arduino"""
        self.parse_arduino_messages(message)

    def on_new_object(self, data):
        """Handle new object detected by Arduino"""
        self.log("New object detected by Arduino")
        # Force a classification
        self.force_classification()

    def on_detection(self, detection):
        """Handle new detection from the camera"""
        # Update the detection display
        self.detection_var.set(f"{detection['class_name']} ({detection['confidence']:.2f})")

        # Auto-send when class changes (add this code)
        current_class = detection['class_idx']
        current_time = time.time()
        min_confidence = 0.60  # Minimum confidence threshold (adjust as needed)

        # Only send if:
        # 1. It's a different class than last sent
        # 2. Enough time has passed
        # 3. It's not class 0 (background)
        # 4. Confidence is above threshold
        if (current_class != self.last_class_sent and
                current_time - self.last_class_time >= self.min_send_interval and
                current_class != 0 and
                detection['confidence'] >= min_confidence):
            # Send command using existing method
            self.direct_send_class(
                detection['class_idx'],
                detection['class_name']
            )

    def on_queue_updated(self, queue):
        """Handle updates to the object queue"""
        self.update_queue_display()

    def update_queue_display(self):
        """Update the queue display in the GUI"""
        # Clear the current display
        for item in self.queue_tree.get_children():
            self.queue_tree.delete(item)

        # Add all objects in the queue
        objects = self.object_queue.get_objects()
        for i, obj in enumerate(objects):
            self.queue_tree.insert(
                "",
                "end",
                values=(
                    f"#{i + 1}",
                    obj['class_name'],
                    obj['class_idx'],
                    obj['timestamp'],
                    obj['status']
                ),
                tags=(obj['class_name'].lower(),)
            )

        # Add colors to rows based on class
        self.queue_tree.tag_configure('blue', background='#d0e0ff')
        self.queue_tree.tag_configure('red', background='#ffd0d0')
        self.queue_tree.tag_configure('black', background='#d0d0d0')
        self.queue_tree.tag_configure('yellow', background='#ffffd0')

    def direct_send_class(self, class_idx, class_name):
        """Send a specific class directly to Arduino"""
        current_time = time.time()

        # Skip if not enough time has passed
        if current_time - self.last_class_time < self.min_send_interval:
            self.log(f"Too soon to send another class. Please wait.")
            return

        # Add to queue
        self.object_queue.add(
            class_idx,
            class_name,
            1.0  # Maximum confidence
        )

        # Send directly to Arduino
        command = str(class_idx)

        # Ensure we send it properly
        if self.arduino.send_command(command):
            self.last_class_sent = class_idx
            self.last_class_time = current_time
            self.last_sent_var.set(f"{class_name} ({class_idx})")
            self.log(f"Directly sent class {class_idx} ({class_name}) to Arduino")

            # Update the queue item status
            if self.object_queue.get_size() > 0:
                self.object_queue.update_status(0, "Processing")

            return True
        else:
            self.log(f"Failed to send class {class_idx}")
            return False

    def force_classification(self):
        """Force classification and send to Arduino"""
        if not self.detector.last_detection:
            self.log("No detection available")
            return False

        detection = self.detector.last_detection
        current_time = time.time()

        # Skip class 0
        if detection['class_idx'] == 0:
            self.log(f"Ignoring class 0 ({detection['class_name']})")
            return False

        # Check if enough time has passed since last send
        if current_time - self.last_class_time < self.min_send_interval:
            self.log(f"Too soon to send another class. Please wait.")
            return False


        # Return the direct send method with the class info
        return self.direct_send_class(
            detection['class_idx'],
            detection['class_name']
        )

    def send_manual_command(self, event=None):
        """Send a manual command to Arduino"""
        command = self.cmd_var.get().strip()
        if command:
            self.arduino.send_command(command)
            self.cmd_var.set("")  # Clear the entry

    def update_camera(self):
        """Update the camera display"""
        if self.detector.last_frame is not None:
            try:
                # Convert frame to a format Tkinter can display
                frame = cv2.cvtColor(self.detector.last_frame, cv2.COLOR_BGR2RGB)

                # Draw detection info if available
                if self.detector.last_detection:
                    detection = self.detector.last_detection
                    text = f"{detection['class_name']}: {detection['confidence']:.2f}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Show if this was sent
                    if self.last_class_sent == detection['class_idx'] and time.time() - self.last_class_time < 2.0:
                        cv2.putText(frame, "SENT", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Resize image to fit canvas
                canvas_width = self.camera_canvas.winfo_width()
                canvas_height = self.camera_canvas.winfo_height()

                if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has been drawn
                    frame = cv2.resize(frame, (canvas_width, canvas_height))

                    # Convert to PhotoImage
                    img = tk.PhotoImage(data=cv2.imencode('.ppm', frame)[1].tobytes())

                    # Keep a reference to avoid garbage collection
                    self.current_image = img

                    # Update canvas
                    self.camera_canvas.create_image(0, 0, image=img, anchor=tk.NW)
            except Exception as e:
                self.log(f"Error updating camera display: {e}")

        # Schedule next update
        self.root.after(33, self.update_camera)  # ~30 FPS

    def update_status(self, message):
        """Update the status display"""
        self.status_var.set(message)
        self.log(message)

    def log(self, message):
        """Add a message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

        # Also log to console
        print(f"[{timestamp}] {message}")

    def parse_arduino_messages(self, message):
        """Parse messages from Arduino to update the UI"""
        if "Drop complete" in message:
            # When Arduino signals that it completed processing an object
            if self.object_queue.get_size() > 0:
                # Mark the object as completed
                obj = self.object_queue.remove(0)
                self.log(f"Object {obj['class_name']} (index {obj['class_idx']}) processing completed")

                # Check if there are more objects in the queue
                if self.object_queue.get_size() > 0:
                    self.object_queue.update_status(0, "Processing")

    def on_closing(self):
        """Handle window closing"""
        if self.detector.running:
            self.detector.stop_detection()

        if self.arduino.connected:
            self.arduino.disconnect()

        self.root.destroy()


def main():
    root = tk.Tk()
    app = ConveyorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
