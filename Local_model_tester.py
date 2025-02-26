import cv2
import numpy as np
import tensorflow as tf
import json
import serial
import time




class ModelTester:
    def __init__(self):
        self.image_size = (224, 224)
        self.model = None
        self.class_names = []
        self.show_controls = True
        self.serial_port = None
        self.previous_class_idx = None
        self.last_sent_time = 0
        self.min_send_interval = 0  # Minimum time between sends in seconds

    def load_model(self, model_path='interactive_model.h5', class_names_path='class_names.json'):
        """Load the trained model and class names"""
        try:
            print("Loading model...")
            self.model = tf.keras.models.load_model(model_path)

            print("Loading class names...")
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)

            print(f"Model loaded successfully!")
            print(f"Available classes: {', '.join(self.class_names)}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def connect_serial(self, port='COM7', baud_rate=9600):
        """Connect to the serial port"""
        try:
            self.serial_port = serial.Serial(port, baud_rate, timeout=1)
            print(f"Connected to serial port {port} at {baud_rate} baud")
            return True
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            return False

    def send_class_to_serial(self, class_idx, class_name):
        """Send class index to serial port if conditions are met"""
        # Don't send class 0
        if class_idx == 0:
            print(f"Not sending class {class_idx} ({class_name}) - class 0 is filtered out")
            return False

        # Only send if class has changed from previous detection
        if class_idx == self.previous_class_idx:
            print(f"Not sending class {class_idx} ({class_name}) - same as previous detection")
            return False

        # Only send if enough time has passed since last send
        current_time = time.time()
        if current_time - self.last_sent_time < self.min_send_interval:
            print(f"Not sending class {class_idx} ({class_name}) - minimum interval not reached")
            return False

        try:
            # Send the class index as a single byte
            self.serial_port.write(bytes([class_idx]))
            print(f"SUCCESS: Sent class index {class_idx} ({class_name}) to serial port")
            self.last_sent_time = current_time
            return True
        except Exception as e:
            print(f"Error sending to serial port: {e}")
            return False

    def draw_controls(self, frame):
        """Draw control information on the frame"""
        if self.show_controls:
            h, w = frame.shape[:2]
            # Draw semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (w - 250, 0), (w, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Add control text
            controls = [
                "Controls:",
                "H - Hide/Show Controls",
                "Q - Quit",
                f"Classes: {len(self.class_names)}",
                f"Serial: {'Connected' if self.serial_port else 'Disconnected'}",
                f"Prev Class: {self.previous_class_idx}"
            ]
            y = 25
            for text in controls:
                cv2.putText(frame, text, (w - 240, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y += 20

    def run_recognition(self, camera_index=1, serial_port='COM3', baud_rate=9600):
        """Run continuous recognition with the loaded model"""
        if self.model is None:
            print("No model loaded!")
            return

        # Connect to serial port
        self.connect_serial(serial_port, baud_rate)

        # Initialize camera
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            camera = cv2.VideoCapture(0)  # Fallback to default camera
            if not camera.isOpened():
                print("Error: No camera available!")
                return

        cv2.namedWindow('Model Recognition Test', cv2.WINDOW_NORMAL)

        while True:
            ret, frame = camera.read()
            if not ret:
                print("Camera error!")
                break

            # Process frame
            processed = cv2.resize(frame, self.image_size)
            processed = np.expand_dims(processed, axis=0) / 255.0

            # Get prediction
            prediction = self.model.predict(processed, verbose=0)
            class_idx = np.argmax(prediction[0])
            confidence = prediction[0][class_idx]
            predicted_class = self.class_names[class_idx]

            # Send to serial if conditions are met
            if self.serial_port:
                sent = self.send_class_to_serial(class_idx, predicted_class)
                if sent:
                    # Update previous class after successful send
                    self.previous_class_idx = class_idx
                    print(f"Updated previous class to {class_idx} ({predicted_class})")
                else:
                    # Still update previous class to track the current detection
                    # Remove this if you want to only update after successful sends
                    if self.previous_class_idx is None:
                        self.previous_class_idx = class_idx
                        print(f"Initialized previous class to {class_idx} ({predicted_class})")

            # Draw prediction
            # Main prediction
            cv2.putText(frame, f"{predicted_class}: {confidence:.2f}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            # Indicate serial status
            just_sent = False  # Track if we just sent in this iteration

            # Determine serial status display
            if class_idx == 0:
                serial_status = "NOT SENT (class 0)"
                color = (0, 0, 255)  # Red for not sent
            elif class_idx == self.previous_class_idx and self.previous_class_idx is not None:
                serial_status = "ALREADY SENT"
                color = (255, 165, 0)  # Orange for already sent
            else:
                if just_sent:
                    serial_status = "JUST SENT"
                    color = (0, 255, 0)  # Green for just sent
                else:
                    serial_status = "NOT SENT"
                    color = (0, 0, 255)  # Red for not sent

            cv2.putText(frame, f"Serial: {serial_status}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Show top 3 predictions
            sorted_idx = np.argsort(prediction[0])[::-1][:3]  # Top 3 predictions
            y = 130
            for idx in sorted_idx:
                conf = prediction[0][idx]
                if conf > 0.01:  # Only show if confidence > 1%
                    class_name = self.class_names[idx]
                    cv2.putText(frame, f"{class_name}: {conf:.2f}",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                    y += 35

            # Draw controls
            self.draw_controls(frame)

            # Show frame
            cv2.imshow('Model Recognition Test', frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                self.show_controls = not self.show_controls

        # Clean up
        if self.serial_port:
            self.serial_port.close()
        camera.release()
        cv2.destroyAllWindows()


def main():
    tester = ModelTester()
    if tester.load_model():
        # You can customize the camera index and serial port here
        tester.run_recognition(camera_index=1, serial_port='COM3', baud_rate=9600)
    else:
        print("Failed to load model. Make sure model files exist in the current directory.")


if __name__ == "__main__":
    main()