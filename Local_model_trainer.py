import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import json
import time


class EnhancedTrainer:
    def __init__(self):
        self.image_size = (224, 224)
        self.training_data = {}  # Dictionary to store images for each class
        self.model = None
        self.class_names = []
        self.current_class = None
        self.is_training = False
        self.show_controls = True

    def draw_controls_panel(self, frame):
        """Draw control information on the frame"""
        h, w = frame.shape[:2]
        if self.show_controls:
            # Draw semi-transparent overlay for controls
            overlay = frame.copy()
            cv2.rectangle(overlay, (w - 300, 0), (w, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Add control text
            controls = [
                "Controls:",
                "N - New Class",
                "C - Capture Image",
                "D - Delete Last",
                "T - Train Model",
                "R - Test Recognition",
                "H - Hide/Show Controls",
                "Q - Quit"
            ]
            y = 30
            for text in controls:
                cv2.putText(frame, text, (w - 290, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y += 20

    def capture_images(self, camera_index=1):
        """Enhanced interactive image capture with live preview"""
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            camera = cv2.VideoCapture(0)  # Fallback to default camera
            if not camera.isOpened():
                print("Error: No camera available!")
                return

        print("\n=== Image Capture Mode ===")
        cv2.namedWindow('Interactive Trainer', cv2.WINDOW_NORMAL)

        while True:
            ret, frame = camera.read()
            if not ret:
                print("Camera error!")
                break

            # Create info frame with current class and count
            info_frame = frame.copy()

            # Draw class info
            if self.current_class:
                count = len(self.training_data.get(self.current_class, []))
                cv2.putText(info_frame, f"Current Class: {self.current_class} | Images: {count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(info_frame, "Press 'N' to add new class",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw controls panel
            self.draw_controls_panel(info_frame)

            # Show all classes and their image counts
            y = 70
            for class_name in self.class_names:
                count = len(self.training_data[class_name])
                cv2.putText(info_frame, f"{class_name}: {count} images",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y += 25

            cv2.imshow('Interactive Trainer', info_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('n'):
                # Add new class
                cv2.putText(info_frame, "Enter class name in terminal",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Interactive Trainer', info_frame)
                self.current_class = input("\nEnter class name: ").strip()
                if self.current_class not in self.training_data:
                    self.training_data[self.current_class] = []
                    self.class_names.append(self.current_class)

            elif key == ord('c') and self.current_class:
                # Capture image
                processed_frame = cv2.resize(frame, self.image_size)
                self.training_data[self.current_class].append(processed_frame)
                print(f"Captured image for {self.current_class} (Total: {len(self.training_data[self.current_class])})")

            elif key == ord('d') and self.current_class:
                # Delete last capture
                if self.training_data[self.current_class]:
                    self.training_data[self.current_class].pop()
                    print(f"Deleted last image from {self.current_class}")

            elif key == ord('t'):
                # Train model
                camera.release()
                cv2.destroyAllWindows()
                self.train_model()
                camera = cv2.VideoCapture(camera_index)
                if not camera.isOpened():
                    camera = cv2.VideoCapture(0)

            elif key == ord('r'):
                # Test recognition
                camera.release()
                cv2.destroyAllWindows()
                self.test_recognition(camera_index)
                camera = cv2.VideoCapture(camera_index)
                if not camera.isOpened():
                    camera = cv2.VideoCapture(0)

            elif key == ord('h'):
                # Toggle controls visibility
                self.show_controls = not self.show_controls

            elif key == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

    def train_model(self):
        """Train the model with progress visualization"""
        if not self.training_data:
            print("No training data available!")
            return

        print("\n=== Training Model ===")

        # Prepare data
        X = []
        y = []
        for class_idx, class_name in enumerate(self.class_names):
            for img in self.training_data[class_name]:
                X.append(img)
                y.append(class_idx)

        X = np.array(X) / 255.0  # Normalize
        y = tf.keras.utils.to_categorical(y)

        # Create model
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(self.class_names), activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # Create progress window
        cv2.namedWindow('Training Progress', cv2.WINDOW_NORMAL)
        progress_image = np.zeros((100, 500, 3), dtype=np.uint8)

        # Custom callback for progress visualization
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, total_epochs):
                self.total_epochs = total_epochs

            def on_epoch_end(self, epoch, logs=None):
                progress = int((epoch + 1) / self.total_epochs * 500)
                progress_image[:] = (0, 0, 0)
                cv2.rectangle(progress_image, (0, 0), (progress, 100), (0, 255, 0), -1)
                cv2.putText(progress_image, f"Epoch {epoch + 1}/{self.total_epochs}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(progress_image, f"Accuracy: {logs['accuracy']:.2f}",
                            (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Training Progress', progress_image)
                cv2.waitKey(1)

        # Train
        self.model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2,
                       callbacks=[ProgressCallback(10)])

        # Save model and class names
        self.model.save('interactive_model.h5')
        with open('class_names.json', 'w') as f:
            json.dump(self.class_names, f)
        print("\nModel saved as 'interactive_model.h5'")

        cv2.destroyWindow('Training Progress')

    def test_recognition(self, camera_index=1):
        """Test the trained model with live camera feed"""
        if self.model is None:
            print("No trained model available!")
            return

        print("\n=== Testing Mode ===")
        print("Press 'Q' to quit testing")

        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            camera = cv2.VideoCapture(0)

        cv2.namedWindow('Recognition Test', cv2.WINDOW_NORMAL)

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            # Preprocess
            processed = cv2.resize(frame, self.image_size)
            processed = np.expand_dims(processed, axis=0) / 255.0

            # Predict
            prediction = self.model.predict(processed, verbose=0)
            predicted_class = self.class_names[np.argmax(prediction[0])]
            confidence = np.max(prediction[0])

            # Display result
            cv2.putText(frame, f"{predicted_class}: {confidence:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw controls
            cv2.putText(frame, "Press 'Q' to quit testing",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Recognition Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()


def main():
    trainer = EnhancedTrainer()
    trainer.capture_images()


if __name__ == "__main__":
    main()