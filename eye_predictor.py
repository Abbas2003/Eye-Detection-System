# eye_predictor.py
import cv2
import numpy as np
from typing import Union, Tuple
import keras

class EyePredictor:
    """
    A class to handle eye state predictions (open/closed) using a trained model.
    """
    
    def __init__(self, model_path: str = None, img_size: int = 50):
        """
        Initialize the EyePredictor.
        
        Args:
            model_path (str): Path to the trained model file
            img_size (int): Size to resize images to
        """
        self.img_size = img_size
        self.model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the model file
        """
        try:
            self.model = keras.models.load_model(model_path)
            print(f"\033[92mModel loaded successfully from {model_path}\033[0m")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def preprocess_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Preprocess an image for prediction.
        
        Args:
            image_input: Either path to image file or numpy array
            
        Returns:
            np.ndarray: Preprocessed image ready for prediction
        """
        if isinstance(image_input, str):
            # Load image from file path
            img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image from {image_input}")
        else:
            # Use provided numpy array
            img = image_input
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        img = np.expand_dims(img, axis=0)   # Add batch dimension
        
        return img
    
    def predict_single_image(self, image_input: Union[str, np.ndarray], 
                           threshold: float = 0.5) -> Tuple[str, float]:
        """
        Predict eye state for a single image.
        
        Args:
            image_input: Either path to image file or numpy array
            threshold (float): Threshold for classification
            
        Returns:
            Tuple[str, float]: (prediction_label, confidence_score)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        processed_img = self.preprocess_image(image_input)
        
        # Make prediction
        prediction = self.model.predict(processed_img, verbose=0)[0][0]
        
        # Determine label
        if prediction > threshold:
            label = "Open"
        else:
            label = "Closed"
        
        return label, float(prediction)
    
    def predict_batch(self, image_paths: list, threshold: float = 0.5) -> list:
        """
        Predict eye states for multiple images.
        
        Args:
            image_paths (list): List of image paths
            threshold (float): Threshold for classification
            
        Returns:
            list: List of tuples (image_path, prediction_label, confidence_score)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        
        for img_path in image_paths:
            try:
                label, confidence = self.predict_single_image(img_path, threshold)
                results.append((img_path, label, confidence))
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                results.append((img_path, "Error", 0.0))
        
        return results
    
    def predict_from_webcam(self, duration: int = 30, threshold: float = 0.5):
        """
        Real-time eye state prediction from webcam.
        
        Args:
            duration (int): Duration in seconds to run prediction
            threshold (float): Threshold for classification
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise Exception("Could not open webcam")
        
        print(f"Starting webcam prediction for {duration} seconds...")
        print("Press 'q' to quit early")
        
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for display
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            try:
                # Make prediction
                label, confidence = self.predict_single_image(gray_frame, threshold)
                
                # Display result on frame
                color = (0, 255, 0) if label == "Open" else (0, 0, 255)
                cv2.putText(frame, f"Eye: {label} ({confidence:.2f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
            except Exception as e:
                cv2.putText(frame, f"Error: {str(e)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Eye State Detection', frame)
            
            # Check for exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Check duration
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed_time >= duration:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam prediction ended")
    
    def get_prediction_statistics(self, image_paths: list, threshold: float = 0.5) -> dict:
        """
        Get statistics from batch predictions.
        
        Args:
            image_paths (list): List of image paths
            threshold (float): Threshold for classification
            
        Returns:
            dict: Statistics dictionary
        """
        results = self.predict_batch(image_paths, threshold)
        
        total_images = len(results)
        open_eyes = sum(1 for _, label, _ in results if label == "Open")
        closed_eyes = sum(1 for _, label, _ in results if label == "Closed")
        errors = sum(1 for _, label, _ in results if label == "Error")
        
        avg_confidence = np.mean([conf for _, label, conf in results if label != "Error"])
        
        stats = {
            'total_images': total_images,
            'open_eyes': open_eyes,
            'closed_eyes': closed_eyes,
            'errors': errors,
            'open_percentage': (open_eyes / (total_images - errors)) * 100 if total_images > errors else 0,
            'closed_percentage': (closed_eyes / (total_images - errors)) * 100 if total_images > errors else 0,
            'average_confidence': float(avg_confidence) if not np.isnan(avg_confidence) else 0.0
        }
        
        return stats