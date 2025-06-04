# data_preprocessing.py
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

class DataPreprocessor:
    """
    A class to handle data preprocessing for eye detection system.
    Handles loading, resizing, and normalizing image data.
    """
    
    def __init__(self, img_size: int = 50):
        """
        Initialize the DataPreprocessor.
        
        Args:
            img_size (int): Size to resize images to (default: 50x50)
        """
        self.img_size = img_size
        
    def load_images_from_directory(self, path: str) -> np.ndarray:
        """
        Load and preprocess images from a directory.
        
        Args:
            path (str): Path to the directory containing images
            
        Returns:
            np.ndarray: Preprocessed image data
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")
            
        data = []
        
        for img_name in os.listdir(path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(path, img_name)
                
                # Load and preprocess image
                img = self._preprocess_image(img_path)
                if img is not None:
                    data.append(img)
        
        if not data:
            raise ValueError(f"No valid images found in directory: {path}")
            
        # Convert to numpy array and reshape
        data = np.array(data)
        data = np.reshape(data, (data.shape[0], self.img_size, self.img_size, 1))
        
        return data
    
    def _preprocess_image(self, img_path: str) -> Optional[np.ndarray]:
        """
        Preprocess a single image.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            np.ndarray or None: Preprocessed image or None if error
        """
        try:
            # Load image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                return None
                
            # Resize image
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Normalize pixel values to [0, 1]
            img = img.astype('float32') / 255.0
            
            return img
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            return None
    
    def create_labels(self, num_images: int, balanced: bool = True) -> np.ndarray:
        """
        Create labels for the dataset.
        
        Args:
            num_images (int): Total number of images
            balanced (bool): Whether to create balanced labels (half 0s, half 1s)
            
        Returns:
            np.ndarray: Label array
        """
        if balanced:
            # Create balanced labels: half 1s (open eyes), half 0s (closed eyes)
            labels = np.concatenate([
                np.ones(num_images // 2),
                np.zeros(num_images // 2)
            ])
        else:
            # All labels as 1 (modify as needed)
            labels = np.ones(num_images)
            
        return labels
    
    def save_data(self, data: np.ndarray, labels: np.ndarray, 
                  data_filename: str, labels_filename: str) -> None:
        """
        Save preprocessed data and labels to files.
        
        Args:
            data (np.ndarray): Image data
            labels (np.ndarray): Labels
            data_filename (str): Filename for data
            labels_filename (str): Filename for labels
        """
        np.save(data_filename, data)
        np.save(labels_filename, labels)
        print(f"Data saved to {data_filename}")
        print(f"Labels saved to {labels_filename}")
    
    def load_data(self, data_filename: str, labels_filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load preprocessed data and labels from files.
        
        Args:
            data_filename (str): Filename for data
            labels_filename (str): Filename for labels
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Data and labels
        """
        data = np.load(data_filename)
        labels = np.load(labels_filename)
        return data, labels