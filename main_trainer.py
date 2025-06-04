# main_trainer.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessing import DataPreprocessor
from model_architecture import EyeDetectionModel
from eye_predictor import EyePredictor
import matplotlib.pyplot as plt

class EyeDetectionTrainer:
    """
    Main trainer class that orchestrates the entire training process.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the trainer with configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        # Default configuration
        self.config = {
            'img_size': 50,
            'batch_size': 32,
            'epochs': 20,
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'validation_split': 0.2,
            'model_save_path': 'best_eye_model.h5',
            'data_save_dir': 'processed_data'
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Initialize components
        self.preprocessor = DataPreprocessor(img_size=self.config['img_size'])
        self.model = EyeDetectionModel(input_shape=(self.config['img_size'], self.config['img_size'], 1))
        
        # Create directories
        os.makedirs(self.config['data_save_dir'], exist_ok=True)
    
    def prepare_data(self, train_closed_path: str, train_open_path: str, 
                    test_closed_path: str = None, test_open_path: str = None):
        """
        Prepare training and testing data.
        
        Args:
            train_closed_path (str): Path to training closed eye images
            train_open_path (str): Path to training open eye images
            test_closed_path (str): Path to testing closed eye images (optional)
            test_open_path (str): Path to testing open eye images (optional)
        """
        print("Preparing training data...")
        
        # Load training data
        try:
            closed_data = self.preprocessor.load_images_from_directory(train_closed_path)
            open_data = self.preprocessor.load_images_from_directory(train_open_path)
            
            # Combine data
            train_data = np.concatenate([closed_data, open_data], axis=0)
            
            # Create labels (0 for closed, 1 for open)
            closed_labels = np.zeros(len(closed_data))
            open_labels = np.ones(len(open_data))
            train_labels = np.concatenate([closed_labels, open_labels], axis=0)
            
            # Shuffle data
            indices = np.random.permutation(len(train_data))
            train_data = train_data[indices]
            train_labels = train_labels[indices]
            
            print(f"Training data shape: {train_data.shape}")
            print(f"Training labels shape: {train_labels.shape}")
            
            # Save training data
            train_data_path = os.path.join(self.config['data_save_dir'], 'train_data.npy')
            train_labels_path = os.path.join(self.config['data_save_dir'], 'train_labels.npy')
            self.preprocessor.save_data(train_data, train_labels, train_data_path, train_labels_path)
            
            # Prepare test data if paths provided
            if test_closed_path and test_open_path:
                print("Preparing test data...")
                test_closed_data = self.preprocessor.load_images_from_directory(test_closed_path)
                test_open_data = self.preprocessor.load_images_from_directory(test_open_path)
                
                test_data = np.concatenate([test_closed_data, test_open_data], axis=0)
                test_closed_labels = np.zeros(len(test_closed_data))
                test_open_labels = np.ones(len(test_open_data))
                test_labels = np.concatenate([test_closed_labels, test_open_labels], axis=0)
                
                # Shuffle test data
                test_indices = np.random.permutation(len(test_data))
                test_data = test_data[test_indices]
                test_labels = test_labels[test_indices]
                
                print(f"Test data shape: {test_data.shape}")
                print(f"Test labels shape: {test_labels.shape}")
                
                # Save test data
                test_data_path = os.path.join(self.config['data_save_dir'], 'test_data.npy')
                test_labels_path = os.path.join(self.config['data_save_dir'], 'test_labels.npy')
                self.preprocessor.save_data(test_data, test_labels, test_data_path, test_labels_path)
            
            return train_data, train_labels
            
        except Exception as e:
            raise Exception(f"Error preparing data: {str(e)}")
    
    def train_model(self, train_data=None, train_labels=None, 
                   test_data=None, test_labels=None):
        """
        Train the eye detection model.
        
        Args:
            train_data: Training data (if None, loads from saved files)
            train_labels: Training labels (if None, loads from saved files)
            test_data: Test data (optional)
            test_labels: Test labels (optional)
        """
        # Load data if not provided
        if train_data is None or train_labels is None:
            train_data_path = os.path.join(self.config['data_save_dir'], 'train_data.npy')
            train_labels_path = os.path.join(self.config['data_save_dir'], 'train_labels.npy')
            
            if os.path.exists(train_data_path) and os.path.exists(train_labels_path):
                train_data, train_labels = self.preprocessor.load_data(train_data_path, train_labels_path)
            else:
                raise ValueError("Training data not found. Run prepare_data() first.")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_data, train_labels, 
            test_size=self.config['validation_split'], 
            random_state=42, 
            stratify=train_labels
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        
        # Build model
        print("Building model...")
        self.model.build_model(
            learning_rate=self.config['learning_rate'],
            dropout_rate=self.config['dropout_rate']
        )
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.get_model_summary()
        
        # Get callbacks
        callbacks = self.model.get_callbacks(self.config['model_save_path'])
        
        # Train model
        print(f"\nStarting training for {self.config['epochs']} epochs...")
        history = self.model.train_model(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks
        )
        
        # Evaluate on test data if available
        if test_data is not None and test_labels is not None:
            print("\nEvaluating on test data...")
            test_loss, test_accuracy = self.model.evaluate_model(test_data, test_labels)
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Save final model
        final_model_path = os.path.join(self.config['data_save_dir'], 'final_eye_model.h5')
        self.model.save_model(final_model_path)
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """
        Plot training history.
        
        Args:
            history: Keras training history object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['data_save_dir'], 'training_history.png'))
        plt.show()
    
    def test_model(self, model_path: str = None, test_image_path: str = None):
        """
        Test the trained model.
        
        Args:
            model_path (str): Path to the saved model
            test_image_path (str): Path to test image
        """
        if model_path is None:
            model_path = self.config['model_save_path']
        
        predictor = EyePredictor(model_path)
        
        if test_image_path:
            label, confidence = predictor.predict_single_image(test_image_path)
            print(f"Prediction: {label} (Confidence: {confidence:.4f})")
        else:
            print("No test image provided. Use webcam for real-time testing.")
            # predictor.predict_from_webcam(duration=10)

def main():
    """
    Main function to demonstrate usage.
    """
    # Configuration
    config = {
        'img_size': 50,
        'batch_size': 32,
        'epochs': 15,
        'learning_rate': 0.001,
        'dropout_rate': 0.3,
        'validation_split': 0.2,
        'model_save_path': 'best_eye_model.h5',
        'data_save_dir': 'processed_data'
    }
    
    # Initialize trainer
    trainer = EyeDetectionTrainer(config)
    
    # Example usage (uncomment and modify paths as needed)
    
    # Prepare data
    train_data, train_labels = trainer.prepare_data(
        train_closed_path='Data/Dataset/train_closed',
        train_open_path='Data/Dataset/Open_Eyes',
        test_closed_path='Data/Dataset/test_closed',
        test_open_path='Data/Dataset/Open_Eyes'
    )
    
    # Train model
    history = trainer.train_model()
    
    # Test model
    trainer.test_model(test_image_path='Data/Dataset/Open_Eyes/s0001_02334_0_0_1_0_0_01.png')
    
    
    print("Eye Detection System - Modular Training Framework")
    print("Modify the paths in main() function and uncomment the code to start training.")

if __name__ == "__main__":
    main()