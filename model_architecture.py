# model_architecture.py
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras

class EyeDetectionModel:
    """
    A class to define and manage the eye detection CNN model.
    """
    
    def __init__(self, input_shape: tuple = (50, 50, 1)):
        """
        Initialize the EyeDetectionModel.
        
        Args:
            input_shape (tuple): Shape of input images (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self, learning_rate: float = 0.001, 
                   dropout_rate: float = 0.3) -> Sequential:
        """
        Build the CNN model architecture.
        
        Args:
            learning_rate (float): Learning rate for optimizer
            dropout_rate (float): Dropout rate for regularization
            
        Returns:
            Sequential: Compiled Keras model
        """
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Flatten and Dense Layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_callbacks(self, model_save_path: str = 'best_eye_model.h5'):
        """
        Get training callbacks for better training performance.
        
        Args:
            model_save_path (str): Path to save the best model
            
        Returns:
            list: List of callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001,
                verbose=1
            )
        ]
        return callbacks
    
    def train_model(self, train_data, train_labels, 
                   validation_data=None, epochs=10, batch_size=32,
                   callbacks=None, verbose=1):
        """
        Train the model.
        
        Args:
            train_data: Training images
            train_labels: Training labels
            validation_data: Validation data tuple (X_val, y_val)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            callbacks: Training callbacks
            verbose (int): Verbosity level
            
        Returns:
            History: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        history = self.model.fit(
            train_data, train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def evaluate_model(self, test_data, test_labels, verbose=1):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test images
            test_labels: Test labels
            verbose (int): Verbosity level
            
        Returns:
            tuple: (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        loss, accuracy = self.model.evaluate(test_data, test_labels, verbose=verbose)
        return loss, accuracy
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a pre-trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict(self, image_data):
        """
        Make predictions on image data.
        
        Args:
            image_data: Input image data
            
        Returns:
            predictions: Model predictions
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Call build_model() or load_model() first.")
            
        return self.model.predict(image_data)
    
    def get_model_summary(self):
        """
        Print model summary.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        return self.model.summary()