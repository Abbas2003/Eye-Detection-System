# usage_example.py
"""
Eye Detection System - Usage Examples
This script demonstrates how to use the modular eye detection system.
"""

from main_trainer import EyeDetectionTrainer
from eye_predictor import EyePredictor
from data_preprocessing import DataPreprocessor
from model_architecture import EyeDetectionModel
import os

def example_1_full_training_pipeline():
    """
    Example 1: Complete training pipeline from scratch
    """
    print("=" * 60)
    print("EXAMPLE 1: Full Training Pipeline")
    print("=" * 60)
    
    # Configuration
    config = {
        'img_size': 50,
        'batch_size': 16,
        'epochs': 10,
        'learning_rate': 0.001,
        'dropout_rate': 0.3,
        'validation_split': 0.2,
        'model_save_path': 'models/eye_detection_model.h5',
        'data_save_dir': 'data/processed'
    }
    
    # Initialize trainer
    trainer = EyeDetectionTrainer(config)
    
    # Step 1: Prepare data (modify paths according to your dataset)
    try:
        print("Step 1: Preparing data...")
        train_data, train_labels = trainer.prepare_data(
            train_closed_path='Data/Dataset/train_closed',  # Your closed eyes training folder
            train_open_path='Data/Dataset/Open_Eyes',      # Your open eyes training folder
            test_closed_path='Data/Dataset/test_closed',    # Your closed eyes test folder
            test_open_path='Data/Dataset/Open_Eyes'         # Your open eyes test folder
        )
        print("✓ Data preparation completed")
        
        # Step 2: Train model
        print("\nStep 2: Training model...")
        history = trainer.train_model()
        print("✓ Model training completed")
        
        # Step 3: Test model
        print("\nStep 3: Testing model...")
        trainer.test_model(test_image_path='Data/Dataset/test_closed/141.jpg')  # Your test image
        print("✓ Model testing completed")
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        print("Make sure your dataset paths are correct!")

def example_2_data_preprocessing_only():
    """
    Example 2: Only data preprocessing
    """
    print("=" * 60)
    print("EXAMPLE 2: Data Preprocessing Only")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(img_size=50)
    
    try:
        # Load and preprocess images
        print("Loading closed eye images...")
        closed_data = preprocessor.load_images_from_directory('Data/Dataset/train_closed')
        print(f"Loaded {len(closed_data)} closed eye images")
        
        print("Loading open eye images...")
        open_data = preprocessor.load_images_from_directory('Data/Dataset/Open_Eyes')
        print(f"Loaded {len(open_data)} open eye images")
        
        # Create labels
        closed_labels = preprocessor.create_labels(len(closed_data), balanced=False)  # All 0s
        open_labels = preprocessor.create_labels(len(open_data), balanced=False)      # All 1s
        open_labels.fill(1)  # Set to 1 for open eyes
        
        # Save processed data
        preprocessor.save_data(closed_data, closed_labels, 'closed_data.npy', 'closed_labels.npy')
        preprocessor.save_data(open_data, open_labels, 'open_data.npy', 'open_labels.npy')
        
        print("✓ Data preprocessing completed")
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")

def example_3_model_training_only():
    """
    Example 3: Model training with pre-processed data
    """
    print("=" * 60)
    print("EXAMPLE 3: Model Training Only")
    print("=" * 60)
    
    # Initialize model
    model = EyeDetectionModel(input_shape=(50, 50, 1))
    
    try:
        # Load preprocessed data
        preprocessor = DataPreprocessor()
        train_data, train_labels = preprocessor.load_data('train_data.npy', 'train_labels.npy')
        
        print(f"Loaded training data: {train_data.shape}")
        print(f"Loaded training labels: {train_labels.shape}")
        
        # Build and train model
        model.build_model(learning_rate=0.001, dropout_rate=0.3)
        
        print("Model architecture:")
        model.get_model_summary()
        
        # Get callbacks for better training
        callbacks = model.get_callbacks('trained_model.h5')
        
        # Train model
        history = model.train_model(
            train_data, train_labels,
            epochs=10,
            batch_size=32,
            callbacks=callbacks
        )
        
        # Save model
        model.save_model('final_trained_model.h5')
        print("✓ Model training completed")
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")

def example_4_prediction_only():
    """
    Example 4: Using trained model for predictions
    """
    print("=" * 60)
    print("EXAMPLE 4: Prediction Only")
    print("=" * 60)
    
    try:
        # Initialize predictor
        predictor = EyePredictor('eye_model.h5', img_size=50)
        
        # Single image prediction
        print("Single image prediction:")
        image_path = 'Data/Dataset/test_closed/141.jpg'  # Closed eye test image
        if os.path.exists(image_path):
            label, confidence = predictor.predict_single_image(image_path)
            print(f"Image: {image_path}")
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence:.4f}")
        else:
            print(f"Test image not found: {image_path}")
        
        # Batch prediction
        print("\nBatch prediction:")
        test_images = [
            'Data/Dataset/test_closed/145.jpg', # Closed eye test image
            'Data/Dataset/Open_Eyes/s0001_02599_0_1_1_0_0_01.png', # Open eye test image
            'Data/Dataset/test_closed/143.jpg'  # Closed eye test image
        ]
        
        # Filter existing images
        existing_images = [img for img in test_images if os.path.exists(img)]
        
        if existing_images:
            results = predictor.predict_batch(existing_images)
            
            for img_path, label, confidence in results:
                print(f"{os.path.basename(img_path)}: {label} ({confidence:.4f})")
            
            # Get statistics
            stats = predictor.get_prediction_statistics(existing_images)
            print(f"\nStatistics:")
            print(f"Total images: {stats['total_images']}")
            print(f"Open eyes: {stats['open_eyes']} ({stats['open_percentage']:.1f}%)")
            print(f"Closed eyes: {stats['closed_eyes']} ({stats['closed_percentage']:.1f}%)")
            print(f"Average confidence: {stats['average_confidence']:.4f}")
        else:
            print("No test images found for batch prediction")
        
        # Real-time webcam prediction (uncomment to use)
        print("\nStarting webcam prediction (press 'q' to quit)...")
        predictor.predict_from_webcam(duration=10)
        
        print("✓ Prediction examples completed")
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")

def example_5_custom_configuration():
    """
    Example 5: Custom configuration for different use cases
    """
    print("=" * 60)
    print("EXAMPLE 5: Custom Configuration")
    print("=" * 60)
    
    # Configuration for high accuracy (slower training)
    high_accuracy_config = {
        'img_size': 64,           # Larger image size
        'batch_size': 16,         # Smaller batch size
        'epochs': 50,             # More epochs
        'learning_rate': 0.0005,  # Lower learning rate
        'dropout_rate': 0.5,      # Higher dropout for regularization
        'validation_split': 0.25,
        'model_save_path': 'models/high_accuracy_model.h5',
        'data_save_dir': 'data/high_accuracy'
    }
    
    # Configuration for fast training (lower accuracy)
    fast_training_config = {
        'img_size': 32,           # Smaller image size
        'batch_size': 64,         # Larger batch size
        'epochs': 10,             # Fewer epochs
        'learning_rate': 0.01,    # Higher learning rate
        'dropout_rate': 0.2,      # Lower dropout
        'validation_split': 0.15,
        'model_save_path': 'models/fast_model.h5',
        'data_save_dir': 'data/fast_training'
    }
    
    print("High accuracy configuration:", high_accuracy_config)
    print("\nFast training configuration:", fast_training_config)
    
    # You can use either configuration with the trainer
    # trainer = EyeDetectionTrainer(high_accuracy_config)
    # trainer = EyeDetectionTrainer(fast_training_config)

def example_6_real_time_webcam_prediction():
    """
    Example 6: Real-time webcam prediction using trained model
    """
    print("=" * 60)
    print("EXAMPLE 6: Real-time Webcam Prediction")
    print("=" * 60)
    try:
        predictor = EyePredictor('models/eye_detection_model.h5', img_size=50)
        print("Starting webcam prediction (press 'q' to quit)...")
        predictor.predict_from_webcam(duration=60)  # 60 seconds, change as needed
        print("✓ Real-time webcam prediction completed")
    except Exception as e:
        print(f"Error in real-time webcam prediction: {str(e)}")

def main():
    """
    Main function to run examples
    """
    print("Eye Detection System - Usage Examples")
    print("Choose an example to run:")
    print("1. Full training pipeline")
    print("2. Data preprocessing only")
    print("3. Model training only")
    print("4. Prediction only")
    print("5. Custom configuration examples")
    print("6. Real-time webcam prediction")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-6): ").strip()

            if choice == '0':
                print("Goodbye!")
                break
            elif choice == '1':
                example_1_full_training_pipeline()
            elif choice == '2':
                example_2_data_preprocessing_only()
            elif choice == '3':
                example_3_model_training_only()
            elif choice == '4':
                example_4_prediction_only()
            elif choice == '5':
                example_5_custom_configuration()
            elif choice == '6':
                example_6_real_time_webcam_prediction()
            else:
                print("Invalid choice. Please enter 0-6.")

        except KeyboardInterrupt:
            print("\nProgram interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()