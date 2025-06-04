# Eye Detection System - Modular Framework

Yeh ek complete modular eye detection system hai jo CNN (Convolutional Neural Network) use karta hai eyes ki state detect karne ke liye (open/closed).

## 🚀 Features

- **Modular Architecture**: Har component alag module mein organized
- **Easy Training**: Simple configuration ke sath model train kar sakte hain
- **Real-time Prediction**: Webcam se live eye detection
- **Batch Processing**: Multiple images ko ek sath process kar sakte hain
- **Flexible Configuration**: Different use cases ke liye customizable settings
- **Data Preprocessing**: Automatic image loading, resizing, aur normalization
- **Model Callbacks**: Early stopping, model checkpointing, learning rate scheduling

## 📁 Project Structure

```
eye-detection-system/
├── data_preprocessing.py    # Data loading aur preprocessing
├── model_architecture.py   # CNN model definition
├── eye_predictor.py        # Prediction utilities
├── main_trainer.py         # Main training orchestrator  
├── usage_example.py        # Usage examples
├── requirements.txt        # Dependencies
├── README.md              # Documentation
└── data/                  # Data directories
    ├── Dataset/
    │   ├── train_closed/
    │   ├── train_open/
    │   ├── test_closed/
    │   └── test_open/
    └── processed/
```

## 🛠️ Installation

### Step 1: Clone Repository
```bash
git clone <your-repo-url>
cd eye-detection-system
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Dataset
Apne dataset ko is structure mein organize karein:
```
Dataset/
├── train_closed/    # Closed eyes training images
├── train_open/      # Open eyes training images  
├── test_closed/     # Closed eyes test images
└── test_open/       # Open eyes test images
```

## 🎯 Quick Start

### Method 1: Complete Training Pipeline
```python
from main_trainer import EyeDetectionTrainer

# Configuration
config = {
    'img_size': 50,
    'batch_size': 32,
    'epochs': 15,
    'learning_rate': 0.001,
    'model_save_path': 'my_eye_model.h5'
}

# Initialize trainer
trainer = EyeDetectionTrainer(config)

# Prepare data
trainer.prepare_data(
    train_closed_path='Dataset/train_closed',
    train_open_path='Dataset/train_open',
    test_closed_path='Dataset/test_closed',
    test_open_path='Dataset/test_open'
)

# Train model
history = trainer.train_model()

# Test model
trainer.test_model(test_image_path='test_image.jpg')
```

### Method 2: Step-by-step Process

#### Data Preprocessing
```python
from data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(img_size=50)

# Load images
train_data = preprocessor.load_images_from_directory('Dataset/train_closed')
labels = preprocessor.create_labels(len(train_data))

# Save processed data
preprocessor.save_data(train_data, labels, 'train_data.npy', 'train_labels.npy')
```

#### Model Training
```python
from model_architecture import EyeDetectionModel

model = EyeDetectionModel(input_shape=(50, 50, 1))
model.build_model(learning_rate=0.001)

# Train model
history = model.train_model(train_data, train_labels, epochs=10)
model.save_model('trained_model.h5')
```

#### Making Predictions
```python
from eye_predictor import EyePredictor

predictor = EyePredictor('trained_model.h5')

# Single image prediction
label, confidence = predictor.predict_single_image('test_image.jpg')
print(f"Eye state: {label} (Confidence: {confidence:.4f})")

# Batch prediction
results = predictor.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# Real-time webcam prediction
predictor.predict_from_webcam(duration=30)
```

## ⚙️ Configuration Options

```python
config = {
    'img_size': 50,              # Image resize dimension
    'batch_size': 32,            # Training batch size
    'epochs': 15,                # Number of training epochs
    'learning_rate': 0.001,      # Learning rate for optimizer
    'dropout_rate': 0.3,         # Dropout rate for regularization
    'validation_split': 0.2,     # Validation data percentage
    'model_save_path': 'model.h5', # Model save path
    'data_save_dir': 'processed'  # Processed data directory
}
```

## 📊 Model Architecture

```
Input (50x50x1)
↓
Conv2D(32) + BatchNorm + MaxPool
↓
Conv2D(64) + BatchNorm + MaxPool  
↓
Conv2D(128) + BatchNorm + MaxPool
↓
Flatten
↓
Dense(128) + Dropout
↓
Dense(64) + Dropout
↓
Dense(1, sigmoid) - Output
```

## 🎮 Usage Examples

Usage examples run karne ke liye:
```bash
python usage_example.py
```

Yeh script different examples provide karta hai:
1. Complete training pipeline
2. Data preprocessing only
3. Model training only
4. Prediction only
5. Custom configurations

## 📈 Performance Tips

### High Accuracy Configuration
```python
config = {
    'img_size': 64,        # Larger images
    'batch_size': 16,      # Smaller batches
    'epochs': 50,          # More epochs
    'learning_rate': 0.0005, # Lower learning rate
    'dropout_rate': 0.5    # Higher regularization
}
```

### Fast Training Configuration
```python
config = {
    'img_size': 32,        # Smaller images
    'batch_size': 64,      # Larger batches
    'epochs': 10,          # Fewer epochs
    'learning_rate': 0.01, # Higher learning rate
    'dropout_rate': 0.2    # Lower regularization
}
```

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: cv2**
   ```bash
   pip install opencv-python
   ```

2. **Low accuracy**
   - Dataset ko balance karein (equal open/closed images)
   - More epochs train karein
   - Learning rate adjust karein
   - Data augmentation add karein

3. **Memory issues**
   - Batch size reduce karein
   - Image size kam karein
   - GPU use karein agar available hai

4. **File not found errors**
   - Dataset paths check karein
   - File extensions verify karein (.jpg, .png, etc.)

## 📱 Real-time Usage

Webcam se real-time eye detection:
```python
from eye_predictor import EyePredictor

predictor = EyePredictor('trained_model.h5')
predictor.predict_from_webcam(duration=60)  # 60 seconds
# Press 'q' to quit early
```

## 🎯 Advanced Features

### Custom Data Loading
```python
# Custom image preprocessing
def custom_preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (50, 50))
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Additional blur
    return img / 255.0
```

### Model Callbacks
System automatically uses these callbacks:
- **EarlyStopping**: Training stop hoga agar improvement nahi ho raha
- **ModelCheckpoint**: Best model automatically save hoga
- **ReduceLROnPlateau**: Learning rate automatically reduce hoga

## 📊 Evaluation Metrics

```python
# Get detailed statistics
stats = predictor.get_prediction_statistics(image_paths)
print(f"Accuracy: {stats['accuracy']:.2f}%")
print(f"Open eyes: {stats['open_percentage']:.1f}%")
print(f"Closed eyes: {stats['closed_percentage']:.1f}%")
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- OpenCV community for computer vision tools
- Contributors and users of this project