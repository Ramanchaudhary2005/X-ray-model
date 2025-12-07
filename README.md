# X-ray Disease Classification System

A comprehensive machine learning project for classifying diseases from X-ray images using Convolutional Neural Networks (CNN). This project is designed according to the Predictive Analytics syllabus and demonstrates various machine learning concepts.

## 📚 Syllabus Coverage

This project incorporates concepts from multiple units of the Predictive Analytics syllabus:

### Unit I: Introduction and Data Preparation
- ✅ Data preprocessing (image normalization, resizing)
- ✅ Data augmentation (rotation, shifting, zooming)
- ✅ Train/validation split

### Unit III: Supervised Learning - Classification
- ✅ Convolutional Neural Networks (CNN) for image classification
- ✅ Model evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

### Unit V: Dimensionality Reduction and Neural Networks
- ✅ Convolutional Neural Networks (CNN)
- ✅ Multi-layer Perceptron (MLP) components
- ✅ Deep learning architecture with multiple layers

### Unit VI: Model Performance
- ✅ Cross-validation ready implementation
- ✅ Comprehensive evaluation metrics
- ✅ Performance visualization (confusion matrix, training history)

## 🏗️ Project Structure

```
X-ray model/
│
├── train_images/          # Training images directory
├── test_images/           # Test images directory
├── labels_train.csv       # Training labels
│
├── train_model.py         # Model training script
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
├── xray_disease_model.h5  # Trained model (generated after training)
├── class_names.pkl        # Class names (generated after training)
├── confusion_matrix.png   # Confusion matrix visualization (generated)
└── training_history.png   # Training history plots (generated)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Train the CNN model on your dataset:

```bash
python train_model.py
```

This will:
- Load and preprocess the X-ray images
- Create and train a CNN model
- Evaluate the model performance
- Save the trained model and visualizations

### 3. Run the Streamlit App

Launch the interactive web application:

```bash
streamlit run app.py
```

The app will open in your browser where you can:
- Upload X-ray images for prediction
- View prediction results with confidence scores
- See model information and evaluation metrics

## 📊 Dataset Information

- **Training Images:** 4,672 images
- **Test Images:** 1,168 images
- **Classes:** 3 disease categories
  - Class 0: Normal (1,227 images)
  - Class 1: Pneumonia (2,238 images)
  - Class 2: COVID-19 (1,207 images)

## 🧠 Model Architecture

The CNN model consists of:

1. **Convolutional Layers:**
   - 4 convolutional blocks with increasing filters (32, 64, 128, 256)
   - Batch normalization after each convolution
   - Max pooling for dimensionality reduction
   - Dropout for regularization

2. **Fully Connected Layers (MLP):**
   - Flatten layer
   - Dense layers (512, 256 neurons)
   - Dropout layers
   - Softmax output layer (3 classes)

3. **Training Configuration:**
   - Input size: 224x224 RGB images
   - Batch size: 32
   - Optimizer: Adam
   - Loss function: Sparse Categorical Crossentropy
   - Early stopping and learning rate reduction callbacks

## 📈 Model Evaluation

The training script automatically evaluates the model using:

- **Accuracy:** Overall prediction correctness
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1 Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Visual representation of classification performance

## 🖥️ Streamlit Application Features

### 1. Predict Disease Page
- Upload X-ray images
- Real-time disease prediction
- Confidence scores for each class
- Probability distribution visualization

### 2. Model Information Page
- Syllabus coverage details
- Model architecture description
- Class labels information

### 3. Evaluation Metrics Page
- Confusion matrix visualization
- Training history plots
- Metric explanations

## 🔧 Configuration

You can modify the following parameters in `train_model.py`:

```python
IMG_SIZE = 224          # Image input size
BATCH_SIZE = 32         # Training batch size
EPOCHS = 50             # Maximum training epochs
NUM_CLASSES = 3         # Number of disease classes
```

## 📝 Usage Example

### Training the Model

```python
python train_model.py
```

Output:
- Trained model: `xray_disease_model.h5`
- Class names: `class_names.pkl`
- Confusion matrix: `confusion_matrix.png`
- Training history: `training_history.png`

### Using the Streamlit App

1. Run: `streamlit run app.py`
2. Navigate to "Predict Disease" page
3. Upload an X-ray image
4. View prediction results with confidence scores

## 🎯 Key Features

- ✅ **Deep Learning:** CNN architecture for image classification
- ✅ **Data Preprocessing:** Comprehensive image preprocessing pipeline
- ✅ **Evaluation Metrics:** Multiple classification metrics
- ✅ **Interactive UI:** User-friendly Streamlit interface
- ✅ **Visualization:** Confusion matrix and training history plots
- ✅ **Real-time Prediction:** Instant disease classification from uploaded images

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.15.0
- Streamlit 1.28.0
- NumPy, Pandas, Matplotlib, Seaborn
- OpenCV, Pillow

See `requirements.txt` for complete list.

## 🎓 Educational Value

This project demonstrates:

1. **Image Classification:** Using CNNs for medical image analysis
2. **Data Preprocessing:** Handling image data for machine learning
3. **Model Evaluation:** Understanding classification metrics
4. **Deep Learning:** Building and training neural networks
5. **Web Application:** Creating interactive ML applications

## ⚠️ Important Notes

- This is an educational project for academic purposes
- Medical diagnoses should always be confirmed by qualified healthcare professionals
- The model predictions are for educational demonstration only
- Ensure you have sufficient computational resources for training

## 🤝 Contributing

Feel free to enhance this project by:
- Adding more evaluation metrics
- Implementing additional models (SVM, Random Forest, etc.)
- Improving the UI/UX
- Adding more visualization features

## 📄 License

This project is created for educational purposes as part of the Predictive Analytics course.

## 👨‍💻 Author

Created for Predictive Analytics course project.

---

**Note:** Make sure to train the model before running the Streamlit app. The app requires the trained model file (`xray_disease_model.h5`) to make predictions.

