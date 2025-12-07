"""
X-ray Disease Classification Streamlit App
Interactive web application for X-ray disease prediction
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Page configuration
st.set_page_config(
    page_title="X-ray Disease Classification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = keras.models.load_model('xray_disease_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please train the model first by running: python train_model.py")
        return None

@st.cache_data
def load_class_names():
    """Load class names"""
    try:
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        return class_names
    except:
        return ['Normal', 'Pneumonia', 'COVID-19']  # Default class names

def preprocess_image(image):
    """
    Unit I: Data Preprocessing
    Preprocess uploaded image for prediction
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Normalize
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_resized

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 X-ray Disease Classification System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict diseases from X-ray images using Deep Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🔍 Predict Disease", "📊 Model Information", "📈 Evaluation Metrics"]
    )
    
    # Load model and class names
    model = load_model()
    class_names = load_class_names()
    
    if page == "🔍 Predict Disease":
        st.header("Upload X-ray Image for Prediction")
        st.markdown("---")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a chest X-ray image in JPG, JPEG, or PNG format"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, caption="Input X-ray Image", use_container_width=True)
            
            with col2:
                if model is not None:
                    # Preprocess image
                    img_batch, img_processed = preprocess_image(image)
                    
                    # Make prediction
                    with st.spinner("Analyzing X-ray image..."):
                        predictions = model.predict(img_batch, verbose=0)
                        predicted_class = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_class]
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    # Predicted disease
                    st.markdown(f"### 🎯 Predicted Disease: **{class_names[predicted_class]}**")
                    st.markdown(f"### 📊 Confidence: **{confidence*100:.2f}%**")
                    
                    # Probability distribution
                    st.markdown("### Probability Distribution:")
                    prob_df = pd.DataFrame({
                        'Disease': class_names,
                        'Probability': predictions[0] * 100
                    })
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    # Display probabilities
                    for idx, row in prob_df.iterrows():
                        st.progress(row['Probability'] / 100)
                        st.write(f"{row['Disease']}: {row['Probability']:.2f}%")
                    
                    # Visualize probabilities
                    fig, ax = plt.subplots(figsize=(8, 6))
                    bars = ax.barh(prob_df['Disease'], prob_df['Probability'], color='steelblue')
                    ax.set_xlabel('Probability (%)', fontsize=12)
                    ax.set_ylabel('Disease', fontsize=12)
                    ax.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
                    ax.set_xlim(0, 100)
                    
                    # Add value labels on bars
                    for i, (idx, row) in enumerate(prob_df.iterrows()):
                        ax.text(row['Probability'] + 1, i, f'{row["Probability"]:.2f}%',
                               va='center', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Warning for low confidence
                    if confidence < 0.7:
                        st.warning("⚠️ Low confidence prediction. Please consult a medical professional.")
                else:
                    st.error("Model not found. Please train the model first.")
    
    elif page == "📊 Model Information":
        st.header("Model Information")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Syllabus Coverage")
            st.markdown("""
            This project incorporates concepts from the Predictive Analytics syllabus:
            
            **Unit I: Data Preprocessing**
            - Image normalization
            - Data augmentation
            - Train/validation split
            
            **Unit III: Classification**
            - Convolutional Neural Networks (CNN)
            - Evaluation metrics: Accuracy, Precision, Recall, F1 Score
            - Confusion Matrix
            
            **Unit V: Neural Networks**
            - Convolutional Neural Networks (CNN)
            - Multi-layer Perceptron (MLP) components
            - Deep learning architecture
            
            **Unit VI: Model Performance**
            - Cross-validation ready
            - Model evaluation metrics
            - Performance visualization
            """)
        
        with col2:
            st.subheader("🔬 Model Architecture")
            if model is not None:
                st.success("✅ Model loaded successfully!")
                st.info("""
                **CNN Architecture:**
                - 4 Convolutional Blocks
                - Batch Normalization
                - Max Pooling
                - Dropout layers
                - Dense layers (MLP)
                - Softmax output layer
                
                **Input:** 224x224 RGB images
                **Output:** 3 disease classes
                """)
            else:
                st.warning("Model not loaded. Train the model first.")
        
        st.subheader("📋 Class Labels")
        class_df = pd.DataFrame({
            'Class ID': range(len(class_names)),
            'Disease Name': class_names
        })
        st.table(class_df)
    
    elif page == "📈 Evaluation Metrics":
        st.header("Model Evaluation Metrics")
        st.markdown("---")
        
        if model is not None:
            st.info("📝 To view detailed evaluation metrics, please run the training script which generates confusion matrix and classification report.")
            
            # Check if evaluation files exist
            if os.path.exists('confusion_matrix.png'):
                st.subheader("Confusion Matrix")
                st.image('confusion_matrix.png', use_container_width=True)
            
            if os.path.exists('training_history.png'):
                st.subheader("Training History")
                st.image('training_history.png', use_container_width=True)
            
            st.markdown("""
            ### Evaluation Metrics Explained:
            
            **Accuracy:** Overall correctness of predictions
            
            **Precision:** Ratio of true positives to all predicted positives
            
            **Recall:** Ratio of true positives to all actual positives
            
            **F1 Score:** Harmonic mean of precision and recall
            
            **Confusion Matrix:** Visual representation of classification performance
            """)
        else:
            st.warning("Model not found. Please train the model first to see evaluation metrics.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "X-ray Disease Classification System | Built with Streamlit & TensorFlow"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

