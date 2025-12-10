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
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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

@st.cache_data
def load_dataset_info():
    """Load dataset information"""
    try:
        df = pd.read_csv('labels_train.csv')
        train_files = [f for f in os.listdir('train_images') if f.endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists('train_images') else []
        test_files = [f for f in os.listdir('test_images') if f.endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists('test_images') else []
        
        return {
            'labels_df': df,
            'train_count': len(train_files),
            'test_count': len(test_files),
            'total_samples': len(df),
            'class_distribution': df['class_id'].value_counts().sort_index()
        }
    except Exception as e:
        return None

@st.cache_data
def load_sample_images_for_analysis(n_samples=100):
    """Load sample images for analysis"""
    try:
        df = pd.read_csv('labels_train.csv')
        sample_df = df.sample(min(n_samples, len(df)))
        images_data = []
        
        for idx, row in sample_df.iterrows():
            img_path = os.path.join('train_images', row['file_name'])
            if os.path.exists(img_path):
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images_data.append({
                            'file': row['file_name'],
                            'class': row['class_id'],
                            'shape': img_rgb.shape,
                            'mean': img_rgb.mean(),
                            'std': img_rgb.std()
                        })
                except:
                    pass
        return pd.DataFrame(images_data)
    except:
        return pd.DataFrame()

def preprocess_image(image):
    """
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
        [
            "🔍 Predict Disease",
            "📊 Dataset Overview",
            "🔧 Dataset Preprocessing",
            "📈 Regression Models",
            "🎯 Classification Models",
            "⚖️ Model Comparison"
        ]
    )
    
    # Load model and class names
    model = load_model()
    class_names = load_class_names()
    dataset_info = load_dataset_info()
    
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
    
    elif page == "📊 Dataset Overview":
        st.header("Dataset Overview")
        st.markdown("---")
        
        if dataset_info:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", dataset_info['total_samples'])
            with col2:
                st.metric("Training Images", dataset_info['train_count'])
            with col3:
                st.metric("Test Images", dataset_info['test_count'])
            with col4:
                st.metric("Classes", len(dataset_info['class_distribution']))
            
            st.markdown("### Class Distribution")
            class_dist_df = pd.DataFrame({
                'Class ID': dataset_info['class_distribution'].index,
                'Class Name': [class_names[i] if i < len(class_names) else f'Class {i}' for i in dataset_info['class_distribution'].index],
                'Count': dataset_info['class_distribution'].values,
                'Percentage': (dataset_info['class_distribution'].values / dataset_info['total_samples'] * 100).round(2)
            })
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(class_dist_df, use_container_width=True)
            
            with col2:
                # Bar chart
                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.bar(class_dist_df['Class Name'], class_dist_df['Count'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                ax.set_xlabel('Disease Class', fontsize=12)
                ax.set_ylabel('Number of Images', fontsize=12)
                ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Additional visualizations
            st.markdown("### Detailed Visualizations")
            
            # Pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 8))
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                ax.pie(class_dist_df['Count'], labels=class_dist_df['Class Name'], 
                      autopct='%1.1f%%', startangle=90, colors=colors)
                ax.set_title('Class Distribution (Pie Chart)', fontsize=14, fontweight='bold')
                st.pyplot(fig)
            
            with col2:
                # Load sample images for analysis
                with st.spinner("Loading image statistics..."):
                    img_stats = load_sample_images_for_analysis(200)
                    if not img_stats.empty:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Image size distribution
                        sizes = img_stats['shape'].apply(lambda x: x[0] * x[1] if isinstance(x, tuple) else 0)
                        axes[0].hist(sizes, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
                        axes[0].set_xlabel('Image Size (pixels)', fontsize=10)
                        axes[0].set_ylabel('Frequency', fontsize=10)
                        axes[0].set_title('Image Size Distribution', fontsize=12, fontweight='bold')
                        axes[0].grid(True, alpha=0.3)
                        
                        # Pixel intensity distribution
                        axes[1].hist(img_stats['mean'], bins=30, color='coral', alpha=0.7, edgecolor='black')
                        axes[1].set_xlabel('Mean Pixel Intensity', fontsize=10)
                        axes[1].set_ylabel('Frequency', fontsize=10)
                        axes[1].set_title('Pixel Intensity Distribution', fontsize=12, fontweight='bold')
                        axes[1].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
            
            # Class distribution by statistics
            if not img_stats.empty and 'class' in img_stats.columns:
                st.markdown("### Image Statistics by Class")
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                for class_id in sorted(img_stats['class'].unique()):
                    class_data = img_stats[img_stats['class'] == class_id]
                    class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f'Class {int(class_id)}'
                    
                    axes[0, 0].hist(class_data['mean'], bins=20, alpha=0.6, label=class_name)
                    axes[0, 1].hist(class_data['std'], bins=20, alpha=0.6, label=class_name)
                
                axes[0, 0].set_xlabel('Mean Pixel Intensity')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('Mean Intensity by Class')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                axes[0, 1].set_xlabel('Standard Deviation')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Pixel Std Dev by Class')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Box plots
                box_data = [img_stats[img_stats['class'] == c]['mean'].values 
                           for c in sorted(img_stats['class'].unique())]
                box_labels = [class_names[int(c)] if int(c) < len(class_names) else f'Class {int(c)}' 
                             for c in sorted(img_stats['class'].unique())]
                
                axes[1, 0].boxplot(box_data, labels=box_labels)
                axes[1, 0].set_ylabel('Mean Pixel Intensity')
                axes[1, 0].set_title('Pixel Intensity Distribution by Class')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Class count visualization
                class_counts = img_stats['class'].value_counts().sort_index()
                axes[1, 1].bar(range(len(class_counts)), class_counts.values, 
                             color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                axes[1, 1].set_xticks(range(len(class_counts)))
                axes[1, 1].set_xticklabels([class_names[int(c)] if int(c) < len(class_names) else f'Class {int(c)}' 
                                           for c in class_counts.index])
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].set_title('Sample Distribution by Class')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Sample images
            st.markdown("### Sample Images from Dataset")
            if os.path.exists('train_images'):
                sample_files = [f for f in os.listdir('train_images') if f.endswith(('.jpg', '.jpeg', '.png'))][:9]
                cols = st.columns(3)
                for idx, filename in enumerate(sample_files):
                    with cols[idx % 3]:
                        try:
                            img_path = os.path.join('train_images', filename)
                            img = Image.open(img_path)
                            # Get class label
                            label_row = dataset_info['labels_df'][dataset_info['labels_df']['file_name'] == filename]
                            if not label_row.empty:
                                class_id = label_row.iloc[0]['class_id']
                                class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f'Class {int(class_id)}'
                                st.image(img, caption=f"{class_name}", use_container_width=True)
                            else:
                                st.image(img, caption=filename, use_container_width=True)
                        except:
                            pass
        else:
            st.warning("Dataset information not available. Please ensure labels_train.csv exists.")
    
    elif page == "🔧 Dataset Preprocessing":
        st.header("Dataset Preprocessing")
        st.markdown("---")
        st.markdown("**Unit I: Data Preprocessing**")
        
        st.markdown("### Preprocessing Steps")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 1. Image Loading
            - Read X-ray images from directory
            - Handle different image formats (JPG, PNG)
            - Convert to RGB format
            
            #### 2. Image Resizing
            - Resize all images to 224x224 pixels
            - Maintain aspect ratio considerations
            - Standardize input dimensions
            
            #### 3. Normalization
            - Convert pixel values to float32
            - Normalize to range [0, 1]
            - Divide by 255.0
            """)
        
        with col2:
            st.markdown("""
            #### 4. Data Augmentation
            - **Rotation:** ±15 degrees
            - **Shifting:** Width/Height ±10%
            - **Zooming:** ±10% scale
            - **Shearing:** ±10% transformation
            - **Flipping:** Horizontal flip
            
            #### 5. Train/Validation Split
            - 80% training data
            - 20% validation data
            - Stratified split (maintains class distribution)
            """)
        
        st.markdown("### Preprocessing Code Example")
        st.code("""
# Image preprocessing pipeline
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
img = cv2.resize(img, (224, 224))           # Resize
img = img.astype('float32') / 255.0         # Normalize
        """, language='python')
        
        st.markdown("### Data Augmentation Configuration")
        aug_config = {
            'Rotation Range': '15 degrees',
            'Width Shift': '10%',
            'Height Shift': '10%',
            'Shear Range': '10%',
            'Zoom Range': '10%',
            'Horizontal Flip': 'True',
            'Fill Mode': 'Nearest'
        }
        st.json(aug_config)
        
        # Interactive preprocessing visualization
        st.markdown("### Interactive Preprocessing Visualization")
        
        if os.path.exists('train_images'):
            sample_files = [f for f in os.listdir('train_images') if f.endswith(('.jpg', '.jpeg', '.png'))]
            if sample_files:
                selected_file = st.selectbox("Select an image to preprocess", sample_files[:20])
                
                if selected_file:
                    img_path = os.path.join('train_images', selected_file)
                    img = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Original image
                    st.markdown("#### Preprocessing Pipeline")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown("**1. Original Image**")
                        st.image(img_rgb, caption=f'Original ({img_rgb.shape[1]}x{img_rgb.shape[0]})', use_container_width=True)
                    
                    # Resized
                    img_resized = cv2.resize(img_rgb, (224, 224))
                    with col2:
                        st.markdown("**2. Resized (224x224)**")
                        st.image(img_resized, caption='Resized', use_container_width=True)
                    
                    # Normalized
                    img_normalized = (img_resized.astype('float32') / 255.0)
                    with col3:
                        st.markdown("**3. Normalized [0-1]**")
                        st.image(img_normalized, caption='Normalized', use_container_width=True)
                    
                    # Grayscale version
                    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                    # Convert grayscale to 3-channel for display
                    img_gray_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                    with col4:
                        st.markdown("**4. Grayscale**")
                        st.image(img_gray_rgb, caption='Grayscale', use_container_width=True)
                    
                    # Detailed comparison
                    st.markdown("#### Detailed Analysis")
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    # Original
                    axes[0, 0].imshow(img_rgb)
                    axes[0, 0].set_title(f'Original ({img_rgb.shape[1]}x{img_rgb.shape[0]})')
                    axes[0, 0].axis('off')
                    
                    # Resized
                    axes[0, 1].imshow(img_resized)
                    axes[0, 1].set_title('Resized (224x224)')
                    axes[0, 1].axis('off')
                    
                    # Normalized
                    axes[0, 2].imshow(img_normalized)
                    axes[0, 2].set_title('Normalized [0-1]')
                    axes[0, 2].axis('off')
                    
                    # Pixel distribution - Original
                    axes[1, 0].hist(img_rgb.flatten(), bins=50, color='red', alpha=0.7, edgecolor='black')
                    axes[1, 0].set_title('Original Pixel Distribution')
                    axes[1, 0].set_xlabel('Pixel Value')
                    axes[1, 0].set_ylabel('Frequency')
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Pixel distribution - Resized
                    axes[1, 1].hist(img_resized.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
                    axes[1, 1].set_title('Resized Pixel Distribution')
                    axes[1, 1].set_xlabel('Pixel Value')
                    axes[1, 1].set_ylabel('Frequency')
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # Pixel distribution - Normalized
                    axes[1, 2].hist(img_normalized.flatten(), bins=50, color='green', alpha=0.7, edgecolor='black')
                    axes[1, 2].set_title('Normalized Pixel Distribution')
                    axes[1, 2].set_xlabel('Pixel Value [0-1]')
                    axes[1, 2].set_ylabel('Frequency')
                    axes[1, 2].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Data Augmentation Examples
                    st.markdown("#### Data Augmentation Examples")
                    try:
                        from tensorflow.keras.preprocessing.image import ImageDataGenerator
                    except:
                        from keras.preprocessing.image import ImageDataGenerator
                    
                    # Create augmentation generator
                    datagen = ImageDataGenerator(
                        rotation_range=15,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        shear_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True,
                        fill_mode='nearest'
                    )
                    
                    # Generate augmented images
                    img_batch = np.expand_dims(img_resized, axis=0)
                    aug_iter = datagen.flow(img_batch, batch_size=1)
                    
                    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                    axes[0, 0].imshow(img_resized)
                    axes[0, 0].set_title('Original')
                    axes[0, 0].axis('off')
                    
                    for i in range(1, 8):
                        aug_img = next(aug_iter)[0].astype('uint8')
                        row = i // 4
                        col = i % 4
                        axes[row, col].imshow(aug_img)
                        axes[row, col].set_title(f'Augmented {i}')
                        axes[row, col].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Statistics comparison
                    st.markdown("#### Preprocessing Statistics")
                    stats_data = {
                        'Metric': ['Width', 'Height', 'Channels', 'Min Value', 'Max Value', 'Mean', 'Std Dev'],
                        'Original': [
                            img_rgb.shape[1], img_rgb.shape[0], img_rgb.shape[2],
                            img_rgb.min(), img_rgb.max(), img_rgb.mean(), img_rgb.std()
                        ],
                        'Resized': [
                            img_resized.shape[1], img_resized.shape[0], img_resized.shape[2],
                            img_resized.min(), img_resized.max(), img_resized.mean(), img_resized.std()
                        ],
                        'Normalized': [
                            img_normalized.shape[1], img_normalized.shape[0], img_normalized.shape[2],
                            img_normalized.min(), img_normalized.max(), img_normalized.mean(), img_normalized.std()
                        ]
                    }
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True)
    
    elif page == "📈 Regression Models":
        st.header("Regression Models")
        st.markdown("---")
        st.markdown("**Unit II: Supervised Learning - Regression**")
        
        st.info("""
        **Note:** This project focuses on classification, but we'll demonstrate regression models 
        using image features (pixel statistics) to predict continuous values like image complexity scores.
        """)
        
        # Load sample data for regression
        if dataset_info:
            with st.spinner("Preparing regression data..."):
                img_stats = load_sample_images_for_analysis(500)
                
                if not img_stats.empty:
                    # Create features for regression
                    X_reg = img_stats[['mean', 'std']].values
                    # Create target: image complexity score (combination of mean and std)
                    y_reg = (img_stats['mean'] * 0.5 + img_stats['std'] * 0.5).values
                    
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import PolynomialFeatures
                    from sklearn.preprocessing import StandardScaler
                    
                    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                        X_reg, y_reg, test_size=0.2, random_state=42
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_reg)
                    X_test_scaled = scaler.transform(X_test_reg)
                    
                    st.markdown("### 1. Simple Linear Regression")
                    st.code("""
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
                    """, language='python')
                    
                    # Simple Linear Regression
                    lr_model = LinearRegression()
                    lr_model.fit(X_train_scaled[:, 0:1], y_train_reg)  # Using only mean
                    y_pred_lr = lr_model.predict(X_test_scaled[:, 0:1])
                    
                    mae_lr = mean_absolute_error(y_test_reg, y_pred_lr)
                    mse_lr = mean_squared_error(y_test_reg, y_pred_lr)
                    rmse_lr = np.sqrt(mse_lr)
                    r2_lr = r2_score(y_test_reg, y_pred_lr)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(X_test_reg[:, 0], y_test_reg, alpha=0.5, label='Actual', color='blue')
                        sorted_idx = np.argsort(X_test_scaled[:, 0])
                        ax.plot(X_test_reg[sorted_idx, 0], y_pred_lr[sorted_idx], 
                               'r-', linewidth=2, label='Predicted')
                        ax.set_xlabel('Mean Pixel Intensity')
                        ax.set_ylabel('Image Complexity Score')
                        ax.set_title('Simple Linear Regression')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("**Metrics:**")
                        st.metric("MAE", f"{mae_lr:.4f}")
                        st.metric("MSE", f"{mse_lr:.4f}")
                        st.metric("RMSE", f"{rmse_lr:.4f}")
                        st.metric("R² Score", f"{r2_lr:.4f}")
                    
                    st.markdown("### 2. Multiple Linear Regression")
                    st.code("""
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)  # Multiple features
y_pred = model.predict(X_test)
                    """, language='python')
                    
                    # Multiple Linear Regression
                    mlr_model = LinearRegression()
                    mlr_model.fit(X_train_scaled, y_train_reg)
                    y_pred_mlr = mlr_model.predict(X_test_scaled)
                    
                    mae_mlr = mean_absolute_error(y_test_reg, y_pred_mlr)
                    mse_mlr = mean_squared_error(y_test_reg, y_pred_mlr)
                    rmse_mlr = np.sqrt(mse_mlr)
                    r2_mlr = r2_score(y_test_reg, y_pred_mlr)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(y_test_reg, y_pred_mlr, alpha=0.5, color='green')
                        min_val = min(y_test_reg.min(), y_pred_mlr.min())
                        max_val = max(y_test_reg.max(), y_pred_mlr.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
                        ax.set_xlabel('Actual Complexity Score')
                        ax.set_ylabel('Predicted Complexity Score')
                        ax.set_title('Multiple Linear Regression')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("**Metrics:**")
                        st.metric("MAE", f"{mae_mlr:.4f}")
                        st.metric("MSE", f"{mse_mlr:.4f}")
                        st.metric("RMSE", f"{rmse_mlr:.4f}")
                        st.metric("R² Score", f"{r2_mlr:.4f}")
                    
                    st.markdown("### 3. Polynomial Regression")
                    st.code("""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X_train)
model = LinearRegression()
model.fit(X_poly, y_train)
                    """, language='python')
                    
                    # Polynomial Regression
                    poly_features = PolynomialFeatures(degree=2)
                    X_train_poly = poly_features.fit_transform(X_train_scaled)
                    X_test_poly = poly_features.transform(X_test_scaled)
                    
                    poly_model = LinearRegression()
                    poly_model.fit(X_train_poly, y_train_reg)
                    y_pred_poly = poly_model.predict(X_test_poly)
                    
                    mae_poly = mean_absolute_error(y_test_reg, y_pred_poly)
                    mse_poly = mean_squared_error(y_test_reg, y_pred_poly)
                    rmse_poly = np.sqrt(mse_poly)
                    r2_poly = r2_score(y_test_reg, y_pred_poly)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(y_test_reg, y_pred_poly, alpha=0.5, color='purple')
                        min_val = min(y_test_reg.min(), y_pred_poly.min())
                        max_val = max(y_test_reg.max(), y_pred_poly.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
                        ax.set_xlabel('Actual Complexity Score')
                        ax.set_ylabel('Predicted Complexity Score')
                        ax.set_title('Polynomial Regression (Degree=2)')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("**Metrics:**")
                        st.metric("MAE", f"{mae_poly:.4f}")
                        st.metric("MSE", f"{mse_poly:.4f}")
                        st.metric("RMSE", f"{rmse_poly:.4f}")
                        st.metric("R² Score", f"{r2_poly:.4f}")
                    
                    # Comparison
                    st.markdown("### Regression Models Comparison")
                    comparison_data = {
                        'Model': ['Simple Linear', 'Multiple Linear', 'Polynomial (deg=2)'],
                        'MAE': [mae_lr, mae_mlr, mae_poly],
                        'MSE': [mse_lr, mse_mlr, mse_poly],
                        'RMSE': [rmse_lr, rmse_mlr, rmse_poly],
                        'R² Score': [r2_lr, r2_mlr, r2_poly]
                    }
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Visualization
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    
                    models = ['Simple LR', 'Multiple LR', 'Polynomial']
                    mae_values = [mae_lr, mae_mlr, mae_poly]
                    mse_values = [mse_lr, mse_mlr, mse_poly]
                    rmse_values = [rmse_lr, rmse_mlr, rmse_poly]
                    r2_values = [r2_lr, r2_mlr, r2_poly]
                    
                    axes[0, 0].bar(models, mae_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                    axes[0, 0].set_ylabel('MAE')
                    axes[0, 0].set_title('Mean Absolute Error Comparison')
                    axes[0, 0].grid(True, alpha=0.3, axis='y')
                    
                    axes[0, 1].bar(models, mse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                    axes[0, 1].set_ylabel('MSE')
                    axes[0, 1].set_title('Mean Squared Error Comparison')
                    axes[0, 1].grid(True, alpha=0.3, axis='y')
                    
                    axes[1, 0].bar(models, rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                    axes[1, 0].set_ylabel('RMSE')
                    axes[1, 0].set_title('Root Mean Squared Error Comparison')
                    axes[1, 0].grid(True, alpha=0.3, axis='y')
                    
                    axes[1, 1].bar(models, r2_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                    axes[1, 1].set_ylabel('R² Score')
                    axes[1, 1].set_title('R² Score Comparison')
                    axes[1, 1].grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Unable to load image statistics for regression analysis.")
        else:
            st.warning("Dataset information not available.")
    
    elif page == "🎯 Classification Models":
        st.header("Classification Models")
        st.markdown("---")
        st.markdown("**Unit III: Supervised Learning - Classification**")
        
        if dataset_info:
            with st.spinner("Preparing classification data..."):
                # Load sample data
                img_stats = load_sample_images_for_analysis(1000)
                
                if not img_stats.empty and 'class' in img_stats.columns:
                    # Prepare features and labels
                    X_clf = img_stats[['mean', 'std']].values
                    y_clf = img_stats['class'].values
                    
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
                        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
                    )
                    
                    scaler_clf = StandardScaler()
                    X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
                    X_test_clf_scaled = scaler_clf.transform(X_test_clf)
                    
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["CNN", "KNN", "Naive Bayes", "Decision Tree", "SVM"])
                    
                    with tab1:
                        st.markdown("#### Convolutional Neural Network (CNN) - **Primary Model**")
                        if model is not None:
                            st.success("✅ CNN Model is trained and available!")
                            st.code("""
# CNN Architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    # ... more layers
    Dense(3, activation='softmax')
])
                            """, language='python')
                            
                            if os.path.exists('confusion_matrix.png'):
                                st.image('confusion_matrix.png', use_container_width=True)
                        else:
                            st.warning("CNN model not trained yet. Run train_model.py first.")
                    
                    with tab2:
                        st.markdown("#### K-Nearest Neighbors (KNN)")
                        st.code("""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
                        """, language='python')
                        
                        k_value = st.slider("Select k value", 3, 15, 5)
                        knn_model = KNeighborsClassifier(n_neighbors=k_value)
                        knn_model.fit(X_train_clf_scaled, y_train_clf)
                        y_pred_knn = knn_model.predict(X_test_clf_scaled)
                        
                        acc_knn = accuracy_score(y_test_clf, y_pred_knn)
                        prec_knn = precision_score(y_test_clf, y_pred_knn, average='weighted')
                        rec_knn = recall_score(y_test_clf, y_pred_knn, average='weighted')
                        f1_knn = f1_score(y_test_clf, y_pred_knn, average='weighted')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{acc_knn:.4f}")
                            st.metric("Precision", f"{prec_knn:.4f}")
                            st.metric("Recall", f"{rec_knn:.4f}")
                            st.metric("F1 Score", f"{f1_knn:.4f}")
                        
                        with col2:
                            cm_knn = confusion_matrix(y_test_clf, y_pred_knn)
                            fig, ax = plt.subplots(figsize=(6, 5))
                            sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=ax,
                                       xticklabels=class_names[:len(np.unique(y_test_clf))],
                                       yticklabels=class_names[:len(np.unique(y_test_clf))])
                            ax.set_title('KNN Confusion Matrix')
                            ax.set_ylabel('True Label')
                            ax.set_xlabel('Predicted Label')
                            st.pyplot(fig)
                    
                    with tab3:
                        st.markdown("#### Naive Bayes")
                        st.code("""
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
                        """, language='python')
                        
                        nb_model = GaussianNB()
                        nb_model.fit(X_train_clf_scaled, y_train_clf)
                        y_pred_nb = nb_model.predict(X_test_clf_scaled)
                        
                        acc_nb = accuracy_score(y_test_clf, y_pred_nb)
                        prec_nb = precision_score(y_test_clf, y_pred_nb, average='weighted')
                        rec_nb = recall_score(y_test_clf, y_pred_nb, average='weighted')
                        f1_nb = f1_score(y_test_clf, y_pred_nb, average='weighted')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{acc_nb:.4f}")
                            st.metric("Precision", f"{prec_nb:.4f}")
                            st.metric("Recall", f"{rec_nb:.4f}")
                            st.metric("F1 Score", f"{f1_nb:.4f}")
                        
                        with col2:
                            cm_nb = confusion_matrix(y_test_clf, y_pred_nb)
                            fig, ax = plt.subplots(figsize=(6, 5))
                            sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', ax=ax,
                                       xticklabels=class_names[:len(np.unique(y_test_clf))],
                                       yticklabels=class_names[:len(np.unique(y_test_clf))])
                            ax.set_title('Naive Bayes Confusion Matrix')
                            ax.set_ylabel('True Label')
                            ax.set_xlabel('Predicted Label')
                            st.pyplot(fig)
                    
                    with tab4:
                        st.markdown("#### Decision Tree")
                        st.code("""
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=10)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
                        """, language='python')
                        
                        max_depth = st.slider("Max Depth", 3, 20, 10)
                        dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                        dt_model.fit(X_train_clf_scaled, y_train_clf)
                        y_pred_dt = dt_model.predict(X_test_clf_scaled)
                        
                        acc_dt = accuracy_score(y_test_clf, y_pred_dt)
                        prec_dt = precision_score(y_test_clf, y_pred_dt, average='weighted')
                        rec_dt = recall_score(y_test_clf, y_pred_dt, average='weighted')
                        f1_dt = f1_score(y_test_clf, y_pred_dt, average='weighted')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{acc_dt:.4f}")
                            st.metric("Precision", f"{prec_dt:.4f}")
                            st.metric("Recall", f"{rec_dt:.4f}")
                            st.metric("F1 Score", f"{f1_dt:.4f}")
                        
                        with col2:
                            cm_dt = confusion_matrix(y_test_clf, y_pred_dt)
                            fig, ax = plt.subplots(figsize=(6, 5))
                            sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Oranges', ax=ax,
                                       xticklabels=class_names[:len(np.unique(y_test_clf))],
                                       yticklabels=class_names[:len(np.unique(y_test_clf))])
                            ax.set_title('Decision Tree Confusion Matrix')
                            ax.set_ylabel('True Label')
                            ax.set_xlabel('Predicted Label')
                            st.pyplot(fig)
                    
                    with tab5:
                        st.markdown("#### Support Vector Machine (SVM)")
                        st.code("""
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
                        """, language='python')
                        
                        kernel_type = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
                        svm_model = SVC(kernel=kernel_type, random_state=42)
                        svm_model.fit(X_train_clf_scaled, y_train_clf)
                        y_pred_svm = svm_model.predict(X_test_clf_scaled)
                        
                        acc_svm = accuracy_score(y_test_clf, y_pred_svm)
                        prec_svm = precision_score(y_test_clf, y_pred_svm, average='weighted')
                        rec_svm = recall_score(y_test_clf, y_pred_svm, average='weighted')
                        f1_svm = f1_score(y_test_clf, y_pred_svm, average='weighted')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{acc_svm:.4f}")
                            st.metric("Precision", f"{prec_svm:.4f}")
                            st.metric("Recall", f"{rec_svm:.4f}")
                            st.metric("F1 Score", f"{f1_svm:.4f}")
                        
                        with col2:
                            cm_svm = confusion_matrix(y_test_clf, y_pred_svm)
                            fig, ax = plt.subplots(figsize=(6, 5))
                            sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', ax=ax,
                                       xticklabels=class_names[:len(np.unique(y_test_clf))],
                                       yticklabels=class_names[:len(np.unique(y_test_clf))])
                            ax.set_title('SVM Confusion Matrix')
                            ax.set_ylabel('True Label')
                            ax.set_xlabel('Predicted Label')
                            st.pyplot(fig)
                    
                    # Store results for comparison
                    st.session_state['clf_results'] = {
                        'KNN': {'acc': acc_knn, 'prec': prec_knn, 'rec': rec_knn, 'f1': f1_knn},
                        'Naive Bayes': {'acc': acc_nb, 'prec': prec_nb, 'rec': rec_nb, 'f1': f1_nb},
                        'Decision Tree': {'acc': acc_dt, 'prec': prec_dt, 'rec': rec_dt, 'f1': f1_dt},
                        'SVM': {'acc': acc_svm, 'prec': prec_svm, 'rec': rec_svm, 'f1': f1_svm}
                    }
                else:
                    st.warning("Unable to load data for classification models.")
        else:
            st.warning("Dataset information not available.")
    
    elif page == "⚖️ Model Comparison":
        st.header("Model Comparison")
        st.markdown("---")
        st.markdown("**Compare different classification models**")
        
        if 'clf_results' in st.session_state:
            results = st.session_state['clf_results']
            
            st.markdown("### Classification Models Performance Comparison")
            
            # Extract metrics
            models = list(results.keys())
            accuracies = [results[m]['acc'] for m in models]
            precisions = [results[m]['prec'] for m in models]
            recalls = [results[m]['rec'] for m in models]
            f1_scores = [results[m]['f1'] for m in models]
            
            # Comparison table
            comparison_data = {
                'Model': models,
                'Accuracy': accuracies,
                'Precision': precisions,
                'Recall': recalls,
                'F1 Score': f1_scores
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualizations
            st.markdown("### Performance Metrics Visualization")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Accuracy comparison
            axes[0, 0].bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Accuracy Comparison')
            axes[0, 0].set_ylim([0, 1])
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(accuracies):
                axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # Precision comparison
            axes[0, 1].bar(models, precisions, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision Comparison')
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(precisions):
                axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # Recall comparison
            axes[1, 0].bar(models, recalls, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].set_title('Recall Comparison')
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(recalls):
                axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # F1 Score comparison
            axes[1, 1].bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_title('F1 Score Comparison')
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(f1_scores):
                axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Radar chart style comparison
            st.markdown("### Comprehensive Metrics Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(models))
            width = 0.2
            
            ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#1f77b4')
            ax.bar(x - 0.5*width, precisions, width, label='Precision', color='#ff7f0e')
            ax.bar(x + 0.5*width, recalls, width, label='Recall', color='#2ca02c')
            ax.bar(x + 1.5*width, f1_scores, width, label='F1 Score', color='#d62728')
            
            ax.set_ylabel('Score')
            ax.set_title('All Metrics Comparison Across Models')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best model
            best_model_idx = np.argmax(accuracies)
            best_model = models[best_model_idx]
            st.success(f"🏆 **Best Performing Model:** {best_model} with Accuracy: {accuracies[best_model_idx]:.4f}")
            
        elif model is not None:
            st.info("""
            Please visit the 'Classification Models' page first to train and evaluate different models.
            The results will then be available here for comparison.
            """)
            
            if os.path.exists('confusion_matrix.png'):
                st.markdown("### CNN Model (Current)")
                st.image('confusion_matrix.png', use_container_width=True)
        else:
            st.warning("Please train models first. Visit 'Classification Models' page to train different models.")
    
    elif page == "📊 Model Information":
        st.header("Model Information")
        st.markdown("---")
        
        # Model Status
        if model is not None:
            st.success("✅ **CNN Model Loaded Successfully!**")
            try:
                total_params = model.count_params()
                st.metric("Total Parameters", f"{total_params:,}")
            except:
                pass
        else:
            st.warning("⚠️ Model not loaded. Please train the model first by running: `python train_model.py`")
        
        st.markdown("---")
        
        # Model Architecture Details
        st.subheader("🔬 CNN Model Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Architecture Overview:**
            
            **Input Layer:**
            - Shape: (224, 224, 3)
            - RGB images
            
            **Convolutional Blocks (4 blocks):**
            1. Conv2D(32 filters) + BatchNorm + MaxPool + Dropout(0.25)
            2. Conv2D(64 filters) + BatchNorm + MaxPool + Dropout(0.25)
            3. Conv2D(128 filters) + BatchNorm + MaxPool + Dropout(0.25)
            4. Conv2D(256 filters) + BatchNorm + MaxPool + Dropout(0.25)
            
            **Fully Connected Layers (MLP):**
            - Flatten layer
            - Dense(512) + BatchNorm + Dropout(0.5)
            - Dense(256) + Dropout(0.5)
            - Dense(3, softmax) - Output layer
            """)
        
        with col2:
            # Visual architecture diagram
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.axis('off')
            
            # Draw architecture diagram
            layers = [
                ('Input\n224×224×3', 0, 0.9),
                ('Conv2D(32)\n+BN+Pool', 0, 0.75),
                ('Conv2D(64)\n+BN+Pool', 0, 0.6),
                ('Conv2D(128)\n+BN+Pool', 0, 0.45),
                ('Conv2D(256)\n+BN+Pool', 0, 0.3),
                ('Flatten', 0, 0.15),
                ('Dense(512)\n+Dropout', 0, 0.05),
                ('Dense(256)\n+Dropout', 0, -0.05),
                ('Output\nDense(3)', 0, -0.15)
            ]
            
            for i, (label, x, y) in enumerate(layers):
                # Draw box
                box = plt.Rectangle((x-0.15, y-0.03), 0.3, 0.06, 
                                   fill=True, facecolor='lightblue', 
                                   edgecolor='black', linewidth=1)
                ax.add_patch(box)
                ax.text(x, y, label, ha='center', va='center', 
                       fontsize=8, fontweight='bold')
                
                # Draw arrow
                if i < len(layers) - 1:
                    ax.arrow(x, y-0.03, 0, -0.06, head_width=0.02, 
                           head_length=0.01, fc='black', ec='black')
            
            ax.set_xlim(-0.3, 0.3)
            ax.set_ylim(-0.2, 1.0)
            ax.set_title('CNN Architecture Diagram', fontsize=12, fontweight='bold', pad=20)
            st.pyplot(fig)
        
        # Model Code
        st.markdown("### 📝 Model Architecture Code")
        st.code("""
from tensorflow.keras import layers, models

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
        """, language='python')
        
        # Training Configuration
        st.markdown("### ⚙️ Training Configuration")
        config_col1, config_col2, config_col3, config_col4 = st.columns(4)
        
        with config_col1:
            st.metric("Image Size", "224×224")
            st.metric("Batch Size", "32")
        with config_col2:
            st.metric("Epochs", "50")
            st.metric("Optimizer", "Adam")
        with config_col3:
            st.metric("Loss Function", "Sparse Categorical\nCrossentropy")
            st.metric("Learning Rate", "Adaptive")
        with config_col4:
            st.metric("Early Stopping", "Yes (patience=10)")
            st.metric("Data Augmentation", "Yes")
        
        # Syllabus Coverage
        st.markdown("---")
        st.subheader("📚 Syllabus Coverage")
        
        syllabus_tab1, syllabus_tab2, syllabus_tab3, syllabus_tab4 = st.tabs(
            ["Unit I", "Unit III", "Unit V", "Unit VI"]
        )
        
        with syllabus_tab1:
            st.markdown("""
            **Unit I: Introduction and Data Preparation**
            
            ✅ **Data Preprocessing:**
            - Image loading and format conversion
            - Image resizing (224×224)
            - Pixel normalization (0-1 range)
            - Data augmentation (rotation, shift, zoom, flip)
            - Train/validation split (80/20)
            - Stratified sampling
            """)
        
        with syllabus_tab2:
            st.markdown("""
            **Unit III: Supervised Learning - Classification**
            
            ✅ **Classification Models:**
            - Convolutional Neural Networks (CNN) - Primary
            - K-Nearest Neighbors (KNN)
            - Naive Bayes
            - Decision Trees
            - Support Vector Machine (SVM)
            
            ✅ **Evaluation Metrics:**
            - Accuracy
            - Precision
            - Recall
            - F1 Score
            - Confusion Matrix
            """)
        
        with syllabus_tab3:
            st.markdown("""
            **Unit V: Dimensionality Reduction and Neural Networks**
            
            ✅ **Neural Networks:**
            - Convolutional Neural Networks (CNN)
            - Multi-layer Perceptron (MLP) components
            - Feedforward architecture
            - Deep learning layers
            
            ✅ **Dimensionality Reduction:**
            - Max Pooling layers
            - Feature extraction
            """)
        
        with syllabus_tab4:
            st.markdown("""
            **Unit VI: Model Performance**
            
            ✅ **Bias-Variance Trade-off:**
            - Dropout layers for regularization
            - Batch normalization
            - Early stopping
            
            ✅ **Cross-validation:**
            - Train/validation split
            - Stratified sampling
            - Ready for K-fold CV
            
            ✅ **Model Evaluation:**
            - Comprehensive metrics
            - Visualization tools
            """)
        
        # Class Labels
        st.markdown("---")
        st.subheader("📋 Class Labels")
        class_df = pd.DataFrame({
            'Class ID': range(len(class_names)),
            'Disease Name': class_names,
            'Description': [
                'Normal chest X-ray with no abnormalities',
                'Pneumonia infection detected',
                'COVID-19 infection detected'
            ]
        })
        st.dataframe(class_df, use_container_width=True, hide_index=True)
        
        # Model Summary (if available)
        if model is not None:
            st.markdown("---")
            st.subheader("📊 Model Summary")
            with st.expander("View Model Summary"):
                try:
                    from io import StringIO
                    import sys
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    model.summary()
                    summary_text = sys.stdout.getvalue()
                    sys.stdout = old_stdout
                    st.text(summary_text)
                except:
                    st.info("Model summary not available. Model architecture is loaded successfully.")
    
    elif page == "📈 Evaluation Metrics":
        st.header("Model Evaluation Metrics")
        st.markdown("---")
        st.markdown("**Unit III & Unit VI: Model Performance Evaluation**")
        
        # Check if model exists
        if model is not None:
            st.success("✅ Model loaded. Evaluation metrics available below.")
        else:
            st.warning("⚠️ Model not found. Please train the model first by running: `python train_model.py`")
        
        # Evaluation Files
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists('confusion_matrix.png'):
                st.subheader("📊 Confusion Matrix")
                st.image('confusion_matrix.png', use_container_width=True)
                st.caption("Confusion matrix showing true vs predicted labels")
            else:
                st.info("""
                **Confusion Matrix not found.**
                
                To generate:
                1. Run `python train_model.py`
                2. The confusion matrix will be saved as `confusion_matrix.png`
                """)
        
        with col2:
            if os.path.exists('training_history.png'):
                st.subheader("📈 Training History")
                st.image('training_history.png', use_container_width=True)
                st.caption("Training and validation accuracy/loss over epochs")
            else:
                st.info("""
                **Training History not found.**
                
                To generate:
                1. Run `python train_model.py`
                2. The training history will be saved as `training_history.png`
                """)
        
        # Metrics Explanation
        st.markdown("---")
        st.subheader("📖 Evaluation Metrics Explained")
        
        metrics_tab1, metrics_tab2, metrics_tab3, metrics_tab4, metrics_tab5 = st.tabs(
            ["Accuracy", "Precision", "Recall", "F1 Score", "Confusion Matrix"]
        )
        
        with metrics_tab1:
            st.markdown("""
            ### Accuracy
            
            **Formula:**
            ```
            Accuracy = (TP + TN) / (TP + TN + FP + FN)
            ```
            
            **Definition:**
            - Overall correctness of the model
            - Ratio of correct predictions to total predictions
            - Range: 0 to 1 (higher is better)
            
            **Interpretation:**
            - 1.0 = Perfect predictions
            - 0.5 = Random guessing (for binary classification)
            - < 0.5 = Worse than random
            
            **Use Case:**
            - Good for balanced datasets
            - Quick overall performance indicator
            """)
            
            # Visual example
            fig, ax = plt.subplots(figsize=(8, 6))
            accuracy_example = 0.85
            ax.bar(['Correct', 'Incorrect'], 
                  [accuracy_example * 100, (1-accuracy_example) * 100],
                  color=['green', 'red'], alpha=0.7)
            ax.set_ylabel('Percentage (%)')
            ax.set_title(f'Example: Accuracy = {accuracy_example:.2%}')
            ax.set_ylim([0, 100])
            for i, v in enumerate([accuracy_example * 100, (1-accuracy_example) * 100]):
                ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
        
        with metrics_tab2:
            st.markdown("""
            ### Precision
            
            **Formula:**
            ```
            Precision = TP / (TP + FP)
            ```
            
            **Definition:**
            - Ratio of true positives to all predicted positives
            - Measures how many of the predicted positives are actually positive
            - Range: 0 to 1 (higher is better)
            
            **Interpretation:**
            - 1.0 = All positive predictions are correct
            - 0.5 = Half of positive predictions are correct
            - Low precision = Many false positives
            
            **Use Case:**
            - Important when false positives are costly
            - Example: Medical diagnosis (don't want false alarms)
            """)
            
            # Visual example
            fig, ax = plt.subplots(figsize=(8, 6))
            tp, fp = 85, 15
            precision_example = tp / (tp + fp)
            ax.bar(['True Positives', 'False Positives'], 
                  [tp, fp],
                  color=['green', 'red'], alpha=0.7)
            ax.set_ylabel('Count')
            ax.set_title(f'Example: Precision = {precision_example:.2%} (TP={tp}, FP={fp})')
            for i, v in enumerate([tp, fp]):
                ax.text(i, v + 2, str(v), ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
        
        with metrics_tab3:
            st.markdown("""
            ### Recall (Sensitivity)
            
            **Formula:**
            ```
            Recall = TP / (TP + FN)
            ```
            
            **Definition:**
            - Ratio of true positives to all actual positives
            - Measures how many actual positives were found
            - Range: 0 to 1 (higher is better)
            
            **Interpretation:**
            - 1.0 = All actual positives were found
            - 0.5 = Only half of actual positives were found
            - Low recall = Many false negatives (missed cases)
            
            **Use Case:**
            - Important when false negatives are costly
            - Example: Disease detection (don't want to miss cases)
            """)
            
            # Visual example
            fig, ax = plt.subplots(figsize=(8, 6))
            tp, fn = 80, 20
            recall_example = tp / (tp + fn)
            ax.bar(['True Positives', 'False Negatives'], 
                  [tp, fn],
                  color=['green', 'orange'], alpha=0.7)
            ax.set_ylabel('Count')
            ax.set_title(f'Example: Recall = {recall_example:.2%} (TP={tp}, FN={fn})')
            for i, v in enumerate([tp, fn]):
                ax.text(i, v + 2, str(v), ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
        
        with metrics_tab4:
            st.markdown("""
            ### F1 Score
            
            **Formula:**
            ```
            F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
            ```
            
            **Definition:**
            - Harmonic mean of Precision and Recall
            - Balances both metrics
            - Range: 0 to 1 (higher is better)
            
            **Interpretation:**
            - 1.0 = Perfect precision and recall
            - 0.5 = Moderate performance
            - Useful when you need to balance precision and recall
            
            **Use Case:**
            - When both false positives and false negatives matter
            - Single metric to optimize
            - Better than accuracy for imbalanced datasets
            """)
            
            # Visual example
            fig, ax = plt.subplots(figsize=(8, 6))
            precision_val = 0.85
            recall_val = 0.80
            f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)
            
            metrics_names = ['Precision', 'Recall', 'F1 Score']
            metrics_values = [precision_val, recall_val, f1_val]
            colors = ['blue', 'green', 'purple']
            
            bars = ax.bar(metrics_names, [v * 100 for v in metrics_values], 
                         color=colors, alpha=0.7)
            ax.set_ylabel('Score (%)')
            ax.set_title('F1 Score as Harmonic Mean of Precision and Recall')
            ax.set_ylim([0, 100])
            for bar, val in zip(bars, metrics_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{val:.2%}', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
        
        with metrics_tab5:
            st.markdown("""
            ### Confusion Matrix
            
            **Structure:**
            ```
                    Predicted
                   Class0  Class1  Class2
            Class0   TP     FP     FP
            Class1   FN     TP     FP
            Class2   FN     FN     TP
            ```
            
            **Components:**
            - **TP (True Positive):** Correctly predicted positive
            - **TN (True Negative):** Correctly predicted negative
            - **FP (False Positive):** Incorrectly predicted as positive
            - **FN (False Negative):** Incorrectly predicted as negative
            
            **Interpretation:**
            - Diagonal values = Correct predictions
            - Off-diagonal values = Incorrect predictions
            - Higher diagonal = Better model
            
            **Use Case:**
            - Visual representation of model performance
            - Identify which classes are confused
            - Understand error patterns
            """)
            
            # Example confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            example_cm = np.array([[120, 5, 2],
                                  [8, 115, 3],
                                  [3, 4, 118]])
            class_names_ex = ['Normal', 'Pneumonia', 'COVID-19']
            sns.heatmap(example_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=class_names_ex, yticklabels=class_names_ex)
            ax.set_title('Example Confusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)
        
        # Metrics Comparison Table
        st.markdown("---")
        st.subheader("📊 Metrics Comparison Guide")
        
        comparison_guide = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Best Value': ['1.0', '1.0', '1.0', '1.0'],
            'Worst Value': ['0.0', '0.0', '0.0', '0.0'],
            'When to Use': [
                'Balanced datasets, overall performance',
                'False positives are costly',
                'False negatives are costly',
                'Need balance between precision and recall'
            ],
            'Formula': [
                '(TP+TN)/(TP+TN+FP+FN)',
                'TP/(TP+FP)',
                'TP/(TP+FN)',
                '2×(Precision×Recall)/(Precision+Recall)'
            ]
        })
        st.dataframe(comparison_guide, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "X-ray Disease Classification System | Built with Streamlit & TensorFlow by Raman Kumar"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

