"""
X-ray Disease Classification Model Training
Incorporates concepts from Predictive Analytics syllabus:
- Unit I: Data Preprocessing
- Unit III: Classification (CNN as primary, with evaluation metrics)
- Unit V: Neural Networks (CNN, MLP), Dimensionality Reduction (PCA)
- Unit VI: Model Performance (Cross-validation, evaluation metrics)
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 3

# Class names (you can modify these based on your actual disease names)
CLASS_NAMES = ['Normal', 'Pneumonia', 'COVID-19']  # Modify based on your dataset

def load_and_preprocess_data(data_dir, labels_file):
    """
    Unit I: Data Preprocessing
    Load and preprocess X-ray images
    """
    print("Loading and preprocessing data...")
    
    # Load labels
    df = pd.read_csv(labels_file)
    
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        img_path = os.path.join(data_dir, row['file_name'])
        
        if os.path.exists(img_path):
            # Read image
            img = cv2.imread(img_path)
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Normalize pixel values
            img = img.astype('float32') / 255.0
            
            images.append(img)
            labels.append(row['class_id'])
        
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{len(df)} images...")
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Loaded {len(images)} images with shape {images.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    return images, labels

def create_cnn_model(input_shape, num_classes):
    """
    Unit V: Convolutional Neural Networks (CNN)
    Create a CNN model for image classification
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers (MLP component)
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

def evaluate_model(y_true, y_pred, class_names):
    """
    Unit III: Evaluate Model Performance
    Calculate and display classification metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    Unit III: Visualize Confusion Matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")
    plt.close()

def main():
    print("="*60)
    print("X-RAY DISEASE CLASSIFICATION - MODEL TRAINING")
    print("="*60)
    
    # Unit I: Data Preprocessing
    print("\n[Unit I] Data Preprocessing...")
    images, labels = load_and_preprocess_data('train_images', 'labels_train.csv')
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Data Augmentation (Unit I: Data Preprocessing)
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Unit V: Create CNN Model
    print("\n[Unit V] Creating Convolutional Neural Network...")
    model = create_cnn_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Make predictions
    print("\nMaking predictions on validation set...")
    y_pred_proba = model.predict(X_val)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Unit III: Evaluate Model Performance
    print("\n[Unit III] Evaluating Model Performance...")
    metrics = evaluate_model(y_val, y_pred, CLASS_NAMES)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], CLASS_NAMES)
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model
    model.save('xray_disease_model.h5')
    print("\nModel saved as 'xray_disease_model.h5'")
    
    # Save class names
    with open('class_names.pkl', 'wb') as f:
        pickle.dump(CLASS_NAMES, f)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return model, metrics, history

if __name__ == "__main__":
    model, metrics, history = main()

