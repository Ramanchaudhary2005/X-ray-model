"""
Example script showing how to use the trained model programmatically
(Alternative to using the Streamlit app)
"""

import numpy as np
import cv2
from tensorflow import keras
import pickle
import os

def load_model_and_classes():
    """Load the trained model and class names"""
    try:
        model = keras.models.load_model('xray_disease_model.h5')
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        return model, class_names
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first by running: python train_model.py")
        return None, None

def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (224, 224))
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img, axis=0)
    
    return img_batch

def predict_disease(image_path, model, class_names):
    """Predict disease from an image"""
    # Preprocess image
    img_batch = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_batch, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Get all probabilities
    results = {}
    for i, class_name in enumerate(class_names):
        results[class_name] = predictions[0][i] * 100
    
    return {
        'predicted_class': class_names[predicted_class],
        'confidence': confidence * 100,
        'all_probabilities': results
    }

def main():
    """Example usage"""
    print("="*60)
    print("X-ray Disease Prediction - Example Script")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model, class_names = load_model_and_classes()
    
    if model is None:
        return
    
    print(f"Model loaded successfully!")
    print(f"Classes: {class_names}")
    
    # Example: Predict on a test image
    test_images_dir = 'test_images'
    if os.path.exists(test_images_dir):
        test_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(test_files) > 0:
            # Predict on first test image
            test_image_path = os.path.join(test_images_dir, test_files[0])
            print(f"\nPredicting on: {test_files[0]}")
            
            try:
                result = predict_disease(test_image_path, model, class_names)
                
                print("\n" + "="*60)
                print("PREDICTION RESULTS")
                print("="*60)
                print(f"Predicted Disease: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.2f}%")
                print("\nAll Probabilities:")
                for disease, prob in sorted(result['all_probabilities'].items(), 
                                          key=lambda x: x[1], reverse=True):
                    print(f"  {disease}: {prob:.2f}%")
                
            except Exception as e:
                print(f"Error during prediction: {e}")
        else:
            print(f"\nNo test images found in {test_images_dir}")
    else:
        print(f"\nTest images directory not found: {test_images_dir}")
        print("\nTo use this script:")
        print("1. Train the model: python train_model.py")
        print("2. Provide an image path to predict_disease() function")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()

