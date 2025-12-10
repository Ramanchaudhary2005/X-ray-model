# Quick Reference: Exact Changes for MRI Brain Project

## File-by-File Changes

### 1. train_model.py

**Line ~33:**
```python
# OLD:
CLASS_NAMES = ['Normal', 'Pneumonia', 'COVID-19']

# NEW:
CLASS_NAMES = ['Normal', 'Brain Tumor', 'Stroke']  # Adjust based on your dataset
```

**Line ~27-30:**
```python
# OLD:
IMG_SIZE = 224
NUM_CLASSES = 3

# NEW (if needed):
IMG_SIZE = 256  # Common for MRI, adjust if needed
NUM_CLASSES = 3  # Update based on your number of classes
```

**Line ~55-56 (in load_and_preprocess_data function):**
```python
# Add after reading image if grayscale:
if len(img.shape) == 2:  # If grayscale
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
```

**Line ~204:**
```python
# OLD:
print("X-RAY DISEASE CLASSIFICATION - MODEL TRAINING")

# NEW:
print("BRAIN MRI CLASSIFICATION - MODEL TRAINING")
```

### 2. app.py

**Line ~100-101:**
```python
# OLD:
st.markdown('<h1 class="main-header">🏥 X-ray Disease Classification System</h1>', ...)
st.markdown('<p class="sub-header">Predict diseases from X-ray images using Deep Learning</p>', ...)

# NEW:
st.markdown('<h1 class="main-header">🧠 Brain MRI Classification System</h1>', ...)
st.markdown('<p class="sub-header">Classify brain conditions from MRI images using Deep Learning</p>', ...)
```

**Line ~71:**
```python
# OLD:
return ['Normal', 'Pneumonia', 'COVID-19']

# NEW:
return ['Normal', 'Brain Tumor', 'Stroke']  # Your classes
```

**Line ~119-122:**
```python
# OLD:
uploaded_file = st.file_uploader(
    "Choose an X-ray image...",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a chest X-ray image in JPG, JPEG, or PNG format"
)

# NEW:
uploaded_file = st.file_uploader(
    "Choose a brain MRI image...",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a brain MRI scan in JPG, JPEG, or PNG format"
)
```

**Line ~150:**
```python
# OLD:
st.markdown(f"### 🎯 Predicted Disease: **{class_names[predicted_class]}**")

# NEW:
st.markdown(f"### 🎯 Predicted Condition: **{class_names[predicted_class]}**")
```

**Line ~154:**
```python
# OLD:
st.markdown("### Probability Distribution:")
prob_df = pd.DataFrame({
    'Disease': class_names,
    'Probability': predictions[0] * 100
})

# NEW:
st.markdown("### Probability Distribution:")
prob_df = pd.DataFrame({
    'Condition': class_names,  # Changed from 'Disease'
    'Probability': predictions[0] * 100
})
```

**Line ~170:**
```python
# OLD:
ax.set_ylabel('Disease', fontsize=12)

# NEW:
ax.set_ylabel('Condition', fontsize=12)
```

**Line ~1234:**
```python
# OLD:
"X-ray Disease Classification System | Built with Streamlit & TensorFlow"

# NEW:
"Brain MRI Classification System | Built with Streamlit & TensorFlow"
```

### 3. README.md

**Title:**
```markdown
# OLD:
# X-ray Disease Classification System

# NEW:
# Brain MRI Classification System
```

**Description:**
```markdown
# OLD:
A comprehensive machine learning project for classifying diseases from X-ray images

# NEW:
A comprehensive machine learning project for classifying brain conditions from MRI images
```

**Dataset Information:**
```markdown
# OLD:
- **Training Images:** 4,672 images
- **Test Images:** 1,168 images
- **Classes:** 3 disease categories
  - Class 0: Normal (1,227 images)
  - Class 1: Pneumonia (2,238 images)
  - Class 2: COVID-19 (1,207 images)

# NEW:
- **Training Images:** [Your count] images
- **Test Images:** [Your count] images
- **Classes:** 3 brain condition categories
  - Class 0: Normal ([count] images)
  - Class 1: Brain Tumor ([count] images)
  - Class 2: Stroke ([count] images)
```

**Replace all instances of:**
- "X-ray" → "Brain MRI"
- "disease" → "condition" or "abnormality"
- "chest X-ray" → "brain MRI scan"
- "Pneumonia/COVID-19" → "Brain Tumor/Stroke"

### 4. check_dataset.py

**Line ~11:**
```python
# OLD:
print("DATASET CHECKER")

# NEW:
print("BRAIN MRI DATASET CHECKER")
```

### 5. predict_example.py

**Line ~67:**
```python
# OLD:
print("X-ray Disease Prediction - Example Script")

# NEW:
print("Brain MRI Classification - Example Script")
```

**Line ~46:**
```python
# OLD:
def predict_disease(image_path, model, class_names):

# NEW:
def predict_condition(image_path, model, class_names):
```

### 6. app.py - All Page Descriptions

**Dataset Overview Page:**
```python
# OLD:
st.markdown("### Sample Images from Dataset")

# NEW:
st.markdown("### Sample Brain MRI Images from Dataset")
```

**Preprocessing Page:**
```python
# OLD:
st.markdown("**Unit I: Data Preprocessing**")
# Update descriptions to mention MRI instead of X-ray
```

**Regression Models Page:**
```python
# OLD:
st.info("""
**Note:** This project focuses on classification (disease categories), but regression concepts 
can be applied for continuous predictions like disease severity scores or confidence levels.
""")

# NEW:
st.info("""
**Note:** This project focuses on classification (brain conditions), but regression concepts 
can be applied for continuous predictions like condition severity scores or confidence levels.
""")
```

**Classification Models Page:**
```python
# OLD:
st.markdown("**Unit III: Supervised Learning - Classification**")

# Keep same, but update descriptions
```

**Model Information Page:**
```python
# OLD:
st.markdown("""
**Unit I: Data Preprocessing**
- Image normalization
- Data augmentation
- Train/validation split
""")

# NEW:
st.markdown("""
**Unit I: Data Preprocessing**
- MRI image normalization
- Data augmentation
- Train/validation split
""")
```

**Evaluation Metrics Page:**
```python
# OLD:
st.markdown("**Unit III & Unit VI: Model Performance Evaluation**")

# Keep same
```

### 7. SYLLABUS_MAPPING.md

**Update examples:**
```markdown
# OLD:
- Project demonstrates predictive analytics for medical image classification
- Real-world application: Disease detection from X-ray images

# NEW:
- Project demonstrates predictive analytics for medical image classification
- Real-world application: Brain condition detection from MRI images
```

## Global Find & Replace

Use these find-and-replace operations across all files:

1. **"X-ray"** → **"Brain MRI"** (case-sensitive)
2. **"x-ray"** → **"brain MRI"** (lowercase)
3. **"chest X-ray"** → **"brain MRI scan"**
4. **"disease"** → **"condition"** (be careful with "disease" in medical context)
5. **"Pneumonia"** → **"Brain Tumor"**
6. **"COVID-19"** → **"Stroke"** (or your class name)
7. **"🏥"** → **"🧠"** (optional, for emoji)

## Quick Checklist

- [ ] Update all class names in train_model.py
- [ ] Update IMG_SIZE if needed (256 for MRI)
- [ ] Update app.py headers and titles
- [ ] Update file uploader text
- [ ] Update README.md title and description
- [ ] Update dataset information in README
- [ ] Update all "X-ray" references to "Brain MRI"
- [ ] Update "disease" to "condition" where appropriate
- [ ] Test dataset loading
- [ ] Test model training
- [ ] Test Streamlit app

## Testing Steps

1. **Verify Dataset:**
   ```bash
   python check_dataset.py
   ```

2. **Train Model:**
   ```bash
   python train_model.py
   ```

3. **Run App:**
   ```bash
   streamlit run app.py
   ```

4. **Test Each Page:**
   - Dataset Overview
   - Dataset Preprocessing
   - Regression Models
   - Classification Models
   - Model Comparison
   - Model Information
   - Evaluation Metrics

## Notes

- Keep all functionality the same
- Only change medical imaging context
- Maintain all syllabus coverage
- Keep same evaluation metrics
- Preserve all navigation structure

