# Prompt: Brain MRI Classification Project

## Project Overview
Create a comprehensive machine learning project for classifying brain MRI images into different categories (e.g., Normal, Tumor, Stroke, etc.) using the same structure and syllabus coverage as the X-ray Disease Classification project.

## Dataset Requirements

### Dataset Structure
```
brain-mri-project/
├── train_images/          # Training MRI images
├── test_images/           # Test MRI images
├── labels_train.csv       # Training labels (file_name, class_id)
└── labels_test.csv        # Test labels (optional)
```

### Label Format (labels_train.csv)
```csv
file_name,class_id
brain_001.jpg,0
brain_002.jpg,1
brain_003.jpg,2
...
```

### Class Categories (Example)
- Class 0: Normal
- Class 1: Brain Tumor
- Class 2: Stroke
- Class 3: Hemorrhage
(Adjust based on your dataset)

## Project Structure to Create

```
brain-mri-project/
│
├── train_images/          # Training MRI images
├── test_images/           # Test MRI images
├── labels_train.csv       # Training labels
│
├── train_model.py         # Model training script
├── app.py                 # Streamlit web application
├── check_dataset.py       # Dataset verification script
├── predict_example.py     # Example prediction script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── QUICKSTART.md         # Quick start guide
├── SYLLABUS_MAPPING.md   # Syllabus coverage
└── .gitignore           # Git ignore file
```

## Key Adaptations Needed

### 1. Update Class Names
**In train_model.py and app.py:**
```python
# Change from:
CLASS_NAMES = ['Normal', 'Pneumonia', 'COVID-19']

# To (example):
CLASS_NAMES = ['Normal', 'Brain Tumor', 'Stroke', 'Hemorrhage']
```

### 2. Update Image Preprocessing
**MRI images are typically:**
- Grayscale (single channel) or RGB
- Different sizes (commonly 256x256, 512x512)
- May need different normalization

**In train_model.py:**
```python
# Adjust IMG_SIZE if needed
IMG_SIZE = 256  # or 224, depending on your dataset

# If MRI images are grayscale, convert to RGB:
if len(img.shape) == 2:  # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
```

### 3. Update Project Descriptions
**Replace all mentions of:**
- "X-ray" → "Brain MRI"
- "chest X-ray" → "brain MRI scan"
- "disease" → "condition" or "abnormality"
- "Pneumonia/COVID-19" → "Brain Tumor/Stroke/etc."

### 4. Update README.md
**Change:**
- Project title
- Dataset description
- Class descriptions
- Use case examples

### 5. Update App Titles and Headers
**In app.py:**
```python
# Change:
st.markdown('<h1 class="main-header">🏥 X-ray Disease Classification System</h1>', ...)
# To:
st.markdown('<h1 class="main-header">🧠 Brain MRI Classification System</h1>', ...)
```

## Step-by-Step Implementation

### Step 1: Prepare Your Dataset
1. Organize MRI images into `train_images/` and `test_images/` folders
2. Create `labels_train.csv` with format: `file_name,class_id`
3. Ensure class_id is numeric (0, 1, 2, ...)

### Step 2: Copy and Adapt Files
1. Copy all files from X-ray project
2. Rename project folder to `brain-mri-project`
3. Update all file references

### Step 3: Update train_model.py
**Key changes:**
```python
# Line ~33: Update class names
CLASS_NAMES = ['Normal', 'Brain Tumor', 'Stroke']  # Your classes

# Line ~27-29: Adjust image size if needed
IMG_SIZE = 256  # Common for MRI
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 3  # Update based on your classes
```

### Step 4: Update app.py
**Key changes:**
1. **Line ~100-101:** Update header
```python
st.markdown('<h1 class="main-header">🧠 Brain MRI Classification System</h1>', ...)
st.markdown('<p class="sub-header">Classify brain conditions from MRI images using Deep Learning</p>', ...)
```

2. **Line ~71:** Update default class names
```python
return ['Normal', 'Brain Tumor', 'Stroke']  # Your classes
```

3. **Line ~122:** Update file uploader text
```python
"Choose a brain MRI image...",
help="Upload a brain MRI scan in JPG, JPEG, or PNG format"
```

4. **Update all page descriptions** to mention MRI instead of X-ray

### Step 5: Update Documentation Files

**README.md changes:**
- Title: "Brain MRI Classification System"
- Description: "Classify brain conditions from MRI images"
- Update class descriptions
- Update dataset information

**SYLLABUS_MAPPING.md:**
- Update examples to mention MRI
- Keep all syllabus coverage the same

### Step 6: Test the Project
1. Run `python check_dataset.py` to verify dataset
2. Run `python train_model.py` to train model
3. Run `streamlit run app.py` to test the app

## Specific Code Changes Checklist

### train_model.py
- [ ] Update `CLASS_NAMES` list
- [ ] Adjust `IMG_SIZE` if needed (256 for MRI is common)
- [ ] Update `NUM_CLASSES`
- [ ] Check image loading handles grayscale/RGB correctly
- [ ] Update print statements mentioning "X-ray" → "MRI"

### app.py
- [ ] Update page title and headers
- [ ] Update class names in `load_class_names()`
- [ ] Update file uploader descriptions
- [ ] Update all text mentioning "X-ray" → "MRI"
- [ ] Update disease names → condition names
- [ ] Update emoji from 🏥 to 🧠 (optional)

### README.md
- [ ] Update project title
- [ ] Update description
- [ ] Update dataset information
- [ ] Update class labels
- [ ] Update use case examples

### requirements.txt
- [ ] Keep the same (no changes needed)

## Example Prompt for AI Assistant

Use this prompt when creating the project:

```
I want to create a brain MRI classification project similar to my X-ray disease classification project. 

The project should:
1. Classify brain MRI images into categories: Normal, Brain Tumor, Stroke
2. Use the same structure as the X-ray project
3. Cover the same syllabus units (I, III, V, VI)
4. Include Streamlit app with all navigation pages
5. Have regression models, classification models, and model comparison
6. Include dataset overview, preprocessing, and evaluation metrics

My dataset structure:
- train_images/ folder with MRI images
- test_images/ folder with MRI images  
- labels_train.csv with file_name and class_id columns

Classes:
- 0: Normal
- 1: Brain Tumor
- 2: Stroke

Please adapt all the code from the X-ray project to work with brain MRI images, updating:
- All references from "X-ray" to "Brain MRI"
- Class names
- Image preprocessing (handle grayscale if needed)
- All descriptions and documentation
- App headers and titles

Keep the same functionality, structure, and syllabus coverage.
```

## Additional Considerations for MRI

### Image Characteristics
- MRI images may be grayscale (single channel)
- Common sizes: 256x256, 512x512
- May need different preprocessing
- Consider 3D MRI volumes (if applicable)

### Medical Context
- Brain MRI shows different anatomy than chest X-ray
- Different abnormalities to detect
- May need different augmentation strategies

### Model Adjustments
- May need different architecture for MRI
- Consider transfer learning from medical imaging models
- Adjust input size based on your MRI dimensions

## Quick Start Commands

```bash
# 1. Check dataset
python check_dataset.py

# 2. Train model
python train_model.py

# 3. Run app
streamlit run app.py
```

## Notes
- Keep all the same functionality and features
- Maintain syllabus coverage
- Use same evaluation metrics
- Keep same navigation structure
- Adapt only the medical imaging context

---

**This prompt provides everything needed to create a similar project for brain MRI classification while maintaining the same educational value and syllabus coverage.**

