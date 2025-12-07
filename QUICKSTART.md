# Quick Start Guide

## Step-by-Step Instructions

### Step 1: Check Your Dataset
First, verify that your dataset is properly set up:

```bash
python check_dataset.py
```

This will show you:
- Number of training images
- Class distribution
- Any missing files

### Step 2: Install Dependencies
Install all required Python packages:

```bash
pip install -r requirements.txt
```

**Note:** If you encounter any installation issues, you may need to install TensorFlow separately:
```bash
pip install tensorflow==2.15.0
```

### Step 3: Train the Model
Train the CNN model on your dataset:

```bash
python train_model.py
```

**What happens during training:**
- Images are loaded and preprocessed
- Data is split into training and validation sets
- CNN model is created and trained
- Model performance is evaluated
- Results are saved

**Expected output files:**
- `xray_disease_model.h5` - Trained model
- `class_names.pkl` - Class labels
- `confusion_matrix.png` - Confusion matrix visualization
- `training_history.png` - Training/validation curves

**Training time:** Approximately 30-60 minutes depending on your hardware.

### Step 4: Run the Streamlit App
Launch the web application:

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Step 5: Use the Application

1. **Navigate to "Predict Disease" page**
2. **Upload an X-ray image** (JPG, JPEG, or PNG format)
3. **View the prediction results:**
   - Predicted disease class
   - Confidence score
   - Probability distribution for all classes

## Troubleshooting

### Issue: Model file not found
**Solution:** Make sure you've trained the model first by running `python train_model.py`

### Issue: Out of memory during training
**Solution:** Reduce `BATCH_SIZE` in `train_model.py` (try 16 or 8)

### Issue: Training is too slow
**Solution:** 
- Reduce `EPOCHS` for quick testing
- Use a GPU if available
- Reduce `IMG_SIZE` (try 128 or 160)

### Issue: Streamlit app won't start
**Solution:** 
- Make sure Streamlit is installed: `pip install streamlit`
- Check if port 8501 is already in use
- Try: `streamlit run app.py --server.port 8502`

## Project Structure Summary

```
X-ray model/
├── train_images/          # Your training images
├── test_images/           # Your test images  
├── labels_train.csv       # Training labels
├── train_model.py         # Training script
├── app.py                 # Streamlit app
├── check_dataset.py       # Dataset checker
├── requirements.txt       # Dependencies
└── README.md             # Full documentation
```

## Next Steps

After successfully running the project:

1. **Experiment with model architecture** - Modify CNN layers in `train_model.py`
2. **Try different hyperparameters** - Adjust learning rate, batch size, etc.
3. **Add more evaluation metrics** - Implement additional metrics from Unit III
4. **Enhance the UI** - Customize the Streamlit app appearance
5. **Add more models** - Implement SVM, Random Forest, etc. from the syllabus

## Need Help?

- Check the full `README.md` for detailed documentation
- Review the code comments in `train_model.py` and `app.py`
- Ensure all dependencies are correctly installed

Happy coding! 🚀

