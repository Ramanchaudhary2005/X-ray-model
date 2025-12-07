"""
Quick script to check dataset structure and statistics
"""

import pandas as pd
import os
from collections import Counter

def check_dataset():
    print("="*60)
    print("DATASET CHECKER")
    print("="*60)
    
    # Check labels file
    if os.path.exists('labels_train.csv'):
        print("\n✅ Labels file found: labels_train.csv")
        df = pd.read_csv('labels_train.csv')
        print(f"   Total samples: {len(df)}")
        print(f"   Columns: {df.columns.tolist()}")
        
        # Check class distribution
        class_dist = df['class_id'].value_counts().sort_index()
        print(f"\n   Class distribution:")
        for class_id, count in class_dist.items():
            print(f"      Class {class_id}: {count} samples ({count/len(df)*100:.2f}%)")
    else:
        print("\n❌ Labels file not found: labels_train.csv")
    
    # Check training images
    if os.path.exists('train_images'):
        train_files = [f for f in os.listdir('train_images') if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\n✅ Training images directory found")
        print(f"   Total images: {len(train_files)}")
        if len(train_files) > 0:
            print(f"   Sample files: {train_files[:3]}")
    else:
        print("\n❌ Training images directory not found: train_images")
    
    # Check test images
    if os.path.exists('test_images'):
        test_files = [f for f in os.listdir('test_images') if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\n✅ Test images directory found")
        print(f"   Total images: {len(test_files)}")
    else:
        print("\n❌ Test images directory not found: test_images")
    
    # Check if labels match images
    if os.path.exists('labels_train.csv') and os.path.exists('train_images'):
        df = pd.read_csv('labels_train.csv')
        train_files = set([f for f in os.listdir('train_images') if f.endswith(('.jpg', '.jpeg', '.png'))])
        label_files = set(df['file_name'].tolist())
        
        missing_images = label_files - train_files
        extra_images = train_files - label_files
        
        print(f"\n📊 File matching:")
        print(f"   Images in labels: {len(label_files)}")
        print(f"   Images in directory: {len(train_files)}")
        print(f"   Missing images: {len(missing_images)}")
        print(f"   Extra images: {len(extra_images)}")
        
        if len(missing_images) > 0:
            print(f"\n   ⚠️  Warning: {len(missing_images)} images in labels but not in directory")
            if len(missing_images) <= 5:
                print(f"   Missing files: {list(missing_images)[:5]}")
    
    print("\n" + "="*60)
    print("Dataset check complete!")
    print("="*60)

if __name__ == "__main__":
    check_dataset()

