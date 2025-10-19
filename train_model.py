import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import json
import os
from pathlib import Path

# Buat folder model jika belum ada
Path('model').mkdir(exist_ok=True)

# Dataset path
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Data generator dengan augmentasi
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Gunakan class_mode='categorical' untuk multi-class classification
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Tetap konsisten dengan input_shape model
    batch_size=32,
    class_mode='categorical',  # Ubah dari 'binary' ke 'categorical'
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # Ubah dari 'binary' ke 'categorical'
    shuffle=False
)

# Dapatkan jumlah kelas dari train_data
num_classes = len(train_data.class_indices)

print(f"Number of classes: {num_classes}")
print(f"Class indices: {train_data.class_indices}")

# Model CNN untuk multi-class classification
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),
    
    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Ubah ke softmax untuk multi-class
])

model.compile(
    optimizer=Adam(learning_rate=0.0001), 
    loss='categorical_crossentropy',  # Ubah dari binary_crossentropy
    metrics=['accuracy']
)

# Tampilkan summary model
model.summary()

# Training model
print("Starting training...")
history = model.fit(
    train_data, 
    epochs=10, 
    validation_data=test_data,
    verbose=1
)

# Simpan model
model.save('model/cnn_model.h5')
print("Model saved successfully to model/cnn_model.h5")

# Simpan class indices untuk mapping
class_mapping = {v: k for k, v in train_data.class_indices.items()}
with open('class_indices.json', 'w') as f:
    json.dump(class_mapping, f, indent=4)

print("Class indices saved to class_indices.json")
print(f"Class mapping: {class_mapping}")

# Evaluasi model
print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Print informasi penting untuk aplikasi Streamlit
print("\n" + "="*50)
print("STREAMLIT APP COMPATIBILITY INFO:")
print(f"Model input shape: {model.layers[0].input_shape}")
print(f"Number of classes: {num_classes}")
print(f"Expected image size: 150x150 pixels")
print("Class indices saved to: class_indices.json")
print("="*50)