import numpy as np
import cv2
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Load dataset with steering angles embedded in filenames (e.g., image_123_steering_0.35.png)
def load_data(image_folder):
    images = []
    angles = []

    # Regex pattern to extract steering angle from filename (adjust if your naming scheme is different)
    pattern = re.compile(r".*_steering_([-+]?[0-9]*\.?[0-9]+)\.png")  # e.g., image_123_steering_0.35.png

    for filename in os.listdir(image_folder):
        match = pattern.match(filename)
        if match:
            # Extract steering angle
            angle = float(match.group(1))
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            images.append(img)
            angles.append(angle)

    return np.array(images), np.array(angles)

# Preprocess the images (resize and normalize)
def preprocess_images(images):
    processed_images = []
    for img in images:
        img = cv2.resize(img, (200, 66))  # Resize to (200, 66)
        img = img / 255.0  # Normalize to [0, 1]
        processed_images.append(img)
    return np.array(processed_images)

# Set the path to your image folder
image_folder = 'C:/Users/raouf/Downloads/data_road_224/training/'  # Path to your images (no labels.txt needed)

# Load and preprocess data
images, angles = load_data(image_folder)
images = preprocess_images(images)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(horizontal_flip=True)
datagen.fit(X_train)

# Define the model
model = Sequential([
    Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)),
    Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
    Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1)  # Steering angle output
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_val, y_val),
                    epochs=10,  # Change as needed
                    steps_per_epoch=len(X_train) // 32)

# Save the model
model.save('steering_angle_model.h5')

# Plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
