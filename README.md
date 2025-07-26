# Deep Learning Project: Handwritten Digit Classification with CNN & MNIST Dataset ✍️📸

## 📚 Overview  
This project covers building, training, and evaluating a convolutional neural network (CNN) for image classification. Using the classic MNIST dataset of handwritten digits, you'll learn how to preprocess image data, design a CNN model with Keras, and assess its performance.

---

## 🎯 Objectives  
By the end of this project, you will:

- Define a clear machine learning image classification problem  
- Import and explore the MNIST handwritten digit image dataset  
- Visualize sample images to understand the data  
- Prepare image data for CNN modeling (normalization, reshaping)  
- Build and train a convolutional neural network using Keras  
- Evaluate model performance on training and test data  

---

## 🧠 Problem Statement  
Classify grayscale images of handwritten digits (0–9) using a CNN architecture trained on the MNIST dataset.

---

## 🛠️ Project Steps  

### 1. Data Loading and Exploration  
- Load the MNIST dataset using Keras utilities  
- Visualize sample images and their labels  
- Split data into training and testing sets  

### 2. Data Preparation  
- Normalize pixel values  
- Reshape data to fit CNN input requirements  
- Convert labels to categorical format  

### 3. CNN Construction and Training  
- Build a CNN architecture with convolutional, pooling, and dense layers  
- Compile the model with appropriate loss and optimizer  
- Train the model on the training set with validation  

### 4. Evaluation  
- Assess model accuracy and loss on training and test sets  
- Visualize training history (accuracy/loss curves)  
- Generate classification reports or confusion matrices (optional)  

---

## 💻 Sample Code Snippet  
```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Visualize a sample image
plt.imshow(X_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

# Preprocess data
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train_cat, epochs=5, validation_split=0.1)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"Test accuracy: {test_acc:.4f}")
 
