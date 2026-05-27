# Handwritten Digit Recognizer using MNIST Dataset

# Import libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = keras.datasets.mnist

# Split data into training and testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data (0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Display one sample image
plt.imshow(x_train[0], cmap='gray')
plt.title("Sample Digit")
plt.show()

# Build neural network model
model = keras.Sequential([
    
    # Convert 28x28 image into 1D array
    keras.layers.Flatten(input_shape=(28, 28)),
    
    # Hidden layer
    keras.layers.Dense(128, activation='relu'),
    
    # Output layer (10 digits)
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(x_train, y_train, epochs=5)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", test_acc)

# Predict first test image
prediction = model.predict(x_test)

# Show prediction
print("\nPredicted Digit:", prediction[0].argmax())

# Display image
plt.imshow(x_test[0], cmap='gray')
plt.title("Test Image")
plt.show()