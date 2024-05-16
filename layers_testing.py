import tensorflow as tf

# Define a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the input image to a 1D array
    tf.keras.layers.Dense(128, activation='relu'),  # Dense hidden layer with ReLU activation
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model (code for training not shown here)
