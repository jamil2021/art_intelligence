# ### TensorFlow with  keras
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.datasets import mnist

# # Load the MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# # Define the model architecture using the Keras Sequential API
# model = models.Sequential([
#     layers.Flatten(input_shape=(28, 28)),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(10, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# model.fit(x_train, y_train, epochs=5)

# # Evaluate the model
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print('Test accuracy:', test_acc)


# ### PyTorch example
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms

# # Define the model architecture using PyTorch's nn.Module
# class NeuralNet(nn.Module):
#     def __init__(self):
#         super(NeuralNet, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(28*28, 128)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# # Load the MNIST dataset
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# # Initialize the model
# model = NeuralNet()

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Train the model
# for epoch in range(5):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch {epoch+1}, loss: {running_loss/len(trainloader)}")

# # Evaluate the model
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         inputs, labels = data
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Test accuracy:', correct / total)


# ### Example of TensorFlow without Keras
# import tensorflow as tf
# from tensorflow.keras.datasets import mnist
# import numpy as np

# # Load the MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# # Define the model architecture
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# # Compile the model
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam()
# model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# # Convert NumPy arrays to TensorFlow tensors
# x_train_tf = tf.convert_to_tensor(x_train, dtype=tf.float32)
# y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.int32)

# # Train the model
# batch_size = 64
# epochs = 5
# for epoch in range(epochs):
#     for i in range(0, len(x_train_tf), batch_size):
#         x_batch = x_train_tf[i:i+batch_size]
#         y_batch = y_train_tf[i:i+batch_size]
#         with tf.GradientTape() as tape:
#             predictions = model(x_batch, training=True)
#             loss = loss_fn(y_batch, predictions)
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     print(f"Epoch {epoch+1}, loss: {loss.numpy()}")

# # Evaluate the model
# x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
# y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.int32)
# test_loss, test_acc = model.evaluate(x_test_tf, y_test_tf)
# print('Test accuracy:', test_acc)
