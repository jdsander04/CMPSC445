import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import mnist

TF_ENABLE_ONEDNN_OPTS=0

# Load the MNIST dataset
def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Normalize data between 0 and 1 and flatten images
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = X_train.reshape(-1, 28 * 28)  # Flatten images to vectors
    X_test = X_test.reshape(-1, 28 * 28)
    return X_train, y_train, X_test, y_test

# Compute pairwise Euclidean distances
def pairwise_distances(X):
    sum_X = tf.reduce_sum(tf.square(X), 1)
    D = tf.add(tf.expand_dims(sum_X, axis=1), tf.expand_dims(sum_X, axis=0)) - 2 * tf.matmul(X, tf.transpose(X))
    return D

# Compute similarity matrix
def similarity_matrix(D, sigma=1.0):
    P = tf.exp(-D / (2 * sigma ** 2))
    P /= tf.reduce_sum(P, axis=1, keepdims=True)
    return P

# Neural network for Parametric t-SNE
def build_parametric_tsne(input_dim, output_dim=2):
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(input_dim,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(output_dim, activation=None))  # Output in 2D
    return model

# Custom loss inspired by KL-divergence in t-SNE
def tsne_loss(P, Q):
    epsilon = 1e-10  # Small constant to avoid log(0)
    loss = tf.reduce_sum(P * tf.math.log((P + epsilon) / (Q + epsilon)))
    return loss

# Training function
def train_parametric_tsne(X, model, epochs=100, learning_rate=0.01, sigma=1.0):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # Forward pass: get 2D embeddings
            Y = model(X)

            # Compute pairwise distances in original space and 2D space
            D_high = pairwise_distances(X)
            D_low = pairwise_distances(Y)

            # Compute similarity matrices (P for high-dimensional, Q for 2D space)
            P = similarity_matrix(D_high, sigma)
            Q = similarity_matrix(D_low, sigma)

            # Compute t-SNE loss (KL divergence between P and Q)
            loss = tsne_loss(P, Q)

        # Compute gradients and update weights
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")

    return model

# Compute t-SNE embeddings using Scikit-Learn
def compute_tsne(X, y, perplexity=30, learning_rate=200, n_iter=1000):
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

# Visualize the original and reduced data
def visualize(X_tsne, y, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
    plt.colorbar()
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    # Load the MNIST dataset
    X_train, y_train, X_test, y_test = load_mnist_data()

    # Subsample for faster training
    X_train, y_train = X_train[:10000], y_train[:10000]
    X_test, y_test = X_test[:2000], y_test[:2000]

    # Convert to TensorFlow tensors
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)

    # Build the parametric t-SNE model
    parametric_tsne_model = build_parametric_tsne(input_dim=28*28, output_dim=2)

    # Train the parametric t-SNE model
    trained_model = train_parametric_tsne(X_train_tensor, parametric_tsne_model, epochs=100, sigma=1.0)

    # Get the 2D embeddings for both train and test data
    X_train_2D = trained_model.predict(X_train)
    X_test_2D = trained_model.predict(X_test)

    # Visualize the 2D embeddings using Scikit-Learn
    X_train_tsne = compute_tsne(X_train, y_train, perplexity=30, learning_rate=200, n_iter=1000)
    X_test_tsne = compute_tsne(X_test, y_test, perplexity=30, learning_rate=200, n_iter=1000)

    # Visualize the improvements between epochs against actual t-SNE of the MNIST data
    visualize(X_train_tsne, y_train, title="2D Embeddings of MNIST Training Data using t-SNE")
    visualize(X_test_tsne, y_test, title="2D Embeddings of MNIST Test Data using t-SNE")
    visualize(X_train_2D, y_train, title="2D Embeddings of MNIST Training Data using Parametric t-SNE")
    visualize(X_test_2D, y_test, title="2D Embeddings of MNIST Test Data using Parametric t-SNE")

