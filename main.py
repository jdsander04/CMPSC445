import numpy as np
import tensorflow as tf


def t_sne(x, dim=2, learning_rate=0.01, iterations=1000):
    """
    A Tensorflow implementation of t-SNE.

    Parameters
    ----------
    x : array, shape (n_samples, n_features)
        The input data.
    dim : int, optional
        The desired number of dimensions in the output.
    learning_rate : float, optional
        The learning rate of the optimization algorithm.
    iterations : int, optional
        The number of iterations of the optimization algorithm.

    Returns
    -------
    y : array, shape (n_samples, n_dim)
        The output data.
    """
    n_samples = x.shape[0]
    x = tf.Variable(x, dtype=tf.float32)

    # Compute the similarity matrix
    def similarity_matrix(x):
        sum_x = tf.reduce_sum(tf.square(x), 1)
        D = tf.add(tf.subtract(sum_x, 2 * tf.matmul(x, tf.transpose(x))), tf.transpose(sum_x))
        return tf.exp(-D / 2)

    # Compute the gradient of the Kullback-Leibler divergence
    def gradient(y):
        Q = similarity_matrix(y) / tf.reduce_sum(similarity_matrix(y))
        P = similarity_matrix(x) / tf.reduce_sum(similarity_matrix(x))
        grad = 4 * (tf.add(P, Q) - 2 * P * Q)
        return grad

    # Optimize the KL-divergence
    y = tf.Variable(tf.random.normal([n_samples, dim]), dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for _ in range(iterations):
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(gradient(y))
        grads = tape.gradient(loss, [y])
        optimizer.apply_gradients(zip(grads, [y]))  # Apply the gradients to the variable `y`

    return y.numpy()


def generate_high_dimensional_model(n_samples, n_features, n_classes):
    """
    Generate a high-dimensional model.

    Parameters
    ----------
    n_samples : int
        The number of samples.
    n_features : int
        The number of features.
    n_classes : int
        The number of classes.

    Returns
    -------
    X : array, shape (n_samples, n_features)
        The input data.
    y : array, shape (n_samples,)
        The labels.
    """
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y


def visualize(X, y, X_tsne):
    """
    Visualize the high-dimensional model and the t-SNE reduced data.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The input data.
    y : array, shape (n_samples,)
        The labels.
    X_tsne : array, shape (n_samples, 2)
        The t-SNE reduced data.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(X[:, 0], X[:, 1], c=y)
    ax[0].set_title('High-dimensional model')
    ax[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
    ax[1].set_title('t-SNE reduced data')
    plt.show()


if __name__ == '__main__':
    n_samples = 1000
    n_features = 20
    n_classes = 5
    X, y = generate_high_dimensional_model(n_samples, n_features, n_classes)
    X_tsne = t_sne(X, dim=2)
    visualize(X, y, X_tsne)
