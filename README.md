after git clone:
- make venv
```cmd
py -m venv venv
```
- enter venv

There are several ways to potentially improve the training of the neural network (NN) for parametric t-SNE, aiming for better results or faster convergence. Below are some strategies:

### 1. **Architecture and Model Adjustments**
   - **Increase the Model Complexity**: If the current architecture is too simple, increasing the number of layers or the number of units in each layer may allow the model to capture more complex relationships in the data.
     ```python
     def build_parametric_tsne(input_dim, output_dim=2):
         model = models.Sequential()
         model.add(layers.Dense(256, activation='relu', input_shape=(input_dim,)))
         model.add(layers.Dense(128, activation='relu'))
         model.add(layers.Dense(64, activation='relu'))
         model.add(layers.Dense(output_dim, activation=None))  # Output in 2D
         return model
     ```
   - **Use Different Activation Functions**: Try alternatives like Leaky ReLU, ELU, or SELU, which can help the network deal with vanishing gradients better than traditional ReLU.
     ```python
     model.add(layers.Dense(128, activation='selu'))
     ```

   - **Batch Normalization**: Adding batch normalization layers can stabilize and speed up training by normalizing activations, which can help avoid issues with exploding/vanishing gradients.
     ```python
     model.add(layers.BatchNormalization())
     ```

### 2. **Improved Loss Function**
   - **KL-Divergence Smoothing**: The current loss function uses a small constant `epsilon` to avoid division by zero. You could experiment with different values for this constant or use log-sum-exp trick to improve numerical stability.
     ```python
     epsilon = 1e-7  # Smaller value to avoid log(0)
     ```

   - **Regularization Terms**: Add regularization terms to prevent overfitting and encourage smoother embeddings:
     - **L2 Regularization**: Add a penalty on the magnitude of the weights (L2 regularization).
       ```python
       model.add(layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
       ```
     - **Attractive and Repulsive Forces**: Introduce terms in the loss that explicitly attract similar points and repel dissimilar ones in the 2D space.

### 3. **Optimization Adjustments**
   - **Learning Rate Scheduling**: Use a learning rate scheduler that decreases the learning rate as the training progresses. This helps fine-tune the model after the initial fast convergence.
     ```python
     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
         initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9
     )
     optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
     ```

   - **AdamW Optimizer**: Use the AdamW optimizer, which decouples the weight decay (L2 regularization) from the gradient update step. This tends to provide better generalization.
     ```python
     optimizer = tf.keras.optimizers.AdamW(learning_rate=0.01, weight_decay=1e-4)
     ```

   - **Gradient Clipping**: If the model is suffering from exploding gradients, applying gradient clipping can stabilize the training.
     ```python
     tf.clip_by_value(grads, -1.0, 1.0)
     ```

### 4. **Mini-Batch Training**
   - **Use Mini-Batches**: Instead of computing the loss over the entire dataset at each step, break the data into mini-batches. This can reduce the computational cost per iteration and make the training faster and more stable.
     ```python
     dataset = tf.data.Dataset.from_tensor_slices((X_train_tensor, y_train))
     dataset = dataset.shuffle(buffer_size=1024).batch(128)
     for epoch in range(epochs):
         for step, (batch_X, batch_y) in enumerate(dataset):
             with tf.GradientTape() as tape:
                 # Forward pass and loss computation for each mini-batch
     ```

### 5. **Data Augmentation**
   - Although data augmentation is often used in image classification, techniques such as adding noise to the input data might help the model generalize better in this case, as MNIST is visual data. 

   - **Gaussian Noise**: Apply small amounts of noise to the images to help the model learn more robust features.
     ```python
     model.add(layers.GaussianNoise(0.1))
     ```

### 6. **Advanced Similarity Matrix**
   - **Learnable σ (Sigma) Parameter**: Instead of using a fixed value of `sigma` in the similarity matrix, you can make it a learnable parameter. The model will adjust the scale of the Gaussian kernel based on the training data.
     ```python
     sigma = tf.Variable(1.0, trainable=True)
     ```

### 7. **Early Stopping and Model Checkpointing**
   - **Early Stopping**: Stop training once the loss on a validation set stops improving, which can prevent overfitting.
     ```python
     callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
     model.fit(X_train, y_train, epochs=200, callbacks=[callback])
     ```

   - **Model Checkpointing**: Save the best model during training by monitoring the loss and restore the best weights at the end of training.
     ```python
     checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', save_best_only=True)
     model.fit(X_train, y_train, epochs=200, callbacks=[checkpoint])
     ```

### 8. **Preprocessing Enhancements**
   - **Principal Component Analysis (PCA)**: Before feeding data to the parametric t-SNE network, apply PCA to reduce the dimensions of the input data to a smaller size (e.g., from 784 to 100). This can help reduce noise and focus on the most important features.
     ```python
     from sklearn.decomposition import PCA
     pca = PCA(n_components=100)
     X_train_pca = pca.fit_transform(X_train)
     ```

### 9. **Increase Dataset Size**
   - **Train on More Data**: Although the script subsamples the dataset, training on more data (or the full MNIST dataset) could improve the network's performance by exposing it to more variability in the digits.

### Summary of Key Ideas:
1. **Model Adjustments**: Add more layers, different activation functions, and regularization.
2. **Optimization**: Use learning rate scheduling, the AdamW optimizer, and gradient clipping.
3. **Batching**: Implement mini-batch training for better generalization and faster training.
4. **Training Regularization**: Introduce noise, augmentations, and early stopping to improve generalization.
5. **Preprocessing**: Use PCA to reduce input dimensionality before training.

By experimenting with these techniques, you can likely improve the training speed, generalization, and stability of your parametric t-SNE model.