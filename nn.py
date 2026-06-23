import numpy as np
import pickle
from sklearn.model_selection import train_test_split

#Load the dataset
with open('dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Now we have a dictionary with 'images' and 'labels'
images = data['images'] # Shape: (num_samples=3060, height=10, width=10, channels=3)
labels = data['labels'] # Shape: (num_samples=3060,) -values: 0 (heart), 1 (sun), 2 (snake)

# Check the data 
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Unique labels: {np.unique(labels)}")
print(f"Label counts: {np.bincount(labels)}")

# Flatten images
X = images.reshape(len(images), -1) # (3060, 300)

# Split train/test data
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# One-hot encode labels
def one_hot_encode(labels, num_classes=3):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

y_train_onehot = one_hot_encode(y_train)
y_test_onehot = one_hot_encode(y_test)

print(f"Training data: {X_train.shape}")
print(f"Training labels: {y_train_onehot.shape}")
print(f"Test data: {X_test.shape}")
print(f"Test labels: {y_test_onehot.shape}")


"""
Actual NN building
"""

# 1. Create NN architecture:
L = 3
n = [300, 128, 64, 3]

# 2. Prepare data for training (TRANSPOSE for matrix multiplication)
A0 = X_train.T  # (300, 2448) - transposed input
Y = y_train_onehot.T  # (3, 2448) - transposed labels
m = Y.shape[1]  # number of training samples

print(f"\nPrepared for training:")
print(f"A0 shape: {A0.shape}")
print(f"Y shape: {Y.shape}")
print(f"m (num samples): {m}")

# 3. Weights and biases
W1 = np.random.randn(n[1], n[0]) * 0.07  # (128, 300)
W2 = np.random.randn(n[2], n[1]) * 0.07  # (64, 128)
W3 = np.random.randn(n[3], n[2]) * 0.07  # (3, 64)

b1 = np.zeros((n[1], 1))  # (128, 1)
b2 = np.zeros((n[2], 1))  # (64, 1)
b3 = np.zeros((n[3], 1))  # (3, 1)

# 4. Activation functions
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # stability trick
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

# 5. Cost function
def cost(y_hat, Y):
    """
    Categorical cross-entropy loss
    y_hat: predictions (3, m)
    Y: true labels (3, m)
    """
    
    # Add small epsilon to prevent log(0)
    loss = -np.sum(Y * np.log(y_hat + 1e-8)) / m
    return loss

# 6. Feed forward
def feed_forward(A0):
    """
    Forward pass through the network
    Returns predictions and cache of intermediate values
    """
    # Layer 1 (hidden) - use ReLU
    Z1 = W1 @ A0 + b1
    A1 = relu(Z1)
    
    # Layer 2 (hidden) - use ReLU
    Z2 = W2 @ A1 + b2
    A2 = relu(Z2)
    
    # Layer 3 (output) - use Softmax
    Z3 = W3 @ A2 + b3
    A3 = softmax(Z3)
    
    cache = {
        "A0": A0,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "A3": A3
    }
    
    return A3, cache

# 7. Backpropagation
def backprop_layer_3(y_hat, Y, A2, W3):
    """
    Backprop for output layer (softmax + categorical cross-entropy)
    """
    A3 = y_hat
    m = Y.shape[1]  # Get m from Y
    
    # Step 1: Calculate dC/dZ3 (simplified for softmax + cross-entropy)
    dC_dZ3 = (A3 - Y) / m
    assert dC_dZ3.shape == (n[3], m)
    
    # Step 2: Calculate dC/dW3
    dC_dW3 = dC_dZ3 @ A2.T
    assert dC_dW3.shape == (n[3], n[2])
    
    # Step 3: Calculate dC/db3
    dC_db3 = np.sum(dC_dZ3, axis=1, keepdims=True)
    assert dC_db3.shape == (n[3], 1)
    
    # Step 4: Calculate propagator dC/dA2
    dC_dA2 = W3.T @ dC_dZ3
    assert dC_dA2.shape == (n[2], m)
    
    return dC_dW3, dC_db3, dC_dA2

def backprop_layer_2(propagator_dC_dA2, Z2, A1, A2, W2):
    """
    Backprop for hidden layer 2 (ReLU activation)
    """
    m = propagator_dC_dA2.shape[1]  # Get m from input
    
    # Step 1: Calculate dC/dZ2 (using ReLU derivative)
    dA2_dZ2 = relu_derivative(Z2)
    dC_dZ2 = propagator_dC_dA2 * dA2_dZ2
    assert dC_dZ2.shape == (n[2], m)
    
    # Step 2: Calculate dC/dW2
    dC_dW2 = dC_dZ2 @ A1.T
    assert dC_dW2.shape == (n[2], n[1])
    
    # Step 3: Calculate dC/db2
    dC_db2 = np.sum(dC_dZ2, axis=1, keepdims=True)
    assert dC_db2.shape == (n[2], 1)
    
    # Step 4: Calculate propagator dC/dA1
    dC_dA1 = W2.T @ dC_dZ2
    assert dC_dA1.shape == (n[1], m)
    
    return dC_dW2, dC_db2, dC_dA1

def backprop_layer_1(propagator_dC_dA1, Z1, A0, W1):
    """
    Backprop for hidden layer 1 (ReLU activation)
    """
    m = propagator_dC_dA1.shape[1]  # Get m from input
    
    # Step 1: Calculate dC/dZ1 (using ReLU derivative)
    dA1_dZ1 = relu_derivative(Z1)
    dC_dZ1 = propagator_dC_dA1 * dA1_dZ1
    assert dC_dZ1.shape == (n[1], m)
    
    # Step 2: Calculate dC/dW1
    dC_dW1 = dC_dZ1 @ A0.T
    assert dC_dW1.shape == (n[1], n[0])
    
    # Step 3: Calculate dC/db1
    dC_db1 = np.sum(dC_dZ1, axis=1, keepdims=True)
    assert dC_db1.shape == (n[1], 1)
    
    return dC_dW1, dC_db1

# 8. Training function
def train(epochs=1000, alpha=0.07):
    """
    Train the neural network
    
    epochs: number of training iterations
    alpha: learning rate
    """
    global W1, W2, W3, b1, b2, b3
    
    costs = []
    
    print(f"\nStarting training for {epochs} epochs with learning rate {alpha}...")
    print("=" * 60)
    
    for e in range(epochs):
        # 1. FEED FORWARD
        y_hat, cache = feed_forward(A0)
        
        # 2. CALCULATE COST
        error = cost(y_hat, Y)
        costs.append(error)
        
        # 3. BACKPROPAGATION
        dC_dW3, dC_db3, dC_dA2 = backprop_layer_3(
            y_hat, 
            Y, 
            A2=cache["A2"], 
            W3=W3
        )
        
        dC_dW2, dC_db2, dC_dA1 = backprop_layer_2(
            propagator_dC_dA2=dC_dA2,
            Z2=cache["Z2"],
            A1=cache["A1"],
            A2=cache["A2"],
            W2=W2
        )
        
        dC_dW1, dC_db1 = backprop_layer_1(
            propagator_dC_dA1=dC_dA1,
            Z1=cache["Z1"],
            A0=cache["A0"],
            W1=W1
        )
        
        # 4. UPDATE WEIGHTS AND BIASES
        W3 = W3 - (alpha * dC_dW3)
        W2 = W2 - (alpha * dC_dW2)
        W1 = W1 - (alpha * dC_dW1)
        
        b3 = b3 - (alpha * dC_db3)
        b2 = b2 - (alpha * dC_db2)
        b1 = b1 - (alpha * dC_db1)
        
        # Print progress
        if e % 100 == 0:
            print(f"Epoch {e:4d}: Cost = {error:.4f}")
    
    print("=" * 60)
    print(f"Training complete! Final cost: {costs[-1]:.4f}")
    
    return costs

# 9. Prediction and evaluation functions
def predict(X):
    """
    Make predictions on new data
    X: input data (num_samples, 300)
    Returns: predicted class labels (num_samples,)
    """
    A0_test = X.T  # (300, num_samples)
    y_hat, _ = feed_forward(A0_test)  # (3, num_samples)
    predictions = np.argmax(y_hat, axis=0)  # Get class with highest probability
    return predictions

def evaluate_accuracy(X, y_true):
    """
    Calculate accuracy on a dataset
    """
    predictions = predict(X)
    accuracy = np.mean(predictions == y_true) * 100
    return accuracy

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Train the network
    costs = train(epochs=1000, alpha=0.07)
    
    # Evaluate on training data
    train_accuracy = evaluate_accuracy(X_train, y_train)
    print(f"\nTraining Accuracy: {train_accuracy:.2f}%")
    
    # Evaluate on test data
    test_accuracy = evaluate_accuracy(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Show some predictions
    print("\n" + "=" * 60)
    print("Sample Predictions:")
    print("=" * 60)
    
    shape_names = ['Heart', 'Sun', 'Snake']
    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    
    for idx in sample_indices:
        true_label = y_test[idx]
        pred_label = predict(X_test[idx:idx+1])[0]
        
        symbol = "✓" if true_label == pred_label else "✗"
        print(f"{symbol} True: {shape_names[true_label]:6s} | Predicted: {shape_names[pred_label]:6s}")



# ==================== SAVE AND VISUALIZE ====================

import matplotlib.pyplot as plt

# 1. Save the trained model
model_data = {
    'W1': W1, 'W2': W2, 'W3': W3,
    'b1': b1, 'b2': b2, 'b3': b3,
    'costs': costs,
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy
}

with open('trained_model_nn.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("\n✓ Model saved to 'trained_model_nn.pkl'")

# 2. Plot training curve
plt.figure(figsize=(10, 6))
plt.plot(costs)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Cost (Loss)', fontsize=12)
plt.title('Neural Network Training Progress', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('training_curve_nn.png', dpi=150, bbox_inches='tight')
print("✓ Training curve saved to 'training_curve_nn.png'")

# 3. Show some test predictions with images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
shape_names = ['Heart', 'Sun', 'Snake']

for idx, ax in enumerate(axes.flat):
    random_idx = np.random.randint(0, len(X_test))
    test_image = X_test[random_idx]
    true_label = y_test[random_idx]
    prediction = predict(test_image.reshape(1, -1))[0]
    
    # Get original image
    image_rgb = test_image.reshape(10, 10, 3)
    
    ax.imshow(image_rgb)
    color = 'green' if prediction == true_label else 'red'
    ax.set_title(f"True: {shape_names[true_label]}\nPred: {shape_names[prediction]}", 
                 color=color, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('predictions_grid_nn.png', dpi=150, bbox_inches='tight')
print("✓ Predictions grid saved to 'predictions_grid_nn.png'")

print("\n" + "="*60)
print("ALL RESULTS SAVED!")
print("="*60)
print(f"✓ Model: trained_model_nn.pkl")
print(f"✓ Training curve: training_curve_nn.png")
print(f"✓ Predictions: predictions_grid_nn.png")
print(f"✓ Test Accuracy: {test_accuracy:.2f}%")
print("="*60)