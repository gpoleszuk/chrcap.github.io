import json
import os
import numpy as np
import matplotlib.pyplot as plt
from data import get_mnist

# File to store network parameters
PARAMS_FILE = "network_params.npz"

# Network structure
INPUT_SIZE = 784
HIDDEN_SIZE = 32
OUTPUT_SIZE = 10
LEARN_RATE = 1.0/8192
LEARN_RATE = 1.0/4096
LEARN_RATE = 1.0/2048
LEARN_RATE = 1.0/128
LEARN_RATE = 1.0/256
LEARN_RATE = 1.0/512
EPOCHS = 0
BATCH = 15
# Prevent Overflow in Sigmoid: Replace the direct computation of the sigmoid
# function with a numerically stable implementation:
# def sigmoid(x):
#     return np.where(x >= 0, 
#                     1 / (1 + np.exp(-x)), 
#                     np.exp(x) / (1 + np.exp(x)))
def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def save_init_params(weights, biases):
    """Save weights and biases to a binary file."""
    np.savez("network_init_params_full.npz", weights=weights, biases=biases)
    print("Network init parameters saved successfully to binary file.")

    """Save weights and biases to a text file."""
    output_file = "network_init_params_full.txt"
    np.set_printoptions(threshold=np.inf)  # Disable truncation
    with open(output_file, "w") as f:
        for i, (weight_matrix, bias_vector) in enumerate(zip(weights, biases)):
            f.write(f"Layer {i+1} Weights:\n")
            np.savetxt(f, weight_matrix, fmt="%.6f")
            f.write(f"Layer {i+1} Biases:\n")
            np.savetxt(f, bias_vector, fmt="%.6f")
    print(f"Initial weights and biases exported to {output_file}.")


def save_params_full(weights, biases, output_file="network_params_full_precision.txt"):
    """
    Save weights and biases to a text file without truncation,
    using a specific format for numbers.
    """
    np.set_printoptions(threshold=np.inf, formatter={'float_kind': lambda x: f"{x:22.18E}"})  # Ensure no truncation and specific format

    with open(output_file, "w") as f:
        for i, (weight_matrix, bias_vector) in enumerate(zip(weights, biases)):
            f.write(f"Layer {i+1} Weights:\n")
            for row in weight_matrix:
                f.write(" ".join([f"{val:25.18E}" for val in row]) + "\n")
            f.write(f"\nLayer {i+1} Biases:\n")
            for bias in bias_vector.flatten():
                f.write(f"{bias:25.18E}\n")
            f.write("\n")
    print(f"Weights and biases exported to {output_file}.")

# Example use case
#weights = [np.random.randn(4, 3), np.random.randn(3, 2)]
#biases = [np.random.randn(4, 1), np.random.randn(3, 1)]
#
#save_params_full(weights, biases)


def save_params(weights, biases):
    """Save weights and biases to a binary file."""
    np.savez(PARAMS_FILE, weights=weights, biases=biases)
    print("Network parameters saved successfully to binary file.")

    """Save weights and biases to a text file."""
    output_file = "network_params_full.txt"
    np.set_printoptions(threshold=np.inf)  # Disable truncation
    with open(output_file, "w") as f:
        for i, (weight_matrix, bias_vector) in enumerate(zip(weights, biases)):
            f.write(f"Layer {i+1} Weights:\n")
            np.savetxt(f, weight_matrix, fmt="%.6f")
            f.write(f"Layer {i+1} Biases:\n")
            np.savetxt(f, bias_vector, fmt="%.6f")
    print(f"Weights and biases exported to {output_file}.")


def load_params():
    """Load weights and biases from a binary file."""
    if not os.path.exists(PARAMS_FILE):
        return None, None
    data = np.load(PARAMS_FILE, allow_pickle=True)
    weights = data["weights"]
    biases = data["biases"]
    print("Network parameters loaded successfully.")
    return weights, biases

# Run this function after training your model. The model is saved in model.json
def export_model_to_json(weights, biases, output_file="model.json"):
    model = {
        "weights": [w.tolist() for w in weights],
        "biases": [b.tolist() for b in biases]
    }
    with open(output_file, "w") as f:
        json.dump(model, f)
    print(f"Model exported to {output_file}.")

# https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization
# two types of simple initialization: random initialization and normal (naïve) initialization
# ToDo: Weight Initialization: Xavier Initialization: For weights, use the
# following formula to prevent gradients from exploding or vanishing:
# w_i_h = np.random.uniform(-np.sqrt(6/(HIDDEN_SIZE+INPUT_SIZE)), \
#                            np.sqrt(6/(HIDDEN_SIZE+INPUT_SIZE)), (HIDDEN_SIZE, INPUT_SIZE))
# w_h_o = np.random.uniform(-np.sqrt(6/(OUTPUT_SIZE+HIDDEN_SIZE)), \
#                            np.sqrt(6/(OUTPUT_SIZE+HIDDEN_SIZE)), (OUTPUT_SIZE, HIDDEN_SIZE))
# Initialize or load network parameters
weights, biases = load_params()
if weights is None or biases is None:
    weights = [
        np.random.uniform(-0.5, 0.5, (HIDDEN_SIZE, INPUT_SIZE)),
        np.random.uniform(-0.5, 0.5, (OUTPUT_SIZE, HIDDEN_SIZE)),
    ]
    biases = [
        np.ones((HIDDEN_SIZE, 1))*0.5,
        np.ones((OUTPUT_SIZE, 1))*0.5,
    ]
    #biases = [
    #    np.zeros((HIDDEN_SIZE, 1)),
    #    np.zeros((OUTPUT_SIZE, 1)),
    #]
    save_init_params(weights, biases)

images, labels = get_mnist()

#"""Save images and labels to a text file."""
#output_file = "images_labels.txt"
#np.set_printoptions(threshold=np.inf)  # Disable truncation
#with open(output_file, "w") as f:
#    for i, (images_vector, labels_vector) in enumerate(zip(images, labels)):
#        f.write(f"Layer {i+1} Weights:\n")
#        np.savetxt(f, images_vector, fmt="%.6f")
#        f.write(f"Layer {i+1} Biases:\n")
#        np.savetxt(f, labels_vector, fmt="%.6f")
#print(f"Images and labels exported to {output_file}.")


#    biases = [
#        np.ones((OUTPUT_SIZE, 1)),
#        np.ones((OUTPUT_SIZE, 1)),
#    ]


print(f"Network details:")
print(f"- Input size: {INPUT_SIZE}")
print(f"- Hidden layer neurons: {HIDDEN_SIZE}")
print(f"- Output size: {OUTPUT_SIZE}")
print(f"- Learning rate: {LEARN_RATE}")
print(f"- Epochs: {EPOCHS}")
print(f"- Bathc: {BATCH}")

export_model_to_json(weights, biases)

# ToDo: Mini-Batches: Instead of training on one sample at a time, process
# data in batches (e.g., 32 samples per batch). This improves computational
# efficiency and stabilizes training:
# batch_size = 32
# for epoch in range(epochs):
#    for i in range(0, len(images), batch_size):
#        batch_images = images[i:i + batch_size]
#        batch_labels = labels[i:i + batch_size]
#        # Apply forward/backpropagation on batches

batch_size = BATCH
for epoch in range(EPOCHS):
    nr_correct = 0
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        for img, l in zip(batch_images, batch_labels):
            # ToDo: Optimize matrix multiplications by ensuring shapes are correct and
            # avoid reshaping repeatedly:
            # img = img[:, None]  # Convert to column vector once at the start
            img = img.reshape(-1, 1)
            l = l.reshape(-1, 1)

            # Forward propagation
            h_pre = biases[0] + weights[0] @ img
            #h = 1 / (1 + np.exp(-h_pre))  # Sigmoid activation
            h = sigmoid(h_pre)  # Sigmoid activation
            o_pre = biases[1] + weights[1] @ h
            #o = 1 / (1 + np.exp(-o_pre))
            o = sigmoid(o_pre)

            # Cost / Error calculation
            e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
            nr_correct += int(np.argmax(o) == np.argmax(l))

            # ToDo: Add Momentum: Incorporate momentum to smooth updates and
            # accelerate convergence:
            # momentum = 0.9
            # velocity_w_h_o = momentum * velocity_w_h_o - learn_rate * delta_o @ h.T
            # w_h_o += velocity_w_h_o
        
            # ToDo: Learning Rate Scheduling: Decrease the learning rate as training
            # progresses to refine convergence:
            # learn_rate = initial_rate / (1 + decay * epoch)

            # ToDo: Eliminate explicit loops for backpropagation by using
            # vectorized calculations. For example:
            # delta_h = (w_h_o.T @ delta_o) * (h * (1 - h))
            # w_i_h -= learn_rate * (delta_h @ img.T)
            # b_i_h -= learn_rate * delta_h
        
            # Backpropagation
            delta_o = o - l
            weights[1] -= LEARN_RATE * delta_o @ h.T
            biases[1] -= LEARN_RATE * delta_o
            delta_h = weights[1].T @ delta_o * (h * (1 - h))
            weights[0] -= LEARN_RATE * delta_h @ img.T
            biases[0] -= LEARN_RATE * delta_h

    # ToDo: improve the calculation of loss. Hardcoded to !zero!
    loss = 0.0
    accuracy = round((nr_correct / len(images)) * 100, 2)
    #print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Epoch {epoch+1}, Accuracy: {accuracy:.2f}%")


# Save parameters after training
save_params(weights, biases)
save_params_full(weights, biases)

# Inference
while True:
    index = int(input("Enter a number of an image (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Blues")
    img = img.reshape(-1, 1)

    print(f"{img}")

    # Forward propagation
    h_pre = biases[0] + weights[0] @ img
    h = 1 / (1 + np.exp(-h_pre))
    #h = sigmoid(h_pre)
    o_pre = biases[1] + weights[1] @ h
    o = 1 / (1 + np.exp(-o_pre))
    #o = sigmoid(o_pre)

    # Display probabilities in the terminal
    print("\nDigit probabilities:")
    for i, prob in enumerate(o.flatten()):
        print(f"{i}: {prob * 100:.2f}%")

    plt.title(f"Prediction: {o.argmax()} (Confidence: {o.max():.2f})")
    plt.show()
