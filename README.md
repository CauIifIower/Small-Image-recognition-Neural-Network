# From-Scratch-Small-Image-recognition-Neural-Network-Using-Only-NumPy

A three-layer neural network built from scratch using only NumPy, trained to classify 10×10 pixel images into three categories: **hearts**, **snakes**, and **suns** without PyTorch or TensorFlow

Built as a practical part for a secondary grammar school seminary project exploring how linear algebra underlies neural networks. The neural network and dataset are purposely simple to simplify the demonstration of mathematical operations happening during forward propagation.

---

## Results

| Metric | Value |
|---|---|
| Training accuracy | 99.35% |
| Test accuracy | **99.67%** |
| Misclassified (out of 612) | 2 |
| Epochs | 1,000 |
| Final loss | 0.05 (down from 1.12) |

The dataset was synthetically generated with controlled structure and limited variation, which makes the classification task relatively straightforward. The near-perfect accuracy should be interpreted in that context; results would likely differ significantly on real-world image data.

---

## Architecture

- **Input layer:** 300 nodes — each 10×10 RGB image flattened to a vector (10×10×3)
- **Hidden layers:** 128 and 64 nodes with ReLU activation
- **Output layer:** 3 nodes with Softmax (one per class)
- **Total parameters:** 46,979

---

## Dataset

3,060 synthetically generated images across 3 classes:

- Hearts — red and pink tones
- Suns — yellow tones
- Snakes — green tones

Images were created from 19 hand-drawn templates, then augmented with random colour noise (±60 RGB). Split: 80% training (2,448 images) / 20% test (612 images).

---

## How it works

Each forward pass is a chain of matrix operations:

Z[l] = W[l] · A[l-1] + b[l]

A[l] = g(Z[l])

Weight matrices perform linear transformations between layers. ReLU activation introduces non-linearity without which the whole network would collapse into a single linear transformation regardless of depth. Training uses gradient descent with backpropagation (chain rule) to minimise categorical cross-entropy loss.

---

## Dependencies

```bash
pip install numpy matplotlib scikit-learn
```

---

## Usage

```bash
# Generate the dataset
python create_dataset.py
```
Takes the templates from dataset.py, multiplies them and adds random colour noise ±60 RGB.

```bash
# Train the network
python nn.py
```

---

## Project structure

─ create_dataset.py          # Generates and saves the image dataset

─ dataset.py                 # Dataset templates and augmentation logic

─ generate_and_check_data.py # Data validation and inspection

─ nn.py                      # Network architecture, training loop, evaluation

─ dataset.pkl                # Serialised dataset (generated)

─ trained_model_nn.pkl       # Saved model weights (generated)

─ training_curve_nn.png      # Loss curve plot (generated)

---

## References

- Weidman, S. *Deep Learning from Scratch*. O'Reilly, 2019
- Aggarwal, C. C. *Neural Networks and Deep Learning*. Springer, 2018
- Ng, A. [Stanford CS230 notation](https://cs230.stanford.edu/files/Notation.pdf), 2018
- Harris et al. [Array programming with NumPy](https://doi.org/10.1038/s41586-020-2649-2). *Nature*, 2020
