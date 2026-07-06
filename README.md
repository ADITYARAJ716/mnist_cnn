# MNIST Digit Classification with CNN (PyTorch)
 
A simple Convolutional Neural Network built with PyTorch to classify handwritten digits (0–9) from the MNIST dataset. Trained on GPU (Google Colab T4).
 
## Overview
 
- **Dataset:** MNIST (60,000 training images, 10,000 test images, 28x28 grayscale)
- **Framework:** PyTorch
- **Model:** Simple CNN (2 conv layers + 2 fully-connected layers)
- **Test Accuracy:** ~98–99%
## Model Architecture
 
```
Input (1x28x28)
  -> Conv2d(1, 32, kernel=3, padding=1) -> ReLU -> MaxPool2d(2,2)   -> (32x14x14)
  -> Conv2d(32, 64, kernel=3, padding=1) -> ReLU -> MaxPool2d(2,2)  -> (64x7x7)
  -> Flatten                                                        -> (3136)
  -> Linear(3136, 128) -> ReLU
  -> Linear(128, 10)
```
 
## Requirements
 
```
torch
torchvision
scikit-learn
matplotlib
seaborn
numpy
```
 
Install with:
```bash
pip install torch torchvision scikit-learn matplotlib seaborn numpy
```
 
## Usage
 
1. Run the notebook/script top to bottom (GPU runtime recommended — in Colab: `Runtime -> Change runtime type -> GPU`).
2. The MNIST dataset auto-downloads on first run.
3. Model trains for 5 epochs by default.
4. Evaluation prints accuracy, a full classification report, and a confusion matrix heatmap.
5. Trained weights are saved to `mnist_cnn.pth`.
## Training
 
```python
epochs = 5
optimizer = Adam(lr=0.001)
loss = CrossEntropyLoss
```
 
## Evaluation Metrics
 
- **Accuracy Score** — overall % of correct predictions
- **Classification Report** — precision, recall, F1-score per digit class
- **Confusion Matrix** — visual breakdown of which digits get misclassified as which
## Loading Saved Weights
 
```python
model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()
```
 
## Notes
 
- Trained and tested on Google Colab's free T4 GPU.
- Model and training loop kept intentionally simple for learning purposes — no data augmentation, learning rate scheduling, or advanced regularization.
