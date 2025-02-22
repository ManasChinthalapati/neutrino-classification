# Neutrino Event Classification

## Overview

This project focuses on classifying neutrino interactions using machine learning techniques, specifically convolutional neural networks (CNNs). The dataset consists of simulated neutrino interactions in a detector similar to that of the NOvA experiment. The primary goal is to differentiate charged-current neutrino interactions from other interactions based on image data.

## Dataset

The dataset is composed of multiple `.h5` files containing event images and metadata:

- **Event images**: Represent energy deposited by neutrino interactions.
- **Metadata variables**:
  - `neutrino/nuenergy`: Neutrino Energy (GeV)
  - `neutrino/lepenergy`: Lepton Energy (GeV)
  - `neutrino/finalstate`: Interaction Final State
  - `neutrino/interaction`: Interaction Type (Encoded as PDG codes)

### Interaction Types

The dataset contains various neutrino interactions, categorized into:

- **Charged-Current (CC) Interactions**:
  - Quasi-elastic (QE)
  - Resonance (RES)
  - Deep Inelastic Scattering (DIS)
- **Neutral-Current (NC) Interactions**
- **Cosmic Ray Background**

## Machine Learning Tasks

### Task 1: Binary Classification

Develop a CNN model to classify events as neutrino charged-current interactions or non-neutrino events.

### Task 2: Performance Analysis

Investigate how classifier performance varies based on metadata features such as neutrino energy and final state.

### Extensions:

- Predict **neutrino energy** using regression techniques.
- Determine the **flavor of the neutrino** (electron, muon, tau).
- Predict the **lepton energy over neutrino energy ratio**.
- Classify the **number of protons or pions** in the final state.
- Determine the **interaction mode** (QE, RES, DIS, NC, etc.).

## Model Architecture

A **Convolutional Neural Network (CNN)** is used for feature extraction and classification. The final model (`model2`) consists of:

1. **Two convolutional layers** with ReLU activation and max pooling.
2. **Dropout layers** to prevent overfitting.
3. **Fully connected dense layers** with sigmoid activation for binary classification.

### Training and Testing

- The dataset is split into **75% training, 25% testing**.
- A batch size of **100** is used.
- **20 epochs** were found to balance performance and training time.
- Both **balanced and unbalanced datasets** are tested to evaluate model performance.

## Results

- **98% accuracy** on unbalanced test data.
- **94% accuracy** on balanced test data (indicating sensitivity to class imbalance).
- **Interaction-specific performance**:
  - **QE:** 99.54%
  - **RES:** 99.79%
  - **DIS:** 99.62%

## Key Findings

- **CNNs effectively classify neutrino events**, even in imbalanced datasets.
- **Balancing training data improves fairness across interaction types**.
- **Model performance varies by interaction type**, likely due to differences in data distribution.

## Dependencies

- Python 3.8+
- TensorFlow/Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- h5py

## Running the Code

1. Install dependencies using:
   ```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn h5py
   ```
2. Download the dataset (`.h5` files) and place them in the project directory.
3. Run the main script to train the model:
   ```bash
   python neutrino_classifier.py
   ```

## Future Work

- Improve performance on imbalanced classes using **data augmentation**.
- Implement **multi-class classification** for different neutrino flavors.
- Explore **alternative architectures**, such as Transformer-based models.

---

**Author:** Manas Chinthalapati**Project:** PHAS0056 Mini-Project**Institution:** UCL
