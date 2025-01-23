# Plant Disease Classification with Custom ResNet

## Project Overview
This project implements a custom ResNet-18 architecture from scratch for classifying plant diseases using the PlantVillage dataset.

## Key Features
- Custom ResNet implementation
- Skip connections to mitigate vanishing gradient problem
- Advanced training techniques
- High-accuracy plant disease classification

## Architectural Flexibility
- Supports multiple ResNet variants (18, 34, 50, etc.)
- Modular design allows easy network depth customization
- Implemented without relying on pre-built frameworks

### Bottleneck Consideration
While standard ResNet-50 and higher versions use bottleneck blocks, this implementation:
- Focuses on simpler, more interpretable architecture
- Avoids bottleneck complexity for lower-depth networks
- Provides a clean, extensible base for further modifications

## Technical Details

### ResNet Architecture
The core of this project is the Residual Network (ResNet) architecture, which introduces skip connections to enable training of very deep neural networks. 

#### Skip Connections
Skip connections (residual connections) solve two critical problems:
- Vanishing Gradient: By providing direct paths for gradient flow
- Degradation Problem: Allowing networks to go deeper without performance collapse

### Skip Connection Benefits
- Mitigates vanishing gradient problem
- Enables training of deeper neural networks
- Provides direct information flow through the network

## Key Features
- Custom ResNet implementation from scratch
- Configurable network depth
- Residual blocks with:
  - Skip connections
  - Dropout
  - Batch normalization

### Implementation Highlights
- Residual Block with Dropout
- BatchNormalization
- Dynamic Learning Rate Reduction
- Stratified Data Splitting

### Performance Metrics
- Training Accuracy: 99.76%
- Validation Accuracy: 97.96%
- Macro Average F1-Score: 0.98

## Data Preparation
- Used PlantVillage dataset
- Stratified train-test split to maintain class distribution
- Image preprocessing with ImageDataGenerator

## Training Techniques
- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau for adaptive learning rates
- Dropout for regularization

## Future Improvements
- Implement bottleneck architecture (ResNet-50)
- Experiment with more advanced augmentation techniques
- Cross-validation

## Requirements
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Usage
1. Clone the repository
2. Install requirements
3. Run the Jupyter notebook
