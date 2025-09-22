# Self-Supervised Learning (SSL) for Image Classification

This notebook implements a **SimCLR (Simple Contrastive Learning of Representations)** approach for self-supervised learning on the CIFAR-10 dataset, followed by linear evaluation to assess the learned representations.

## Overview

The implementation demonstrates how to:
1. **Pre-train** a ResNet-18 encoder using contrastive learning (SimCLR)
2. **Evaluate** the learned representations through linear classification
3. Compare self-supervised learning performance on CIFAR-10

## Key Components

### 1. SimCLR Framework
- **Contrastive Learning**: Learns representations by maximizing agreement between differently augmented views of the same image
- **Data Augmentation**: Random resized crops, horizontal flips, and color jittering
- **NT-Xent Loss**: Normalized temperature-scaled cross-entropy loss for contrastive learning

### 2. Model Architecture
- **Backbone**: ResNet-18 (without pre-trained weights)
- **Projection Head**: Removed final classification layer for representation learning
- **Linear Classifier**: Simple linear layer for downstream evaluation

### 3. Training Process

#### Phase 1: Self-Supervised Pre-training
- **Dataset**: CIFAR-10 training set with SimCLR augmentations
- **Loss**: NT-Xent contrastive loss (temperature=0.5)
- **Optimizer**: SGD (lr=0.03, momentum=0.9, weight_decay=1e-4)
- **Epochs**: 10
- **Batch Size**: 256

#### Phase 2: Linear Evaluation
- **Frozen Encoder**: Pre-trained ResNet-18 encoder (no gradient updates)
- **Linear Classifier**: Single linear layer (512 → 10 classes)
- **Loss**: Cross-entropy loss
- **Optimizer**: Adam (lr=1e-3)
- **Epochs**: 10

## Results

### Pre-training Results
- **Final Loss**: 5.1871 (after 10 epochs)
- **Training Progress**: Consistent loss reduction from 5.5659 to 5.1871

### Linear Evaluation Results
- **Training Accuracy**: 41.51% (after 10 epochs)
- **Test Accuracy**: 41.18% (v1) 
- **Performance**: Demonstrates that the self-supervised representations can be used for downstream classification

## Technical Details

### Data Augmentation Strategy
```python
transforms.Compose([
    transforms.RandomResizedCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor()
])
```

### NT-Xent Loss Implementation
- **Temperature Scaling**: 0.5
- **Positive Pairs**: Different augmentations of the same image
- **Negative Pairs**: All other images in the batch
- **Normalization**: L2 normalization of embeddings

### Hardware Requirements
- **GPU**: CUDA-compatible (automatically detected)
- **CPU Fallback**: Available if CUDA is not present
- **Memory**: ~2GB GPU memory recommended for batch size 256

## Usage

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision tqdm
   ```

2. **Run the Notebook**:
   - Execute cells sequentially
   - Monitor training progress with progress bars
   - Results will be displayed after each phase

3. **Customization**:
   - Adjust `batch_size` for memory constraints
   - Modify `temperature` parameter for NT-Xent loss
   - Change number of epochs for longer/shorter training
   - Experiment with different backbone architectures

## Key Features

- ✅ **Complete SimCLR Implementation**: From data augmentation to loss computation
- ✅ **Linear Evaluation Protocol**: Standard evaluation methodology
- ✅ **Progress Tracking**: Real-time training progress with tqdm
- ✅ **GPU Support**: Automatic CUDA detection and usage
- ✅ **Modular Design**: Easy to modify and extend

## Performance Notes

- **Training Time**: ~5-10 minutes on modern GPU
- **Memory Usage**: Efficient batch processing
- **Convergence**: Stable training with consistent loss reduction
- **Reproducibility**: Deterministic results with fixed random seeds

## Extensions

This implementation can be extended for:
- **Different Datasets**: Easy adaptation to other image datasets
- **Architecture Variations**: Experiment with different backbone networks
- **Advanced Augmentations**: Implement more sophisticated data augmentation strategies
- **Evaluation Metrics**: Add additional evaluation metrics beyond accuracy

## References

- [SimCLR Paper](https://arxiv.org/abs/2002.05709): "A Simple Framework for Contrastive Learning of Visual Representations"
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html): 10-class image classification benchmark
