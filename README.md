# Monet Style Transfer Using GANs and Neural Style Transfer

**[Fall 2024] COMS 4995 Applied Machine Learning** Course Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)](https://pytorch.org)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-Latest-purple)](https://lightning.ai)

A deep learning project exploring multiple approaches for artistic style transfer, transforming photographs into Monet-style paintings using generative models and neural networks.


## Key Results

| Method | FID Score ↓ | SSIM ↑ | Training Configuration | Key Features |
|--------|------------|--------|----------------------|------------|
| **CycleGAN+LSGAN** | **101.97** | **0.3966** | 25 epochs (PyTorch Lightning) | LSGAN stability + ResNet |
| CycleGAN | 114.89 | **0.7647** | 25 epochs (TensorFlow) | U-Net + Skip connections |
| NST | 127.98 | **0.2796** | 1000 steps/image | VGG19 feature extraction |
| DCGAN | 162 | N/A | Early stopping (200 max epochs) | Spectral normalization |

## Tools

### **Deep Learning**
- **Generative Adversarial Networks (GANs)**: DCGAN, CycleGAN implementations
- **Loss Functions**: LSGAN, Cycle Consistency, Identity Loss, Adversarial Loss
- **Neural Style Transfer**: Optimization-based approach using pre-trained VGG19
- **Transfer Learning**: Leveraging pre-trained models for feature extraction

### **Machine Learning**
- **Multi-framework Implementation**: TensorFlow 2.x, PyTorch, PyTorch Lightning
- **Distributed Training**: GPU acceleration with MirroredStrategy
- **Model Optimization**: Spectral normalization, batch normalization, dropout
- **Training Stability**: Early stopping, label smoothing, learning rate scheduling

### **Computer Vision**
- **Image Processing**: Data augmentation, normalization, tensor operations
- **Evaluation Metrics**: FID (Fréchet Inception Distance), SSIM (Structural Similarity)
- **Data Pipeline**: TFRecord processing, custom PyTorch datasets, data loading


## Directory Structure

```
Monet-Style-Using-Gans/
├── README.md                           
├── Project_Report.pdf                  # written report for the project
├── DCGAN/                              # deep convolutional gan implementation
│   ├── DCGan.ipynb                     # dcgan implementation
│   ├── data_augment.py                 # custom data augmentation pipeline
│   ├── dcgan_generator.pth             # trained generator model weights
│   ├── dcgan_discriminator.pth         # trained discriminator model weights
│   ├── data/                           # original dataset (1493 jpg images)
│   └── generated_image_fid/            # sample generated images for evaluation
├── CycleGAN/                           # tensorflow cyclegan implementation
│   ├── CycleGAN.ipynb                  # u-net architecture with skip connections
│   └── cyclegan_generated_image/       # generated samples
├── CycleGAN+LSGAN/                     # pytorch lightning implementation
│   ├── CycleGAN+LSGAN_LOSS.ipynb       # resnet-based with lsgan loss
│   └── cyclegan_lsgan_generated_images/ # model outputs
└── NST/                                # neural style transfer implementation
    ├── NST_500.ipynb                   # optimization-based style transfer (500 images)
    ├── NST_1000.ipynb                  # extended processing (1000 images)
    └── NST_generated_images/           # stylized output samples
```

## Installation & Setup

### Prerequisites
```bash
# Core ML frameworks
pip install tensorflow==2.15.0
pip install torch torchvision pytorch-lightning
pip install numpy matplotlib scikit-image Pillow
pip install pytorch-fid torchmetrics torch-fidelity
```

### Dataset Configuration
- **Primary Dataset**: 7,038 landscape photos + 300 Monet paintings
- **Enhanced Dataset**: Additional 1,193 Berkeley Monet images
- **Format**: TFRecord for optimized loading, JPEG for direct processing
- **Size**: Multi-gigabyte datasets (hosted externally due to size)

**Dataset Links**:
- [Main Dataset](https://drive.google.com/drive/folders/1vRKhb8ApjFsoAybFT_5Ve8PFbo4rgHhz?dmr=1&ec=wgc-drive-hero-goto)
- [Berkeley Monet Dataset](https://www.kaggle.com/datasets/dimitreoliveira/monet-paintings-jpg-berkeley/data)

## Technical Implementation Details

### 1. CycleGAN Implementation ([`CycleGAN.ipynb`](CycleGAN/CycleGAN.ipynb))

**Features**:
- **Custom U-Net Architecture**: 8-layer encoder-decoder with skip connections
- **Group Normalization**: Better than batch norm for artistic tasks (groups=-1)
- **Distributed Training**: TensorFlow MirroredStrategy for multi-GPU
- **Loss Engineering**: Combines adversarial, cycle, and identity losses
- **TFRecord Integration**: Data pipeline with decode_jpg_image preprocessing

**Implementation**:
```python
# Custom downsampling with Group Normalization
def downsample(filters, size, add_gn=True):
    layers = [Conv2D(filters, size, strides=2, padding='same')]
    if add_gn:
        layers.insert(1, GroupNormalization(groups=-1))
    return Sequential(layers)
```

**Optimizations**:
- **TFRecord Processing**: Parallel data loading with AUTOTUNE
- **Memory Efficiency**: Batch size optimization for GPU memory
- **Training Monitoring**: Real-time loss tracking and visualization
- **Evaluation**: SSIM calculation with multichannel support
- **Google Colab Integration**: Drive mounting and GPU detection

### 2. CycleGAN with LSGAN Loss ([`CycleGAN+LSGAN_LOSS.ipynb`](CycleGAN+LSGAN/CycleGAN+LSGAN_LOSS.ipynb))

**Key Features**:
- **PyTorch Lightning Framework**: Training pipeline with automatic optimization
- **ResNet Generator**: 9 residual blocks for deep feature learning
- **LSGAN Loss**: Mean Squared Error (F.mse_loss) for more stable training
- **Instance Normalization**: Optimal for style transfer tasks
- **Dual Dataset Integration**: Combined Monet + Berkeley datasets via ConcatDataset

**Architecture**:
```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)  # Skip connection
```

**Implementation Details**:
- **Custom Data Pipeline**: TFRecord to PyTorch tensor conversion with tf.transpose
- **Memory Management**: Batch processing with RandomSampler
- **Multi-GPU Support**: CUDA detection and device optimization
- **Loss Tracking**: Generator, discriminator, cycle, and identity losses

### 3. DCGAN Implementation ([`DCGan.ipynb`](DCGAN/DCGan.ipynb))

**Techniques**:
- **Spectral Normalization**: Applied to all discriminator Conv2d layers for training stability
- **Early Stopping**: Prevents mode collapse via Nash equilibrium detection (patience=5000)
- **Label Smoothing**: Real labels = 0.9 for better generalization
- **Asymmetric Learning Rates**: Discriminator lr*0.5, Generator lr (balanced training)
- **Dropout Regularization**: 0.3 dropout in discriminator layers

**Stability Methods**:
```python
# Early stopping based on discriminator equilibrium
if (abs(D_x - 0.5) <= 0.05) and (abs(D_G_z1 - 0.5) <= 0.05):
    counter += 1
    if counter >= max_counter:
        print("Early stopping triggered - Nash equilibrium reached")
        break
```

**Additional Features**:
- **Model Persistence**: Automatic checkpoint saving (.pth files)
- **Batch Generation**: 5000 image generation for FID evaluation
- **FID Evaluation**: Quality metrics with pytorch-fid
- **Reproducible Training**: Manual seed setting with deterministic algorithms

### 4. Neural Style Transfer ([`NST_500.ipynb`](NST/NST_500.ipynb), [`NST_1000.ipynb`](NST/NST_1000.ipynb))

**Classical Deep Learning**:
- **VGG19 Feature Extraction**: Pre-trained CNN for style/content separation
- **Gram Matrix Computation**: Mathematical style representation
- **Multi-layer Style Loss**: Hierarchical feature matching across 5 conv layers
- **Optimization-based**: Direct image optimization via Adam optimizer (1000 steps per image)

**Technical Implementation**:
- **Loss Weights**: Style weight = 2e7, Content weight = 5e2
- **Learning Rate**: Adam optimizer with lr=0.01
- **Random Style Selection**: Each content image paired with random Monet painting
- **Progress Tracking**: Loss monitoring every 1000 optimization steps

**Mathematical Foundation**:
```python
def gram_matrix(tensor):
    """Computes Gram matrix for style representation"""
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (d * h * w)  # Normalized
```

**Scalable Processing**:
- **Dataset Variants**: NST_500 processes 500 images, NST_1000 processes 1000 images
- **Memory Optimization**: Single image processing to prevent OOM
- **Evaluation**: FID and MFID metrics calculation
- **Google Colab Integration**: Drive mounting and GPU acceleration

## Performance Analysis & Metrics

### Quantitative Results
- **FID Scores**: Generative model evaluation
- **SSIM Analysis**: Structural similarity preservation
- **Training Efficiency**: Time-to-convergence optimization
- **Memory Usage**: GPU utilization tracking


## Implementation Features

### **Data Engineering**
```python
# Custom TFRecord to PyTorch tensor conversion
def decode_jpg_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1  # Normalize to [-1, 1]
    img = tf.transpose(img, [2, 0, 1])  # CHW format
    return torch.from_numpy(img.numpy())
```

### **Loss Function Engineering**
```python
# Multi-component loss for CycleGAN
total_loss = (
    adversarial_loss + 
    lambda_cycle * cycle_consistency_loss + 
    lambda_identity * identity_loss
)
```

### **Training Optimization**
```python
# Distributed training setup
if tf.config.list_physical_devices('GPU'):
    strategy = tf.distribute.MirroredStrategy()
    print("Multi-GPU training enabled")
```

## Dependencies & Environment

### **Core Frameworks**
```python
tensorflow==2.15.0          # Google's ML platform (CycleGAN)
torch==2.5.1                # PyTorch deep learning (DCGAN, NST)
pytorch-lightning           # High-level PyTorch wrapper (CycleGAN+LSGAN)
torchvision                 # Computer vision utilities
```

### **Evaluation & Metrics**
```python
pytorch-fid                 # FID score calculation
torchmetrics               # ML metrics collection
scikit-image               # Image processing utilities
```

### **Visualization & Analysis**
```python
matplotlib                 # Plotting and visualization
numpy                      # Numerical computing
PIL                        # Image manipulation
```

## Academic & Research Context

This project demonstrates:

- **Generative Modeling**: Understanding of GANs and their variants
- **Computer Vision**: Deep learning for image-to-image translation
- **Optimization Theory**: Loss function design and training dynamics


## References & Citations

- **CycleGAN**: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- **LSGAN**: [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)
- **DCGAN**: [Unsupervised Representation Learning with Deep Convolutional GANs](https://arxiv.org/abs/1511.06434)
- **Neural Style Transfer**: [Image Style Transfer Using Convolutional Neural Networks](https://arxiv.org/abs/1508.06576)

