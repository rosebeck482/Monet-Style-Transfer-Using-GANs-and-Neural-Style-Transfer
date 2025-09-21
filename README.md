# Monet Style Transfer Using GANs and Neural Style Transfer

**[Fall 2024] COMS 4995 Applied Machine Learning** Course Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)](https://pytorch.org)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-Latest-purple)](https://lightning.ai)

A comprehensive deep learning project exploring multiple state-of-the-art approaches for artistic style transfer, transforming photographs into Monet-style paintings using advanced generative models and neural networks.

## Project Highlights

- **4 Different AI Architectures** implemented from scratch
- **Advanced GAN Training** with stability techniques and loss engineering
- **Large-scale Dataset Processing** (7,038+ images with TFRecord optimization)
- **Multi-framework Expertise** (TensorFlow, PyTorch, PyTorch Lightning)
- **Production-ready Code** with comprehensive evaluation metrics
- **Research-level Implementation** of state-of-the-art papers

## Key Results

| Method | FID Score ↓ | SSIM ↑ | Training Efficiency | Innovation |
|--------|------------|--------|-------------------|------------|
| **CycleGAN+LSGAN** | **101.97** | 0.397 | 25 epochs | LSGAN stability + ResNet |
| CycleGAN | 114.89 | **0.7647** | 25 epochs | U-Net + Skip connections |
| NST (1000 iter) | 127.63 | - | Optimization-based | VGG19 feature extraction |
| DCGAN | 162.81 | - | Early stopping | Spectral normalization |

## Technical Skills Demonstrated

### **Deep Learning & AI**
- **Generative Adversarial Networks (GANs)**: DCGAN, CycleGAN implementations
- **Advanced Loss Functions**: LSGAN, Cycle Consistency, Identity Loss, Adversarial Loss
- **Neural Style Transfer**: Optimization-based approach using pre-trained VGG19
- **Transfer Learning**: Leveraging pre-trained models for feature extraction

### **Machine Learning Engineering**
- **Multi-framework Proficiency**: TensorFlow 2.x, PyTorch, PyTorch Lightning
- **Distributed Training**: GPU acceleration with MirroredStrategy
- **Model Optimization**: Spectral normalization, batch normalization, dropout
- **Training Stability**: Early stopping, label smoothing, learning rate scheduling

### **Computer Vision**
- **Image Processing**: Data augmentation, normalization, tensor operations
- **Evaluation Metrics**: FID (Fréchet Inception Distance), SSIM (Structural Similarity)
- **Data Pipeline**: TFRecord processing, custom PyTorch datasets, efficient data loading

### **Software Engineering**
- **Code Architecture**: Modular design, custom classes, inheritance
- **Performance Optimization**: Parallel processing, memory management
- **Version Control**: Git workflow, reproducible experiments
- **Documentation**: Comprehensive README, inline documentation

## Directory Structure

```
Monet-Style-Using-Gans/
├── README.md                           
├── Project_Report.pdf                  # written report for the project
├── DCGAN/                              # deep convolutional gan implementation
│   ├── DCGan.ipynb                     # main dcgan notebook with advanced techniques
│   ├── data_augment.py                 # custom data augmentation pipeline
│   ├── dcgan_generator.pth             # trained generator model weights
│   ├── dcgan_discriminator.pth         # trained discriminator model weights
│   ├── data/                           # original dataset (1493 jpg images)
│   └── generated_image_fid/            # sample generated images for evaluation
├── CycleGAN/                           # tensorflow cyclegan implementation
│   ├── CycleGAN.ipynb                  # u-net architecture with skip connections
│   └── cyclegan_generated_image/       # high-quality generated samples
├── CycleGAN+LSGAN/                     # pytorch lightning advanced implementation
│   ├── CycleGAN+LSGAN_LOSS.ipynb       # resnet-based with lsgan loss
│   └── cyclegan_lsgan_generated_images/ # best performing model outputs
└── NST/                                # neural style transfer implementation
    ├── NST_500.ipynb                   # optimization-based style transfer (500 iter)
    ├── NST_1000.ipynb                  # extended optimization (1000 iter)
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

**Advanced Features**:
- **Custom U-Net Architecture**: 8-layer encoder-decoder with skip connections
- **Group Normalization**: Better than batch norm for artistic tasks
- **Distributed Training**: TensorFlow MirroredStrategy for multi-GPU
- **Advanced Loss Engineering**: Combines adversarial, cycle, and identity losses

**Technical Innovations**:
```python
# Custom downsampling with Group Normalization
def downsample(filters, size, add_gn=True):
    layers = [Conv2D(filters, size, strides=2, padding='same')]
    if add_gn:
        layers.insert(1, GroupNormalization(groups=-1))
    return Sequential(layers)
```

**Performance Optimizations**:
- **TFRecord Processing**: Parallel data loading with AUTOTUNE
- **Memory Efficiency**: Batch size optimization for GPU memory
- **Training Monitoring**: Real-time loss tracking and visualization

### 2. CycleGAN with LSGAN Loss ([`CycleGAN+LSGAN_LOSS.ipynb`](CycleGAN+LSGAN/CycleGAN+LSGAN_LOSS.ipynb))

**State-of-the-art Features**:
- **PyTorch Lightning Framework**: Production-ready training pipeline
- **ResNet Generator**: 9 residual blocks for deep feature learning
- **LSGAN Loss**: Mean Squared Error for more stable training
- **Instance Normalization**: Optimal for style transfer tasks

**Advanced Architecture**:
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

**Engineering Excellence**:
- **Custom Data Pipeline**: TFRecord to PyTorch tensor conversion
- **Memory Management**: Efficient batch processing with samplers
- **Logging & Monitoring**: Comprehensive metrics tracking

### 3. DCGAN Implementation ([`DCGan.ipynb`](DCGAN/DCGan.ipynb))

**Advanced Techniques**:
- **Spectral Normalization**: Stabilizes discriminator training
- **Early Stopping**: Prevents mode collapse via Nash equilibrium detection
- **Label Smoothing**: Real labels = 0.9 for better generalization
- **Asymmetric Learning Rates**: Balanced adversarial training

**Stability Innovations**:
```python
# Early stopping based on discriminator equilibrium
if (abs(D_x - 0.5) <= 0.05) and (abs(D_G_z1 - 0.5) <= 0.05):
    counter += 1
    if counter >= max_counter:
        print("Early stopping triggered - Nash equilibrium reached")
        break
```

**Production Features**:
- **Model Persistence**: Automatic checkpoint saving
- **Batch Generation**: Efficient image generation for evaluation
- **FID Evaluation**: Industry-standard quality metrics

### 4. Neural Style Transfer ([`NST_500.ipynb`](NST/NST_500.ipynb), [`NST_1000.ipynb`](NST/NST_1000.ipynb))

**Classical Deep Learning**:
- **VGG19 Feature Extraction**: Pre-trained CNN for style/content separation
- **Gram Matrix Computation**: Mathematical style representation
- **Multi-layer Style Loss**: Hierarchical feature matching
- **Optimization-based**: Direct image optimization via backpropagation

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
- **Batch Processing**: 1000+ images with progress tracking
- **Memory Optimization**: Single image processing to prevent OOM
- **Random Style Selection**: Diverse style application

## Performance Analysis & Metrics

### Quantitative Results
- **FID Scores**: Industry-standard generative model evaluation
- **SSIM Analysis**: Structural similarity preservation
- **Training Efficiency**: Time-to-convergence optimization
- **Memory Usage**: GPU utilization tracking

### Model Comparison
| Aspect | CycleGAN | CycleGAN+LSGAN | DCGAN | NST |
|--------|----------|----------------|-------|-----|
| **Architecture** | U-Net + PatchGAN | ResNet + PatchGAN | ConvTranspose + Conv | VGG19 Features |
| **Training Stability** | Good | Excellent | Moderate | N/A |
| **Output Quality** | High | Highest | Moderate | Good |
| **Computational Efficiency** | High | High | Very High | Low |
| **Memory Requirements** | Moderate | Moderate | Low | Very Low |

## Key Technical Achievements

### **Research Implementation**
- **Paper Reproduction**: Faithful implementation of 4 major research papers
- **Architecture Innovation**: Custom modifications for improved stability
- **Ablation Studies**: Systematic comparison of different approaches

### **Engineering Excellence**
- **Multi-framework Expertise**: Seamless switching between TensorFlow and PyTorch
- **Scalable Data Pipelines**: Efficient processing of large datasets
- **Production Readiness**: Error handling, logging, and monitoring

### **Performance Optimization**
- **GPU Acceleration**: Proper utilization of CUDA/MPS backends
- **Memory Management**: Efficient batch processing and gradient accumulation
- **Training Stability**: Advanced techniques to prevent mode collapse

## Advanced Features

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
tensorflow==2.15.0          # Google's ML platform
torch==2.5.1                # PyTorch deep learning
pytorch-lightning           # High-level PyTorch wrapper
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

This project demonstrates mastery of:

- **Generative Modeling**: Advanced understanding of GANs and their variants
- **Computer Vision**: Deep learning for image-to-image translation
- **Optimization Theory**: Loss function design and training dynamics
- **Software Engineering**: Production-quality ML code development
- **Research Implementation**: Ability to reproduce and extend academic work

## Future Enhancements

- **StyleGAN Integration**: Modern GAN architectures
- **Attention Mechanisms**: Transformer-based style transfer
- **Real-time Processing**: Optimized inference for deployment
- **Web Interface**: User-friendly style transfer application
- **Mobile Deployment**: Edge computing optimization

## References & Citations

- **CycleGAN**: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- **LSGAN**: [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)
- **DCGAN**: [Unsupervised Representation Learning with Deep Convolutional GANs](https://arxiv.org/abs/1511.06434)
- **Neural Style Transfer**: [Image Style Transfer Using Convolutional Neural Networks](https://arxiv.org/abs/1508.06576)

---

*This project showcases advanced machine learning engineering skills, research implementation capabilities, and production-ready code development for computer vision applications.*