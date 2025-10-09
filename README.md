# Deep Embedded K-Means Clustering (DEKM)

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A high-quality PyTorch implementation of the **Deep Embedded K-Means Clustering** algorithm, as described in the paper:

> **Deep Embedded K-Means Clustering**  
> Wengang Guo, Kaiyan Lin, Wei Ye  
> *2021 International Conference on Data Mining Workshops (ICDMW)*  
> [[Paper](https://arxiv.org/pdf/2109.15149)]

## 🚀 Features

- **Complete Implementation**: Full DEKM algorithm with autoencoder pretraining and joint clustering optimization
- **Multiple Datasets**: Support for MNIST, CIFAR-10, Fashion-MNIST, KMNIST, STL-10, and USPS
- **Comprehensive Evaluation**: Built-in evaluation pipeline with visualization and metrics
- **Professional Code**: Well-documented, modular, and maintainable codebase
- **Easy to Use**: Simple command-line interface and example scripts
- **Flexible Configuration**: Extensive hyperparameter tuning options

## 📋 Requirements

```bash
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.6.0
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/deep-embedded-kmeans-clustering-pytorch.git
cd deep-embedded-kmeans-clustering-pytorch
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Training a Model

Train DEKM on MNIST with 10 clusters:

```bash
python main.py --dataset mnist --n-clusters 10 --verbose
```

### Using Example Scripts

We provide several example scripts in the `scripts/` directory:

```bash
# Quick test (reduced epochs)
./scripts/run_quick_test.sh

# Train on different datasets
./scripts/run_mnist.sh
./scripts/run_cifar10.sh
./scripts/run_fashion_mnist.sh

# Complete pipeline (train + evaluate)
./scripts/train_and_evaluate.sh mnist 10
```

### Evaluating a Trained Model

```bash
python evaluate.py --model-path pretrained_weights/mnist_dekm.pth --dataset mnist
```

## 📊 Supported Datasets

| Dataset | Classes | Image Size | Description |
|---------|---------|------------|-------------|
| MNIST | 10 | 28×28×1 | Handwritten digits |
| Fashion-MNIST | 10 | 28×28×1 | Fashion items |
| CIFAR-10 | 10 | 32×32×3 | Natural images |
| KMNIST | 10 | 28×28×1 | Japanese characters |
| STL-10 | 10 | 96×96×3 | Unlabeled natural images |
| USPS | 10 | 16×16×1 | Handwritten digits |

## 🔧 Configuration Options

### Training Parameters

```bash
python main.py \
    --dataset mnist \
    --n-clusters 10 \
    --batch-size 256 \
    --pretrain-epochs 200 \
    --clustering-epochs 200 \
    --pretrain-lr 0.001 \
    --clustering-lr 0.0001 \
    --embedding-size 10 \
    --save-dir pretrained_weights \
    --results-dir results \
    --verbose
```

### Key Parameters

- `--n-clusters`: Number of clusters (required)
- `--embedding-size`: Size of the embedding layer (default: 10)
- `--batch-size`: Training batch size (default: 256)
- `--pretrain-epochs`: Autoencoder pretraining epochs (default: 200)
- `--clustering-epochs`: Clustering training epochs (default: 200)
- `--pretrain-lr`: Learning rate for pretraining (default: 0.001)
- `--clustering-lr`: Learning rate for clustering (default: 0.0001)

## 📈 Evaluation and Visualization

The evaluation pipeline provides comprehensive analysis:

### Metrics Computed
- **Normalized Mutual Information (NMI)**
- **Adjusted Rand Index (ARI)**
- **Clustering Accuracy (ACC)**

### Visualizations Generated
- Confusion matrices
- t-SNE embeddings visualization
- PCA projections
- Cluster size distributions
- Comprehensive evaluation reports

### Example Evaluation Output

```
==================================================
CLUSTERING RESULTS
==================================================
Normalized Mutual Information (NMI): 0.82345
Adjusted Rand Index (ARI):          0.78912
Clustering Accuracy (ACC):          0.85678
==================================================
```

## 🏗️ Project Structure

```
deep-embedded-kmeans-clustering-pytorch/
├── main.py                 # Main training script
├── evaluate.py             # Evaluation pipeline
├── DEKM.py                 # Core DEKM algorithm
├── DEKM_AE.py             # Autoencoder architecture
├── dataset.py             # Dataset loading utilities
├── utils.py               # Utility functions
├── scripts/               # Example run scripts
│   ├── run_mnist.sh
│   ├── run_cifar10.sh
│   ├── evaluate_model.sh
│   └── ...
├── pretrained_weights/    # Saved model weights
├── results/               # Training results
├── plots/                 # Evaluation visualizations
└── data/                  # Dataset storage
```

## 🔬 Algorithm Overview

DEKM combines deep learning with traditional k-means clustering:

1. **Autoencoder Pretraining**: Learn meaningful feature representations
2. **Initial Clustering**: Apply k-means on learned embeddings
3. **Joint Optimization**: Alternately update cluster centers and autoencoder parameters
4. **Eigenvalue Decomposition**: Use scatter matrices for better cluster separation

### Key Components

- **Convolutional Autoencoder**: Encoder-decoder architecture for feature learning
- **K-means Integration**: Traditional clustering on learned embeddings
- **Joint Training**: Simultaneous optimization of reconstruction and clustering objectives
- **Adaptive Learning**: Dynamic adjustment of clustering parameters

## 📚 Usage Examples

### Custom Configuration

```python
from DEKM import DEKM
from dataset import DatasetWrapper

# Load dataset
dataset = DatasetWrapper('mnist')

# Initialize model with custom parameters
model = DEKM(
    n_clusters=10,
    embedding_size=16,
    batch_size=128,
    pretrain_epochs=100,
    clustering_epochs=150,
    pretrain_learning_rate=0.001,
    clustering_learning_rate=0.0001
)

# Train the model
model.fit(dataset)

# Make predictions
predictions = model.predict(dataset.get_full_data().data)
```

### Loading Pretrained Models

```python
# Load a trained model
model = DEKM(n_clusters=10)
model.load_model('pretrained_weights/mnist_dekm.pth', dataset)

# Get model information
summary = model.get_model_summary()
print(f"Model has {summary['total_parameters']:,} parameters")
```

## 🎨 Visualization Examples

The evaluation pipeline generates several types of visualizations:

- **t-SNE plots** showing cluster separation in 2D
- **Confusion matrices** comparing true vs predicted labels
- **Cluster distributions** showing class balance
- **PCA projections** for faster visualization of high-dimensional embeddings

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{guo2021deep,
    title={Deep Embedded K-Means Clustering},
    author={Guo, Wengang and Lin, Kaiyan and Ye, Wei},
    booktitle={2021 International Conference on Data Mining Workshops (ICDMW)},
    pages={686--694},
    year={2021},
    organization={IEEE}
}
```

## 🙏 Acknowledgments

- Original paper authors for the DEKM algorithm
- PyTorch team for the excellent deep learning framework
- The open-source community for various utility libraries

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-username/deep-embedded-kmeans-clustering-pytorch/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

---

**Happy Clustering! 🎯**
