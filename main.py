import argparse
import os
import numpy as np
import torch
from DEKM import DEKM
from dataset import DatasetWrapper
from utils import metrics


def get_args():
    """Parse command line arguments with improved structure and validation."""
    parser = argparse.ArgumentParser(
        description='Deep Embedded K-Means Clustering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument('--dataset', '-d', default='mnist', type=str,
                              choices=['mnist', 'cifar10', 'fashionmnist', 'kmnist', 'stl10', 'usps'],
                              help='Dataset to use for clustering')
    dataset_group.add_argument('--data-root', default='data', type=str,
                              help='Root directory for datasets')
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--n-clusters', '-n', type=int, required=True,
                            help='Number of clusters')
    model_group.add_argument('--embedding-size', type=int, default=10,
                            help='Size of the embedding layer')
    model_group.add_argument('--autoencoder-layers', nargs=3, type=int, default=[32, 64, 128],
                            help='Autoencoder layer sizes (encoder)')
    
    # Training arguments
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument('--batch-size', '-bs', default=256, type=int,
                               help='Batch size for training')
    training_group.add_argument('--pretrain-epochs', type=int, default=200,
                               help='Number of epochs for autoencoder pretraining')
    training_group.add_argument('--clustering-epochs', type=int, default=200,
                               help='Number of epochs for clustering training')
    training_group.add_argument('--pretrain-lr', type=float, default=0.001,
                               help='Learning rate for pretraining')
    training_group.add_argument('--clustering-lr', type=float, default=0.0001,
                               help='Learning rate for clustering')
    training_group.add_argument('--cluster-loss-weight', type=float, default=1.0,
                               help='Weight of clustering loss vs reconstruction loss')
    
    # System arguments
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument('--device', default='auto', type=str,
                             choices=['auto', 'cpu', 'cuda'],
                             help='Device to use for training')
    system_group.add_argument('--seed', type=int, default=42,
                             help='Random seed for reproducibility')
    system_group.add_argument('--num-workers', type=int, default=4,
                             help='Number of workers for data loading')
    
    # Output arguments
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--save-dir', default='pretrained_weights', type=str,
                             help='Directory to save model weights')
    output_group.add_argument('--results-dir', default='results', type=str,
                             help='Directory to save results and plots')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='Enable verbose output')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup and return the appropriate device for training."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    return device


def setup_seed(seed):
    """Setup random seeds for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Changed to False for reproducibility


def create_directories(args):
    """Create necessary directories if they don't exist."""
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)


def print_dataset_info(dataset_wrapper, verbose=False):
    """Print dataset information."""
    print(f"Dataset: {dataset_wrapper.dataset_name}")
    print(f"Image shape: {dataset_wrapper.get_image_shape()}")
    
    if verbose:
        train_data = dataset_wrapper.get_train_data()
        test_data = dataset_wrapper.get_test_data()
        full_data = dataset_wrapper.get_full_data()
        print(f"Train data shape: {train_data.data.shape}")
        print(f"Test data shape: {test_data.data.shape}")
        print(f"Full data shape: {full_data.data.shape}")


def print_results(nmi, ari, acc):
    """Print clustering results in a formatted way."""
    print("\n" + "="*50)
    print("CLUSTERING RESULTS")
    print("="*50)
    print(f"Normalized Mutual Information (NMI): {nmi:.5f}")
    print(f"Adjusted Rand Index (ARI):           {ari:.5f}")
    print(f"Clustering Accuracy (ACC):           {acc:.5f}")
    print("="*50)


def main():
    """Main function to run Deep Embedded K-Means clustering."""
    args = get_args()
    
    # Setup device and seed
    device = setup_device(args.device)
    setup_seed(args.seed)
    
    # Create directories
    create_directories(args)
    
    if args.verbose:
        print(f"Using device: {device}")
        print(f"Random seed: {args.seed}")
    
    # Load dataset
    dataset_wrapper = DatasetWrapper(
        dataset_name=args.dataset,
        root=args.data_root
    )
    print_dataset_info(dataset_wrapper, args.verbose)
    
    # Initialize and train model
    model = DEKM(
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs,
        clustering_epochs=args.clustering_epochs,
        pretrain_learning_rate=args.pretrain_lr,
        clustering_learning_rate=args.clustering_lr,
        embedding_size=args.embedding_size,
        cluster_loss_weight=args.cluster_loss_weight,
        save_dir=os.path.join(args.save_dir, f"{args.dataset}_dekm.pth"),
        random_state=args.seed
    )
    
    print(f"\nStarting DEKM training on {args.dataset} with {args.n_clusters} clusters...")
    model.fit(dataset_wrapper)
    
    # Evaluate model
    print("Evaluating model...")
    predicted_labels = model.predict(dataset_wrapper.get_full_data().data)
    true_labels = dataset_wrapper.get_full_data().target
    
    # Calculate metrics
    nmi = metrics.nmi(true_labels, predicted_labels)
    ari = metrics.ari(true_labels, predicted_labels)
    acc = metrics.acc(true_labels, predicted_labels)
    
    print_results(nmi, ari, acc)
    
    # Save results
    results = {
        'nmi': nmi,
        'ari': ari,
        'acc': acc,
        'predicted_labels': predicted_labels,
        'true_labels': true_labels,
        'args': vars(args)
    }
    
    results_file = os.path.join(args.results_dir, f"{args.dataset}_results.npz")
    np.savez(results_file, **{k: v for k, v in results.items() if k != 'args'})
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
