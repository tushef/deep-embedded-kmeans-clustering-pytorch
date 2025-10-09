"""
Evaluation Pipeline for Deep Embedded K-Means Clustering

This module provides evaluation functionality for trained DEKM models,
including loading pretrained weights, generating visualizations, and
computing comprehensive clustering metrics.

Author: Deep Embedded K-Means Implementation
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import torch
from DEKM import DEKM
from dataset import DatasetWrapper
from utils import metrics


def get_evaluation_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained DEKM models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--model-path', required=True, type=str,
                       help='Path to the trained model file')
    parser.add_argument('--dataset', '-d', default='mnist', type=str,
                       choices=['mnist', 'cifar10', 'fashionmnist', 'kmnist', 'stl10', 'usps'],
                       help='Dataset to evaluate on')
    
    # Optional arguments
    parser.add_argument('--data-root', default='data', type=str,
                       help='Root directory for datasets')
    parser.add_argument('--output-dir', default='plots', type=str,
                       help='Directory to save evaluation plots')
    parser.add_argument('--embedding-size', type=int, default=10,
                       help='Embedding size (will be loaded from model if available)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', default='auto', type=str,
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for evaluation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup and return the appropriate device for evaluation."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    return device


def load_trained_model(model_path, dataset_wrapper, embedding_size=None):
    """Load a trained DEKM model from file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading trained model from: {model_path}")
    
    # Initialize model with default parameters
    model = DEKM(
        n_clusters=10,  # Will be updated from saved model
        embedding_size=embedding_size or 10,  # Will be updated from saved model
        batch_size=256,
        pretrain_epochs=200,
        clustering_epochs=200
    )
    
    # Load the trained model
    model.load_model(model_path, dataset_wrapper)
    
    return model


def compute_clustering_metrics(true_labels, predicted_labels):
    """Compute comprehensive clustering evaluation metrics."""
    print("Computing clustering metrics...")
    
    # Basic clustering metrics
    nmi = metrics.nmi(true_labels, predicted_labels)
    ari = metrics.ari(true_labels, predicted_labels)
    acc = metrics.acc(true_labels, predicted_labels)
    
    # Additional metrics
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    # Note: These require feature vectors, so we'll compute them later
    # when we have the embeddings
    
    metrics_dict = {
        'NMI': nmi,
        'ARI': ari,
        'Accuracy': acc,
    }
    
    return metrics_dict


def plot_clustering_results(true_labels, predicted_labels, embeddings, dataset_name, output_dir):
    """Generate comprehensive clustering visualization plots."""
    print("Generating clustering visualization plots...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name.upper()}')
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # t-SNE visualization
    if len(embeddings) > 1000:
        # Subsample for t-SNE if dataset is too large
        indices = np.random.choice(len(embeddings), 1000, replace=False)
        embeddings_sample = embeddings[indices]
        true_labels_sample = true_labels[indices]
        predicted_labels_sample = predicted_labels[indices]
    else:
        embeddings_sample = embeddings
        true_labels_sample = true_labels
        predicted_labels_sample = predicted_labels
    
    # t-SNE for true labels
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)//4))
    embeddings_tsne = tsne.fit_transform(embeddings_sample)
    
    plt.figure(figsize=(15, 6))
    
    # True labels
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                         c=true_labels_sample, cmap='tab10', alpha=0.7, s=20)
    plt.title(f't-SNE: True Classes - {dataset_name.upper()}')
    plt.colorbar(scatter)
    
    # Predicted labels
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                         c=predicted_labels_sample, cmap='tab10', alpha=0.7, s=20)
    plt.title(f't-SNE: Predicted Clusters - {dataset_name.upper()}')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_tsne_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PCA visualization (faster alternative)
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(15, 6))
    
    # True labels
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                         c=true_labels, cmap='tab10', alpha=0.7, s=20)
    plt.title(f'PCA: True Classes - {dataset_name.upper()}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter)
    
    # Predicted labels
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                         c=predicted_labels, cmap='tab10', alpha=0.7, s=20)
    plt.title(f'PCA: Predicted Clusters - {dataset_name.upper()}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_pca_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cluster size distribution
    plt.figure(figsize=(12, 5))
    
    # True class distribution
    plt.subplot(1, 2, 1)
    unique_classes, class_counts = np.unique(true_labels, return_counts=True)
    plt.bar(unique_classes, class_counts, alpha=0.7)
    plt.title(f'True Class Distribution - {dataset_name.upper()}')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Predicted cluster distribution
    plt.subplot(1, 2, 2)
    unique_clusters, cluster_counts = np.unique(predicted_labels, return_counts=True)
    plt.bar(unique_clusters, cluster_counts, alpha=0.7)
    plt.title(f'Predicted Cluster Distribution - {dataset_name.upper()}')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_evaluation_report(model, dataset_wrapper, true_labels, predicted_labels, 
                             embeddings, metrics_dict, output_dir):
    """Generate a comprehensive evaluation report."""
    print("Generating evaluation report...")
    
    # Get model summary
    model_summary = model.get_model_summary()
    
    # Create report
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("DEEP EMBEDDED K-MEANS CLUSTERING EVALUATION REPORT")
    report_lines.append("="*60)
    report_lines.append("")
    
    # Dataset information
    report_lines.append("DATASET INFORMATION:")
    report_lines.append(f"  Dataset: {dataset_wrapper.dataset_name.upper()}")
    report_lines.append(f"  Image Shape: {dataset_wrapper.get_image_shape()}")
    report_lines.append(f"  Total Samples: {len(true_labels)}")
    report_lines.append(f"  Number of Classes: {len(np.unique(true_labels))}")
    report_lines.append("")
    
    # Model information
    report_lines.append("MODEL CONFIGURATION:")
    report_lines.append(f"  Number of Clusters: {model_summary['n_clusters']}")
    report_lines.append(f"  Embedding Size: {model_summary['embedding_size']}")
    report_lines.append(f"  Batch Size: {model_summary['batch_size']}")
    report_lines.append(f"  Pretrain Epochs: {model_summary['pretrain_epochs']}")
    report_lines.append(f"  Clustering Epochs: {model_summary['clustering_epochs']}")
    report_lines.append(f"  Total Parameters: {model_summary.get('total_parameters', 'N/A'):,}")
    report_lines.append("")
    
    # Clustering metrics
    report_lines.append("CLUSTERING PERFORMANCE:")
    for metric_name, metric_value in metrics_dict.items():
        report_lines.append(f"  {metric_name}: {metric_value:.5f}")
    report_lines.append("")
    
    # Cluster statistics
    report_lines.append("CLUSTER STATISTICS:")
    unique_clusters, cluster_counts = np.unique(predicted_labels, return_counts=True)
    for cluster_id, count in zip(unique_clusters, cluster_counts):
        percentage = (count / len(predicted_labels)) * 100
        report_lines.append(f"  Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")
    report_lines.append("")
    
    # Save report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Evaluation report saved to: {report_path}")
    
    # Also print to console
    print('\n'.join(report_lines))


def main():
    """Main evaluation function."""
    args = get_evaluation_args()
    
    # Setup device and seed
    device = setup_device(args.device)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    if args.verbose:
        print(f"Using device: {device}")
        print(f"Random seed: {args.seed}")
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset_wrapper = DatasetWrapper(
        dataset_name=args.dataset,
        root=args.data_root
    )
    
    # Load trained model
    model = load_trained_model(args.model_path, dataset_wrapper, args.embedding_size)
    
    # Get full dataset
    full_data = dataset_wrapper.get_full_data()
    true_labels = full_data.target
    
    # Get embeddings and predictions
    print("Computing embeddings and predictions...")
    model.autoencoder.eval()
    with torch.no_grad():
        data_tensor = torch.from_numpy(full_data.data).float()
        embeddings = model.autoencoder.encode(data_tensor).cpu().numpy()
    
    predicted_labels = model.predict(full_data.data)
    
    # Compute metrics
    metrics_dict = compute_clustering_metrics(true_labels, predicted_labels)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric_name, metric_value in metrics_dict.items():
        print(f"{metric_name}: {metric_value:.5f}")
    print("="*50)
    
    # Generate visualizations
    plot_clustering_results(true_labels, predicted_labels, embeddings, 
                          args.dataset, args.output_dir)
    
    # Generate evaluation report
    generate_evaluation_report(model, dataset_wrapper, true_labels, predicted_labels,
                             embeddings, metrics_dict, args.output_dir)
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
