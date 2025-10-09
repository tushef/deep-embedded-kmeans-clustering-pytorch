"""
Deep Embedded K-Means Clustering Implementation

This module implements the DEKM (Deep Embedded K-Means) clustering algorithm
as described in the paper "Deep Embedded K-Means Clustering" by Guo et al.

The algorithm combines deep learning with traditional k-means clustering by:
1. Pretraining an autoencoder for feature learning
2. Using the learned embeddings for k-means clustering
3. Jointly optimizing reconstruction and clustering objectives

"""

import numpy as np
import torch
import os
from typing import Optional, Tuple, Dict, Any
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from torch import linalg
from torch.nn import Parameter
from ConvAutoencoder import ConvAutoencoder
from dataset import DatasetWrapper


class DEKM(BaseEstimator, ClusterMixin):
    """
    The DEKM: Deep Embedded K-Means Clustering
    PyTorch implementation for DEKM

    Parameters
    ----------
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    batch_size : int
        size of the data batches (default: 256)
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder (default: 100)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 150)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    pretrain_learning_rate: float
        the learning rate for the pretraining optimizer (default: 0.001)
    clustering_learning_rate
        the learning rate for the clustering optimizer (default: 0.0001)
    loss_fn : torch.nn.modules.loss._Loss
        loss function for the reconstruction (default: torch.nn.MSELoss())
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created (default: None)
    embedding_size : int
        size of the embedding within the autoencoder (default: 10)
    cluster_loss_weight : float
        weight of the clustering loss compared to the reconstruction loss (default: 1)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used (default: None)
    random_state : int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    save_dir: str
        saving directory to save the pretrained autoencoder weights (default: None)

    Attributes
    ----------
    n_clusters : int
        number of clusters
    autoencoder : torch.nn.Module
        The final autoencoder

    References
    ----------
    @inproceedings{guo2021deep,
        title={Deep Embedded K-Means Clustering},
        author={Guo, Wengang and Lin, Kaiyan and Ye, Wei},
        booktitle={2021 International Conference on Data Mining Workshops (ICDMW)},
        pages={686--694},
        year={2021},
        organization={IEEE}
    }

    """

    def __init__(self, n_clusters: int, batch_size: int = 256,
                 pretrain_epochs: int = 200, clustering_epochs: int = 200,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 pretrain_learning_rate: float = 0.001, clustering_learning_rate: float = 0.0001,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), 
                 autoencoder: Optional[torch.nn.Module] = None,
                 embedding_size: int = 10, cluster_loss_weight: float = 1, 
                 custom_dataloaders: Optional[tuple] = None,
                 random_state: int = 42, save_dir: Optional[str] = None):

        # Validate inputs
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if pretrain_epochs <= 0:
            raise ValueError("pretrain_epochs must be positive")
        if clustering_epochs <= 0:
            raise ValueError("clustering_epochs must be positive")
        if pretrain_learning_rate <= 0:
            raise ValueError("pretrain_learning_rate must be positive")
        if clustering_learning_rate <= 0:
            raise ValueError("clustering_learning_rate must be positive")
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        if cluster_loss_weight <= 0:
            raise ValueError("cluster_loss_weight must be positive")

        # Store parameters
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.pretrain_learning_rate = pretrain_learning_rate
        self.clustering_learning_rate = clustering_learning_rate
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.cluster_loss_weight = cluster_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.random_state = random_state
        self.save_dir = save_dir
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Model attributes (initialized during training)
        self.center = None
        self.labels_ = None
        self._is_fitted = False

    def fit(self, dataset: DatasetWrapper) -> 'DEKM':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        dataset : DatasetWrapper
            The given dataset wrapper containing the data to cluster

        Returns
        -------
        self : DEKM
            This instance of the DEKM algorithm (for method chaining)
        """
        print(f"Starting DEKM training with {self.n_clusters} clusters...")
        
        # Create autoencoder if not provided
        if self.autoencoder is None:
            self._create_autoencoder(dataset)

        # Pretrain autoencoder if not already fitted
        if not self.autoencoder.fitted:
            self._pretrain_autoencoder(dataset)
        else:
            print("Using pre-trained autoencoder")

        # Perform clustering training
        self._train_clustering(dataset.get_full_data().data)
        
        self._is_fitted = True
        print("DEKM training completed!")
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels of the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data to cluster

        Returns
        -------
        predicted_labels : np.ndarray
            The predicted cluster labels
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.center is None:
            raise ValueError("Model has not been properly trained")
        
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            X_encoded = self.autoencoder.encode(X_tensor)
            distances = self._compute_distances_to_centers(X_encoded)
            predicted_labels = distances.min(1)[1].cpu().numpy()
        
        return predicted_labels

    def _create_autoencoder(self, dataset: DatasetWrapper):
        """Create the autoencoder architecture."""
        print('Creating Autoencoder')
        input_shape = dataset.get_image_shape()
        self.autoencoder = ConvAutoencoder(
            input_shape=input_shape,
            layers=[32, 64, 128], 
            embedding_size=self.embedding_size
        )
        
        # Print model information
        model_info = self.autoencoder.get_model_info()
        print(f"Autoencoder created with {model_info['total_parameters']:,} parameters")

    def _pretrain_autoencoder(self, dataset: DatasetWrapper):
        """Pretrain the autoencoder for feature learning."""
        print("Pretraining DEKM Autoencoder")
        
        # Get dataloader
        if self.custom_dataloaders is not None:
            dataloader = self.custom_dataloaders[0]
        else:
            dataloader = dataset.get_dataloader(batch_size=self.batch_size, num_workers=4)

        # Setup optimizer and device
        optimizer = self.optimizer_class(self.autoencoder.parameters(), lr=self.pretrain_learning_rate)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(device)
        self.autoencoder.train()

        # Training loop
        for epoch in range(1, self.pretrain_epochs + 1):
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (x,) in enumerate(dataloader, 1):
                x = x.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                _, reconstruction = self.autoencoder(x)
                batch_loss = self.loss_fn(x, reconstruction)
                
                # Backward pass
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item() * x.shape[0]
                num_batches += 1

            # Print progress
            if epoch % 50 == 0 or epoch == self.pretrain_epochs:
                avg_loss = total_loss / len(dataloader.dataset)
                print(f"Pretraining Epoch {epoch}/{self.pretrain_epochs}, Loss: {avg_loss:.6f}")

        # Mark as fitted and save
        self.autoencoder.fitted = True
        if self.save_dir is not None:
            os.makedirs(os.path.dirname(self.save_dir), exist_ok=True)
            torch.save(self.autoencoder.state_dict(), self.save_dir)
            print(f"Pretrained autoencoder saved to: {self.save_dir}")

    def _compute_distances_to_centers(self, X: torch.Tensor) -> torch.Tensor:
        """Compute distances from data points to cluster centers."""
        return linalg.norm(X[:, None, :] - self.center, dim=2)

    def _compute_eigenvalues_and_vectors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sorted eigenvalues and eigenvectors."""
        e_vals, e_vecs = np.linalg.eig(X)
        idx = np.argsort(e_vals)
        e_vecs = e_vecs[:, idx]
        e_vals = e_vals[idx]
        return e_vals, e_vecs

    def _train_clustering(self, x: np.ndarray):
        """Train the clustering component of DEKM."""
        print("Starting clustering training...")
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(device)
        
        # Setup optimizer and training variables
        optimizer = self.optimizer_class(self.autoencoder.parameters(), lr=self.clustering_learning_rate)
        index = 0
        kmeans_n_init = 100
        assignment = np.array([-1] * len(x))
        index_array = np.arange(x.shape[0])
        
        # Convert data to tensor
        x_tensor = torch.from_numpy(x).float().to(device)

        for iteration in range(self.clustering_epochs):
            # Update cluster centers every 10 iterations
            if iteration % 10 == 0:
                with torch.no_grad():
                    hidden_layer = self.autoencoder.encode(x_tensor).cpu().detach().numpy()
                
                # Perform k-means clustering
                kmeans = KMeans(n_clusters=self.n_clusters, n_init=kmeans_n_init, random_state=self.random_state)
                kmeans.fit(hidden_layer)
                kmeans_n_init = int(kmeans.n_iter_ * 2)

                # Update cluster centers
                cluster_centers_ = kmeans.cluster_centers_
                self.center = Parameter(torch.tensor(cluster_centers_, device=device), requires_grad=False)
                assignment_new = kmeans.labels_
                self.labels_ = kmeans.labels_

                # Compute assignment consistency matrix
                w = np.zeros((self.n_clusters, self.n_clusters), dtype=np.int64)
                for i in range(len(assignment_new)):
                    if assignment[i] != -1:  # Skip initial assignment
                        w[assignment_new[i], assignment[i]] += 1
                
                # Find optimal assignment mapping
                from scipy.optimize import linear_sum_assignment as linear_assignment
                ind = linear_assignment(-w)
                temp = np.array(assignment)
                for i in range(self.n_clusters):
                    assignment[temp == ind[1][i]] = i
                
                n_change_assignment = np.sum(assignment_new != assignment)
                assignment = assignment_new

                # Compute scatter matrices for eigenvalue decomposition
                S_i = []
                for i in range(self.n_clusters):
                    cluster_data = hidden_layer[assignment == i] - cluster_centers_[i]
                    if len(cluster_data) > 0:
                        scatter_matrix = np.matmul(cluster_data.T, cluster_data)
                        S_i.append(scatter_matrix)
                
                if len(S_i) > 0:
                    S_i = np.array(S_i)
                    S = np.sum(S_i, 0)
                    Evals, V = self._compute_eigenvalues_and_vectors(S)
                    H_vt = np.matmul(hidden_layer, V)
                    U_vt = np.matmul(cluster_centers_, V)
                else:
                    print("Warning: No valid clusters found, skipping iteration")
                    continue

                # Print progress
                if iteration % 50 == 0:
                    print(f"Clustering iteration {iteration}/{self.clustering_epochs}, "
                          f"assignment changes: {n_change_assignment}")

            # Early stopping condition
            if n_change_assignment <= len(x) * 0.005:
                print('Convergence reached, ending training early')
                break

            # Training step
            batch_indices = index_array[index * self.batch_size: min((index + 1) * self.batch_size, x.shape[0])]
            y_true = H_vt[batch_indices].copy()
            temp_assignment = assignment[batch_indices]
            
            # Update target values based on cluster centers
            for i in range(len(batch_indices)):
                y_true[i, -1] = U_vt[temp_assignment[i], -1]
            
            y_true = torch.tensor(y_true, device=device)

            # Forward pass
            self.autoencoder.train()
            optimizer.zero_grad()
            outputs = self.autoencoder.encode(x_tensor[batch_indices])
            y_pred_cluster = torch.matmul(outputs, torch.tensor(V, device=device))
            
            # Compute loss and backpropagate
            loss_value = self.loss_fn(y_true, y_pred_cluster)
            loss_value.backward()
            optimizer.step()

            # Update batch index
            index = (index + 1) if (index + 1) * self.batch_size <= x.shape[0] else 0

        print(f"Clustering training completed after {iteration + 1} iterations")
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'center': self.center.detach().cpu().numpy(),
            'labels_': self.labels_,
            'n_clusters': self.n_clusters,
            'embedding_size': self.embedding_size,
            'model_params': {
                'batch_size': self.batch_size,
                'pretrain_epochs': self.pretrain_epochs,
                'clustering_epochs': self.clustering_epochs,
                'pretrain_learning_rate': self.pretrain_learning_rate,
                'clustering_learning_rate': self.clustering_learning_rate,
                'cluster_loss_weight': self.cluster_loss_weight,
                'random_state': self.random_state
            }
        }
        
        torch.save(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str, dataset: DatasetWrapper):
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = torch.load(filepath, map_location='cpu')
        
        # Create autoencoder if not exists
        if self.autoencoder is None:
            self._create_autoencoder(dataset)
        
        # Load autoencoder weights
        self.autoencoder.load_state_dict(model_data['autoencoder_state_dict'])
        self.autoencoder.fitted = True
        
        # Load clustering parameters
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.center = Parameter(torch.tensor(model_data['center'], device=device), requires_grad=False)
        self.labels_ = model_data['labels_']
        
        # Update model parameters if provided
        if 'model_params' in model_data:
            params = model_data['model_params']
            self.n_clusters = params.get('n_clusters', self.n_clusters)
            self.embedding_size = params.get('embedding_size', self.embedding_size)
        
        self._is_fitted = True
        print(f"Model loaded from: {filepath}")
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get the cluster centers."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before accessing cluster centers")
        return self.center.detach().cpu().numpy()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model configuration and training status."""
        summary = {
            'is_fitted': self._is_fitted,
            'n_clusters': self.n_clusters,
            'embedding_size': self.embedding_size,
            'batch_size': self.batch_size,
            'pretrain_epochs': self.pretrain_epochs,
            'clustering_epochs': self.clustering_epochs,
            'pretrain_learning_rate': self.pretrain_learning_rate,
            'clustering_learning_rate': self.clustering_learning_rate,
            'cluster_loss_weight': self.cluster_loss_weight,
            'random_state': self.random_state,
            'autoencoder_fitted': self.autoencoder.fitted if self.autoencoder else False
        }
        
        if self.autoencoder:
            summary.update(self.autoencoder.get_model_info())
        
        return summary
