import numpy as np
import torch
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from torch import linalg
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset

import utils
from DEKM_AE import DEKM_AE
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
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, cluster_loss_weight: float = 1, custom_dataloaders: tuple = None,
                 random_state: int = 42, save_dir: str = None):

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
        np.random.seed(random_state)
        # Attributes
        self.center = None
        self.labels_ = None

    def fit(self, dataset: DatasetWrapper) -> 'DEKM':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        dataset : DatasetWrapper
            the given dataset

        Returns
        -------
        self : DEKM
            this instance of the DEKM algorithm
        """
        if self.autoencoder is None:
            self._create_autoencoder(dataset)

        if not self.autoencoder.fitted:
            self._pretrain_autoencoder(dataset)

        self.train(dataset.get_full_data().data)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels of the input data.

        Parameters
        ----------
        X : np.ndarray
            input data

        Returns
        -------
        predicted_labels : np.ndarray
            The predicted labels
        """
        self.autoencoder.eval()
        X = torch.from_numpy(X)
        X = self.autoencoder.encode(X)
        predicted_labels = self._cluster(X).min(1)[1].cpu().numpy()
        return predicted_labels

    def _create_autoencoder(self, X):
        print('Creating Autoencoder')
        self.autoencoder = DEKM_AE(input_shape=X.get_image_shape(),
                                   layers=[32, 64, 128], embedding_size=self.embedding_size)

    def _pretrain_autoencoder(self, X):
        print("Pretrain DEKM Autoencoder")

        dataloader = X.get_dataloader(batch_size=self.batch_size, num_workers=4)

        optimizer = self.optimizer_class(self.autoencoder.parameters(), lr=self.pretrain_learning_rate)

        self.autoencoder.train()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for epoch in range(1, self.pretrain_epochs + 1):
            loss = 0.
            for i, (x) in enumerate(dataloader, 1):
                x = x[0].to(device)
                optimizer.zero_grad()
                _, reconstruction = self.autoencoder(x)
                batch_loss = self.loss_fn(x, reconstruction)
                batch_loss.backward()
                optimizer.step()
                loss += batch_loss * x.shape[0]

        self.autoencoder.fitted = True
        torch.save(self.autoencoder.state_dict(), self.save_dir)

    def _cluster(self, X):
        return linalg.norm(X[:, None, :] - self.center, dim=2)

    def _sorted_eig(self, X):
        e_vals, e_vecs = np.linalg.eig(X)
        idx = np.argsort(e_vals)
        e_vecs = e_vecs[:, idx]
        e_vals = e_vals[idx]
        return e_vals, e_vecs

    def train(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        optimizer = self.optimizer_class(self.autoencoder.parameters(), lr=self.clustering_learning_rate)
        index = 0
        kmeans_n_init = 100
        assignment = np.array([-1] * len(x))
        index_array = np.arange(x.shape[0])

        for ite in range(self.clustering_epochs):
            if ite % 10 == 0:
                hidden_layer = self.autoencoder.encode(x).detach().numpy()
                kmeans = KMeans(n_clusters=self.n_clusters, n_init=kmeans_n_init).fit(hidden_layer)
                kmeans_n_init = int(kmeans.n_iter_ * 2)

                cluster_centers_ = kmeans.cluster_centers_
                self.center = Parameter(torch.tensor(cluster_centers_, device=device), False)
                assignment_new = kmeans.labels_
                self.labels_ = kmeans.labels_

                w = np.zeros((self.n_clusters, self.n_clusters), dtype=np.int64)
                for i in range(len(assignment_new)):
                    w[assignment_new[i], assignment[i]] += 1
                from scipy.optimize import linear_sum_assignment as linear_assignment
                ind = linear_assignment(-w)
                temp = np.array(assignment)
                for i in range(self.n_clusters):
                    assignment[temp == ind[1][i]] = i
                n_change_assignment = np.sum(assignment_new != assignment)
                assignment = assignment_new

                S_i = []
                for i in range(self.n_clusters):
                    temp = hidden_layer[assignment == i] - cluster_centers_[i]
                    temp = np.matmul(np.transpose(temp), temp)
                    S_i.append(temp)
                S_i = np.array(S_i)
                S = np.sum(S_i, 0)
                Evals, V = self._sorted_eig(S)
                H_vt = np.matmul(hidden_layer, V)
                U_vt = np.matmul(cluster_centers_, V)

            if n_change_assignment <= len(x) * 0.005:
                print('Ending Training')
                break

            idx = index_array[index * self.batch_size: min((index + 1) * self.batch_size, x.shape[0])]
            y_true = H_vt[idx]
            temp = assignment[idx]
            for i in range(len(idx)):
                y_true[i, -1] = U_vt[temp[i], -1]
            y_true = torch.Tensor(y_true)

            self.autoencoder.train()
            optimizer.zero_grad()
            outputs = self.autoencoder.encode(x[idx])
            y_pred_cluster = torch.matmul(outputs, torch.Tensor(V))
            loss_value = self.loss_fn(y_true, y_pred_cluster)
            loss_value.backward()
            optimizer.step()

            index = index + 1 if (index + 1) * self.batch_size <= x.shape[0] else 0
