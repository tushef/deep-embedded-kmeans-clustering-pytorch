import numpy as np
import torch
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from torch import linalg
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset

import utils_preprocess
from algorithms.DEKM.DEKM_AE import DEKM_AE


class DEKM(BaseEstimator, ClusterMixin):
    """
    The DEKM: Deep Embedded K-Means Clustering
    PyTorch implementation for DEKM

    Parameters
    ----------
    n_clusters : int
        number of clusters. Can be None if a corresponding initial_clustering_class is given, e.g. DBSCAN
    alpha : float
        alpha value for the prediction (default: 1.0)
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder (default: 100)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 150)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
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
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining (default: KMeans)
    initial_clustering_params : dict
        parameters for the initial clustering class (default: {})
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    n_clusters : int
        number of clusters
    autoencoder : torch.nn.Module
        The final autoencoder

    References
    ----------

    """

    def __init__(self, n_clusters: int, alpha: float = 1.0, batch_size: int = 256,
                 pretrain_optimizer_params: dict = None, clustering_optimizer_params: dict = None,
                 pretrain_epochs: int = 100, clustering_epochs: int = 150,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, cluster_loss_weight: float = 1, custom_dataloaders: tuple = None,
                 augmentation_invariance: bool = False, initial_clustering_class: ClusterMixin = KMeans,
                 initial_clustering_params: dict = None, random_state: np.random.RandomState = None):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.pretrain_optimizer_params = pretrain_optimizer_params
        self.clustering_optimizer_params = clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.cluster_loss_weight = cluster_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.initial_clustering_class = initial_clustering_class
        self.initial_clustering_params = initial_clustering_params
        self.random_state = random_state
        self.center = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DEKM':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : DEKM
            this instance of the DEKM algorithm
        """
        if self.autoencoder is None:
            self.create_autoencoder(X)

        if not self.autoencoder.fitted:
            self._pretrain_autoencoder(X)

        self.train(X)

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

    def _pretrain_autoencoder(self, X):
        print("Pretrain DEKM Autoencoder")
        ### Prepare Data
        channels = self.autoencoder.input_shape[2]
        width = self.autoencoder.input_shape[0]
        height = self.autoencoder.input_shape[1]

        dataset = TensorDataset(torch.from_numpy(X).view(X.shape[0], channels, height, width) / 255.0)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        dataset_size = X.shape[0]

        optimizer = self.optimizer_class(self.autoencoder.parameters())

        self.autoencoder = self._train_autoencoder(dataset_size,
                                                   optimizer,
                                                   dataloader)
        self.autoencoder.fitted = True

    def _cluster(self, X):
        return linalg.norm(X[:, None, :] - self.center, dim=2)

    def create_autoencoder(self, X):
        print('Creating Autoencoder')
        image_shape = utils_preprocess.get_input_shape(X)
        ae = DEKM_AE(input_shape=[image_shape[1], image_shape[2], image_shape[3]],
                     layers=[32, 64, 128], embedding_size=self.embedding_size)
        self.autoencoder = ae

    def _train_autoencoder(self,
                           dataset_size,
                           optimizer,
                           train_dataloader, ):

        self.autoencoder.train()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for epoch in range(1, self.pretrain_epochs + 1):
            loss = 0.
            for i, (x) in enumerate(train_dataloader, 1):
                x = x[0].to(device)
                optimizer.zero_grad()
                _, reconstruction = self.autoencoder(x)
                batch_loss = self.loss_fn(x, reconstruction)
                batch_loss.backward()
                optimizer.step()
                loss += batch_loss * x.shape[0]

            loss /= dataset_size
        return self.autoencoder

    def sorted_eig(self, X):
        e_vals, e_vecs = np.linalg.eig(X)
        idx = np.argsort(e_vals)
        e_vecs = e_vecs[:, idx]
        e_vals = e_vals[idx]
        return e_vals, e_vecs

    def train(self, x):
        model = self.autoencoder
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        optimizer = self.optimizer_class(self.autoencoder.parameters())
        loss_value = 0
        index = 0
        kmeans_n_init = 100
        assignment = np.array([-1] * len(x))
        index_array = np.arange(x.shape[0])
        x_tensor = torch.Tensor(x)

        for ite in range(self.clustering_epochs):
            if ite % 10 == 0:
                H = model.encode(x_tensor).detach().numpy()
                ans_kmeans = KMeans(n_clusters=self.n_clusters, n_init=kmeans_n_init).fit(H)
                kmeans_n_init = int(ans_kmeans.n_iter_ * 2)

                U = ans_kmeans.cluster_centers_
                self.center = Parameter(torch.tensor(U, device=device), False)
                assignment_new = ans_kmeans.labels_
                self.labels_ = ans_kmeans.labels_

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
                    temp = H[assignment == i] - U[i]
                    temp = np.matmul(np.transpose(temp), temp)
                    S_i.append(temp)
                S_i = np.array(S_i)
                S = np.sum(S_i, 0)
                Evals, V = self.sorted_eig(S)
                H_vt = np.matmul(H, V)
                U_vt = np.matmul(U, V)

            if n_change_assignment <= len(x) * 0.005:
                print('Ending Training')
                break

            idx = index_array[index * self.batch_size: min((index + 1) * self.batch_size, x.shape[0])]
            y_true = H_vt[idx]
            temp = assignment[idx]
            for i in range(len(idx)):
                y_true[i, -1] = U_vt[temp[i], -1]
            y_true = torch.Tensor(y_true)

            model.train()
            optimizer.zero_grad()
            outputs = model.encode(x_tensor[idx])
            y_pred_cluster = torch.matmul(outputs, torch.Tensor(V))
            loss_value = torch.nn.functional.mse_loss(y_true, y_pred_cluster)
            loss_value.backward()
            optimizer.step()

            index = index + 1 if (index + 1) * self.batch_size <= x.shape[0] else 0
