import argparse

import numpy as np
import torch

from DEKM import DEKM
from DEKM_AE import DEKM_AE
from dataset import DatasetWrapper
from utils import metrics


def get_arg():
    arg = argparse.ArgumentParser()
    arg.add_argument('-bs', default=256, type=int, help='batch size')
    arg.add_argument('-pre_epoch', type=int, default=200, help='epochs for train Autoencoder')
    arg.add_argument('-epoch', type=int, default=200, help='epochs to train DEKM')
    arg.add_argument('-n', type=int, help='num of clusters')
    arg.add_argument('-save_dir', default='pretrained_weights', help='location where model will be saved')
    arg.add_argument('-seed', type=int, default=42, help='torch random seed')
    arg = arg.parse_args()
    return arg


def main():
    arg = get_arg()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if arg.seed is not None:
        np.random.seed(arg.seed)
        torch.manual_seed(arg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    dataset_wrapper = DatasetWrapper('mnist')
    print("Image shape:", dataset_wrapper.get_image_shape())
    train_data = dataset_wrapper.get_train_data()
    test_data = dataset_wrapper.get_test_data()
    full_data = dataset_wrapper.get_full_data()
    print(train_data.data.shape)
    print(test_data.data.shape)
    print(full_data.data.shape)

    model = DEKM(n_clusters=10)
    model.fit(dataset_wrapper)
    predicted_labels = model.predict(dataset_wrapper.get_full_data().data)
    nmi = metrics.nmi(dataset_wrapper.get_full_data().target, predicted_labels)
    ari = metrics.ari(dataset_wrapper.get_full_data().target, predicted_labels)
    acc = metrics.acc(dataset_wrapper.get_full_data().target, predicted_labels)
    print('NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\n'.format(nmi, ari, acc))


if __name__ == '__main__':
    main()
