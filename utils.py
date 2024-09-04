import numpy as np
import sklearn
import torch
from torchvision.transforms import transforms


def get_input_shape(x):
    """
         x: ndarray
        The data which needs to be interpreted
    """
    # assume square images
    width = int(np.sqrt(x.shape[-1]))
    if width * width == x.shape[-1]:  # gray
        im_shape = [-1, width, width, 1]
    else:  # RGB
        width = int(np.sqrt(x.shape[-1] / 3.0))
        im_shape = [-1, width, width, 3]
    return im_shape


# Simple tensor to image translation
def tensor2img(tensor):
    img = tensor.cpu().data[0]
    if img.shape[0] != 1:
        img = inv_normalize(img)
    img = torch.clamp(img, 0, 1)
    return img


inv_normalize = transforms.Normalize(
    mean=[-0.485 / .229, -0.456 / 0.224, -0.406 / 0.255],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)


# Metrics class was copied from DCEC article authors repository
class metrics:
    nmi = sklearn.metrics.normalized_mutual_info_score
    ari = sklearn.metrics.adjusted_rand_score
    @staticmethod
    def acc(labels_true, labels_pred):
        labels_true = labels_true.astype(np.int64)
        assert labels_pred.size == labels_true.size
        D = max(labels_pred.max(), labels_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(labels_pred.size):
            w[labels_pred[i], labels_true[i]] += 1
        from scipy.optimize import linear_sum_assignment
        ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
        return sum([w[i, j] for i, j in ind]) * 1.0 / labels_pred.size
