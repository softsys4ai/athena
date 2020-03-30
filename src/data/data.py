"""
Implement operations related to data (dataset).
Functions regarding color images are adapted from https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/data.py
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

import logging

import keras
import matplotlib.pyplot as plt
import torchvision
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.models import load_model
from sklearn.model_selection import StratifiedShuffleSplit
from theconf import Config as C
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import SubsetRandomSampler
from torchvision.transforms import transforms as tv_transforms

import utils.data_utils as data_utils
from models.transformation import transform
from utils.archive import arsaug_policy, autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10
from utils.augmentations import *
from utils.config import *
from utils.logger import get_logger

"""
set random seed for replication
"""
np.random.seed(1000)

logger = get_logger('Athena')
logger.setLevel(logging.INFO)

# For complex data sets like CIFAR-100
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}

# For CIFAR-10 and CIFAR-100
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def load_data(dataset, trans_type=TRANSFORMATION.clean, channel_first=False, trans_set='both'):
    assert dataset in DATA.get_supported_datasets()
    assert trans_set is None or trans_set in ['none', 'train', 'test', 'both']

    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    img_rows = 0
    img_cols = 0
    nb_channels = 0
    nb_classes = 0

    if DATA.mnist == dataset:
        """
        Dataset of 60,000 28x28 grayscale images of the 10 digits,
        along with a test set of 10,000 images.
        """
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

        nb_examples, img_rows, img_cols = X_test.shape
        nb_channels = 1
        nb_classes = 10
    elif DATA.fation_mnist == dataset:
        """
        Dataset of 60,000 28x28 grayscale images of 10 fashion categories,
        along with a test set of 10,000 images. The class labels are:
        Label   Description
        0       T-shirt/top
        1       Trouser
        2       Pullover
        3       Dress
        4       Coat
        5       Sandal
        6       Shirt
        7       Sneaker
        8       Bag
        9       Ankle boot
        """
        (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

        nb_examples, img_rows, img_cols = X_test.shape
        nb_channels = 1
        nb_classes = 10
    elif DATA.cifar_10 == dataset:
        """
        Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
        """
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

        nb_examples, img_rows, img_cols, nb_channels = X_test.shape
        nb_classes = 10
    elif DATA.cifar_100 == dataset:
        (X_train, Y_train), (X_test, Y_test) = cifar100.load_data(label_mode='fine')
        nb_examples, img_rows, img_cols, nb_channels = X_test.shape
        nb_classes = 100

    X_train = X_train.reshape(-1, img_rows, img_cols, nb_channels)
    X_test = X_test.reshape(-1, img_rows, img_cols, nb_channels)

    """
    cast pixels to floats, normalize to [0, 1] range
    """
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_train = data_utils.rescale(X_train, range=(0., 1.))
    X_test = data_utils.rescale(X_test, range=(0., 1.))

    """
    one-hot-encode the labels
    """
    Y_train = keras.utils.to_categorical(Y_train, nb_classes)
    Y_test = keras.utils.to_categorical(Y_test, nb_classes)

    """
    transform images
    """
    if trans_set is not None:
        if trans_set in ['train', 'both']:
            X_train = transform(X_train, trans_type)
            X_train = data_utils.rescale(X_train, range=(0., 1.))

        if trans_set in ['test', 'both']:
            X_test = transform(X_test, trans_type)
            X_test = data_utils.rescale(X_test, range=(0., 1.))

    if channel_first:
        X_train = data_utils.set_channels_first(X_train)
        X_test = data_utils.set_channels_first(X_test)
    """
    summarize data set
    """
    print('Dataset({}) Summary:'.format(dataset.upper()))
    print('Train set: {}, {}'.format(X_train.shape, Y_train.shape))
    print('Test set: {}, {}'.format(X_test.shape, Y_test.shape))
    return (X_train, Y_train), (X_test, Y_test)


def get_augmentation(dataset):
    if dataset in [DATA.mnist, DATA.fation_mnist]:
        transform_train = None
        transform_test = None

    elif dataset in [DATA.cifar_10, DATA.cifar_100]:
        transform_train = tv_transforms.Compose([
            tv_transforms.RandomCrop(32, padding=4),
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
        ])

        transform_test = tv_transforms.Compose([
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])

    else:
        raise NotImplementedError('Not yet implemented for dataset={}.'.format(dataset))

    return transform_train, transform_test


def get_augmented_aeloaders(dataset, batch, dataroot, ae_file, trans_type=TRANSFORMATION.clean):
    train_sampler, trainloader, validloader, _ = get_dataloaders(dataset,
                                                                 batch,
                                                                 dataroot,
                                                                 trans_type)
    _, test_aug = get_augmentation(dataset)
    _, (_, y_test) = load_data(dataset)

    x_ae = load_model(ae_file)
    x_ae = transform(x_ae, trans_type)
    x_ae = data_utils.rescale(x_ae)

    x_ae = data_utils.set_channels_first(x_ae)

    testset = MyDataset(x_ae, y_test, aug=test_aug)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=32, pin_memory=torch.cuda.is_available(),
        drop_last=False)

    return train_sampler, trainloader, validloader, testloader


def get_dataloaders(dataset, batch, trans_type=TRANSFORMATION.clean, trans_set='both', **kwargs):
    train_aug, test_aug = get_augmentation(dataset)

    split = kwargs.get('split', 0.15)
    split_idx = kwargs.get('split_idx', 0)
    target_lb = kwargs.get('target_lb', -1)
    aug = kwargs.get('aug', 'default')
    cutout = kwargs.get('cutout', 0)

    if isinstance(aug, list):
        logger.debug('Processing data with custom augmentation [{}].'.format(aug))
        train_aug.transforms.insert(0, Augmentation(aug))
    else:
        logger.debug('Processing data with pre-defined augmentation [{}].'.format(aug))
        if aug == 'fa_reduced_cifar10':
            train_aug.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        elif aug == 'arsaug':
            train_aug.transforms.insert(0, Augmentation(arsaug_policy()))
        elif aug == 'autoaug_cifar10':
            train_aug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        elif aug == 'autoaug_extend':
            train_aug.transforms.insert(0, Augmentation(autoaug_policy()))
        elif aug in ['default', 'inception', 'inception320']:
            pass
        else:
            raise ValueError('Augmentation [{}] is not supported.'.format(aug))

    if cutout > 0:
        train_aug.transforms.append(CutoutDefault(cutout))

    (x_train, y_train), (x_test, y_test) = load_data(dataset=dataset, trans_type=trans_type, trans_set=trans_set)
    if dataset in DATA.get_supported_datasets():
        total_trainset = MyDataset(x_train, y_train, aug=train_aug)
        testset = MyDataset(x_test, y_test, aug=test_aug)
    else:
        raise ValueError("Dataset [{}] is not supported yet.".format(dataset))

    train_sampler = None
    if split > 0.0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)

        train_idx = None
        valid_idx = None
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        if target_lb >= 0:
            train_idx = [i for i in train_idx if total_trainset.targets[i] == target_lb]
            valid_idx = [i for i in valid_idx if total_trainset.targets[i] == target_lb]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)
    else:
        valid_sampler = SubsetSampler([])

    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False, num_workers=32,
        pin_memory=torch.cuda.is_available(), sampler=train_sampler, drop_last=True)
    validloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=False, num_workers=16, pin_memory=torch.cuda.is_available(),
        sampler=valid_sampler, drop_last=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=32, pin_memory=torch.cuda.is_available(), drop_last=False)

    # --------------------------------
    # for test
    """
    for b_idx, (d, t) in enumerate(trainloader):
        print('--- batch idx {}, data shape {}, target shape {}.'.format(b_idx, d.shape, t.shape))
        print('---- type: ', type(t), t.dtype)
        print('---- values: ', torch.max(d[0]), torch.min(d[0]))
        show_gridimg(d[0])
        labels = []
        for i in range(9):
            grid_img = torchvision.utils.make_grid(d[i], normalize=True)
            plt.subplot(330 + 1 + i)
            plt.imshow(grid_img.permute(1, 2, 0), interpolation='nearest')
            labels.append(np.argmax(t[i]))

        plt.show()
        print('labels:', labels)
        break
    """
    # --------------------------------

    return train_sampler, trainloader, validloader, testloader


def show_gridimg(img):
    grid_img = torchvision.utils.make_grid(img, normalize=True)
    plt.imshow(grid_img.permute(1, 2, 0), interpolation='nearest')
    plt.show()


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class Augmentation(object):
    def __init__(self, polices):
        self.policies = polices

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)

        return img


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class MyDataset(Dataset):
    def __init__(self, data, targets, aug=None, target_aug=None, as_image=False):
        self.data = torch.from_numpy(data).float()
        self.targets = torch.LongTensor(targets)
        self.aug = aug
        self.target_aug = target_aug
        self.as_image = as_image

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.as_image:
            # Expect a PIL image as an item
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = torchvision.transforms.ToPILImage()(img).convert('RGB')

        if self.aug is not None:
            img = self.aug(img)

        if self.target_aug is not None:
            target = self.target_aug(target)

        return img, target

    def __len__(self):
        return len(self.data)


"""
for testing
"""
def main(args):
    load_data(args)

if __name__ == "__main__":
    main(DATA.cifar_10)