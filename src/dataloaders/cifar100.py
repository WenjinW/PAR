from asyncio.log import logger
import logging
import os,sys
from types import new_class
import numpy as np
import pickle
from sklearn import utils
import torch
from PIL import Image
from torchvision import datasets,transforms
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from sklearn.utils import shuffle

import utils
from dataloaders.my_dataset import MyDataset
# from my_dataset import MyDataset


class MyCIFAR100(datasets.CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data["fine_label_names"]
            self.coarse_classes = data["coarse_label_names"]

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        self.coarse_class_to_idx = {_coarse_class: i for i, _coarse_class in enumerate(self.coarse_classes)}


train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

def _map_new_class_index(y, order):
    """Transforms targets for new class order."""

    return np.array(list(map(lambda x: order.index(x), y)))

def _select(x, y, low_range=0, high_range=0):
    idxes = sorted(np.where(np.logical_and(y >= low_range, y < high_range))[0])
    if isinstance(x, list):
        selected_x = [x[idx] for idx in idxes]
    else:
        selected_x = x[idxes]
    return selected_x, y[idxes]

def _split_per_class(x, y, validation_split=0.0):
    """Splits train data for a subset of validation data.
    Split is done so that each class has a much data.
    """
    shuffled_indexes = np.random.permutation(x.shape[0])
    x = x[shuffled_indexes]
    y = y[shuffled_indexes]

    x_val, y_val = [], []
    x_train, y_train = [], []

    for class_id in np.unique(y):
        class_indexes = np.where(y == class_id)[0]
        nb_val_elts = int(class_indexes.shape[0] * validation_split)

        val_indexes = class_indexes[:nb_val_elts]
        train_indexes = class_indexes[nb_val_elts:]

        x_val.append(x[val_indexes])
        y_val.append(y[val_indexes])
        x_train.append(x[train_indexes])
        y_train.append(y[train_indexes])

    # !list
    x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
    x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

    return x_val, y_val, x_train, y_train


coarse_to_fine = {
    "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish":	["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household electrical devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}


def get(path, seed=0, pc_valid=0.10, args=None):
    """

    """
    coarse_lable_list = [
        'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
        'household electrical devices', 'household furniture', 'insects', 'large carnivores',
        'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
        'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
        'trees', 'vehicles 1', 'vehicles 2']
    
    label_list = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy',
        'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
        'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
        'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
        'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
        'willow_tree', 'wolf', 'woman', 'worm'
    ]

    data = {}
    taskcla = []
    size = [3, 32, 32]

    
    if args.experiment in [
        "cifar100_coarse",
        "cifar100__carnivores__omnivores_and_herbivores",
        "cifar100__carnivores__flowers",
        "cifar100__flowers__fruits_and_vegetables",
        "cifar100__flowers__large_man-made_outdoor_things",
        "cifar100__aquatic_mammals__medium-sized_mammals",
        "cifar100__aquatic_mammals__household_furniture",
        ]:
        coarse_order = True
    else:
        coarse_order = False
    if coarse_order:
        increments = 5
    else:
        if args.experiment == "cifar100_10":
            increments = 10
        elif args.experiment == "cifar100_20":
            increments = 5

    num_ckasses = 100
    num_tasks = num_ckasses // increments
    data_path = path
    # name = "binary_cifar100_{}".format(seed)

    train_dataset = datasets.CIFAR100(data_path, train=True, download=True)
    test_dataset = datasets.CIFAR100(data_path, train=False, download=True)

    fine_label_to_id = train_dataset.class_to_idx

    train_x, train_y = train_dataset.data, np.array(train_dataset.targets)
    val_x, val_y, train_x, train_y = _split_per_class(train_x, train_y, validation_split=pc_valid)
    test_x, test_y = test_dataset.data, np.array(test_dataset.targets)

    if coarse_order:
        coarse_classes = {
            "0": 'vehicles 1',
            "1": 'vehicles 2',
            "2": 'aquatic mammals',
            "3": 'fish',
            "4": 'flowers',
            "5": 'food containers',
            "6": 'fruit and vegetables',
            "7": 'household electrical devices',
            "8": 'household furniture',
            "9": 'insects',
            "10": 'large carnivores',
            "11": 'large man-made outdoor things',
            "12": 'large natural outdoor scenes',
            "13": 'large omnivores and herbivores',
            "14": 'medium-sized mammals',
            "15": 'non-insect invertebrates',
            "16": 'people',
            "17": 'reptiles',
            "18": 'small mammals',
            "19": 'trees',
        }

        class_order = []
        for coarse_label in coarse_classes.values():
            utils.mkdir(os.path.join(path,"cifar100_examples/{}".format(coarse_label)))
            for fine_label in coarse_to_fine[coarse_label]:
                class_order.append(fine_label_to_id[fine_label])
                utils.mkdir(
                    os.path.join(path, "cifar100_examples/{}/{}".format(
                        coarse_label, fine_label)))
                x, y = _select(train_x, train_y,
                    fine_label_to_id[fine_label], fine_label_to_id[fine_label]+1)
                # for i in range(x[i].shape[0]):
                for i in range(5):
                    im = Image.fromarray(x[i])
                    im.save(os.path.join(path, "cifar100_examples/{}/{}/{}.jpg".format(
                        coarse_label, fine_label, i
                    )))
        
        print(class_order)
    else:
        class_order = [i for i in range(len(np.unique(train_y)))]
        np.random.shuffle(class_order)
        label_order = [label_list[i] for i in class_order]

        dataset_info = []
        for i in range(num_tasks):
            task_info = "Task {}: {}".format(
                i, ','.join(label_order[i * increments: (i + 1) * increments]))
            args.logger.info(task_info)
            dataset_info.append(task_info)

        dataset_info_file_path = os.path.join(args.result_path, "dataset_info.txt")
        with open(dataset_info_file_path, 'w') as f:
            for line in dataset_info:
                f.write(line + '\n')
        args.logger.info("Write dataset information in {}".format(dataset_info_file_path))
    
    train_y = _map_new_class_index(train_y, class_order)
    val_y = _map_new_class_index(val_y, class_order)
    test_y = _map_new_class_index(test_y, class_order)

    for i in range(num_tasks):
        train_x_i, train_y_i = _select(train_x, train_y, i*increments, (i+1)*increments)
        train_y_i -= i * increments
        val_x_i, val_y_i = _select(val_x, val_y, i*increments, (i+1)*increments)
        val_y_i -= i * increments
        test_x_i, test_y_i = _select(test_x, test_y, i*increments, (i+1)*increments)
        test_y_i -= i * increments
        if coarse_order:
            name = f'cifar100-{coarse_classes[str(i)]}'
        else:
            name = f'cifar100-{i}'
        data[i] = {
            'train': MyDataset({'x': train_x_i, 'y':train_y_i}, transforms=train_transforms),
            'val': MyDataset({'x': val_x_i, 'y':val_y_i}, transforms=test_transforms),
            'test': MyDataset({'x': test_x_i, 'y':test_y_i}, transforms=test_transforms),
            'ncla': increments,
            'name': name,
        }

    if args.experiment == "cifar100__carnivores__omnivores_and_herbivores":
        data = {
            0: data[10],
            1: data[13],
        }
    elif args.experiment == "cifar100__carnivores__flowers":
        data = {
            0: data[10],
            1: data[4],
        }

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']

    data['ncla'] = n

    return data, taskcla, size

if __name__ == '__main__':
    get('../dat/')