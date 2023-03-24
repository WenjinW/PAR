import os, sys
from tkinter import image_names
from turtle import width
from idna import valid_label_length
from matplotlib import image
import numpy as np
import pickle
import torch
# import utils
from MLclf import MLclf
from torchvision import datasets, transforms
from sklearn.utils import shuffle
import shutil
from MLclf import MLclf

from dataloaders.my_dataset import MyDataset


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def get(path='../dat/', seed=0, args=None):
    # there are 20 tasks and each task contains 5 classes
    num_task = 20
    data = {}
    taskcla = []

    width = 84
    
    size = [3, width, width]
    data_path = os.path.join(path, "miniimagenet")
    # data_path = path + 'miniimagenet'

    train_path = os.path.join(data_path, "mini-imagenet-cache-train.pkl")
    val_path = os.path.join(data_path, "mini-imagenet-cache-val.pkl")
    test_path = os.path.join(data_path, "mini-imagenet-cache-test.pkl")

    images = []
    # labels = []
    for p in [train_path, val_path, test_path]:
        with open(p,'rb') as fr:
            raw_data = pickle.load(fr)
        
        for _, idx in raw_data["class_dict"].items():
            image = raw_data["image_data"][idx] # 600 x 84 x 84 x 3
            # label = [num_task for i in range(image.shape[0])]
            images.append(image)
            # labels.append(np.array(label, 'int64'))

    num_class = len(images)
    num_class_per_task = num_class // num_task
    args.logger.info("num_class: {}".format(num_class))
    args.logger.info("num_class_per_task: {}".format(num_class_per_task))

    class_order = [i for i in range(num_class)]
    np.random.shuffle(class_order)

    for i in range(num_task):
        train_data, val_data, test_data = [], [], []
        train_label, val_label, test_label = [], [], []
        for idx, j in enumerate(class_order[i * num_class_per_task: (i + 1) * num_class_per_task]):
            train_data.append(images[j][: 480])
            val_data.append(images[j][480: 540])
            test_data.append(images[j][540:])
            labels = np.ones(600, dtype='int64') * idx
            train_label.append(labels[: 480])
            val_label.append(labels[480: 540])
            test_label.append(labels[540:])

            # train_label.append(labels[j][: 480])
            # val_label.append(labels[j][480: 540])
            # test_label.append(labels[j][540:])
        data[i] = {
            'train': MyDataset(
                {'x': np.concatenate(train_data, axis=0),
                 'y': np.concatenate(train_label, axis=0)},
                transforms=transforms,
                ),
            "val": MyDataset(
                {'x': np.concatenate(val_data, axis=0),
                 'y': np.concatenate(val_label, axis=0)},
                transforms=transforms,
                ),
            "test": MyDataset(
                {'x': np.concatenate(test_data, axis=0),
                 'y': np.concatenate(test_label, axis=0)},
                transforms=transforms,
                ),
            'ncla': num_class_per_task,
            'name': 'miniimage-{}'.format(i),
        }

    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']

    data['ncla'] = n

    return data, taskcla, size


if __name__ == "__main__":
    # data, taskcla, size = get()
    # print(data[0])

    MLclf.miniimagenet_download(Download=True)

    train_dataset, validation_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(
        ratio_train=0.6, ratio_val=0.2, seed_value=None, shuffle=False, save_clf_data=True)
