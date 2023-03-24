import os, sys
from turtle import width
import numpy as np
import torch
# import utils
from torchvision import datasets, transforms
from sklearn.utils import shuffle
import shutil


def get(path='../dat/', seed=0):
    # there are 10 tasks and each task contains 20 classes
    data = {}
    taskcla = []
    width = 84
    # resize_width = 256
    if width == 224:
        resize_width = 256
    elif width == 112:
        resize_width = 128
    if width == 84:
        resize_width =96

    size = [3, width, width]
    data_path = path + 'tiny-imagenet-cl'
    # target_path = path + 'tiny-imagenet-cl'
    # if not os.path.isdir(target_path):
    #     os.mkdir(target_path)

    # create binary files
    if not os.path.isdir(data_path):
        print("No data files!")
    else:
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]

        ids = list(shuffle(np.arange(10), random_state=seed))

        for i, task_id in enumerate(ids):
            # print("task {}".format(i))            
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            for name in ['train', 'val', 'test']:
                data[i][name] = datasets.ImageFolder(
                    data_path+'/'+name+'/'+str(task_id)+'/',
                    transform=transforms.Compose([transforms.Resize(resize_width),
                                                  transforms.CenterCrop(width),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)]))
            data[i]['ncla'] = 20
            data[i]['name'] = 'tiny-imagenet-' + str(task_id)

        n = 0
        for t in data.keys():
            taskcla.append((t, data[t]['ncla']))
            n += data[t]['ncla']

        data['ncla'] = n

    return data, taskcla, size


if __name__ == "__main__":
    data, taskcla, size = get()
    print(data[0])
