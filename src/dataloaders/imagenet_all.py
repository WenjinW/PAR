import os, sys
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
    resize_width = 128
    if resize_width == 256:
        width = 224
    elif resize_width == 128:
        width = 112
    size = [3, width, width]
    data_path = path + 'imagenet_cl'

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
                    data_path+'/'+name+'_'+str(task_id)+'/',
                    transform=transforms.Compose([transforms.Resize(resize_width),
                                                  transforms.CenterCrop(width),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)]))
            data[i]['ncla'] = 100
            data[i]['name'] = 'imagenet-' + str(task_id)

        n = 0
        for t in data.keys():
            taskcla.append((t, data[t]['ncla']))
            n += data[t]['ncla']

        data['ncla'] = n

    return data, taskcla, size


if __name__ == "__main__":
    data, taskcla, size = get()
    print(data[0])
