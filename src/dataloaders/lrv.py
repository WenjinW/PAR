"""
File        :
Description :
Author      :XXX
Date        :2019/10/10
Version     :v1.0
"""
import json
import numpy as np
import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.utils import shuffle


def get(path='../dat/', seed=0):

    size = [3, 224, 224]
    data_path = path

    if not os.path.isdir(data_path+'binary_lrv/'):
        os.makedirs(data_path+'binary_lrv')
        print("Generate binary files")
        means = [[0.3823, 0.3809, 0.3556],
                 [0.4288, 0.4297, 0.4064],
                 [0.3854, 0.3863, 0.3633],
                 [0.2977, 0.3025, 0.2778],
                 [0.3427, 0.3445, 0.3193],
                 [0.4284, 0.4303, 0.4088],
                 [0.3567, 0.3594, 0.3357],
                 [0.4251, 0.4270, 0.4051],
                 [0.3787, 0.3797, 0.3553],
                 ]

        for k in range(1, 10):  # k is the batch id
            lrv_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(means[k - 1], [1, 1, 1]),
            ])
            dat = {"train": ImageFolder(data_path + "LRV/train/batch"+str(k), transform=lrv_transform),
                   "valid": ImageFolder(data_path + "LRV/validation/batch"+str(k), transform=lrv_transform),
                   "test": ImageFolder(data_path + "LRV/test/batch"+str(k), transform=lrv_transform)}

            print(dat['train'].class_to_idx)
            if k == 1:
                file_name = data_path + 'binary_lrv/class_to_idx.json'
                with open(file_name, 'w') as f:
                    json.dump(dat['train'].class_to_idx, f)

            for string in ['train', 'valid', 'test']:
                data = {'x': [], 'y': [], 'ncla': 51, 'name': 'lrv'}
                for sample in dat[string]:
                    data['x'].append(sample[0])
                    data['y'].append(sample[1])
                data['x'] = torch.stack(data['x']).view(-1, size[0], size[1], size[2])
                data['y'] = torch.LongTensor(np.array(data['y'], dtype=int)).view(-1)
                print("Task {}, {} data".format(k, string))
                torch.save(data['x'], os.path.join(os.path.expanduser(
                    data_path + 'binary_lrv'), 'data' + str(k) + string + 'x.bin'))
                torch.save(data['y'], os.path.join(os.path.expanduser(
                    data_path + 'binary_lrv'), 'data' + str(k) + string + 'y.bin'))

    # load binary files
    print("Load binary files")
    data = {'ncla': 51}
    taskcla = []
    ids = list(shuffle(np.arange(9)))
    for i in range(9):
        data[i] = {'ncla': 51, 'name': 'lrv'}
        for string in ["train", "valid", "test"]:
            data[i][string] = {
                'x': torch.load(os.path.join(os.path.expanduser(
                    data_path + 'binary_lrv'), 'data' + str(ids[i] + 1) + string + 'x.bin')),
                'y': torch.load(os.path.join(os.path.expanduser(
                    data_path + 'binary_lrv'), 'data' + str(ids[i] + 1) + string + 'y.bin')),
            }
    for t in data.keys():
        if t != 'ncla':
            taskcla.append((t, data[t]['ncla']))
    print("Finish!")

    return data, taskcla, size


if __name__ == '__main__':
    get()
