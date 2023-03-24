import logging
import os,sys
import statistics
from types import new_class
import numpy as np
import pickle
from sklearn import utils
import torch
from torch.utils.data.dataset import TensorDataset
from PIL import Image
from torchvision import datasets,transforms
from sklearn.utils import shuffle

import utils

import ctrl


class CtrlDataset(torch.utils.data.Dataset):
    def __init__(self, datas, debug='False', transforms=None):
        super(CtrlDataset).__init__()
        if eval(debug):
            print("***************Debug*****************")
            self.datas = datas[:100]
        else:
            self.datas = datas

        self.length = len(self.datas)
        self.transforms = transforms

    def __getitem__(self, idx):
        x, y = self.datas[idx]
        if self.transforms is not None:
            
            x = self.transforms(x)

        return x, y[0]
    
    def __len__(self):

        return self.length


def train_transforms(mean_std):
    mean = [float(i.item()) if isinstance(i, torch.Tensor) else float(i) for i in mean_std["mean"]]
    std = [float(i.item()) if isinstance(i, torch.Tensor) else float(i) for i in mean_std["std"]]
    trans = torch.nn.Sequential(
            transforms.RandomCrop(32, padding=[4,]),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean, std)
            )
    scripted_trans = torch.jit.script(trans)

    return scripted_trans


def test_transforms(mean_std):

    mean = [float(i.item()) if isinstance(i, torch.Tensor) else float(i) for i in mean_std["mean"]]
    std = [float(i.item()) if isinstance(i, torch.Tensor) else float(i) for i in mean_std["std"]]
    
    trans = torch.nn.Sequential(
        transforms.Normalize(mean, std)
        )
    scripted_trans = torch.jit.script(trans)

    return scripted_trans

stream_names = {
    "ctrl_s_out": "s_out",
    "ctrl_s_in": "s_in",
    "ctrl_s_plus": "s_plus",
    "ctrl_s_minus": "s_minus",
    "ctrl_s_pl": "s_pl",
    "ctrl_s_long": "s_long",
}


def get(path, seed=0, pc_valid=0.10, args=None):
    """

    """
    data = {}
    taskcla = []
    size = [3, 32, 32]

    # stream_name = args.experiment.split(':')[-1]
    stream_name = stream_names[args.experiment]
    
    args.logger.info("Loading ctrl stream: {}".format(stream_name))
    task_gen = ctrl.get_stream(stream_name, seed)

    dataset_info = []

    for task_id, task_data in enumerate(task_gen):
        if task_id >= 100:
            break
        
        # args.logger.info("type: {}".format(type(task_data.src_concepts[0])))
        # args.logger.info("type: {}".format(type(task_data.src_concepts[0][0])))
        if stream_name == "s_long":
            train_num = task_data.datasets[0].tensors[0].shape[0]
            train2val = {5000:2500, 25:15, 2250:250}
            new_val_num = train2val[train_num]
            new_val_x = []
            new_val_y = []

            per_class_num = new_val_num // 5
            val_tensors = task_data.datasets[1].tensors
            val_pred = val_tensors[1]

            for l in torch.unique(val_pred):
                index_l = val_pred == l
                index_l = torch.squeeze(index_l, -1)
                new_val_x.append(val_tensors[0][index_l][:per_class_num])
                new_val_y.append(val_tensors[1][index_l][:per_class_num])
            
            new_val_x = torch.concat(new_val_x, dim=0)
            new_val_y = torch.concat(new_val_y, dim=0)

            print("new_val_x shape: {}".format(new_val_x.shape))
            print("new_val_y shape: {}".format(new_val_y.shape))
            
            task_data.datasets[1] = TensorDataset(new_val_x, new_val_y)
            # print(type(task_data.datasets[1]))
            # print(task_data.datasets[1][1])

            # print(task_data.datasets[1])

        # break

        n_samples = [len(task_data.datasets[0]), len(task_data.datasets[1]), len(task_data.datasets[2])]

        task_info = "Task id: {},\t Class: {}, n_samples: {}".format(
            task_id,
            " ".join([str(c) for c in task_data.src_concepts]),
            n_samples
            )

        args.logger.info("Loading {}".format(task_info))

        dataset_info.append(task_info)


        mean_std = task_data.compute_statistics()
        data[task_id] = {
            'train': CtrlDataset(task_data.datasets[0], transforms=train_transforms(mean_std)),
            'val': CtrlDataset(task_data.datasets[1], transforms=test_transforms(mean_std)),
            'test': CtrlDataset(task_data.datasets[2], transforms=test_transforms(mean_std)),
            'ncla': task_data.n_classes.item(),
            'name': str(task_id),
        }
    
    dataset_info_file_path = os.path.join(args.result_path, "dataset_info.txt")
    with open(dataset_info_file_path, 'w') as f:
        for line in dataset_info:
            f.write(line + '\n')
    args.logger.info("Write dataset information in {}".format(dataset_info_file_path))
    
    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']

    data['ncla'] = n

    return data, taskcla, size

if __name__ == '__main__':
    get('../dat/')