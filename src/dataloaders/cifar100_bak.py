import logging
import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
from sklearn.utils import shuffle


def get(path, seed=0, pc_valid=0.10):
    # logger = args.logger if args is not None else logging
    data = {}
    taskcla = []
    size = [3, 32, 32]
    data_path = path
    name = "binary_cifar100"

    if not os.path.isdir(data_path+name):
        os.makedirs(data_path+name)

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        dat = {}
        dat['train'] = datasets.CIFAR100(data_path, train=True, download=True,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize(mean, std)]))
        dat['test'] = datasets.CIFAR100(data_path, train=False, download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize(mean, std)]))
        for n in range(10):
            data[n] = {}
            data[n]['name'] = 'cifar100'
            data[n]['ncla'] = 10
            data[n]['train'] = {'x': [],'y': []}
            data[n]['test'] = {'x': [],'y': []}
        for s in ['train','test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1,shuffle=False)
            for image, target in loader:
                n = target.numpy()[0]
                nn = n//10
                data[nn][s]['x'].append(image)
                data[nn][s]['y'].append(n % 10)

        # "Unify" and save
        for t in data.keys():
            for s in ['train','test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser(data_path+name), 'data'+str(t)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser(data_path+name), 'data'+str(t)+s+'y.bin'))

    # Load binary files
    logging.info("Loading")
    data = {}
    ids = list(shuffle(np.arange(10), random_state=seed))

    logging.info('Task order = {}'.format(ids))
    for i in range(10):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x':[],'y':[]}
            data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser(data_path+name),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser(data_path+name),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100-'+str(ids[i])

    # Validation
    for t in data.keys():
        r = np.arange(data[t]['train']['x'].size(0))
        r = np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid = int(pc_valid*len(r))
        ivalid = torch.LongTensor(r[:nvalid])
        itrain = torch.LongTensor(r[nvalid:])
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
        data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()

    # Others
    n = 0
    task_p = []
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']

    data['ncla'] = n

    return data, taskcla, size

if __name__ == '__main__':
    get('../dat/')