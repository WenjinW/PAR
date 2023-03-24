import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torchvision import datasets,transforms
import json
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from PIL import Image

from dataloaders.my_dataset import MyDataset


def get(path, seed=0, pc_valid=0.10, args=None):
    size=[3,32,32]
    # n_tasks = 10
    # n_tasks = 5
    if args.experiment == 'mix_cifar100_celeba':
        dis_ntasks = 10
        sim_ntasks = 10
    elif args.experiment == 'mix_cifar100-20_celeba':
        dis_ntasks = 20
        sim_ntasks = 10

    data={}
    taskcla=[]

    data_cifar100, taskcla_cifar100, size_cifar100 = read_cifar100(path=path, seed=seed, n_tasks=dis_ntasks, args=args)
    data_celeba, taskcla_celeba, size_celeba = read_celeba(path=path, seed=seed, n_tasks=sim_ntasks, args=args)

    all_cifar100 = [data_cifar100[x]['name'] for x in range(dis_ntasks)]
    all_celeba = [data_celeba[x]['name'] for x in range(sim_ntasks)]

    f_name = path + 'celeba/mixceleba_random_'+str(dis_ntasks + sim_ntasks)

    with open(f_name, 'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.task_seed].split()

    print(random_sep)
    for task_id in range(dis_ntasks+sim_ntasks):
        if 'cifar100' in random_sep[task_id]:# Even
            cifar100_id = all_cifar100.index(random_sep[task_id])
            data[task_id] = data_cifar100[cifar100_id]
            taskcla.append((task_id,data_cifar100[cifar100_id]['ncla']))

        elif 'celeba'in random_sep[task_id]:
            celeba_id = all_celeba.index(random_sep[task_id])
            data[task_id] = data_celeba[celeba_id]
            taskcla.append((task_id,data_celeba[celeba_id]['ncla']))


    print(taskcla)
    return data,taskcla,size

def read_cifar100(path, seed=0, pc_valid=0.10, n_tasks=10, args=0):
    data={}
    taskcla=[]
    size=[3,32,32]
    counter = {}
    # n_tasks = args.dis_ntasks
    # class_per_task = args.classptask
    n_class = 100
    n_tasks = n_tasks
    class_per_task = n_class / n_tasks

    binary_data_path = path + 'binary_cifar100/{}'.format(n_tasks)
    if not os.path.isdir(binary_data_path):
        os.makedirs(binary_data_path)

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]

        # CIFAR100
        dat={}
        dat['train']=datasets.CIFAR100(path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100(path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        n_per_task = len([1 for image,target in torch.utils.data.DataLoader(dat['train'],batch_size=1,shuffle=True) if target.numpy()[0] == 1])
        print('n_per_task: ',n_per_task)

        for n in range(n_tasks):
            data[n]={}
            data[n]['name']='cifar100'
            data[n]['ncla']=class_per_task
            data[n]['train']={'x': [],'y': []}
            data[n]['test']={'x': [],'y': []}
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:

                n=target.numpy()[0]

                if n in counter:
                    counter[n] += 1
                elif n not in counter:
                    counter[n] = 1

                nn=(n//class_per_task)
                data[nn][s]['x'].append(image)
                data[nn][s]['y'].append(n%class_per_task)

        # "Unify" and save
        for t in data.keys():
            for s in ['train','test']:
                data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
                data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser(binary_data_path),'data'+str(t)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser(binary_data_path),'data'+str(t)+s+'y.bin'))

    # Load binary files
    data={}
    ids=list(shuffle(np.arange(n_tasks),random_state=seed))
    print('Task order =',ids)
    for i in range(n_tasks):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser(binary_data_path),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser(binary_data_path),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name']='cifar100-'+str(ids[i])

    # Validation
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['val']={}
        data[t]['val']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['val']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

        data[t]['train'] = MyDataset({'x': data[t]['train']['x'], 'y': data[t]['train']['y']})
        data[t]['val'] = MyDataset({'x': data[t]['val']['x'], 'y': data[t]['val']['y']})
        data[t]['test'] = MyDataset({'x': data[t]['test']['x'], 'y': data[t]['test']['y']})

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data, taskcla, size

def read_celeba(path, seed=0, pc_valid=0.10, n_tasks=10, args=None):
    data={}
    taskcla=[]
    size=[3, 218, 178]
    size=[3, 32, 32]


    # num_task = args.sim_ntasks
    # n_tasks = args.sim_ntasks
    num_task = n_tasks
    n_tasks = n_tasks

    # if 'small' in args.data_size:
    #     data_type = 'small'
    # elif 'full' in args.data_size:
    #     data_type = 'full'
    data_type = 'small'

    if not os.path.isdir(path+data_type+'_binary_celeba/'+str(n_tasks)+'/'):
        os.makedirs(path+data_type+'_binary_celeba/'+str(n_tasks))

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]

        # celeba
        dat={}
        train_dataset = CELEBATrain(root_dir=path+'celeba/'+data_type+'/iid/train/',img_dir=path+'celeba/data/raw/img_align_celeba/',transform=transforms.Compose([transforms.Resize(size=(32,32)),transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['train'] = train_dataset

        test_dataset = CELEBATest(root_dir=path+'celeba/'+data_type+'/iid/test/',img_dir=path+'celeba/data/raw/img_align_celeba/',transform=transforms.Compose([transforms.Resize(size=(32,32)),transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test'] = test_dataset

        users = [x[0] for x in set([user for user,image,target in torch.utils.data.DataLoader(dat['train'],batch_size=1,shuffle=True)])]
        users.sort()
        users = users[:num_task]
        print('users: ',users)
        print('users length: ',len(users))

        # # totally 10 tasks, each tasks 2 classes (whether smiling)
        #
        for task_id,user in enumerate(users):
            data[task_id]={}
            data[task_id]['name'] = 'celeba-'+str(user)
            data[task_id]['ncla'] = 2


        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=True)

            for task_id,user in enumerate(users):
                data[task_id][s]={'x': [],'y': []}

            for user,image,target in loader:
                if user[0] not in users: continue # we dont want too may classes
                label=target.numpy()[0]
                data[users.index(user[0])][s]['x'].append(image)
                data[users.index(user[0])][s]['y'].append(label)


        # # "Unify" and save
        for n,user in enumerate(users):
            for s in ['train','test']:
                data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
                data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)
                torch.save(data[n][s]['x'], os.path.join(os.path.expanduser(path+data_type+'_binary_celeba/'+str(n_tasks)),'data'+str(n)+s+'x.bin'))
                torch.save(data[n][s]['y'], os.path.join(os.path.expanduser(path+data_type+'_binary_celeba/'+str(n_tasks)),'data'+str(n)+s+'y.bin'))


    # number of example
    # need to further slice [:user_num]
    # number of example


    # Load binary files
    data={}
    ids=list(shuffle(np.arange(num_task),random_state=seed))
    print('Task order =',ids)
    for i in range(num_task):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser(path+data_type+'_binary_celeba/'+str(n_tasks)),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser(path+data_type+'_binary_celeba/'+str(n_tasks)),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name']='celeba-'+str(ids[i])


    # Validation
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        print('len r: ',len(r))
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        print('ivalid: ',len(ivalid))

        itrain=torch.LongTensor(r[nvalid:])
        data[t]['val']={}
        data[t]['val']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['val']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

        data[t]['train'] = MyDataset({'x': data[t]['train']['x'], 'y': data[t]['train']['y']})
        data[t]['val'] = MyDataset({'x': data[t]['val']['x'], 'y': data[t]['val']['y']})
        data[t]['test'] = MyDataset({'x': data[t]['test']['x'], 'y': data[t]['test']['y']})
        
    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size


########################################################################################################################



# customize dataset class

class CELEBATrain(Dataset):
    """Federated EMNIST dataset."""

    def __init__(self, root_dir,img_dir, transform=None):
        self.transform = transform
        self.size=[218, 178, 3]

        self.x = []
        self.y = []
        self.user = []
        for file in os.listdir(root_dir):
            with open(root_dir+file) as json_file:
                data = json.load(json_file) # read file and do whatever we need to do.
                for key, value in data['user_data'].items():
                    for type, data in value.items():
                        if type == 'x':
                            for img in data:
                                img_name = img_dir + img
                                im = Image.open(img_name)
                                np_im = np.array(im)
                                self.x.append(torch.from_numpy(np_im))
                        elif type == 'y':
                            self.y.append(data)

                    for _ in range(len(data)):
                        self.user.append(key)

        self.x=torch.cat(self.x,0).view(-1,self.size[0],self.size[1],self.size[2])
        self.y=torch.LongTensor(np.array([d for f in self.y for d in f],dtype=int)).view(-1).numpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        user = self.user[idx]
        x = self.x[idx]
        y = self.y[idx]

        x = x.data.numpy()
        x = Image.fromarray(x)
        # x = Image.fromarray((x * 255).astype(np.uint8))

        if self.transform:
            x = self.transform(x)
        return user,x,y


class CELEBATest(Dataset):
    """Federated EMNIST dataset."""

    def __init__(self, root_dir,img_dir, transform=None):
        self.transform = transform
        self.size=[218, 178, 3]

        self.x = []
        self.y = []
        self.user = []
        for file in os.listdir(root_dir):
            with open(root_dir+file) as json_file:
                data = json.load(json_file) # read file and do whatever we need to do.
                for key, value in data['user_data'].items():
                    for type, data in value.items():
                        if type == 'x':
                            for img in data:
                                img_name = img_dir + img
                                im = Image.open(img_name)
                                np_im = np.array(im)
                                self.x.append(torch.from_numpy(np_im))
                        elif type == 'y':
                            self.y.append(data)

                    for _ in range(len(data)):
                        self.user.append(key)

        self.x=torch.cat(self.x,0).view(-1,self.size[0],self.size[1],self.size[2])
        self.y=torch.LongTensor(np.array([d for f in self.y for d in f],dtype=int)).view(-1).numpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        user = self.user[idx]
        x = self.x[idx]
        y = self.y[idx]

        x = x.data.numpy()
        x = Image.fromarray(x)
        # x = Image.fromarray((x * 255).astype(np.uint8))

        if self.transform:
            x = self.transform(x)
        return user,x,y