"""
File        :
Description :
Author      :XXX
Date        :2019/9/10
Version     :v1.0
"""
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import utils


class Net(torch.nn.Module):
    def __init__(self, inputsize, taskcla):
        super(Net, self).__init__()

        ncha, size, _ = inputsize
        self.inputsize = inputsize
        self.taskcla = taskcla
        s = size
        # 1define the layers
        # 1.1 Conv 1
        self.conv1 = torch.nn.ModuleList([nn.ModuleList([nn.Conv2d(ncha, 64, kernel_size=3, padding=1)])])
        self.ksize_1 = 3
        # self.conv1 = torch.nn.Conv2d(ncha, 64, kernel_size=size//8)
        # 1.2 Conv 2
        # s = utils.compute_conv_output_size(size, size//8)
        s = s//2
        self.conv2 = torch.nn.ModuleList([nn.ModuleList([nn.Conv2d(64, 128, kernel_size=3, padding=1)])])
        self.ksize_2 = 3
        # self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=size//10)
        # 1.3 Conv 3
        # s = utils.compute_conv_output_size(s, size//10)
        s = s//2
        self.conv3 = torch.nn.ModuleList([nn.ModuleList([nn.Conv2d(128, 256, kernel_size=3, padding=1)])])
        self.ksize_3 = 3
        # self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=2)
        # 1.4 Maxpool, relu, and relu
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = nn.ModuleList([nn.ModuleList([nn.Dropout(0.2)])])
        self.drop2 = nn.ModuleList([nn.ModuleList([nn.Dropout(0.2)])])
        self.drop3 = nn.ModuleList([nn.ModuleList([nn.Dropout(0.5)])])
        self.drop4 = nn.ModuleList([nn.Dropout(0.5)])
        self.drop5 = nn.ModuleList([nn.Dropout(0.5)])
        # 1.5 linear 1

        s = s//2
        self.fc1 = torch.nn.ModuleList([nn.Linear(256 * s * s, 2048)])
        # self.fc1 = torch.nn.Linear(256 * s * s, 2048)
        self.fc1_inputsize = 256 * s * s
        # 1.6 linear 2
        self.fc2 = torch.nn.ModuleList([nn.Linear(2048, 2048)])
        # self.fc2 = torch.nn.Linear(2048, 2048)

        # 1.7 task specific layer
        self.last = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(torch.nn.Linear(2048, n))

        # 2 initial architecture
        self.init_archi = {'conv1': [[0, 0]], 'conv2': [[0, 0]], 'conv3': [[0, 0]],
                           'fc1': [0], 'fc2': [0], 'last': [0]}
        self.length = {'conv1': 1, 'conv2': 1, 'conv3': 1, 'fc1': 1, 'fc2': 1}
        self.a = torch.nn.ParameterDict({})
        # modules to be trained in search stage for task t
        self.new_models = None
        # modules to be trained in training stage for task t
        self.model_to_train = None

        return

    def forward(self, x, t, task_arch):
        # 1 conv 1
        idx1, idx2 = task_arch['conv1'][0][0], task_arch['conv1'][0][1]
        h = self.maxpool(self.drop1[idx1][0](self.relu(self.conv1[idx1][0](x))))
        for j in range(1, idx2):
            h += self.maxpool(self.drop1[idx1][j](self.relu(self.conv1[idx1][j](x))))
        x = h

        # 2 conv 2
        idx1, idx2 = task_arch['conv2'][0][0], task_arch['conv2'][0][1]
        h = self.maxpool(self.drop2[idx1][0](self.relu(self.conv2[idx1][0](x))))
        for j in range(1, idx2):
            h += self.maxpool(self.drop2[idx1][j](self.relu(self.conv2[idx1][j](x))))
        x = h

        # conv 3self.conv1
        idx1, idx2 = task_arch['conv3'][0][0], task_arch['conv3'][0][1]
        h = self.maxpool(self.drop3[idx1][0](self.relu(self.conv3[idx1][0](x))))
        for j in range(1, idx2):
            h += self.maxpool(self.drop3[idx1][j](self.relu(self.conv3[idx1][j](x))))
        x = h

        # fc 1
        x = x.view(x.size(0), -1)
        idx1 = task_arch['fc1'][0]
        h = self.drop4[idx1](self.relu(self.fc1[idx1](x)))
        x = h

        # fc 2
        idx1 = task_arch['fc2'][0]
        h = self.drop5[idx1](self.relu(self.fc2[idx1](x)))
        x = h

        # last layer
        y = []
        for t, i in self.taskcla:
            y.append(self.last[t](x))
        return y

    def expand(self, t, device='cuda'):
        # expand the network to a super model
        # 1 expand the conv1 and drop1
        # 1.1 action: new
        self.conv1.append(nn.ModuleList([nn.Conv2d(self.inputsize[0], 64, kernel_size=self.ksize_1, padding=1)]).to(device))
        self.drop1.append(nn.ModuleList([nn.Dropout(0.2)]).to(device))
        # 1.2 action: adaption
        for i in range(self.length['conv1']):
            self.conv1[i].append(nn.Conv2d(self.inputsize[0], 64, kernel_size=1).to(device))
            self.drop1[i].append(nn.Dropout(0.2).to(device))
        # 1.3 generate action parameter
        self.a['conv1'] = nn.Parameter(torch.rand(self.length['conv1'] * 2 + 1).to(device))

        # 2 expand the conv2 and drop2
        # 2.1 action: new
        self.conv2.append(nn.ModuleList([nn.Conv2d(64, 128, kernel_size=self.ksize_2, padding=1)]).to(device))
        self.drop2.append(nn.ModuleList([nn.Dropout(0.2)]).to(device))
        # 2.2 action: adaption
        for i in range(self.length['conv2']):
            self.conv2[i].append(nn.Conv2d(64, 128, kernel_size=1).to(device))
            self.drop2[i].append(nn.Dropout(0.2).to(device))
        # 2.3 generate action parameter
        self.a['conv2'] = nn.Parameter(torch.rand(self.length['conv2'] * 2 + 1).to(device))

        # 3 expand the conv3 and drop3
        # 3.1 action: new
        self.conv3.append(nn.ModuleList([nn.Conv2d(128, 256, kernel_size=self.ksize_3, padding=1)]).to(device))
        self.drop3.append(nn.ModuleList([nn.Dropout(0.5)]).to(device))
        # 3.2 action: adaption
        for i in range(self.length['conv3']):
            self.conv3[i].append(nn.Conv2d(128, 256, kernel_size=1).to(device))
            self.drop3[i].append(nn.Dropout(0.5).to(device))
        # 3.3 generate action parameter
        self.a['conv3'] = nn.Parameter(torch.rand(self.length['conv3'] * 2 + 1).to(device))

        # 4 expand the fc1 and drop4
        # 4.1 action: new
        self.fc1.append(nn.Linear(self.fc1_inputsize, 2048).to(device))
        self.drop4.append(nn.Dropout(0.5).to(device))
        # 4.2 generate action parameter
        self.a['fc1'] = nn.Parameter(torch.rand(self.length['fc1'] + 1).to(device))

        # 5 expand the fc2 and drop5
        # 5.1 action: new
        self.fc2.append(nn.Linear(2048, 2048).to(device))
        self.drop5.append(nn.Dropout(0.5).to(device))
        # 4.2 generate action parameter
        self.a['fc2'] = nn.Parameter(torch.rand(self.length['fc2'] + 1).to(device))

        # get the expanded modules which are need to be trained in search stage of task t
        self.get_model_trained_search(t)

    def get_model_trained_search(self, t):
        """ find modules to be trained in search stage for task t.

        :param t: task t
        :return:
        """
        new_models = {'conv1': [], 'conv2': [], 'conv3': [], 'fc1': [], 'fc2': [], 'last': []}

        # 1 conv 1,2,3
        for (name, conv_layer) in zip(['conv1', 'conv2', 'conv3'], [self.conv1, self.conv2, self.conv3]):
            c = self.length[name]
            # 1.1 new
            new_models[name].append([c, 0])
            # 1.2 adaption
            for j in range(c):
                new_models[name].append([j, len(conv_layer[j])-1])

        # 2 fc 1,2
        for (name, fc_layer) in zip(['fc1', 'fc2'], [self.fc1, self.fc2]):
            c = self.length[name]
            # 2.1 new
            new_models[name].append(c)

        # 3 last
        new_models['last'].append(t)

        # 4 update the model to be trained
        self.new_models = new_models

    def get_archi_param(self):
        """ get the parameters of architecture search

        :return: params
        """
        params = [self.a]

        return params

    def get_param(self, model):
        """ get the parameters of a sub-model

        :param model: the sub-model need to be trained
        :return:
        """
        params = []
        # 1 parameters of conv 1,2,3
        for (name, conv_layer) in zip(['conv1', 'conv2', 'conv3'], [self.conv1, self.conv2, self.conv3]):
            for idx in model[name]:
                params.append({'params': conv_layer[idx[0]][idx[1]].parameters()})
        # 2 parameters of fc 1,2
        for (name, fc_layer) in zip(['fc1', 'fc2'], [self.fc1, self.fc2]):
            for idx in model[name]:
                params.append({'params': fc_layer[idx].parameters()})

        # 3 last layer
        for idx in model['last']:
            params.append({'params': self.last[idx].parameters()})

        return params

    def search_forward(self, x, t):
        # 1 conv 1,2,3
        for (name, conv_layer, drop) in zip(['conv1', 'conv2', 'conv3'], [self.conv1, self.conv2, self.conv3],
                                            [self.drop1, self.drop2, self.drop3]):
            g_conv = torch.exp(self.a[name]) / torch.sum(torch.exp(self.a[name]))

            # 1.1 new
            out_ = g_conv[-1] * self.maxpool(drop[-1][0](self.relu(conv_layer[-1][0](x))))
            # 1.2 reuse and adaption
            c = self.length[name]
            for i in range(c):
                for j in range(len(conv_layer[i]) - 1):
                    # reuse for submodel i and adaption for the submodel i
                    out_ += g_conv[i] * self.maxpool(drop[i][j](self.relu(conv_layer[i][j](x))))
                    out_ += g_conv[c + i] * self.maxpool(drop[i][j](self.relu(conv_layer[i][j](x))))
                # adaption for the submodel j
                out_ += g_conv[c + i] * self.maxpool(drop[i][-1](self.relu(conv_layer[i][-1](x))))


            x = out_

        # 2 fc 1,2
        x = x.view(x.size(0), -1)
        for (name, fc_layer, drop) in zip(['fc1', 'fc2'], [self.fc1, self.fc2],
                                          [self.drop4, self.drop5]):
            g_conv = torch.exp(self.a[name]) / torch.sum(torch.exp(self.a[name]))
            # 2.1 new
            out_ = g_conv[-1] * drop[-1](self.relu(fc_layer[-1](x)))
            # 2.2 reuse and adaption
            c = self.length[name]
            for i in range(c):
                out_ += g_conv[i] * drop[i](self.relu(fc_layer[i](x)))
            x = out_

        # 3 last layer
        y = []
        for t, i in self.taskcla:
            y.append(self.last[t](x))
        return y

    def select(self, t):
        """ select the best model for task t from super model

        :param t: task
        :return:
        """
        # 1 define the container of new models to train and the best sub-model
        model_to_train = {}
        best_archi = {}
        # 2 select the best architecture for conv1, 2, 3
        for (name, conv_layer, drop) in zip(['conv1', 'conv2', 'conv3'], [self.conv1, self.conv2, self.conv3],
                                            [self.drop1, self.drop2, self.drop3]):
            v, arg_v = torch.max(self.a[name].data, dim=0)
            idx = deepcopy(arg_v.item())
            c = self.length[name]

            # select
            best_archi[name], model_to_train[name] = [], []
            if idx < c:  # reuse
                best_archi[name].append([idx, len(conv_layer[idx]) - 2])
            elif idx < 2 * c:  # adaption
                model_to_train[name].append([idx-c, len(conv_layer[idx-c])-1])
                best_archi[name].append([idx-c, len(conv_layer[idx-c])-1])
            elif idx == 2 * c:  # new
                model_to_train[name].append([c, 0])
                best_archi[name].append([c, 0])

            # delete
            for i in range(2 * c + 1):
                if i != idx:  # do not select the action
                    if c <= i < 2*c:  # adaption
                        del conv_layer[i-c][-1]
                        del drop[i-c][-1]
                    elif i == 2 * c:  # new
                        del conv_layer[-1]
                        del drop[-1]
            # update length
            self.length[name] = len(conv_layer)

        # 3 select the best architecture for fc1, fc2
        for (name, fc_layer, drop) in zip(['fc1', 'fc2'], [self.fc1, self.fc2], [self.drop4, self.drop5]):
            v, arg_v = torch.max(self.a[name].data, dim=0)
            idx = deepcopy(arg_v.item())
            c = self.length[name]

            # select
            best_archi[name], model_to_train[name] = [], []
            if idx < c:  # reuse
                best_archi[name].append(idx)
            elif idx == c:  # new
                model_to_train[name].append(c)
                best_archi[name].append(c)

            # delete
            for i in range(c+1):
                if i != idx:  # do not select the action
                    if i == c:  # new
                        del fc_layer[-1]
                        del drop[-1]
            # update length
            self.length[name] = len(fc_layer)

        # 4 the last layer
        model_to_train['last'] = [t]
        best_archi['last'] = [t]

        self.model_to_train = model_to_train

        return best_archi

    def modify_param(self, models, requires_grad=True):
        """freeze or unfreeze the new model. (adaption and new)

        :param models: a dict of submodel
        :param requires_grad:
        :return:
        """
        # 1 parameters of conv 1,2,3
        for (name, conv_layer) in zip(['conv1', 'conv2', 'conv3'], [self.conv1, self.conv2, self.conv3]):
            for idx in models[name]:
                utils.modify_model(conv_layer[idx[0]][idx[1]], requires_grad)
        # 2 parameters of fc 1,2
        for (name, fc_layer) in zip(['fc1', 'fc2'], [self.fc1, self.fc2]):
            for idx in models[name]:
                utils.modify_model(fc_layer[idx], requires_grad)

        # 3 last layer
        for idx in models['last']:
            utils.modify_model(self.last[idx], requires_grad)

    def modify_archi_param(self, requires_grad=True):
        params = self.get_archi_param()
        for param in params:
            if requires_grad:
                utils.unfreeze_parameter(param)
            else:
                utils.freeze_parameter(param)

    def regular_loss(self):
        loss = 0.0
        # 1 conv 1,2,3
        for (name, conv_layer) in zip(['conv1', 'conv2', 'conv3'], [self.conv1, self.conv2, self.conv3]):
            c = self.length[name]
            g_conv = torch.exp(self.a[name]) / torch.sum(torch.exp(self.a[name]))
            for i in range(c):
                loss += g_conv[c + i] * utils.model_size(conv_layer[i][-1])
            loss += g_conv[2 * c] * utils.model_size(conv_layer[c][-1])
        # 2 fc 1,2
        for (name, fc_layer) in zip(['fc1', 'fc2'], [self.fc1, self.fc2]):
            c = self.length[name]
            g_conv = torch.exp(self.a[name]) / torch.sum(torch.exp(self.a[name]))
            loss += g_conv[c] * utils.model_size(fc_layer[c])

        return loss
