import sys
import torch
import torch.nn as nn

import utils


class Net(nn.Module):

    def __init__(self, inputsize, taskcla):
        super(Net, self).__init__()

        ncha, size, _ = inputsize
        self.taskcla = taskcla
        s = size
        self.conv1 = nn.Conv2d(ncha, 64, kernel_size=3, padding=1)
        s = s//2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        s = s//2
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        s = s//2
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256*s*s, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n))

        return

    def forward(self, x):
        h = self.maxpool(self.drop1(self.relu(self.conv1(x))))
        h = self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h = self.maxpool(self.drop2(self.relu(self.conv3(h))))
        h = h.view(x.size(0), -1)
        h = self.drop2(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))
        # y = []
        # for t, i in self.taskcla:
        #     y.append(self.last[t](h))
        # del h

        return self.last[self.current_task](h)
    
    def prepare(self, t):
        self.current_task = t
