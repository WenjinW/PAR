"""
File        :
Description :
Author      :XXX
Date        :2022/03/23
Version     :v1.1
"""
from turtle import forward
import torch
import torch.nn as nn


class Adaption(nn.Module):
    def __init__(self, channel) -> None:
        super(Adaption, self).__init__()

        self.conv = nn.Conv2d(channel, channel, kernel_size=1, bias=False)

    def forward(self, x):
        # x shape: NxCxHxW
        output = self.conv(x)

        return output


class ScalingFactor(nn.Module):
    def __init__(self, channel) -> None:
        super(ScalingFactor, self).__init__()

        self.scale = torch.nn.Parameter(
            data=torch.zeros((1, channel, 1, 1))
        )

    def forward(self, x):

        output = self.scale * x
        return output


class SpatialCalibration(nn.Module):
    def __init__(self, channel) -> None:
        super(SpatialCalibration, self).__init__()

        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=4,bias=False)

    def forward(self, x):
        # print("x shape: {}".format(x.shape))
        c = self.conv(x)
        # print("c shape: {}".format(c.shape))
        return x + c


class ChannelCalibration(nn.Module):
    def __init__(self, channel) -> None:
        super(ChannelCalibration, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv = nn.Conv2d(channel, channel, kernel_size=1, groups=channel,bias=False)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        # print("x shape: {}".format(x.shape))
        c = self.avgpool(x)
        # print("c shape after avgpool: {}".format(c.shape))
        c = self.conv(c)
        # print("c shape after group conv: {}".format(c.shape))
        c = self.bn(c)
        # print("c shape after bn: {}".format(c.shape))
        c = torch.sigmoid(c)
        # print("c shape after sigmoid: {}".format(c.shape))

        return x * c


LightOperations = {
    "Adaption": Adaption, # adaption
    "SF": ScalingFactor, # scaling factor for each map
    "SC": SpatialCalibration, # spatial calibration
    "CC": ChannelCalibration, # channel calibration
}


def LightOperationsSpace(channel):
    """generate the search space for ligth operations
    
    """
    op_list = [op(channel) for op in LightOperations.values()]

    return nn.ModuleList(op_list)