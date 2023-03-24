"""
File        :
Description :
Author      :XXX
Date        :2022/03/23
Version     :v1.1
"""
import torch.nn as nn
import torchvision.models as models


class Alexnet_FE(nn.Module):
    """
	Create a feature extractor model from an Alexnet architecture, that is used to train the autoencoder model
	and get the most related model whilst training a new task in a sequence
	"""
    def __init__(self, alexnet_model):
        super(Alexnet_FE, self).__init__()
        self.fe_model = nn.Sequential(*list(alexnet_model.children())[0][:-2])
        self.fe_model.eval()
        self.fe_model.requires_grad_(False)
    
    def forward(self, x):
        return self.fe_model(x)


class ResNet_FE(nn.Module):
    """
	Create a feature extractor model from an Alexnet architecture, that is used to train the autoencoder model
	and get the most related model whilst training a new task in a sequence
	"""
    def __init__(self, resnet_model):
        super(ResNet_FE, self).__init__()
        self.fe_model = nn.Sequential(*list(resnet_model.children())[:-1])
        self.fe_model.eval()
        self.fe_model.requires_grad_(False)
    
    def forward(self, x):
        return self.fe_model(x)


class CLIP_FE(nn.Module):
    def __init__(self):
        super(ResNet_FE, self).__init__()
        self.fe_model = nn.Sequential(*list(resnet_model.children())[:-1])
        self.fe_model.eval()
        self.fe_model.requires_grad_(False)
    
    def forward(self, x):
        return self.fe_model(x)


def get_pretrained_feat_extractor(name):
    """get the feature extractor pretrained on ImageNet
    
    """
    name = name.lower()
    if name == "alexnet":
        feat_extractor = Alexnet_FE(models.alexnet(pretrained=True))
        # self.logger.info("Using relatedness feature extractor: AlexNet")
    elif name == "resnet18":
        feat_extractor = ResNet_FE(models.resnet18(pretrained=True))
        # self.logger.info("Using relatedness feature extractor: ResNet18")
    elif name == "clip":
        pass
        # feat_extractor = ResNet_FE(models.resnet18(pretrained=True))
    else:
        raise Exception("Unknown relatedness feature extractor !!!")

    return feat_extractor
