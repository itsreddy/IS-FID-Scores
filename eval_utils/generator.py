import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision.datasets.vision import VisionDataset
from torch.autograd import Variable
import torch.autograd as autograd

class Generator(nn.Module):
    '''
        Generator class; replace if architecture changes
    '''
    def __init__(self, opt):
        super(Generator, self).__init__()

        def conv_block(in_filters, out_filters, activation, conv_type="convTranspose2d", bn=True, stride=2, output_padding=1):
            if conv_type == "convTranspose2d":
                block = [nn.ConvTranspose2d(in_filters, out_filters, 3, stride=stride, 
                                            padding=1, output_padding=output_padding)]
            elif conv_type == "conv2d":
                block = [nn.Conv2d(in_filters, out_filters, 3, stride=stride, padding=1)]
            if activation == "relu":
                block.append(nn.ReLU(inplace=True))
            elif activation == "leakyRelu":
                block.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation == "tanh":
                block.append(nn.Tanh())
            if bn == True:
                block.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            return block

        self.init_size = opt.img_size // 8
        
        self.dense = nn.Sequential(
            nn.Linear(opt.latent_dim, 512 * self.init_size ** 2),
            nn.ReLU(inplace=True)
            )

        self.conv = nn.Sequential(
            *conv_block(512, 256, "leakyRelu"),
            # *conv_block(128, 64, "relu", stride=1, output_padding=0),
            *conv_block(256, 128, "leakyRelu"),
            # *conv_block(128, 128, "leakyrelu", stride=1, output_padding=0),
            *conv_block(128, 128, "leakyRelu"),
            *conv_block(128, opt.channels, "tanh", conv_type="conv2d", stride=1, bn=False),
        )

    def forward(self, z):
        out = self.dense(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv(out)
        return img