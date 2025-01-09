import torch
import torch.nn as nn
from torch.nn import Module

# useful modules for injecting fading noise for both critic/generator 
# and AdaIN normalization for generator (as seen in lecture)

class GNoise(Module):                        
    def __init__(self, decay_rate, std=0.1):
        super(GNoise, self).__init__()
        self.std = std
        self.decay = max(std - decay_rate, 0)

    def forward(self, x):
        
        return x + torch.empty_like(x).normal_(std=self.decay)


class AdaIN(Module):
    def __init__(self, channels: int, w_dim: int):
        super(AdaIN, self).__init__()
        self.channels = channels
        self.w_dim = w_dim
        self.instance_norm = nn.InstanceNorm2d(self.channels)
        self.fc_s = nn.Linear(self.w_dim, self.channels)
        self.fc_b = nn.Linear(self.w_dim, self.channels)
        
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:

        x = self.instance_norm(x)
        ys = self.fc_s(w)[..., None, None]
        yb = self.fc_b(w)[..., None, None]

        return x * ys + yb
    
#conv block module for better and neat code in the discriminator class, as seen in previous exercise
#using spectral norm for conv2d modules

class ConvBlock(Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int, stride: int, padding: int, batch_norm: bool = True):
        super(ConvBlock, self).__init__()

        self.convblock_bn = nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.convblock = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        
        self.bn = batch_norm
        self.b_norm = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bn:
            x = self.convblock(x)
            x = self.b_norm(x)
            x = self.act(x)
        else:
            x = self.convblock(x)
            
        return x

#Using spectral norm also for generator - it helped stabilize the training somehow and gave better results

class DeconvBlock(Module):
    def __init__(self, in_c: int, out_c: int, middle: bool = True):
        super(DeconvBlock, self).__init__()
        
        self.deconvblock_bn = nn.Sequential(
                           nn.utils.spectral_norm(nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)),
                           GNoise(0.08),
                           nn.LeakyReLU(0.2, inplace=True))
        
        self.deconvblock = nn.Sequential(
                           nn.utils.spectral_norm(nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)),
                           GNoise(0.09))
        
        self.middle = middle
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.middle:
            x = self.deconvblock_bn(x)
        else:
            x = self.deconvblock(x)
        
        return x