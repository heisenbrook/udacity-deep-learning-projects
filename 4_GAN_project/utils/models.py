import torch
import torch.nn as nn
from torch.nn import Module
from utils.model_utils import GNoise, AdaIN, DeconvBlock, ConvBlock

class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        channels = [3, 64, 128, 256, 512, 1024]
        decay = [0, 0.02, 0.04, 0.06, 0.08]
        self.layers = []

        for i in range(len(channels)-1):
            self.layers.append(GNoise(decay[i]))
            self.layers.append(ConvBlock(channels[i], channels[i+1], 5, 2, 2))
        self.layers.append(nn.Sequential(
                           nn.Flatten(1),
                           nn.Linear(2*2*channels[-1], 1)))

        self.layers = nn.ModuleList(self.layers)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x)
            
        x = x.unsqueeze(-1).unsqueeze(-1)

        return x 
    
    
    
class Generator(Module):
    def __init__(self, latent_dim: int, conv_dim: int = 32):
        super(Generator, self).__init__()

        channels = [1024, 512, 256, 128, 64, 3]

        self.layers = [DeconvBlock(latent_dim, conv_dim*32)]
        self.adains = [AdaIN(conv_dim*32, conv_dim*32)]

        #inspired from styleGAN architecture

        self.fc= nn.Sequential(
                      nn.Linear(latent_dim, conv_dim * 32, bias=False),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Dropout(0.5),
                      nn.Linear(conv_dim * 32, conv_dim * 32, bias=False),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Dropout(0.5),
                      nn.Linear(conv_dim * 32, conv_dim * 32, bias=False),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Dropout(0.5),
                      nn.Linear(conv_dim * 32, conv_dim * 32, bias=False),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Dropout(0.5))

        for i in range(len(channels)-1):
            if channels[i+1] == 3:
                self.layers.append(DeconvBlock(channels[i], channels[i+1], False))
            else:
                self.layers.append(DeconvBlock(channels[i], channels[i+1]))
            self.adains.append(AdaIN(channels[i+1], channels[0]))

        self.layers = nn.ModuleList(self.layers)
        self.adains = nn.ModuleList(self.adains)
        
        self.activation = nn.Tanh()
        
        
    def forward(self, x):

        a = x.squeeze(-1).squeeze(-1)
        a = self.fc(a)

        for layer, adain in zip(self.layers, self.adains):
            x = layer(x)
            x = adain(x, a)

        x = self.activation(x)
        
        return x    