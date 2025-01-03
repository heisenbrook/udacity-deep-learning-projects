import torch
import torch.nn as nn



# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
        

        self.conv1 = nn.Sequential(     
            #first convolution
            nn.Conv2d(3, 16, 3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU())
        
        self.SEBlock1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Sigmoid())
        
        self.max1 = nn.MaxPool2d(2, 2) #112*112*16
             
            
        self.conv2 = nn.Sequential(
            #second convolution
            nn.Conv2d(16, 32, 3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU())
        
        self.SEBlock2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Sigmoid())
        
        self.max2 = nn.MaxPool2d(2, 2) #56*56*32
       
           
        self.conv3 = nn.Sequential(
            #third convolution
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.SEBlock3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid())
        
        self.max3 = nn.MaxPool2d(2, 2) #28*28*64
        
        
        self.conv4 = nn.Sequential(
            #fourth convolution
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        self.SEBlock4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Sigmoid())
        
        self.max4 = nn.MaxPool2d(2, 2) #14*14*128
            
        
        self.conv5 = nn.Sequential(
            #fifth convolution
            nn.Conv2d(128, 256, 3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU())
        
        self.SEBlock5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Sigmoid())
        
        self.max5 = nn.MaxPool2d(2, 2) #7*7*256
            
            
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*256, 512),
            nn.Dropout(dropout),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes))
            
            
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.SEBlock1(x).unsqueeze(-1).unsqueeze(-1) * x
        x = self.max1(x)
        
        x = self.conv2(x)
        x = self.SEBlock2(x).unsqueeze(-1).unsqueeze(-1) * x
        x = self.max2(x)
        
        x = self.conv3(x)
        x = self.SEBlock3(x).unsqueeze(-1).unsqueeze(-1) * x
        x = self.max3(x)
        
        x = self.conv4(x)
        x = self.SEBlock4(x).unsqueeze(-1).unsqueeze(-1) * x
        x = self.max4(x)
        
        x = self.conv5(x)
        x = self.SEBlock5(x).unsqueeze(-1).unsqueeze(-1) * x
        x = self.max5(x)
                
        x = self.head(x)
        
        return x


