
import torch

from resnet1d import ResNet


class Network(torch.nn.Module):
    def __init__(self, input_channels, kernel=5, width=1/4, normalization='bn', dropout=0.0, n_groups=16):
        super().__init__()

        self.resnet = ResNet(in_channels=input_channels, kernel=kernel, width=width, norm_type=normalization, 
                             dropout_rate=dropout, num_groups=n_groups)
        resnet_width = int(width * 2048)
        self.base_out = torch.nn.Linear(in_features=resnet_width, out_features=1, bias=True)
        
        # init weights
        torch.nn.init.xavier_uniform_(self.base_out.weight, gain=0.1)
        
    def forward(self, x):
            
        x = self.resnet(x)
        
        #  L2 normalizacia embeddingu
        x = torch.nn.functional.normalize(x,p=2.0,dim=1)
        
        # output
        x = self.base_out(x)

        return x


