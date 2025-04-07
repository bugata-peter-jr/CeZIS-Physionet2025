
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

class NetworkExt(torch.nn.Module):
    def __init__(self, input_channels, kernel=5, width=1/4, normalization='bn', dropout=0.0, n_groups=16, 
                 n_cats_sex=3, n_cats_age=20, final_dim=64):
        super().__init__()

        self.resnet = ResNet(in_channels=input_channels, kernel=kernel, width=width, norm_type=normalization, 
                             dropout_rate=dropout, num_groups=n_groups)
        resnet_width = int(width * 2048)
        
        self.emb_age = torch.nn.Embedding(num_embeddings=n_cats_age, embedding_dim=32)
        self.emb_gender = torch.nn.Embedding(num_embeddings=n_cats_sex, embedding_dim=16)
        
        self.relu = torch.nn.ReLU()
        
        self.pre_out = torch.nn.Linear(in_features=resnet_width+32+16, out_features=final_dim, bias=True)
        
        self.base_out = torch.nn.Linear(in_features=final_dim, out_features=1, bias=True)
        
        # init weights
        torch.nn.init.xavier_uniform_(self.pre_out.weight, gain=0.1)
        torch.nn.init.xavier_uniform_(self.base_out.weight, gain=0.1)
    
    def forward(self, x, x_age, x_gender):
            
        x = self.resnet(x)
        
        #  L2 normalizacia embeddingu
        x = torch.nn.functional.normalize(x,p=2.0,dim=1)
        
        # additional factors
        x_age = self.emb_age(x_age)
        x_age = torch.nn.functional.normalize(x_age,p=2.0,dim=1)

        x_gender = self.emb_gender(x_gender)
        x_gender = torch.nn.functional.normalize(x_gender,p=2.0,dim=1)
        
        # concat
        #print(x.shape, x_age.shape, x_gender.shape)
        x = torch.cat([x, x_age, x_gender], dim=1)

        # pre-out
        x = self.pre_out(x)

        # relu
        x = self.relu(x)
        
        # output
        x = self.base_out(x)

        return x

