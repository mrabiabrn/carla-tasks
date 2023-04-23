
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


'''
    input --> rgb, speed, high command
    output --> speed, actions 
'''


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        
        super(CILRS, self).__init__()
        
        self.perception = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.input_transforms = ResNet18_Weights.DEFAULT.transforms()

        self.measurements =  nn.Sequential(
                                        nn.Linear(1,128),
                                        nn.ReLU(),
                                        nn.Linear(128,128),
                                        )
        
        self.speed_predictor = nn.Sequential(
                                        nn.Linear(1000,256),
                                        nn.ReLU(),
                                        nn.Linear(256,1),
                                        )

        self.branch_module = nn.Sequential(
                                nn.Linear(1000+128,512),
                                nn.ReLU(),
                                nn.Linear(512,256),
                                nn.Dropout(p=0.5),
                                nn.ReLU(),
                                nn.Linear(256,2)
                                )

        self.command_branches = nn.ModuleList([self.branch_module for i in range(4)])


    def forward(self, img, speed, command, device):

        B = img.shape[0]
        
        p_i = self.perception(img.to(torch.float32))                      # B,1000

        s_enc = self.measurements(speed.to(torch.float32))  # B,128

        s_pred = self.speed_predictor(p_i)              # B,1

        branch_outputs = torch.zeros((B,4,2)).to(device)
        for i, branch in enumerate(self.command_branches):
            branch_outputs[:,i] = branch(torch.cat([p_i,s_enc], dim=1))

        elite_branches = branch_outputs[list(range(B)),list(command),:]   # B,3

        acceleration = torch.tanh(elite_branches[:,0]) 
        steer  = torch.tanh(elite_branches[:,1])

        return (acceleration, steer, s_pred)

    def configure_optimizer(self):

        self.optimizer = Adam(self.parameters(), lr=2e-4) #, weight_decay=1e-4)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=200, cooldown=200)


