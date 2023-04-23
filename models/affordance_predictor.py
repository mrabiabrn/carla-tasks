
import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TaskBlock(nn.Module):

    def __init__(self, n_in, n_out, hidden_size=512, conditional=False):
        super(TaskBlock, self).__init__()

        self.conditional = conditional
        self.hidden_size = hidden_size
        self.n_out = n_out 

        self.module = nn.Sequential(
                                nn.Linear(n_in,hidden_size),
                                nn.Dropout(p=0.5),      # we can apply do before relu
                                nn.ReLU(),
                                nn.Linear(hidden_size,hidden_size),
                                nn.Dropout(p=0.5),
                                nn.ReLU(),
                                )
        
        self.model = nn.ModuleList([self.module for i in range(4)]) if conditional else self.module

        self.ln_layer = nn.Linear(hidden_size,n_out)

    def forward(self, img_enc, command=None):

        B = img_enc.shape[0]

        if self.conditional:
            branch_outputs = torch.zeros((B,4,self.hidden_size)).to(img_enc.device)
            for i, branch in enumerate(self.model):
                branch_outputs[:,i] = branch(img_enc)
            out = branch_outputs[list(range(B)),list(command),:]   # B,3

        else:
            out = self.model(img_enc)

        out = self.ln_layer(out)

        return out



class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self):

        super(AffordancePredictor, self).__init__()
        
        self.perception = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.lane_dist_blk = TaskBlock(n_in=1000, n_out=1, conditional=True)
        self.route_angle_blk = TaskBlock(n_in=1000, n_out=1, conditional=True)
        self.tl_dist_blk = TaskBlock(n_in=1000, n_out=1)
        self.tl_state_blk = TaskBlock(n_in=1000, n_out=2)

    def forward(self, img, command, device):
        
        p_i = self.perception(img.to(torch.float32))  

        lane_dist_pd = self.lane_dist_blk(p_i,command)
        route_angle_pd = self.route_angle_blk(p_i,command)
        tl_dist_pd = self.tl_dist_blk(p_i)
        tl_state_pd = self.tl_state_blk(p_i)

        return lane_dist_pd, route_angle_pd, tl_dist_pd, tl_state_pd

    def configure_optimizer(self):

        self.optimizer = Adam(self.parameters(), lr=2e-3) #, weight_decay=1e-4)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=200, cooldown=200)
