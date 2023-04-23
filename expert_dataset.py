from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import torch 
import json
import os

class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""
    def __init__(self, data_root, task='imitation'):

        self.data_root = data_root

        self.measurements = [os.path.join(data_root,'measurements',f) for f in sorted(os.listdir(os.path.join(data_root,'measurements')))]
        self.rgb = [os.path.join(data_root,'rgb',f) for f in sorted(os.listdir(os.path.join(data_root,'rgb')))]  

        self.transforms = transforms.Compose([
                                            transforms.PILToTensor(),
                                            transforms.Lambda(lambda x: x/255),      # ref: volkanaydingul
                                            transforms.Resize(224),
                                            #transforms.CenterCrop(224),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
        
        self.task = task

        self.commands = {'0':0,'1':0,'2':0,'3':0}


    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, index):
        """Return RGB images and measurements"""

        rgb_path = self.rgb[index]
        msr_path = self.measurements[index]

        bgr = Image.open(rgb_path)
        bgr = self.transforms(bgr)  # 0 --> B, 1 --> G, 2 --> R
        rgb = bgr[[2,1,0],:]
  

        with open(msr_path) as f:
            measurements = json.load(f)

   
        if self.task == 'imitation':
            action = torch.tensor([measurements['throttle'], measurements['brake'], measurements['steer']])
            speed = torch.tensor([measurements['speed']]) 
            command = torch.tensor([measurements['command']]).int()

            self.commands[str(measurements['command'])] += 1

            out = (rgb, action, speed, command)


        if self.task == 'affordances':
            lane_dist = torch.tensor([measurements['lane_dist']])
            route_angle = torch.tensor([measurements['route_angle']])
            tl_dist = torch.tensor([measurements['tl_dist']])
            tl_state = torch.tensor([measurements['tl_state']])
            command = torch.tensor([measurements['command']]).int()

            out = (rgb, lane_dist, route_angle, tl_dist, tl_state, command)


        return out
        

train = ExpertDataset( '/home/mbarin/cvad/train/')
val = ExpertDataset( '/home/mbarin/cvad/val/')

from torch.utils.data import DataLoader

train_loader = DataLoader(train, batch_size=1, shuffle=True,
                              drop_last=True)
val_loader = DataLoader(val, batch_size=1, shuffle=False)

for i, batch in enumerate(train_loader):
    pass

for i, batch in enumerate(val_loader):
    pass


print('val ', val.commands)

print('train ', train.commands)
