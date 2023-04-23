import os

import yaml

import torch

from carla_env.env import Env

from models.cilrs import CILRS
from torchvision.models import resnet18, ResNet18_Weights

from torchvision import transforms
import argparse

def action_helper(acceleration):

    throttle = torch.nn.functional.relu(acceleration) 
    brake = torch.nn.functional.relu(-acceleration) 

    return throttle, brake



class Evaluator():
    def __init__(self, env, config, ckpt):
        
        self.env = env
        self.config = config
        self.ckpt = ckpt
        self.agent = self.load_agent()

        self.transforms = transforms.Compose([
                                            transforms.Lambda(lambda x: x/255),      # ref: volkanaydingul
                                            transforms.Resize(224),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
        

    def load_agent(self):
        
        model = CILRS()
        model = torch.load(self.ckpt)
        model.to('cuda:1')
        model.eval()

        return model

    def generate_action(self, rgb, command, speed):
        
        rgb = self.transforms(torch.Tensor(rgb).permute(2,0,1)).to('cuda:1').unsqueeze(0)
        command = torch.Tensor([command]).int().to('cuda:1').unsqueeze(0)
        speed = torch.Tensor([speed]).to('cuda:1').unsqueeze(0)


        #print('#### Input speed ', type(speed), speed.shape)
        #print('command ', command.item())

        acceleration, steer, _ = self.agent(rgb,speed,command,'cuda:1')

        throttle, brake = action_helper(acceleration)

        #print('pred : throttle, brake, steer ', throttle.item(), brake.item(), steer.item())

        return float(throttle), float(brake), float(steer)


    def take_step(self, state):

        rgb = state["rgb"]
        command = state["command"]
        speed = state["speed"]
        throttle, brake, steer = self.generate_action(rgb, command, speed)
        action = {
            "throttle": throttle,
            "brake": brake,
            "steer": steer
        }
        state, reward_dict, is_terminal = self.env.step(action)
        return state, is_terminal

    def evaluate(self, num_trials=100):
        terminal_histogram = {}
        for i in range(num_trials):
            print('################# Trial ', i + 1, ' #################')
            state, _, is_terminal = self.env.reset()
            for j in range(5000):
                if j % 50 == 0:
                    print('################# Step ', j + 1, ' #################')
                #print('################# Step ', i + 1, ' #################')
                if is_terminal:
                    break
                state, is_terminal = self.take_step(state)
            if not is_terminal:
                is_terminal = ["timeout"]
            terminal_histogram[is_terminal[0]] = (terminal_histogram.get(is_terminal[0], 0)+1)

                
            for key, val in terminal_histogram.items():
                print(f"{key}: {val}/{i+1}")

        print("Evaluation over. Listing termination causes:")
        for key, val in terminal_histogram.items():
            print(f"{key}: {val}/100")


def main():

    parser = argparse.ArgumentParser()
 
    parser.add_argument("-c", "--ckpt", help = "Checkpoint path")
    args = parser.parse_args()

    with open(os.path.join("configs", "cilrs.yaml"), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        evaluator = Evaluator(env, config, args.ckpt)
        evaluator.evaluate()


if __name__ == "__main__":
    main()




