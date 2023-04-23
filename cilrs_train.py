import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from expert_dataset import ExpertDataset
from models.cilrs import CILRS

import wandb
import argparse

from tqdm import tqdm

global train_step, val_step
val_step = 0
train_step = 0

def validate(model, dataloader, loss_fn, device):
    """Validate model performance on the validation dataset"""

    global val_step
    model.eval()
    total_loss = 0
    step  = 0 
    with torch.no_grad():
        with tqdm(total=len(dataloader),desc ='Val Epoch') as pbar:
            for i, batch in enumerate(dataloader):

                batch = [x.to(device) for x in batch]
                rgb, speed_gt, command, action_gt = batch

                acceleration_pd, steer_pd, speed_pd = model(rgb,speed_gt,command,device)
                throttle_pd, brake_pd = action_helper(acceleration_pd)
                
                throttle_loss = loss_fn(throttle_pd,action_gt[:,0])
                brake_loss = loss_fn(brake_pd,action_gt[:,1]) 
                steer_loss = loss_fn(steer_pd,action_gt[:,2]) 
                speed_loss = loss_fn(speed_pd,speed_gt).detach().item()

                loss = throttle_loss + brake_loss + steer_loss + speed_loss

                wandb.log({ 'global_step': val_step,
                            'val_loss/tot': loss,
                            'val_loss/action': (throttle_loss + brake_loss + steer_loss),
                            'val_loss/speed': speed_loss}
                            )

                total_loss += loss
                
                pbar.update(i)
                val_step += 1
                step +=1
    
    return total_loss / step



def train(model, dataloader, loss_fn, device):
    """Train model on the training dataset for one epoch"""
    
    global train_step
    model.train()
    total_loss = 0
    step = 0
    with tqdm(total=len(dataloader),desc ='Train Epoch') as pbar:
        for i, batch in enumerate(dataloader):

            model.optimizer.zero_grad()
        
            batch = [x.to(device) for x in batch]
            rgb, speed_gt, command, action_gt = batch
            
           
            acceleration_pd, steer_pd, speed_pd = model(rgb,speed_gt,command,device)

            throttle_pd, brake_pd = action_helper(acceleration_pd)

            throttle_loss = loss_fn(throttle_pd,action_gt[:,0])
            brake_loss = loss_fn(brake_pd,action_gt[:,1]) 
            steer_loss = loss_fn(steer_pd,action_gt[:,2]) 

            speed_loss = loss_fn(speed_pd,speed_gt)
            action_loss = throttle_loss + brake_loss + steer_loss
            
            loss = 1*action_loss + 1*speed_loss  

            loss.backward()
            model.optimizer.step()

            total_loss += loss.detach().item()

            wandb.log({
                        'global_step': train_step,
                        'train_loss/tot': loss.detach().item(),
                        'train_loss/throttle': throttle_loss.detach().item(),
                        'train_loss/brake': brake_loss.detach().item(),
                        'train_loss/steer': steer_loss.detach().item(),
                        'train_loss/action': action_loss.detach().item(),
                        'train_loss/speed': speed_loss.detach().item()
                        })

            pbar.update(1)
            train_step += 1
            step += 1

    print('train loss ', total_loss / step)

    return total_loss / step



def action_helper(acceleration):

    throttle = torch.nn.functional.relu(acceleration) 
    brake = torch.nn.functional.relu(-acceleration) 

    return throttle, brake


def main():

    parser = argparse.ArgumentParser()
 
    parser.add_argument("-c", "--ckpt", default= None, help = "Checkpoint path")
    args = parser.parse_args()

    run_name = 'cilrs-overfit-fixed'
    wandb.init(project='cvad', 
                name=run_name)
                
    train_root = '/home/mbarin/cvad/train/'
    val_root = '/home/mbarin/cvad/val/' 

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = CILRS()

    if args.ckpt:
        print('Loading model from ', args.ckpt)
        model = torch.load(args.ckpt)

    model.to(device)
    
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    print(f'{len(train_dataset)} number of instances in trainset')
    print(f'{len(val_dataset)} number of instances in valset')

    num_epochs = 20
    batch_size = 64
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loss_fn = nn.L1Loss()
    model.configure_optimizer()
    
    for i in range(1,num_epochs):
        # train
        train_loss = train(model, train_loader, loss_fn, device)

        if i % 1 == 0:
            torch.save(model, run_name+'_epoch'+str(i)+'.ckpt')

        # val
        val_loss = validate(model, val_loader, loss_fn, device)

        if hasattr(model, 'scheduler'):
            model.scheduler.step(val_loss)

        wandb.log({'epoch': i, 
                    'train_loss_avg': train_loss, 
                    'val_loss_avg': val_loss, 
                    '_lr': model.optimizer.param_groups[-1]['lr']
                    })
    


if __name__ == "__main__":
    main()
