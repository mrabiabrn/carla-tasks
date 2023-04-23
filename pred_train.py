
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor

from torchvision import transforms

import wandb
import argparse

from tqdm import tqdm

global train_step, val_step
val_step = 0
train_step = 0

def validate(model, dataloader, cont_loss_fn, bin_loss_fn, device):
    """Validate model performance on the validation dataset"""
    global val_step
    model.eval()
    total_loss = 0
    step  = 0 
    with torch.no_grad():
        with tqdm(total=len(dataloader),desc ='Val Epoch') as pbar:
            for i, batch in enumerate(dataloader):

                batch = [x.to(device) for x in batch]
                rgb, lane_dist, route_angle, tl_dist, tl_state, command = batch
                
                lane_dist_pd, route_angle_pd, tl_dist_pd, tl_state_pd = model(rgb,command,device)

                lane_dist_loss = cont_loss_fn(lane_dist_pd,lane_dist)
                route_angle_loss = cont_loss_fn(route_angle_pd,route_angle) 
                
                tl_dist_loss = cont_loss_fn(tl_dist_pd,tl_dist) 

                tl_state = torch.nn.functional.one_hot(tl_state, num_classes=2).squeeze(1).to(torch.float32)
                tl_state_loss = bin_loss_fn(tl_state_pd,tl_state)
            
                loss = lane_dist_loss + route_angle_loss + 50*tl_dist_loss + tl_state_loss

                wandb.log({ 'global_step': val_step,
                            'val_loss/tot': loss.detach().item(),
                            'val_loss/lane_dist': lane_dist_loss.detach().item(),
                            'val_loss/route_angle': route_angle_loss.detach().item(),
                            'val_loss/tl_dist': tl_dist_loss.detach().item(),
                            'val_loss/tl_state': tl_state_loss.detach().item(),
                        }
                            )

                total_loss += loss
                
                pbar.update(i)
                val_step += 1
                step +=1
    
    return total_loss / step


def train(model, dataloader, cont_loss_fn, bin_loss_fn, device):
    """Train model on the training dataset for one epoch"""
    
    global train_step
    model.train()
    total_loss = 0
    step = 0
    with tqdm(total=len(dataloader),desc ='Train Epoch') as pbar:
        for i, batch in enumerate(dataloader):

            model.optimizer.zero_grad()
        
            batch = [x.to(device) for x in batch]
            rgb, lane_dist, route_angle, tl_dist, tl_state, command = batch
              
            lane_dist_pd, route_angle_pd, tl_dist_pd, tl_state_pd = model(rgb,command,device)

            lane_dist_loss = cont_loss_fn(lane_dist_pd,lane_dist)
            route_angle_loss = cont_loss_fn(route_angle_pd,route_angle) 
            
            tl_dist_loss = cont_loss_fn(tl_dist_pd,tl_dist) 

            tl_state = torch.nn.functional.one_hot(tl_state, num_classes=2).squeeze(1).to(torch.float32)
            tl_state_loss = bin_loss_fn(tl_state_pd,tl_state)

            loss = lane_dist_loss + route_angle_loss + 50*tl_dist_loss + tl_state_loss

            # NOTE: define different optimizers for different affordances
            loss.backward()
            model.optimizer.step()

            total_loss += loss.detach().item()

            wandb.log({
                        'global_step': train_step,
                        'train_loss/tot': loss.detach().item(),
                        'train_loss/lane_dist': lane_dist_loss.detach().item(),
                        'train_loss/route_angle': route_angle_loss.detach().item(),
                        'train_loss/tl_dist': tl_dist_loss.detach().item(),
                        'train_loss/tl_state': tl_state_loss.detach().item(),
                        })

            pbar.update(1)
            train_step += 1
            step += 1

    print('train loss ', total_loss / step)

    return total_loss / step


def main():

    parser = argparse.ArgumentParser()
 
    parser.add_argument("-c", "--ckpt", default= None, help = "Checkpoint path")
    args = parser.parse_args()

    run_name = 'affordances'
    wandb.init(project='cvad', 
                name=run_name)
                
    train_root = '/home/mbarin/cvad/train/'
    val_root = '/home/mbarin/cvad/val/'  # TODO: val

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = AffordancePredictor()
    model.to(device)

    train_dataset = ExpertDataset(train_root,'affordances')
    val_dataset = ExpertDataset(val_root,'affordances')

    num_epochs = 30
    batch_size = 64
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    cont_loss_fn = nn.MSELoss()
    bin_loss_fn = nn.BCEWithLogitsLoss()

    model.configure_optimizer()

    for i in range(1,num_epochs+1):
        # train
        train_loss = train(model, train_loader, cont_loss_fn, bin_loss_fn, device)

        if i % 10== 0:
            torch.save(model, run_name+'_epoch'+str(i)+'.ckpt')

        # val
        val_loss = validate(model, val_loader, cont_loss_fn, bin_loss_fn, device)

        if hasattr(model, 'scheduler'):
            model.scheduler.step(val_loss)

        wandb.log({'epoch': i, 
                    'train_loss_avg': train_loss, 
                    'val_loss_avg': val_loss, 
                    '_lr': model.optimizer.param_groups[-1]['lr']
                    })

if __name__ == "__main__":
    main()
