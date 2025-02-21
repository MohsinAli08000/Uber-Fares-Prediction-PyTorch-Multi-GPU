
import os
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_rank, get_world_size
from rich.progress import track
from pathlib import Path
import argparse


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader,
        criterion,
        optimizer: torch.optim.Optimizer,
        patience: int,
        gpu_id: int,
#         save_every: int,
        save_path: str,
        max_epochs: int,
        world_size: int,
        scheduler = None,
    ) -> None:
        self.gpu_id = gpu_id
#         self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.save_path = save_path
#         self.best_val_loss = float('inf')
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        self.train_losses = np.array([{f'train_losses{i}': np.array([]) for i in range(world_size)}])
        self.val_losses = np.array([{f'val_losses{i}': np.array([]) for i in range(world_size)}])

    def _run_batch(self, source, targets):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(source)
#         print(f"Output shape: {output.shape}, Targets shape: {targets.shape}")
#         loss_fnc = nn.L1Loss().to(self.gpu_id)
#         loss = F.l1_loss(output, targets.unsqueeze(1))
        loss = self.criterion(output, targets.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0, norm_type=2.0)
        self.optimizer.step()
#         self.scheduler.step()
        return loss.item()
    
    def _run_eval(self, epoch):
        self.model.eval()
        total_loss = 0
        self.val_data.sampler.set_epoch(epoch)
        with torch.inference_mode():
            for source, targets in track(self.val_data, description=f"Evaluating...", style='red', complete_style='cyan', finished_style='green'):
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                output = self.model(source)
#                 print(f"Output shape: {output.shape}, Targets shape: {targets.shape}")
#                 loss_fnc = nn.L1Loss().to(self.gpu_id)
#                 loss = F.l1_loss(output, targets.unsqueeze(1))
                loss = self.criterion(output, targets.unsqueeze(1))
                total_loss += loss.item()
#         print(f"val data len: {len(self.val_data)}")
        self.model.train()
        return total_loss / len(self.val_data)

    def _run_epoch(self, epoch, total_epochs):
        b_sz = self.train_data.batch_size
        total_loss = 0
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in track(self.train_data, description=f"[GPU{self.gpu_id}] Epoch {epoch+1}/{total_epochs} | Training: {b_sz}...", style='red', complete_style='cyan', finished_style='green'):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
#             self.scheduler.step(val_loss)
            total_loss += loss
#         print(f"train data len: {len(self.train_data)}")
        return total_loss / len(self.train_data)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f"{self.save_path}/best_model.pt"
#         if self.gpu_id == 0:
        torch.save(ckp, PATH)
        print(f"\t\tNew best model saved at {PATH} from GPU{self.gpu_id}.")
            
    def loss_metric_tensors(self, array: np.ndarray):
        all_tensors = [torch.tensor([[array[0][j][k] for k in range(len(array[0][j]))]], dtype=torch.float32) for j in array[0].keys()]
        b = torch.cat(all_tensors, dim=0)
        return b.transpose(0, 1)
    
    def gather_tensor(self, t):
        gathered_t = [torch.zeros_like(t) for _ in range(get_world_size())]
        torch.distributed.all_gather(gathered_t, t)
        return torch.cat(gathered_t, dim=0)

    def train(self, max_epochs: int):
        b_sz = self.train_data.batch_size  # b_sz = len(next(iter(self.train_data))[0])
        should_stop = torch.zeros(1).to(self.gpu_id)
        patience_count = torch.zeros(1, dtype=torch.int32).to(self.gpu_id)
        for epoch in range(max_epochs):
            train_loss = self._run_epoch(epoch, max_epochs)
            val_loss = self._run_eval(epoch)
            print(f"\t[GPU{self.gpu_id}]  Batch: {b_sz} | Train Step: {len(self.train_data)} | Val Step: {len(self.val_data)} | Loss: {train_loss:.4f} | Val_Loss: {val_loss:.4f} | Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            self.scheduler.step(val_loss)
            
            # Gather losses from all GPUs
            world_size = get_world_size()
            train_losses = [torch.zeros(1).to(self.gpu_id) for _ in range(world_size)]
            val_losses = [torch.zeros(1).to(self.gpu_id) for _ in range(world_size)]
            torch.distributed.all_gather(train_losses, torch.tensor([train_loss]).to(self.gpu_id))
            torch.distributed.all_gather(val_losses, torch.tensor([val_loss]).to(self.gpu_id))
            
            # Save losses for all GPUs
            for i in range(world_size):
                self.train_losses[0][f"train_losses{i}"] = np.append(self.train_losses[0][f"train_losses{i}"], train_losses[i].item())
                self.val_losses[0][f"val_losses{i}"] = np.append(self.val_losses[0][f"val_losses{i}"], val_losses[i].item())

            val_losses_t = self.loss_metric_tensors(self.val_losses)
            
            vl_last_item = np.min(val_losses_t[-1:].squeeze().numpy())
            bval_loss = np.min(val_losses_t.numpy())
                
            # Find the best validation loss across all GPUs
#             best_val_loss = min(val_losses).item()
#             if best_val_loss < self.best_val_loss:
#                 self.best_val_loss = best_val_loss
# #                 if self.gpu_id == 0:  # Only save on the first GPU
#                 self._save_checkpoint(epoch)
            
            improved = torch.tensor([False], dtype=torch.bool).to(self.gpu_id)
            if (len(torch.where(val_losses_t==vl_last_item)[1]) == 1):
                vl_last_gpu = torch.where(val_losses_t==vl_last_item)[1].item()
                if (vl_last_item == bval_loss) and (self.gpu_id == vl_last_gpu):
                    print(f"\t\t1:[GPU{self.gpu_id}] val_loss improved to {vl_last_item:.4f}")
                    self._save_checkpoint(epoch)
                    improved = torch.tensor([True], dtype=torch.bool).to(self.gpu_id)
                    time.sleep(2)
            elif (len(torch.where(val_losses_t==vl_last_item)[1]) > 1):
                if (vm_last_item == bval_metric):
                    vl_last_gpu_min = min(torch.where(val_losses_t==vl_last_item)[1]).item()
                    if (self.gpu_id == vl_last_gpu_min):
                        print(f"\t\t2:[GPU{self.gpu_id}] val_loss: {vm_last_item:.4f}")
                        self._save_checkpoint(epoch)
                        improved = torch.tensor([True], dtype=torch.bool).to(self.gpu_id)
                        time.sleep(2)
            else:
                pass
        
            # Synchronize patience count across all GPUs
            improved_state = self.gather_tensor(improved)
        
            # Update patience count
            if (improved_state[0] and improved_state[1]) or (improved_state[0] or improved_state[1]):
                patience_count.zero_()
#                 print(f"\n[GPU{self.gpu_id}] count zero --> {patience_count}")
            else:
                patience_count += 1
#                 print(f"\n[GPU{self.gpu_id}] count increase --> {patience_count}")

            # Synchronize patience count across all GPUs
            all_patience_counts = self.gather_tensor(patience_count)
            max_patience_count = torch.max(all_patience_counts).item()
            patience_count.fill_(max_patience_count)

#             print(f"\n[GPU{self.gpu_id}] Patience Count --> {patience_count}")
            
            if max_patience_count >= self.patience:
                print(f"\n[GPU{self.gpu_id}] Patience exceeded. Early stopping...")
#                 break
                should_stop[0] = 1
    
            # Synchronize the should_stop tensor across all GPUs
            should_stop_list = [torch.zeros(1).to(self.gpu_id) for _ in range(world_size)]
            torch.distributed.all_gather(should_stop_list, should_stop)

            # If any GPU wants to stop, all GPUs should stop
            if any(stop.item() for stop in should_stop_list):
                break
        
            time.sleep(2)
        
        # Ensure all GPUs exit the training loop together
        torch.distributed.barrier()
        
        if self.gpu_id == 0:
#             print(f"Training completed.")
            
            self.train_losses[0]['epochs'], self.val_losses[0]['epochs'] = np.arange(1, len(self.train_losses[0]['train_losses0'])+1), np.arange(1, len(self.val_losses[0]['val_losses0'])+1)
            
            np.save("train_losses.npy", self.train_losses, allow_pickle=True)
            np.save("val_losses.npy", self.val_losses, allow_pickle=True)


# Custom learning rate scheduler
class CyclicLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.base_lrs = [base_lr] * len(optimizer.param_groups)
        self.max_lrs = [max_lr] * len(optimizer.param_groups)
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down if step_size_down is not None else step_size_up
        self.total_size = step_size_up + self.step_size_down
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.cycle_momentum = cycle_momentum
        self.base_momentums = [base_momentum] * len(optimizer.param_groups)
        self.max_momentums = [max_momentum] * len(optimizer.param_groups)
        super(CyclicLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle = np.floor(1 + self.last_epoch / self.total_size)
        x = 1 + self.last_epoch / self.total_size - cycle
        if x <= 0.5:
            scale_factor = x * 2
        else:
            scale_factor = (1 - x) * 2

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            lr = base_lr + base_height
            lrs.append(lr)

        return lrs

            
class CreateDataset_(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.x = X
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
                
                
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size: int):
        super(LinearRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
          
    
def load_data_objs(batch_size: int, rank: int, world_size: int, epochs: int):
    Xtrain = torch.load('X_train.pt')
    ytrain = torch.load('y_train.pt')
    Xval = torch.load('X_val.pt')
    yval = torch.load('y_val.pt')
    train_dts = CreateDataset_(Xtrain, ytrain)
    val_dts = CreateDataset_(Xval, yval)
    train_dtl = torch.utils.data.DataLoader(train_dts, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=DistributedSampler(train_dts, num_replicas=world_size, rank=rank))
    val_dtl = torch.utils.data.DataLoader(val_dts, batch_size=1, shuffle=False, pin_memory=True, sampler=DistributedSampler(val_dts, num_replicas=world_size, rank=rank))
    model = LinearRegressionModel(len(Xtrain[0]))
#     criterion = nn.L1Loss()
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
#     scheduler = CyclicLRScheduler(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=2000, mode='min')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)  # requires metric to step
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, epochs=epochs, steps_per_epoch=len(train_dtl), anneal_strategy='linear')

    return train_dtl, val_dtl, model, criterion, optimizer, scheduler

def model_eval(model: nn.Module):
    print(f"\n\n{'>'*10}LinearRegression Model Evaluation{'<'*10}\n")
    model.load_state_dict(torch.load("best_model/best_model.pt"))
    X_val = torch.load('X_val.pt')
    y_val = torch.load('y_val.pt')
    val_dts = CreateDataset_(X_val, y_val)
    val_dtl = torch.utils.data.DataLoader(val_dts, batch_size=1, shuffle=False, pin_memory=True,)
    total_loss = 0
    loss_func = nn.L1Loss()
    model.eval()
    with torch.inference_mode():
        for source, target in track(val_dtl, description=f"Evaluating...", style='red', complete_style='cyan', finished_style='green'):
            output = model(source)
            loss = loss_func(output, target.unsqueeze(1))
            total_loss += loss
        print(f"\nLoss --> {(total_loss / len(val_dtl)):.4f}")

def main(rank: int, world_size: int, total_epochs: int, patience: int, batch_size: int, save_path: str):
    if rank == 0:
        print(f"{'>'*10}LinearRegression Model Training{'<'*10}\n")
    ddp_setup(rank, world_size)
    train_dtl, val_dtl, model, criterion, optimizer, scheduler = load_data_objs(batch_size, rank, world_size, total_epochs)
    trainer = Trainer(model, train_dtl, val_dtl, criterion, optimizer, patience, rank, save_path, total_epochs, world_size, scheduler)
    trainer.train(total_epochs)
    destroy_process_group()
    if rank == 0:
        print(f"\n<{'='*10}Training completed & best model saved{'='*10}>\nExiting...")
        model_eval(model.to('cpu'))
        print(f"\n<{'='*10}Evaluation completed{'='*10}>")


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=10, type=int, help='Total epochs to train the model (default: 10)')
    parser.add_argument('--patience', default=5, type=int, help='Patience for increasing val_loss (default: 5)')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--save_path', default='./checkpoints', type=str, help='Path to save the best model (default: ./checkpoints)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    MODEL_PATH = Path(args.save_path)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    mp.spawn(main, args=(world_size, args.total_epochs, args.patience, args.batch_size, MODEL_PATH), nprocs=world_size)
    end_time = time.time()
    print(f'\nTime Elapsed {(end_time - start_time):.2f} sec')
