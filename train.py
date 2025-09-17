import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import tqdm
from grapher import Grapher
from inference import sample_next_img
from config import Config
import torch.distributed as dist
import torch.nn.parallel

if Config.is_colab:
    from google.colab import drive
    drive.mount('/content/drive')

if Config.is_tpu:
    import torch_xla.core.xla_model as xm

class Trainer:
    def __init__(self, model, simulator):
        # --- ADDED: Store rank for convenience ---
        self.rank = 0
        if Config.is_multi_gpu:
            self.rank = dist.get_rank()
            self.device = torch.device('cuda', self.rank)
        elif Config.is_tpu:
            self.device = xm.xla_device()
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        
        # --- MODIFIED: Only rank 0 gets the simulator ---
        self.simulator = simulator if self.rank == 0 else None
        
        self.loss_fn = nn.MSELoss()

        self.batch_count = 0
        self.batch_losses = []
        self.grapher = Grapher()

    def save_checkpoint(self, name):
        if self.rank == 0: # --- Simplified guard ---
            if Config.is_colab:
                check_dir = Path(Config.drive_dir) / 'checkpoints' / Config.model_name
            else:
                check_dir = Path.cwd() / 'checkpoints' / Config.model_name
            if not check_dir.exists():
                check_dir.mkdir(exist_ok=True, parents=True)
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'batch_count': self.batch_count,
                'batch_losses': self.batch_losses,
            }
            checkpoint_path = check_dir / name
            torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self):
        if Config.is_colab:
            checkpoint_path = Path(Config.drive_dir) / 'checkpoints' / Config.load_model / Config.loaded_checkpoint
        else:
            checkpoint_path = Path.cwd() / 'checkpoints' / Config.load_model / Config.loaded_checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device) # --- ADDED map_location ---
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.batch_count = checkpoint['batch_count']
        self.batch_losses = checkpoint['batch_losses']

    def update_graph(self):
        # --- ADDED: Guard to prevent multiple processes from plotting ---
        if self.rank == 0:
            indices = range(len(self.batch_losses))
            losses = self.batch_losses
            recent_losses = self.batch_losses[-1*Config.recent_losses_shown:]
            recent_losses_indices = indices[-1*Config.recent_losses_shown:]
            self.grapher.update_line(1, "Training", recent_losses_indices, recent_losses)

            grouped_losses = []
            grouped_indices = []
            sum = 0
            count = 0
            for i, loss in enumerate(losses):
                if count == Config.loss_bucket_size:
                    grouped_losses.append(sum/Config.loss_bucket_size)
                    grouped_indices.append(i)
                    sum = 0
                    count = 0
                sum += loss
                count += 1
            grouped_losses.append(sum/count)
            grouped_indices.append(indices[-1])
                
            self.grapher.update_line(0, "Training", grouped_indices, grouped_losses) #for performance

    async def train(self):
        self.model.train()
        
        num_samples = len(Config.initial_pages)
        
        # --- MODIFIED: Initialize empty tensors on all devices ---
        prev_img = torch.zeros((num_samples, 3, *Config.model_resolution), device=self.device, dtype=torch.float)
        
        if self.rank == 0:
            print("Rank 0: Getting initial images.", flush=True)
            prev_img = (await self.simulator.get_images()).to(self.device).float()

        # --- ADDED: Broadcast initial image from rank 0 to all other processes ---
        if Config.is_multi_gpu:
            dist.broadcast(prev_img, src=0)
            
        latent_state = torch.randn((num_samples, Config.latent_dimension), device=self.device)
        unrolled_loss = 0
        self.optimizer.zero_grad()
        
        # --- MODIFIED: pbar only on rank 0 ---
        pbar_range = range(Config.max_batches)
        pbar = tqdm.tqdm(pbar_range, desc=f'Batch {self.batch_count}/{Config.max_batches}') if self.rank == 0 else pbar_range

        for idx in pbar:
            # --- ADDED: Initialize empty tensors for data ---
            movement = torch.zeros((num_samples, 6), device=self.device, dtype=torch.float)
            next_img = torch.zeros_like(prev_img)

            # --- MODIFIED: Only rank 0 runs the simulator ---
            if self.rank == 0:
                movement = (await self.simulator.move()).to(self.device).float()
                next_img = (await self.simulator.get_images()).to(self.device).float()
            
            # --- ADDED: Broadcast data from rank 0 to all other processes ---
            if Config.is_multi_gpu:
                dist.broadcast(movement, src=0)
                dist.broadcast(next_img, src=0)
                
            v_target = next_img - prev_img

            total_loss = 0
            for i in range(Config.predictions_per_image):
                time = torch.rand((num_samples), device=self.device)
                pt = prev_img + v_target * time.view(num_samples, 1, 1, 1)
                if Config.is_multi_gpu:
                    v_pred = self.model.module.predict_delta(pt, time, movement, latent_state)
                else:
                    v_pred = self.model.predict_delta(pt, time, movement, latent_state)
                loss = self.loss_fn(v_pred, v_target)
                total_loss += loss / Config.predictions_per_image

            self.batch_losses.append(total_loss.item())
            self.batch_count += 1      
            unrolled_loss += total_loss / Config.latent_persistence_turns

            if idx % Config.sample_every_x_batches == 0:
                with torch.no_grad():
                    # Inference is already guarded for rank 0, so this is safe to call from all processes
                    sample_next_img(self.model, self.device, f"batch_{idx}", prev_img[0:1, :, :, :].detach(), movement[0:1, :].detach(), latent_state[0:1, :].detach(), next_img[0:1, :, :, :].detach())

            if (idx + 1) % Config.latent_persistence_turns == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                unrolled_loss.backward()
                self.optimizer.step()
                unrolled_loss = 0
                self.optimizer.zero_grad()
                latent_state = torch.randn((num_samples, Config.latent_dimension), device=self.device)               

            # --- MODIFIED: pbar update only on rank 0 ---
            if self.rank == 0:
                pbar.set_postfix({"Batch loss": total_loss.item()})
                
            if idx % Config.graph_update_freq == 0:
                self.update_graph()

            if Config.is_multi_gpu:
                latent_state = self.model.module.predict_dynamics(prev_img, latent_state)
            else:
                latent_state = self.model.predict_dynamics(prev_img, latent_state)
            prev_img = next_img.detach().clone()