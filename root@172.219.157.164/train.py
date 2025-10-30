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
import math
from torchvision import transforms

class Trainer:
    def __init__(self, model, simulator, val_simulator):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        self.simulator = simulator
        self.val_simulator = val_simulator
        self.loss_fn = nn.HuberLoss(delta=Config.huber_delta, reduction='sum')

        self.batch_count = 0
        self.batch_losses = []
        self.val_losses = []
        self.val_indices = []
        self.grapher = Grapher()

    def save_checkpoint(self, name):
        check_dir = Path.cwd() / 'checkpoints' / Config.model_name
        if not check_dir.exists():
            check_dir.mkdir(exist_ok=True, parents=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_count': self.batch_count,
            'batch_losses': self.batch_losses,
            'val_losses': self.val_losses,
            'val_indices': self.val_indices
        }
        checkpoint_path = check_dir / name
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self):
        checkpoint_path = Path.cwd() / 'checkpoints' / Config.loaded_model / Config.loaded_checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_lr = self.optimizer.param_groups[0]['lr']
        if checkpoint_lr != Config.learning_rate:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = Config.learning_rate
            print(f"updated optimizer learning rate from {checkpoint_lr} to {Config.learning_rate}")

        self.batch_count = checkpoint['batch_count']
        self.batch_losses = checkpoint['batch_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_indices = checkpoint['val_indices']

    def update_graph(self):
        losses = self.batch_losses
        val_losses = self.val_losses

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
            
        self.grapher.update_line(0, "Training", grouped_indices, grouped_losses) #for performance
        recent_losses = grouped_losses[-1*Config.recent_losses_shown:]
        recent_losses_indices = grouped_indices[-1*Config.recent_losses_shown:]
        self.grapher.update_line(1, "Training", recent_losses_indices, recent_losses)

        if len(val_losses) > 0:
            val_indices = self.val_indices
            self.grapher.update_line(0, "Validation", val_indices, val_losses)
            recent_val_losses = val_losses[-Config.recent_losses_shown:]
            recent_val_indices = val_indices[-Config.recent_losses_shown:]
            self.grapher.update_line(1, "Validation", recent_val_indices, recent_val_losses)

    async def train_batch(self, latent, batch_size, target):
        starting_noise = torch.randn_like(target)
        time = torch.rand((batch_size), device=self.device)
        v_target = (target - starting_noise)
        pt = starting_noise + v_target * time.view(batch_size, 1, 1, 1)
        step_sizes = torch.zeros(Config.batch_size, device=self.device)
        v_pred = self.model.predict_delta(pt, time, latent, step_sizes)
        loss = self.loss_fn(v_pred, v_target)
        return loss

    async def train_batch_shortcut(self, latent, batch_size, target):
        values = torch.tensor([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2])
        indices = torch.randint(0, len(values), (Config.batch_size,), device=self.device)
        step_sizes = values[indices]
        step_sizes_grounded = step_sizes.clone()
        step_sizes_grounded = torch.where(step_sizes == 1/128, 0.0, step_sizes_grounded)
        starting_noise = torch.randn_like(target)
        time = torch.rand((batch_size), device=self.device)
        v_target = target - starting_noise
        pt = starting_noise + v_target * time.view(batch_size, 1, 1, 1)

        first_step = self.model.predict_delta(pt, time, latent, step_sizes_grounded)
        x_after_first_step = pt + first_step * step_sizes.view(Config.batch_size, 1, 1, 1)
        second_step = self.model.predict_delta(x_after_first_step, time + step_sizes, latent, step_sizes_grounded)
        
        v_target = (first_step + second_step).detach() / 2
        v_pred = self.model.predict_delta(pt, time, latent, step_sizes * 2)
        loss = self.loss_fn(v_pred, v_target)
        return loss

    async def validate(self):
        self.model.eval()
        with torch.no_grad():
            latent_state = torch.zeros((1, Config.hidden_size), device=self.device)
            prev_img = (await self.val_simulator.get_images()).to(self.device).float()
            prev_img = self.model.encode_image(prev_img)
            total_loss = 0
            for _ in range(Config.validation_samples):
                await self.val_simulator.increment()
                movement = (await self.val_simulator.get_movement()).to(self.device).float() 
                next_img = (await self.val_simulator.get_images()).to(self.device).float()
                next_img = self.model.encode_image(next_img)
                latent_state = self.model.predict_dynamics(prev_img, movement, latent_state)
                if torch.rand(1).item() < 0.25:
                    loss = await self.train_batch_shortcut(latent_state, 1, next_img)
                else:
                    loss = await self.train_batch(latent_state, 1, next_img)
                total_loss += loss.item()
                prev_img = next_img

            self.val_losses.append(total_loss / Config.validation_samples)
            self.val_indices.append(self.batch_count)

    async def train(self):
        self.model.train()
        prev_img = (await self.simulator.get_images()).to(self.device).float()
        prev_img = self.model.encode_image(prev_img)
        total_loss = 0
        self.optimizer.zero_grad()
        accumulated_batches = 0
        latent_state = torch.zeros((Config.batch_size, Config.hidden_size), device=self.device)

        pbar = tqdm.tqdm(range(self.batch_count, Config.max_batches), initial=self.batch_count, desc=f'Training')
        for idx in pbar:            
            await self.simulator.increment()
            movement = (await self.simulator.get_movement()).to(self.device).float() 
            next_img = (await self.simulator.get_images()).to(self.device).float()
            next_img = self.model.encode_image(next_img)

            latent_state = self.model.predict_dynamics(prev_img, movement, latent_state)
            if idx % Config.shortcut_frequency == 0 :
                loss = await self.train_batch_shortcut(latent_state, Config.batch_size, next_img)
            else:
                loss = await self.train_batch(latent_state, Config.batch_size, next_img)

            self.batch_losses.append(loss.item())
            self.batch_count += 1      
            total_loss += loss 

            if idx % Config.sample_every_x_batches == 0:
                with torch.no_grad():
                    sample_next_img(self.model, self.device, f"batch_{idx}", prev_img[0:1, :, :, :].detach(), latent_state[0:1, :].detach(), next_img[0:1, :, :, :].detach())
                    self.model.train()

            prev_img = next_img.detach().clone()

            if (idx + 1) % Config.latent_persistence_turns == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) #Enable if gradients explode
                total_loss /= Config.effective_batch_size * Config.latent_persistence_turns
                total_loss.backward()
                accumulated_batches += Config.batch_size
                if accumulated_batches >= Config.effective_batch_size:
                    print("Taking an optimiser step.")
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accumulated_batches = 0
                total_loss = 0
                state_base = Path.cwd() / "states" / Config.model_name
                state_base.mkdir(parents=True, exist_ok=True)

                state_path = state_base / f"idx_{(idx+1) % (Config.latent_persistence_turns * Config.latent_reset_turns)}.pt"
                torch.save(latent_state, state_path)
                if (idx + 1) % (Config.latent_persistence_turns * Config.latent_reset_turns) == 0: 
                    latent_state = torch.zeros_like(latent_state, device=self.device) 
                else:
                    latent_state = latent_state.detach()

            if idx % Config.validation_frequency == 0:
                await self.validate()
                self.model.train()
                  
            if idx % Config.save_freq == 0:
                self.save_checkpoint('main')

            pbar.set_postfix({"Batch loss": loss.item()})
            if idx % Config.graph_update_freq == 0:
                self.update_graph()

