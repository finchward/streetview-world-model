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
import copy
from collections import defaultdict
from torch.distributions import Normal, kl_divergence

def print_grad(name):
    def hook(grad):
        print(f"Gradient at {name}: {grad.norm() if grad is not None else 0}")
        return None 
    return hook
 
class Trainer:
    def __init__(self, model, simulator, val_simulator):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        self.simulator = simulator
        self.val_simulator = val_simulator
        self.loss_fn = nn.MSELoss()

        self.batch_count = 1
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
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            print("Could not load optimizer state (likely due to new/changed parameters).")
            print("Reinitializing optimizer. Error was:", e)
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

        if self.val_losses:
            self.grapher.update_line(0, "Validation", self.val_indices, self.val_losses)
            # for i, val_index in enumerate(self.val_indices):
            #     if val_index > self.batch_count - Config.recent_losses_shown:
            #         truncated_val_losses = self.val_losses[i:]
            #         truncated_val_indices = self.val_indices[i:]
            #         self.grapher.update_line(1, "Validation", truncated_val_indices, truncated_val_losses)

            

    async def train_batch(self, latent, batch_size, target):
        starting_noise = torch.randn_like(target)
        time = torch.rand((batch_size), device=self.device)
        v_target = (target - starting_noise)
        pt = starting_noise + v_target * time.view(batch_size, 1, 1, 1)
        step_sizes = torch.zeros(batch_size, device=self.device)
        v_pred = self.model.predict_delta(pt, time, latent, step_sizes)
        loss = self.loss_fn(v_pred, v_target)
        return loss

    async def train_batch_shortcut(self, latent, batch_size, target):
        values = torch.tensor([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2], device=self.device)
        indices = torch.randint(0, len(values), (batch_size,), device=self.device)
        step_sizes = values[indices].to(self.device)
        step_sizes_grounded = step_sizes.clone()
        step_sizes_grounded = torch.where(step_sizes == 1/128, 0.0, step_sizes_grounded).to(self.device)
        starting_noise = torch.randn_like(target)
        time = torch.rand((batch_size), device=self.device)
        time = torch.where(time + step_sizes > 1, 1 - step_sizes, time)

        v_target = target - starting_noise
        pt = starting_noise + v_target * time.view(batch_size, 1, 1, 1)

        first_step = self.ema_model.predict_delta(pt, time, latent, step_sizes_grounded)
        x_after_first_step = pt + first_step * step_sizes.view(batch_size, 1, 1, 1)
        second_step = self.ema_model.predict_delta(x_after_first_step, time + step_sizes, latent, step_sizes_grounded)
        
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
                if torch.rand(1).item() < Config.shortcut_percent:
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
        self.optimizer.zero_grad()
        latent_state = torch.zeros((Config.minibatch_size, Config.hidden_size), device=self.device)

        minibatches_per_batch = math.ceil(Config.batch_size / Config.minibatch_size)
        steps_per_batch = minibatches_per_batch * Config.rollout_steps


        scaler = torch.amp.GradScaler(self.device.type)
        pbar = tqdm.tqdm(range(self.batch_count, Config.max_batches), total=Config.max_batches, initial=self.batch_count, desc=f'Training')
        for batch in pbar:
            sampled_this_batch = False
            total_loss_this_batch = 0
            for minibatch in range(minibatches_per_batch):
                rollout_loss = 0
                for time_step in range(Config.rollout_steps):
                    await self.simulator.increment()
                    movement = (await self.simulator.get_movement()).to(self.device).float() 
                    next_img = (await self.simulator.get_images()).to(self.device).float()

                    with torch.autocast(device_type=self.device.type):
                        next_img = self.model.encode_image(next_img)
                        latent_state = self.model.predict_dynamics(prev_img, movement, latent_state)
                        if torch.rand(1).item() < Config.shortcut_percent:
                            loss = await self.train_batch_shortcut(latent_state, Config.minibatch_size, next_img)
                        else:
                            loss = await self.train_batch(latent_state, Config.minibatch_size, next_img)

                    self.batch_count += 1      
                    rollout_loss += loss
                    total_loss_this_batch += loss.item()

                    if (batch % Config.sample_freq == 0) and not sampled_this_batch:
                        sampled_this_batch = True
                        with torch.no_grad():
                            sample_next_img(self.model, self.device, f"batch_{batch}", prev_img[0:1, :, :, :].detach(), latent_state[0:1, :].detach(), next_img[0:1, :, :, :].detach())
                            self.model.train()

                    prev_img = next_img.detach().clone()           
                
                rollout_loss /= steps_per_batch
                scaler.scale(rollout_loss).backward()

                #if gonna enable clipping make sure it works with amp.
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if (batch * minibatches_per_batch + minibatch) % Config.rollouts_per_episode == 0:
                    latent_state = torch.zeros_like(latent_state, device=self.device) 
                else:
                    latent_state = latent_state.detach()  

            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()
            self.update_ema(self.ema_model, self.model, Config.ema_ratio)

            batch_loss = total_loss_this_batch / steps_per_batch
            self.batch_losses.append(batch_loss)

            if batch % Config.graph_update_freq == 0:
                self.update_graph()

            if batch % Config.save_freq == 0:
                self.save_checkpoint('main')

            if batch % Config.val_freq == 0:
                await self.validate()
                self.model.train() 

            state_base = Path.cwd() / "states" / Config.model_name
            state_base.mkdir(parents=True, exist_ok=True)
            state_path = state_base / f"idx_{batch % 10}.pt"
            torch.save(latent_state, state_path)
                
            pbar.set_postfix({"Batch loss": batch_loss})


    def update_ema(self, ema_model, model, decay):
        with torch.no_grad():
            ema_params = dict(ema_model.named_parameters())
            model_params = dict(model.named_parameters())
            for name in ema_params.keys():
                ema_params[name].data.mul_(decay).add_(model_params[name].data, alpha=1 - decay)

            # Do the same for buffers (e.g., BatchNorm stats)
            ema_buffers = dict(ema_model.named_buffers())
            model_buffers = dict(model.named_buffers())
            for name in ema_buffers.keys():
                ema_buffers[name].copy_(model_buffers[name])


    def debug(self):
        grouped = defaultdict(float)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.norm(2).item()
            else:
                grad = 0
            print("DEBUG START")
            print(f"{name}: {grad}")
            layer = name.split(".")[0]
            grouped[layer] += grad ** 2

        print("GROUPING START")
        print("-"*60)

        for layer, sum in grouped.items():
            print(f"{layer}: {(sum ** 0.5):.4f}")

    def check_model_health(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_norm = param.grad.norm(2).item() if param.grad is not None else 0.0
                param_norm = param.norm(2).item()
                ratio = grad_norm/param_norm if param_norm != 0 else 1 
                print(f"{name} : param={param_norm:.3f}, grad={grad_norm:.8f}, ratio={ratio:.6f}")