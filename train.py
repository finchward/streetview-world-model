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

if Config.is_tpu:
    import torch_xla.core.xla_model as xm

class Trainer:
    def __init__(self, model, simulator):
        if Config.is_tpu:
            self.device = xm.xla_device()
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        self.simulator = simulator
        self.loss_fn = nn.HuberLoss(delta=Config.huber_delta)

        self.batch_count = 0
        self.batch_losses = []
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

    def update_graph(self):
        indices = range(len(self.batch_losses))
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


#every tick, we start by feeding in previous frame and input and previous frame's latent state. New frame produced and that is used (with input maybe) to get new latent state
    async def train(self):
        self.model.train()
        print("about to get_images", flush=True)
        prev_img = (await self.simulator.get_images()).to(self.device).float()
        num_samples = Config.batch_size
        latent_state = torch.zeros((num_samples, Config.latent_dimension), device=self.device)
        unrolled_loss = 0
        self.optimizer.zero_grad()
        accumulated_batches = 0

        pbar = tqdm.tqdm(range(self.batch_count, Config.max_batches), initial=self.batch_count, desc=f'Training')
        for idx in pbar:            
            # latent_state = self.model.predict_dynamics(prev_img, latent_state) #here we are encoding the same image we are using to predict next frame.
            #it might be useful to encode the state of the frame we are using to predict, but maybe redundant.
            #try changing latent state to here during training and see what happens
            await self.simulator.increment()
            movement = (await self.simulator.get_movement()).to(self.device).float() #[num_pages, 6]
            next_img = (await self.simulator.get_images()).to(self.device).float() #[num_pages, 3, w, h]
            if Config.from_noise:
                latent_state = self.model.predict_dynamics(prev_img, movement, latent_state)
                starting_noise = torch.randn_like(prev_img)
                total_loss = 0
                for i in range(Config.predictions_per_image):
                    time = torch.rand((num_samples), device=self.device)
                    time_exp = time.view(num_samples, 1, 1, 1)
                    v_target = (next_img - starting_noise)
                    pt = starting_noise + v_target * time_exp
                    if Config.is_multi_gpu:
                        v_pred = self.model.module.predict_delta(pt, time, latent_state)
                    else:
                        v_pred = self.model.predict_delta(pt, time, latent_state)
                    loss = self.loss_fn(v_pred, v_target)
                    total_loss += loss / Config.predictions_per_image
            else:
                random_erasing = transforms.RandomErasing(p=Config.erasing_p, scale=Config.erasing_scale, ratio=Config.erasing_ratio, value=Config.erasing_value)
                prev_img = torch.stack([random_erasing(img) for img in torch.unbind(prev_img, dim=0)])
                v_target = next_img - prev_img # [n, 3, w, h]

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
                    sample_next_img(self.model, self.device, f"batch_{idx}", prev_img[0:1, :, :, :].detach(), latent_state[0:1, :].detach(), next_img[0:1, :, :, :].detach())

            if (idx + 1) % Config.latent_persistence_turns == 0:
                #Reset latent.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) #Enable if gradients explode
                unrolled_loss /= math.ceil(Config.effective_batch_size / num_samples)
                unrolled_loss.backward()
                accumulated_batches += num_samples
                if accumulated_batches >= Config.effective_batch_size:
                    print("Taking an optimiser step.")
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accumulated_batches = 0
                unrolled_loss = 0
                state_base = Path.cwd() / "states" / Config.model_name
                state_base.mkdir(parents=True, exist_ok=True)

                state_path = state_base / f"idx_{(idx+1) % (Config.latent_persistence_turns * Config.latent_reset_turns)}.pt"
                torch.save(latent_state, state_path)
                if (idx + 1) % (Config.latent_persistence_turns * Config.latent_reset_turns) == 0: 
                    latent_state = torch.zeros_like(latent_state, device=self.device) 
                else:
                    latent_state = latent_state.detach()      
            if idx % Config.save_freq == 0:
                self.save_checkpoint('main')

            pbar.set_postfix({"Batch loss": total_loss.item()})
            if idx % Config.graph_update_freq == 0:
                self.update_graph()

            if not Config.from_noise:
                if Config.is_multi_gpu:
                    latent_state = self.model.module.predict_dynamics(prev_img, latent_state)
                else:
                    latent_state = self.model.predict_dynamics(prev_img, latent_state)
            prev_img = next_img.detach().clone()
            