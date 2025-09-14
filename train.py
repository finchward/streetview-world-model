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

if Config.is_colab:
    from google.colab import drive
    drive.mount('/content/drive')

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
        self.loss_fn = nn.MSELoss()

        self.batch_count = 0
        self.batch_losses = []
        self.grapher = Grapher()

    def save_checkpoint(self, name):
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
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.batch_count = checkpoint['batch_count']
        self.batch_losses = checkpoint['batch_losses']

    def update_graph(self):
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

#every tick, we start by feeding in previous frame and input and previous frame's latent state. New frame produced and that is used (with input maybe) to get new latent state
    async def train(self):
        self.model.train()
        print("about to get_images", flush=True)
        prev_img = (await self.simulator.get_images()).to(self.device).float()
        num_samples = len(Config.initial_pages)
        latent_state = torch.randn((num_samples, Config.latent_dimension), device=self.device)
        unrolled_loss = 0
        self.optimizer.zero_grad()

        pbar = tqdm.tqdm(range(Config.max_batches), desc=f'Batch {self.batch_count}/{Config.max_batches}')
        for idx in pbar:            
            # latent_state = self.model.predict_dynamics(prev_img, latent_state) #here we are encoding the same image we are using to predict next frame.
            #it might be useful to encode the state of the frame we are using to predict, but maybe redundant.
            #try changing latent state to here during training and see what happens
            movement = (await self.simulator.move()).to(self.device).float() #[num_pages, 6]
            next_img = (await self.simulator.get_images()).to(self.device).float() #[num_pages, 3, w, h]
            v_target = next_img - prev_img # [n, 3, w, h]

            total_loss = 0
            for i in range(Config.predictions_per_image):
                time = torch.rand((num_samples), device=self.device)
                pt = prev_img + v_target * time.view(num_samples, 1, 1, 1)
                v_pred = self.model.predict_delta(pt, time, movement, latent_state)
                loss = self.loss_fn(v_pred, v_target)
                total_loss += loss / Config.predictions_per_image

            self.batch_losses.append(total_loss.item())
            self.batch_count += 1      
            unrolled_loss += total_loss / Config.latent_persistence_turns

            if idx % Config.sample_every_x_batches == 0:
                with torch.no_grad():
                    sample_next_img(self.model, self.device, f"batch_{idx}", prev_img[0:1, :, :, :].detach(), movement[0:1, :].detach(), latent_state[0:1, :].detach(), next_img[0:1, :, :, :].detach())

            if (idx + 1) % Config.latent_persistence_turns == 0:
                #Reset latent.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) #Enable if gradients explode
                unrolled_loss.backward()
                self.optimizer.step()
                unrolled_loss = 0
                self.optimizer.zero_grad()
                latent_state = torch.randn((num_samples, Config.latent_dimension), device=self.device)               


            
            pbar.set_postfix({"Batch loss": total_loss.item()})
            if idx % Config.graph_update_freq == 0:
                self.update_graph()

            latent_state = self.model.predict_dynamics(prev_img, latent_state)
            prev_img = next_img.detach().clone()
            




        
