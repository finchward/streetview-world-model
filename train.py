import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Optional, Callable, Any
import logging
from pathlib import Path
import time
import tqdm
from grapher import Grapher
import numpy as np
import bisect
from inference import sample_image

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.current_epoch = 0
        self.max_epochs = config.num_epochs
        self.batch_count = 0
        self.batch_losses = []  
        self.val_losses = []
        self.val_reconstruction_losses = []
        self.val_divergence_losses = []
        self.val_losses_batch_counts = []
        self.grouped_losses = []

        self.model_name = config.model_name
        self.best_val_loss = float('inf')
        self.grapher = Grapher()
        self.graph_update_every = config.graph_update_every
        self.graph_recent_losses_shown = config.graph_recent_losses_shown
        self.group_size = config.graph_grouped_loss_size
        self.save_every_x_epoch = config.save_every_x_epoch
        self.kl_coefficient = config.kl_coefficient

        self.validate_every_x = config.validate_every_x
        self.kl_warmup_steps = config.kl_warmup_steps
    
    def process_batch(self, batch_idx, img):
        img = img.to(self.device)
        out_img, mu, logvar = self.model(img)
        reconstruction_loss = F.mse_loss(out_img, img)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        kl_divergence = 1 + logvar - mu.pow(2) - logvar.exp()
        
        kl_divergence_loss = -0.5 * torch.sum(kl_divergence) * min(1.0, self.batch_count / float(max(1, self.kl_warmup_steps)))
        # kl_divergence_loss = kl_divergence_loss * self.kl_coefficient
        loss = reconstruction_loss + kl_divergence_loss
        return loss, reconstruction_loss, kl_divergence_loss

    def save_checkpoint(self, name):
        check_dir = Path.cwd() / 'checkpoints' / self.model_name
        if not check_dir.exists():
            check_dir.mkdir(exist_ok=True, parents=True)
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_count': self.batch_count,
            'batch_losses': self.batch_losses,
            'val_losses': self.val_losses,
            'val_losses_batch_counts': self.val_losses_batch_counts,
            'best_val_loss': self.best_val_loss,
        }
        checkpoint_path = check_dir / name
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, config):
        checkpoint_path = Path.cwd() / 'checkpoints' / config.loaded_model_name / config.loaded_checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.batch_count = checkpoint['batch_count']
        self.batch_losses = checkpoint['batch_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_losses_batch_counts = checkpoint['val_losses_batch_counts']
        self.best_val_loss = checkpoint['best_val_loss']
        self.update_graph_val()

    def update_graph_training(self):
        indices = range(len(self.batch_losses))
        losses = self.batch_losses
        recent_losses = self.batch_losses[-1*self.graph_recent_losses_shown:]
        recent_losses_indices = indices[-1*self.graph_recent_losses_shown:]
        self.grapher.update_line(1, "Training", recent_losses_indices, recent_losses)

        losses = np.array(losses)
        indices = np.array(indices)
        losses_grouped = losses.reshape(-1, self.group_size)
        indices_grouped = indices.reshape(-1, self.group_size)
        losses_reduced = losses_grouped.mean(axis=1)
        indices_reduced = indices_grouped[:, -1]
        self.grapher.update_line(0, "Training", indices_reduced, losses_reduced) #for performance

    def update_graph_val(self):
        losses = self.val_losses
        indices = self.val_losses_batch_counts
        reconstruction_losses = self.val_reconstruction_losses
        divergence_losses = self.val_divergence_losses
        self.grapher.update_line(0, "Validation", indices, losses)
        self.grapher.update_line(0, "Reconstruction loss", indices, reconstruction_losses)
        self.grapher.update_line(0, "Divergence loss", indices, divergence_losses)
        start = bisect.bisect_right(indices, self.batch_count - self.graph_recent_losses_shown)
        recent_losses = losses[start:]
        recent_reconstruction_losses = reconstruction_losses[start:]
        recent_divergence_losses = reconstruction_losses[start:]
        recent_indices = indices[start:]
        self.grapher.update_line(1, "Validation", recent_indices, recent_losses)
        self.grapher.update_line(1, "Reconstruction loss", recent_indices, recent_reconstruction_losses)
        self.grapher.update_line(1, "Divergence loss", recent_indices, recent_divergence_losses)

    def train_epoch(self):
        self.model.train()
        loader = self.train_loader
        self.optimizer.zero_grad()
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {self.current_epoch}/{self.max_epochs}')
        for batch_idx, (img, _) in pbar:
            loss, _, _ = self.process_batch(batch_idx, img)
            self.batch_losses.append(loss.item())
            if self.validate_every_x is not None:
                if self.batch_count % self.validate_every_x == 0:
                    self.validate()
                    sample_image(self.model.decoder, self.device, f'{self.model_name}_epoch_{self.current_epoch}')

            self.batch_count += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            pbar.set_postfix({"Batch loss": loss.item()})
            if batch_idx % self.graph_update_every:
                self.update_graph_training()

        if self.validate_every_x is None:
            self.validate()
            sample_image(self.model.decoder, self.device, f'{self.model_name}_epoch_{self.current_epoch}')
        if (self.current_epoch) % self.save_every_x_epoch == 0:
            self.save_checkpoint(f'epoch_{self.current_epoch}.pth')
        
        print('New image generated.')
        self.current_epoch += 1

    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        loader = self.val_loader
        total_val_loss = 0
        total_reconstruction_loss = 0
        total_divergence_loss = 0
        for batch_idx, (img, _) in tqdm.tqdm(enumerate(loader), total=len(loader), desc=f'Validating'):
            loss, reconstruction_loss, divergence_loss = self.process_batch(batch_idx, img)
            total_val_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_divergence_loss += divergence_loss.item()

        avg_loss = total_val_loss / len(loader)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(f'best_val_loss.pth')
        self.val_losses.append(total_val_loss / len(loader))
        self.val_reconstruction_losses.append(total_reconstruction_loss / len(loader))
        self.val_divergence_losses.append(total_divergence_loss / len(loader))
        self.val_losses_batch_counts.append(self.batch_count)
        self.update_graph_val()

    def train(self):
        for epoch in range(self.current_epoch, self.max_epochs):
            self.train_epoch()
        
