from config import Config
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
fig, axes, ims = None, None, None

@torch.no_grad()
def sample_next_img(model, device, sample_name, prev_img, movement, latent, next_img=None):
    model.eval()
    base_img = prev_img.clone()
    global fig, axes, ims
    if fig is None:
        plt.ion()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ims = []
        for i, title in enumerate(['Base', 'Generated', 'Target']):
            ims.append(axes[i].imshow(np.zeros((Config.model_resolution[0], Config.model_resolution[1], 3), dtype=np.float32)))
            axes[i].axis("off")
            axes[i].set_title(title)
        plt.tight_layout()
        plt.show()
        
        base = np.transpose(base_img.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
        target = np.transpose(next_img.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
        ims[0].set_data(np.clip(base, 0, 1))
        ims[2].set_data(np.clip(target, 0, 1))
    
    out_img = None
    pbar = tqdm.tqdm(range(Config.inference_samples), desc=f'Sampling')
    for time_step in pbar:
        time_step = torch.tensor([time_step]).to(device)
        dx = Config.inference_step_size / Config.inference_samples
        delta = model.predict_delta(prev_img, time_step, movement, latent)
        prev_img += delta * dx
        prev_img = torch.clamp(prev_img, 0, 1)
        out_img = prev_img.squeeze(0).detach().cpu().numpy()
        out_img = np.transpose(out_img, (1, 2, 0))
        ims[1].set_data(out_img)
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    model_dir = os.path.join(Config.img_dir, Config.model_name)
    if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"sample_{sample_name}.png")
    plt.imsave(save_path, np.clip(out_img, 0, 1))
    target = np.transpose(next_img.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
    save_path = os.path.join(model_dir, f"sample_{sample_name}_target.png")
    plt.imsave(save_path, np.clip(target, 0, 1))
    base = np.transpose(base_img.squeeze(0).detach().cpu().numpy(), (1, 2, 0))
    save_path = os.path.join(model_dir, f"sample_{sample_name}_base.png")
    plt.imsave(save_path, np.clip(base, 0, 1))