from config import Config
from model import get_decoder
import torch
import matplotlib.pyplot as plt
import numpy as np
import os


fig, ax, im = None, None, None

@torch.no_grad()
def sample_image(model, device, name):
    global fig, ax, im
    if fig is None:
        plt.ion()
        fig, ax = plt.subplots()
        im = ax.imshow(np.zeros((Config.image_resolution, Config.image_resolution, 3), dtype=np.float32))
        ax.axis("off")
        plt.show()

    z = torch.randn((1, Config.latent_dim)).to(device)
    out_img = model(z).squeeze(0).detach().cpu().numpy()
    out_img = np.transpose(out_img, (1, 2, 0))

    im.set_data(out_img)
    fig.canvas.draw()
    fig.canvas.flush_events()

    model_dir = os.path.join(Config.img_dir, Config.model_name)
    if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"sample_{name}.png")
    plt.imsave(save_path, np.clip(out_img, 0, 1))

