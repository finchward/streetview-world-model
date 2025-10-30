import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
from pathlib import Path
from config import Config
# Distributed imports
import torch.distributed as dist
from model import WorldModel
import math
from torchdiffeq import odeint

def convert_to_vector(x, y, w, h):
    nx = (2.0 * x - w) / w
    ny = (h - 2.0 * y) / h
    # project to unit sphere
    length2 = nx*nx + ny*ny
    if length2 > 1.0:
        norm = 1.0 / np.sqrt(length2)
        return np.array([nx*norm, ny*norm, 0.0])
    else:
        z = np.sqrt(1.0 - length2)
        return np.array([nx, ny, z])

@torch.no_grad()
def interact():

    def torch_to_image(tensor):
        img = torch.clamp(tensor, 0, 1)
        out_img = img.squeeze(0).detach().cpu().numpy()
        out_img = np.transpose(out_img, (1, 2, 0))
        return out_img
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if Config.is_latent_init_zeros:
        latent = torch.zeros((1, Config.latent_dimension), device=device)
    elif Config.is_latent_init_noise:
        latent = torch.randn((1, Config.latent_dimension), device=device)
    else:
        latent = torch.load(
            Path.cwd() / "states" / Config.interactive_model / Config.interactive_latent, 
            map_location=device
        )[0:1, :]
    checkpoint = torch.load(
        Path.cwd() / "checkpoints" / Config.interactive_model / Config.interactive_checkpoint, 
        map_location=device
    )
    model = WorldModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    height = Config.model_resolution[0]
    width = Config.model_resolution[1]

    out_image = torch.randn((1, 3, height, width), device=device)
    fig, ax = plt.subplots()
    im = ax.imshow(torch_to_image(out_image))
    plt.ion()

    def update_display(tensor):
        im.set_data(torch_to_image(tensor))
        fig.canvas.draw_idle()
        fig.canvas.flush_events()


    class ODEFunc(torch.nn.Module):
        def __init__(self, model, latent):
            super().__init__()
            self.model = model
            self.latent = latent

        def forward(self, t, x):
            time_tensor = torch.ones(x.shape[0], device=x.device) * t
            with torch.no_grad():
                zeros = torch.zeros_like(self.latent)
                delta_uncond = self.model.predict_delta(x, time_tensor, zeros)
                delta_cond = self.model.predict_delta(x, time_tensor, self.latent)
                delta_net = delta_uncond + 3 * (delta_cond - delta_uncond)
                
            return delta_net

    def update_image():
        nonlocal out_image
        model.eval()
        ode_func = ODEFunc(model, latent)
        x0 = torch.randn((1, 3, height, width), device=device)
        t_span = torch.tensor([0.0, 1.0], device=device)

        with torch.no_grad():
            solution = odeint(
                ode_func, 
                x0, 
                t_span, 
                method='dopri5', # A good adaptive solver (RK45)
                rtol=1e-4,       # Relative tolerance
                atol=1e-4        # Absolute tolerance
            )
        out_image = solution[-1]
        
        update_display(out_image)
        print("New image predicted")

    def sample(movement):
        nonlocal latent
        latent = model.predict_dynamics(out_image, movement, latent)
        print("New latent predicted")
        update_image()
        

    update_image()

    start = None
    end = None
    dragging = False
    threshold = 10 #pixels


    def on_press(event):
        nonlocal start, dragging
        if event.inaxes:
            start = (event.xdata, event.ydata)
            dragging = False

    def on_motion(event):
        nonlocal dragging
        if start is not None and event.inaxes:
            dx = event.xdata - start[0]
            dy = event.ydata - start[1]
            distance = math.hypot(dx, dy)
            if distance > threshold:
                dragging = True

    def on_release(event):
        nonlocal start, dragging, end
        if event.inaxes and start is not None:
            end = (event.xdata, event.ydata)
            if dragging:
                on_drag(start, end)
            else:
                on_click(start)
        start = None
        end = None
        dragging = False

    def on_click(coords):
        print("Click detected")
        x, y = coords
        rx = x / width
        ry = y / height
        movement = torch.tensor([1, 0, 0, 0, rx - 0.5, ry - 0.5], device=device, dtype=torch.float32).unsqueeze(0)
        sample(movement)

    def on_drag(start, end):
        print("Drag detected")
        x1, y1 = start
        x2, y2 = end
        v1 = convert_to_vector(x1, y1, width, height)
        v2 = convert_to_vector(x2, y2, width, height)
        axis = np.cross(v1, v2)
        axis /= np.linalg.norm(axis)
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        theta = np.arccos(dot)
        wq = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        xq, yq, zq = axis * s
        movement = torch.tensor([wq, xq, yq, zq, 0, 0], device=device, dtype=torch.float32).unsqueeze(0)
        sample(movement)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    plt.show(block=True)


if __name__ == "__main__":
    interact()