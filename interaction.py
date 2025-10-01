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
    latent = torch.load(
        Path.cwd() / "states" / Config.interactive_model / Config.interactive_latent, 
        map_location=device
    ) # allow noise start too
    checkpoint = torch.load(
        Path.cwd() / "checkpoints" / Config.interactive_model / Config.interactive_checkpoint, 
        map_location=device
    )
    model = WorldModel()
    model.load_state_dict(checkpoint['model_state_dict'])

    out_image = torch.randn(Config.model_resolution, device=device)
    fig, ax = plt.subplots()
    im = ax.imshow(torch_to_image(out_image))
    plt.ion()

    def update_display(tensor):
        im.set_data(torch_to_image(tensor))
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    for time_step in range(Config.inference_samples):
        time_tensor = torch.tensor([time_step/Config.inference_samples], device=device)
        dx = 1/Config.inference_samples
        delta = model.predict_delta(out_image, time_tensor, latent)
        out_image += delta * dx
        update_display(out_image)
    #preparing an initial image for interaction with. 

    def sample(movement):

        


    click_coords = []

    start = None
    end = None
    dragging = False
    threshold = 10 #pixels
    height = Config.model_resolution[0]
    width = Config.model_resolution[1]

    def on_press(event):
        global start, dragging
        if event.inaxes:
            start = (event.xdata, event.ydata)
            dragging = False

    def on_motion(event):
        global dragging
        if start is not None and event.inaxes:
            dx = event.x - start[0]
            dy = event.y - start[1]
            distance = math.hypot(dx, dy)
            if distance > threshold:
                dragging = True

    def on_release(event):
        global start, dragging, end
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
        rx = coords.xdata / width
        ry = coords.ydata / height
        movement = torch.Tensor([1, 0, 0, 0, rx - 0.5, ry - 0.5])

    def on_drag(start, end):
        v1 = convert_to_vector(start.xdata, start.ydata, width, height)
        v2 = convert_to_vector(end.xdata, end.ydata, width, height)
        axis = np.cross(v1, v2)
        axis /= np.linalg.norm(axis)
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        theta = np.arccos(dot)
        wq = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        xq, yq, zq = axis * s
        movement = torch.Tensor([wq, xq, yq, zq, 0, 0], device=device)


    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)


    while True:
       