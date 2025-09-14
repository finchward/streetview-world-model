
import asyncio
from playwright.async_api import async_playwright
from PIL import Image
import io
from torchvision import transforms
from config import Config
import torch
import random
import math
import numpy as np

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

class StreetViewTab:
    def __init__(self, id, page):
        self.id = id
        self.page = page

    async def take_screenshot(self):
        screenshot_bytes = await self.page.screenshot(timeout=180_000)
        image = Image.open(io.BytesIO(screenshot_bytes)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(Config.model_resolution),    
            transforms.ToTensor(),             # Convert to tensor
            # transforms.Normalize(              # Normalize tensor
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # )
        ])

        tensor = transform(image)
        return tensor

    async def move(self):
        await self.page.wait_for_selector('canvas.aFsglc', timeout=180_000)   
        element = self.page.locator('canvas.aFsglc').first
        box = await element.bounding_box()
        if not box:
            return

        async def sample_point():
            """Keep sampling until point is on canvas.aFsglc."""
            while True:
                rx = random.random()
                ry = random.random()
                x = box['x'] + rx * box['width']
                y = box['y'] + ry * box['height']
                hit = await self.page.evaluate(
                    """([x, y]) => document.elementFromPoint(x, y)?.className || null""",
                    [x, y]
                )
                if hit and "aFsglc" in hit:
                    return x, y, rx, ry

        is_rotation = random.random() < Config.rotation_probability
        if is_rotation:
            # Pick two valid points
            x1, y1, _, _ = await sample_point()
            x2, y2, _, _ = await sample_point()

            await self.page.mouse.move(x1, y1)
            await self.page.mouse.down(button="left")
            await self.page.mouse.move(x2, y2, steps=15)
            await self.page.mouse.up(button="left")
            await asyncio.sleep(4)

            # Convert to vectors relative to box
            v1 = convert_to_vector(x1 - box['x'], y1 - box['y'], box['width'], box['height'])
            v2 = convert_to_vector(x2 - box['x'], y2 - box['y'], box['width'], box['height'])
            axis = np.cross(v1, v2)
            axis /= np.linalg.norm(axis)
            dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
            theta = np.arccos(dot)
            wq = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            xq, yq, zq = axis * s
            return [wq, xq, yq, zq, 0, 0]

        else:
            x, y, rx, ry = await sample_point()
            await self.page.mouse.move(x, y)
            await self.page.mouse.click(x, y)
            await asyncio.sleep(4)
            return [1, 0, 0, 0, rx - 0.5, ry - 0.5]

class Simulator():
    def __init__(self):
        self.initial_pages = Config.initial_pages
        self.browser = None
        self.context = None
        self.playwright = None

    async def setup(self):
        self.playwright = await async_playwright().start()
        if Config.is_colab:
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--use-gl=swiftshader",
                    "--disable-software-rasterizer",
                ]
            )
        else:
            self.browser = await self.playwright.chromium.launch(headless=False) # for colab
        self.context = await self.browser.new_context()

        pages = await asyncio.gather(*(self.context.new_page() for _ in self.initial_pages))
        self.tabs = [StreetViewTab(i, page) for i, page in enumerate(pages)]
        await asyncio.gather(*(tab.page.goto(self.initial_pages[i]) for i, tab in enumerate(self.tabs)))
        print("All tabs loaded")

    async def close(self):
        await self.context.close()
        await self.browser.close()
        await self.playwright.stop()

    async def get_images(self):
        images_list = await asyncio.gather(*(tab.take_screenshot() for tab in self.tabs))
        images_tensor = torch.stack(images_list) #[page_num, 3, h, w]
        return images_tensor
    
    async def move(self):
        move_list = await asyncio.gather(*(tab.move() for tab in self.tabs))
        movement_tensor = torch.tensor(move_list)
        return movement_tensor #[page_num, 6]



