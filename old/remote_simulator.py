# remote_simulator.py

import httpx
import torch
import asyncio

# This will be used to get the initial pages to send to the server
from config import Config

class RemoteSimulator:
    """
    A drop-in replacement for the original Simulator class.
    Instead of running a local browser, it communicates with a remote
    server that hosts the actual simulator instance.
    """
    def __init__(self, base_url: str, password: str):
        if not base_url or not password:
            raise ValueError("base_url and password must be provided.")
        
        self.base_url = base_url.rstrip('/')
        self.password = password
        self.initial_pages = Config.initial_pages
        # Use a persistent async client for connection pooling
        self.client = httpx.AsyncClient(timeout=300.0) # 5 minute timeout

    async def _make_request(self, endpoint: str, json_data: dict):
        """Helper function to make POST requests to the server."""
        url = f"{self.base_url}/{endpoint}/"
        try:
            response = await self.client.post(url, json=json_data)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"Error response {e.response.status_code} while requesting {e.request.url!r}.")
            # Print the error detail from the server's response
            print(f"Server detail: {e.response.text}")
            raise
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")
            raise

    async def setup(self):
        """
        Calls the /setup_sim/ endpoint on the remote server to initialize
        the browser simulation with the pages from the local config.
        """
        print("Sending setup request to remote simulator...")
        payload = {
            "pages": self.initial_pages,
            "password": self.password
        }
        response_data = await self._make_request("setup_sim", payload)
        print(f"Remote server response: {response_data.get('message')}")

    async def get_images(self):
        """
        Calls the /get_images/ endpoint, receives the image data as a list,
        and converts it back into a PyTorch tensor.
        """
        print("Requesting images from remote simulator...")
        payload = {"password": self.password}
        images_list = await self._make_request("get_images", payload)
        
        # Convert the received list back to a torch tensor
        images_tensor = torch.tensor(images_list, dtype=torch.float32)
        print(f"Received images tensor with shape: {images_tensor.shape}")
        
        return images_tensor

    async def move(self):
        """
        Calls the /move/ endpoint, receives the movement data as a list,
        and converts it back into a PyTorch tensor.
        """
        print("Requesting moves from remote simulator...")
        payload = {"password": self.password}
        move_list = await self._make_request("move", payload)

        # Convert the received list back to a torch tensor
        movement_tensor = torch.tensor(move_list, dtype=torch.float32)
        print(f"Received movement tensor with shape: {movement_tensor.shape}")

        return movement_tensor
    
    async def close(self):
        """Closes the HTTP client session."""
        await self.client.aclose()
        print("Remote simulator client closed.")