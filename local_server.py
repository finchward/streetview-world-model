# local_server.py

import asyncio
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pyngrok import ngrok
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

# Make sure to have simulator.py and config.py in the same directory
from simulator import Simulator

# --- Configuration & Globals ---

# Load environment variables from a .env file if it exists
load_dotenv()

# Instantiate the FastAPI app
app = FastAPI(
    title="StreetView Simulator Server",
    description="A server to remotely control a Playwright-based StreetView simulator.",
    version="1.0.0"
)

# Global variable to hold our simulator instance
simulator_instance: Simulator = None

# Fetch the password from environment variables
SECRET_PASSWORD = "6Lw9;Q8BUSAnOCDQKBQAOZ/qsBR."
if not SECRET_PASSWORD:
    raise ValueError("FATAL: SIMULATOR_PASSWORD environment variable not set.")

# --- Pydantic Models for Request Bodies ---

class SetupRequest(BaseModel):
    pages: List[str] = Field(..., example=[
        "https://www.google.com/maps/@-36.8485,174.7633,3a,75y,90h,90t/data=..."
    ])
    password: str

class PasswordRequest(BaseModel):
    password: str

# --- Helper Functions ---

def check_password(provided_password: str):
    """Raises an HTTPException if the password is incorrect."""
    if provided_password != SECRET_PASSWORD:
        raise HTTPException(status_code=403, detail="Invalid password.")

async def get_simulator() -> Simulator:
    """Returns the current simulator instance, raising an error if it's not set up."""
    if simulator_instance is None:
        raise HTTPException(
            status_code=409, 
            detail="Simulator not initialized. Please call /setup_sim/ first."
        )
    return simulator_instance

# --- API Endpoints ---

@app.post("/setup_sim/", status_code=200)
async def setup_sim_endpoint(request: SetupRequest):
    """
    Initializes or re-initializes the simulator with a new set of pages.
    This must be called before any other endpoint.
    """
    print("\nReceived request for /setup_sim/")
    global simulator_instance
    check_password(request.password)
    
    initial_pages = request.pages

    print(f"Setting up a new simulator with {len(request.pages)} pages...")
    simulator_instance = Simulator(initial_pages)
    await simulator_instance.setup()
    
    return {"message": f"Simulator initialized successfully with {len(request.pages)} pages."}

@app.post("/get_images/")
async def get_images_endpoint(request: PasswordRequest):
    """
    Takes a screenshot from each browser tab and returns them as a tensor.
    """
    print("\nReceived request for /get_images/")
    check_password(request.password)
    sim = await get_simulator()
    
    print("Capturing images...")
    images_tensor = await sim.get_images()
    # Convert tensor to a list for JSON serialization
    images_list = images_tensor.cpu().numpy().tolist()
    
    print("Sending image data as response.")
    return JSONResponse(content=images_list)

@app.post("/move/")
async def move_endpoint(request: PasswordRequest):
    """
    Performs a random move (pan or translate) in each browser tab.
    """
    print("\nReceived request for /move/")
    check_password(request.password)
    sim = await get_simulator()

    print("Performing moves...")
    movement_tensor = await sim.move()
    # Convert tensor to a list for JSON serialization
    movement_list = movement_tensor.cpu().numpy().tolist()
    
    print("Sending movement data as response.")
    return JSONResponse(content=movement_list)

# --- Server Startup ---

if __name__ == "__main__":
    port = 8000
    
    print("Starting ngrok tunnel...")
    # Start ngrok tunnel
    public_url = ngrok.connect(port)
    print("--- Your Remote Simulator is Live! ---")
    print(f"Public Ngrok URL: {public_url}")
    print("Use this URL and your password in the remote_simulator.py file.")
    print("---------------------------------------")

    # Start the FastAPI server
    print(f"Starting FastAPI server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)