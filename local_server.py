import asyncio
import io
import base64
import subprocess
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import uvicorn
import torch
from PIL import Image
import json
import threading
import requests

# Import your original simulator
from simulator import Simulator as OriginalSimulator

app = FastAPI(title="Street View Simulator Server")

# Global simulator instance
simulator = None
ngrok_url = None

def start_ngrok():
    """Start ngrok tunnel and return the public URL"""
    global ngrok_url
    try:
        # Kill any existing ngrok processes
        subprocess.run(["pkill", "-f", "ngrok"], capture_output=True)
        time.sleep(2)
        
        # Start ngrok tunnel
        process = subprocess.Popen(
            ["ngrok", "http", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for ngrok to start
        time.sleep(3)
        
        # Get the public URL from ngrok API
        response = requests.get("http://127.0.0.1:4040/api/tunnels")
        tunnels = response.json()["tunnels"]
        
        for tunnel in tunnels:
            if tunnel["proto"] == "https":
                ngrok_url = tunnel["public_url"]
                print(f"🚀 Ngrok tunnel active: {ngrok_url}")
                return ngrok_url
                
        raise Exception("No HTTPS tunnel found")
        
    except Exception as e:
        print(f"❌ Failed to start ngrok: {e}")
        print("Make sure ngrok is installed: https://ngrok.com/download")
        return None

@app.on_event("startup")
async def startup_event():
    global simulator, ngrok_url
    
    print("🔧 Starting simulator...")
    simulator = OriginalSimulator()
    await simulator.setup()
    print("✅ Simulator ready")
    
    # Start ngrok in a separate thread
    def ngrok_thread():
        start_ngrok()
    
    threading.Thread(target=ngrok_thread, daemon=True).start()

@app.on_event("shutdown")
async def shutdown_event():
    global simulator
    if simulator:
        await simulator.close()
        print("🛑 Simulator closed")

@app.get("/")
async def root():
    return {
        "message": "Street View Simulator Server",
        "ngrok_url": ngrok_url,
        "endpoints": {
            "screenshots": "/screenshots",
            "move": "/move",
            "status": "/status"
        }
    }

@app.get("/status")
async def status():
    global simulator, ngrok_url
    return {
        "simulator_ready": simulator is not None,
        "ngrok_url": ngrok_url,
        "tabs": len(simulator.tabs) if simulator else 0
    }

@app.get("/screenshots")
async def get_screenshots():
    """Get screenshots from all tabs as base64 encoded images"""
    global simulator
    
    if not simulator:
        raise HTTPException(status_code=503, detail="Simulator not ready")
    
    try:
        # Get images tensor [page_num, 3, h, w]
        images_tensor = await simulator.get_images()
        
        # Convert tensor to base64 encoded images
        images_b64 = []
        for i in range(images_tensor.shape[0]):
            # Convert tensor to PIL Image
            tensor_img = images_tensor[i]  # [3, h, w]
            pil_img = Image.fromarray((tensor_img.permute(1, 2, 0) * 255).numpy().astype('uint8'))
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            images_b64.append(img_b64)
        
        return {
            "images": images_b64,
            "shape": list(images_tensor.shape)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screenshot failed: {str(e)}")

@app.post("/move")
async def move_tabs():
    """Simulate movement on all tabs"""
    global simulator
    
    if not simulator:
        raise HTTPException(status_code=503, detail="Simulator not ready")
    
    try:
        # Get movement tensor [page_num, 6]
        movement_tensor = await simulator.move()
        
        return {
            "movements": movement_tensor.tolist(),
            "shape": list(movement_tensor.shape)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Movement failed: {str(e)}")

def main():
    print("🚀 Starting Street View Simulator Server...")
    print("📋 Make sure you have:")
    print("   1. ngrok installed (https://ngrok.com/download)")
    print("   2. Your config.py file configured")
    print("   3. All dependencies installed")
    print()
    
    # Run the server
    uvicorn.run(
        "local_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True
    )

if __name__ == "__main__":
    main()