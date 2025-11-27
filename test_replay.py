import asyncio
import torch
import matplotlib.pyplot as plt
import sys
import os

# Import your class
# Assumes the previous code is saved in replay_simulator.py
from replaysim import ReplaySimulator

# CONFIGURATION
# Update this path to where your .tar shards are located
DATASET_PATH = "./webdataset_sharded" 
NUM_PARALLEL = 1  # For visualization, 1 is usually easiest to track

async def main():
    # 1. Initialize
    print(f"Initializing simulator from {DATASET_PATH}...")
    try:
        sim = ReplaySimulator(DATASET_PATH, NUM_PARALLEL)
        await sim.setup()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check DATASET_PATH in the script.")
        return

    # 2. Setup Matplotlib for real-time updates
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 6))
    
    print("Simulator ready. Controls:")
    print("  [ENTER] : Next batch")
    print("  'q'     : Quit")

    try:
        while True:
            # --- The Loop ---
            
            # A. Increment
            await sim.increment()
            
            # B. Get Data
            # Tensors are typically shape [Batch_Size, Channels, Height, Width]
            images_tensor = await sim.get_images()
            movements_tensor = await sim.get_movement()

            # Extract the first item in the batch for display
            # Clone to CPU and convert to numpy
            img_to_show = images_tensor[0].cpu()
            movement_val = movements_tensor[0].cpu().numpy()

            # C. Display Image with Matplotlib
            # PyTorch is (C, H, W), Matplotlib needs (H, W, C)
            img_np = img_to_show.permute(1, 2, 0).numpy()
            
            ax.clear()
            ax.imshow(img_np)
            ax.set_title(f"Movement Vector: {movement_val}")
            ax.axis('off') # Hide axes ticks
            
            # Update the plot window
            plt.draw()
            plt.pause(0.01) # Small pause to let GUI events process

            # D. Print Movement & E. Await Input
            print(f"Current Movement: {movement_val}")
            
            user_input = input(">> Press Enter for next (q to quit): ")
            if user_input.lower().strip() == 'q':
                print("Exiting loop.")
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        plt.close(fig)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())