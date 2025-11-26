from config import Config
from simulator import Simulator
import asyncio
import webdataset as wds
from io import BytesIO
from PIL import Image
import os
import json
import glob
import re
import tqdm
def get_start_shard_num(save_dir, page_idx):
    """
    Finds the next shard number to use for a given page by checking existing files.
    This allows the recording to be stopped and resumed.
    """
    pattern = re.compile(f"page_{page_idx:03d}-(\d+).tar")
    max_shard = -1
    
    # Check if the directory exists to avoid errors on the first run
    if not os.path.isdir(save_dir):
        return 0

    for f in os.listdir(save_dir):
        match = pattern.match(f)
        if match:
            shard_num = int(match.group(1))
            if shard_num > max_shard:
                max_shard = shard_num
    
    return max_shard + 1


async def record_webdatasets(initial_pages, steps, save_dir="webdataset", shard_size=1000):
    """
    Records simulation data into sharded webdatasets.
    
    Args:
        initial_pages: List of initial pages for the simulator.
        steps (int): The number of simulation steps to run in this session.
        save_dir (str): The directory to save the datasets in.
        shard_size (int): The number of samples to save in each shard (.tar file).
    """
    os.makedirs(save_dir, exist_ok=True)

    sim = Simulator(initial_pages)
    await sim.setup()

    num_pages = len(initial_pages)

    # Determine the starting shard number for each page to allow resuming
    current_shard_nums = [get_start_shard_num(save_dir, i) for i in range(num_pages)]
    writers = [None] * num_pages

    try:

        for session_step in tqdm.tqdm(range(steps),  desc="Step"):
            # Check if it's time to roll over to a new shard
            if (session_step) % shard_size == 0:
                # Close any existing writers
                for w in writers:
                    if w:
                        w.close()
                
                # Create new writers for the new shard
                writers = [
                    wds.TarWriter(os.path.join(save_dir, f"page_{i:03d}-{current_shard_nums[i]:06d}.tar"))
                    for i in range(num_pages)
                ]
                print(f"--- Starting new shard for step {session_step} ---")
                # Increment shard numbers for the next rollover
                current_shard_nums = [n + 1 for n in current_shard_nums]

            moves = await sim.move()      # [pages, 6]  
            imgs = await sim.get_images()  # [pages, 3, H, W]
            

            for i in range(num_pages):
                img = imgs[i]  # torch tensor [3,H,W]
                move = moves[i]

                # Convert image to JPEG bytes
                img_pil = Image.fromarray((img.numpy() * 255).astype("uint8").transpose(1, 2, 0))
                buffer = BytesIO()
                img_pil.save(buffer, format="JPEG", quality=85)
                jpeg_bytes = buffer.getvalue()

                # Convert movement to list for JSON
                move_list = move.numpy().tolist()

                # Write to the current shard's tar file
                sample_key = f"{session_step:08d}"
                writers[i].write({
                    "__key__": sample_key,
                    "jpg": jpeg_bytes,
                    "json": json.dumps(move_list).encode("utf-8")
                })
                

            if session_step % 100 == 0:
                print(f"Step {session_step} saved for all pages.")

    except Exception as e:
        print("Error", e)

    finally:
        print("--- Closing all writers. ---")
        for w in writers:
            if w:
                w.close()


def replay_webdataset(page_idx, dataset_dir="webdataset"):
    """
    Loads and replays a sharded dataset for a single page.
    It automatically finds all shards for the specified page.
    """
    # Use glob to find all shard files for the given page
    shard_pattern = os.path.join(dataset_dir, f"page_{page_idx:03d}-*.tar")
    dataset_paths = sorted(glob.glob(shard_pattern))

    if not dataset_paths:
        print(f"Warning: No dataset shards found for page {page_idx} in '{dataset_dir}'")
        return

    print(f"Loading {len(dataset_paths)} shards for page {page_idx}...")
    dataset = wds.WebDataset(dataset_paths).decode("pil").to_tuple("jpg", "json")

    # The rest of the function remains the same
    for img, move_json in dataset:
        move = json.loads(move_json)
        yield img, move


if __name__ == '__main__':
    # Run for 100,000 steps, creating a new shard every 10,000 steps.
    # If you stop and restart this, it will create new shards without overwriting.
    asyncio.run(record_webdatasets(
        Config.initial_pages, 
        steps=100_000_000, 
        save_dir="webdataset_sharded",
        shard_size=500 
    ))

    # Example of replaying the data for page 0
    print("\n--- Replaying data for page 0 ---")
    replay_generator = replay_webdataset(page_idx=0, dataset_dir="webdataset_sharded")
    
    # Print the first 5 samples
    for i, (image, move) in enumerate(replay_generator):
        if i >= 5:
            break
        print(f"Sample {i}: Image size {image.size}, Move data {move}")