import os
import glob
import json
import torch
import webdataset as wds
from PIL import Image
from torchvision import transforms
from config import Config
import re
from collections import defaultdict

def safe_json(x):
    # Called when json field exists; you can parse, or return default if missing or invalid
    try:
        # if x is bytes or str
        return json.loads(x) if (isinstance(x, (bytes, str))) else x
    except Exception:
        return None

def has_fields(sample):
    # filter out samples that don’t have a jpg and valid json
    return "jpg" in sample and "json" in sample  # you can also check for sample.get("json") not None

class ReplaySimulator:
    """
    A drop-in replacement for the Simulator that replays recorded data from webdatasets.
    It can read from multiple page sequences in parallel.
    """
    def __init__(self, dataset_dir: str, num_parallel_sequences: int):
        """
        Initializes the ReplaySimulator.
        It automatically discovers the total number of sequences from the dataset_dir.

        Args:
            dataset_dir (str): The directory where the webdataset shards are stored.
            num_parallel_sequences (int): The number of sequences to read in parallel (e.g., 9).
        """
        self.dataset_dir = dataset_dir
        self.num_parallel = num_parallel_sequences
        self.total_sequences = 0  # Will be set by _find_all_sequence_shards

        # Find shards first to determine self.total_sequences
        self._sequence_paths = self._find_all_sequence_shards()
 
        if self.num_parallel > self.total_sequences:
            print(f"Warning: Number of parallel sequences ({self.num_parallel}) "
                  f"exceeds total sequence index range ({self.total_sequences}). "
                  "This is OK if you intend to oversample.")
            # Or raise the error if you prefer:
            # raise ValueError("Number of parallel sequences cannot exceed total available sequences.")

        # Track which sequence index each parallel reader is currently assigned to
        # Use modulo to handle oversampling (e.g., 9 parallel with 4 sequences)
        self.active_sequence_indices = [i % self.total_sequences for i in range(self.num_parallel)]
        
        # The next sequence index to load when one of the active ones finishes
        # Start from the next index after the initial assignments
        self.next_sequence_to_load = self.num_parallel

        self._iterators = [self._create_iterator_for_sequence(i) for i in self.active_sequence_indices]

        self._current_images = None
        self._current_movements = None

        self._image_transform = transforms.Compose([
            transforms.Resize(Config.model_resolution),
            transforms.ToTensor(),
        ])

    def _find_all_sequence_shards(self):
        """
        Finds and groups all shard files by their page index.
        This method scans the directory, parses filenames, and sets
        self.total_sequences based on the *highest sequence index* found.
        """
        print(f"Scanning {self.dataset_dir} for webdataset shards...")
        
        # Regex to find 'page_XXX-' where XXX is 3 digits
        pattern = re.compile(r"page_(\d{3})-.*\.tar")
        
        all_shards = defaultdict(list)
        shard_pattern = os.path.join(self.dataset_dir, "page_*.tar")
        all_files = glob.glob(shard_pattern)

        if not all_files:
            raise FileNotFoundError(f"No webdataset shards found matching '{shard_pattern}' in: {self.dataset_dir}")

        max_idx = -1
        for filepath in all_files:
            filename = os.path.basename(filepath)
            match = pattern.match(filename)
            
            if match:
                seq_idx = int(match.group(1))
                all_shards[seq_idx].append(filepath)
                if seq_idx > max_idx:
                    max_idx = seq_idx
            else:
                print(f"Warning: Found file matching pattern but not regex: {filename}")

        if not all_shards:
            raise FileNotFoundError(f"No valid 'page_XXX-...' shards found in: {self.dataset_dir}")

        # Set the total sequences based on the highest index found + 1
        # This correctly handles sparse sequences (e.g., if only 000 and 005 exist,
        # total_sequences will be 6, and indices 1-4 will be empty)
        self.total_sequences = max_idx + 1
        
        # Sort the shards for each sequence to ensure correct order
        for idx in all_shards:
            all_shards[idx].sort()
            
        print(f"Found {len(all_shards)} unique page sequences. Max index: {max_idx}. Set total_sequences to {self.total_sequences}.")
        return dict(all_shards)

    def _create_iterator_for_sequence(self, sequence_idx: int):
        """Creates a webdataset iterator for a given page/sequence index."""
        if sequence_idx not in self._sequence_paths:
            print(f"Warning: No data found for sequence index {sequence_idx}. Returning empty iterator.")
            return iter([])

        dataset_paths = self._sequence_paths[sequence_idx]
        print(f"Loading {len(dataset_paths)} shards for sequence {sequence_idx}...")
        
        # Create a dataset that decodes JPEGs to PIL images and keeps JSON as is
        dataset = wds.WebDataset(dataset_paths, handler=wds.ignore_and_continue).decode("pil", handler=wds.ignore_and_continue).select(has_fields).to_tuple("jpg", "json")
        return iter(dataset)

    async def setup(self):
        """
        Performs initial setup, including loading the first batch of data.
        This makes it a true drop-in replacement for the original simulator.
        """
        print("ReplaySimulator: Setting up and loading initial data...")
        await self.increment()
        print("ReplaySimulator: Setup complete. Data is ready.")

    async def increment(self):
        images_batch = []
        movements_batch = []

        for i in range(self.num_parallel):
            while True:  # keep advancing until we get a valid sample
                try:
                    img, move_json = next(self._iterators[i])
                except StopIteration:
                    print(f"Sequence {self.active_sequence_indices[i]} finished.")
                    new_sequence_idx = self.next_sequence_to_load % self.total_sequences
                    self.active_sequence_indices[i] = new_sequence_idx
                    self.next_sequence_to_load += 1
                    print(f"Loading next sequence: {new_sequence_idx}")
                    self._iterators[i] = self._create_iterator_for_sequence(new_sequence_idx)
                    continue  # try again with the new iterator

                # Skip bad samples (missing either part)
                if img is None or move_json is None:
                    continue

                # If we reach here, we have both image and json
                break

            # Process the image and movement data
            img_tensor = self._image_transform(img)
            move_data = json.loads(move_json) if isinstance(move_json, (str, bytes)) else move_json

            images_batch.append(img_tensor)
            movements_batch.append(torch.tensor(move_data, dtype=torch.float32))

        self._current_images = torch.stack(images_batch)
        self._current_movements = torch.stack(movements_batch)


    async def get_images(self) -> torch.Tensor:
        """
        Returns the current batch of images.
        Shape: [num_parallel_sequences, 3, H, W]
        """
        if self._current_images is None:
            raise RuntimeError("You must call increment() or setup() before get_images().")
        return self._current_images

    async def get_movement(self) -> torch.Tensor:
        """
        Returns the current batch of movements.
        Shape: [num_parallel_sequences, 6]
        """
        if self._current_movements is None:
            raise RuntimeError("You must call increment() or setup() before get_movement().")
        return self._current_movements
