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
from math import floor

def has_fields(sample):
    # Keep only samples that contain both modalities
    return "jpg" in sample and "json" in sample

class ReplaySimulator:
    """
    Replay recorded data from webdatasets with a single logical queue:

      page_001-000001, page_001-000002, ..., page_001-000500,
      page_002-000001, ..., page_003-..., etc.

    Each parallel sequence starts at an even offset along this flattened queue
    and loops from the start when it reaches the end.
    """

    def __init__(self, dataset_dir: str, num_parallel_sequences: int):
        self.dataset_dir = dataset_dir
        self.num_parallel = num_parallel_sequences

        # Discover shards per page, sorted, and a fully-flattened queue
        self._sequence_paths, self._flat_paths = self._discover_and_flatten()

        if len(self._flat_paths) == 0:
            raise FileNotFoundError(f"No webdataset shards found in: {self.dataset_dir}")

        self.total_sequences = len(self._sequence_paths)  # number of page indices discovered

        # Fixed even-spaced starting offsets across the flattened queue
        N = len(self._flat_paths)
        P = max(1, self.num_parallel)
        self._start_offsets = [round(i * N / P) % N for i in range(P)]

        # One iterator per parallel stream, each looping the flattened queue
        self._iterators = [self._create_iterator_from_offset(off) for off in self._start_offsets]

        self._current_images = None
        self._current_movements = None

        self._image_transform = transforms.Compose([
            transforms.Resize(Config.model_resolution),
            transforms.ToTensor(),
        ])

    def _discover_and_flatten(self):
        """
        Returns:
          sequence_paths: dict[int, list[str]]  # per-page shard lists (sorted)
          flat_paths: list[str]                  # page-major flattened list
        """
        print(f"Scanning {self.dataset_dir} for webdataset shards...")

        # Accept variable digit counts for page and shard numbers.
        # Examples: page_001-000123.tar, page_12-7.tar
        pattern = re.compile(r"page_(\d+)-(\d+)\.tar$")
        shard_glob = os.path.join(self.dataset_dir, "page_*.tar")
        all_files = glob.glob(shard_glob)

        if not all_files:
            raise FileNotFoundError(f"No webdataset shards found matching '{shard_glob}' in: {self.dataset_dir}")

        per_page = defaultdict(list)
        for filepath in all_files:
            name = os.path.basename(filepath)
            m = pattern.match(name)
            if not m:
                print(f"Warning: ignoring unexpected shard name: {name}")
                continue
            page_idx = int(m.group(1))
            shard_idx = int(m.group(2))
            per_page[page_idx].append((shard_idx, filepath))

        if not per_page:
            raise FileNotFoundError(f"No valid 'page_XXX-YYY.tar' shards found in: {self.dataset_dir}")

        # Sort shards numerically within each page; then sort pages
        sequence_paths = {}
        for page_idx in sorted(per_page.keys()):
            per_page[page_idx].sort(key=lambda x: x[0])  # by shard_idx
            sequence_paths[page_idx] = [p for (_, p) in per_page[page_idx]]

        # Flatten page-major
        flat_paths = []
        for page_idx in sorted(sequence_paths.keys()):
            flat_paths.extend(sequence_paths[page_idx])

        print(
            f"Found {len(sequence_paths)} pages and {len(flat_paths)} shards total. "
            f"Flattened queue is page-major and shard-order-preserving."
        )

        return sequence_paths, flat_paths

    def _create_iterator_from_offset(self, offset: int):
        """
        Create a WebDataset iterator over a rotation of the flattened queue
        that starts at 'offset' and runs to the end, then the front.

        When this iterator exhausts, we will recreate it to loop again.
        """
        N = len(self._flat_paths)
        if N == 0:
            return iter([])
        offset = offset % N
        rotated = self._flat_paths[offset:] + self._flat_paths[:offset]
        dataset = (
            wds.WebDataset(rotated, handler=wds.ignore_and_continue)
              .decode("pil", handler=wds.ignore_and_continue)
              .select(has_fields)
              .to_tuple("jpg", "json")
        )
        return iter(dataset)

    async def setup(self):
        print("ReplaySimulator: Setting up and priming first batch...")
        await self.increment()
        print("ReplaySimulator: Ready.")

    async def increment(self):
        images_batch = []
        movements_batch = []

        for i in range(self.num_parallel):
            while True:
                try:
                    img, move_json = next(self._iterators[i])
                except StopIteration:
                    # Loop this iterator from the start of its rotated queue
                    self._iterators[i] = self._create_iterator_from_offset(self._start_offsets[i])
                    continue

                if img is None or move_json is None:
                    continue

                break

            img_tensor = self._image_transform(img)
            move = json.loads(move_json) if isinstance(move_json, (str, bytes)) else move_json

            images_batch.append(img_tensor)
            movements_batch.append(torch.tensor(move, dtype=torch.float32))

        self._current_images = torch.stack(images_batch)
        self._current_movements = torch.stack(movements_batch)

    async def get_images(self) -> torch.Tensor:
        if self._current_images is None:
            raise RuntimeError("Call increment() or setup() before get_images().")
        return self._current_images

    async def get_movement(self) -> torch.Tensor:
        if self._current_movements is None:
            raise RuntimeError("Call increment() or setup() before get_movement().")
        return self._current_movements
