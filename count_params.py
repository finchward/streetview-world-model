import torch
from model import WorldModel

def print_params(module, prefix=""):
    total = 0
    for name, child in module.named_children():
        child_total = sum(p.numel() for p in child.parameters())
        total += child_total
        print(f"{prefix}{name}: {child_total:,} params")
        # Recurse into submodules
        total += print_params(child, prefix + "  ")
    return total

model = WorldModel()
device = "cpu"
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}\n")

# Print hierarchical parameter breakdown
print("Hierarchical parameter breakdown:")
print_params(model)
