from model import WorldModel

model = WorldModel()

model.to("cpu")
param_count = sum(p.numel() for p in model.parameters())
print("Total parameters:", param_count)
quit()