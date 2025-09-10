from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.Resize((config.image_resolution, config.image_resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

    ])
    dataset = datasets.ImageFolder(root='../streetview_images', transform=transform)
    train_dataset = Subset(dataset, range(config.validation_dataset_size, len(dataset)))
    val_dataset = Subset(dataset, range(config.validation_dataset_size))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.data_loader_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.data_loader_workers)
    return train_dataloader, val_dataloader