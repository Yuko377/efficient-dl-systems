import torch
from torch import nn
from tqdm.auto import tqdm

from unet import Unet

from dataset import get_train_data
from scaler import StaticScaler, DynamicScaler

def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaling_type: str = "static"
) -> None:
    model.train()
    scaler = None
    if scaling_type == "static":
        scaler = StaticScaler(2 ** 11)
    elif scaling_type == "dynamic":
        scaler = DynamicScaler(init_scale_factor=2 ** 11, up_multiplier=2, down_multiplier=0.5, threshold=3)
    else:
        raise ValueError("unexpected scaling type")
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device.type, dtype=torch.float16):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        # TODO: your code for loss scaling here
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train(scaling_type: str):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, device=device, scaling_type=scaling_type)
