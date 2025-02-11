import pytest
import torch
from torch.utils.data import DataLoader, Subset # <------------ for CPU testing
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
from pathlib import Path

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, train_epoch, generate_samples
from modeling.unet import UnetModel


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./cifar10",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    torch.manual_seed(1337)
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5

@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])  
def test_training(device, train_dataset):
    # note: implement and test a complete training procedure (including sampling)
    torch.manual_seed(1337)
    if device == "cuda":
        dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    else:
        cropped_dataset_len = int(0.1 * len(train_dataset))
        dataloader = DataLoader(Subset(train_dataset, list(range(cropped_dataset_len))), batch_size=4, shuffle=True)

    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    loss_ema = train_epoch(ddpm, dataloader, optim, device)
    assert loss_ema < 0.1
    
    Path('./samples').mkdir(exist_ok=True)
    _, samples = generate_samples(ddpm, device, f"./samples/test_{device}.png")
    assert samples.shape == (8, 3, 32, 32)
    assert not torch.any(torch.isnan(samples)), "No nans in samples"
    assert not torch.any(torch.isinf(samples)), "No infs in samples"
