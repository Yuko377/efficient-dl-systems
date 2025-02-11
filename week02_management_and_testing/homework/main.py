import torch
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from pathlib import Path

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel
from hparams import config



def main(device: str, num_epochs: int = config["epochs"]):
    wandb.login(key="hehehe")
    wandb.init(config=config, project="ddpm_simple", name="baseline")
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=config["unet_hidden_size"]),
        betas=(config["beta_1"], config["beta_2"]),
        num_timesteps=config["num_timesteps"],
    )
    ddpm.to(device)

    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=False,
        transform=train_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=4, shuffle=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=config["learning_rate"])
    wandb.watch(ddpm)

    Path('./samples').mkdir(exist_ok=True)
    for i in range(num_epochs):
        loss_ema = train_epoch(ddpm, dataloader, optim, device)
        init_noise, samples = generate_samples(ddpm, device, f"samples/{i:02d}_gen_{device}.png", f"samples/{i:02d}_init_{device}.png")
        curr_stats = {
            "curr_epoch_ema_loss": loss_ema,
            "init_noise": wandb.Image(f"samples/{i:02d}_init_{device}.png"),
            "generated_pic": wandb.Image(f"samples/{i:02d}_gen_{device}.png"),
        }
        # wandb.log(curr_stats, step=epoch * len(dataset) + (i + 1) * config["batch_size"])
        wandb.log(curr_stats, step=i+1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(device=device)
