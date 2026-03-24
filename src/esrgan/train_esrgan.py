import os
import yaml
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Generator, Discriminator, initialize_weights
from losses import VGGLoss, gradient_penalty
from dataset import MyImageFolder
from utils import save_checkpoint, load_checkpoint


def train_fn(
    loader,
    disc,
    gen,
    opt_disc,
    opt_gen,
    l1_loss,
    vgg_loss,
    g_scaler,
    d_scaler,
    device,
    lambda_gp=10.0,
):
    loop = tqdm(loader, leave=True)

    for low_res, high_res in loop:
        low_res = low_res.to(device)
        high_res = high_res.to(device)

        # Train discriminator / critic
        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            fake = gen(low_res)
            critic_real = disc(high_res)
            critic_fake = disc(fake.detach())
            gp = gradient_penalty(disc, high_res, fake, device=device)
            loss_disc = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + lambda_gp * gp
            )

        opt_disc.zero_grad()
        d_scaler.scale(loss_disc).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            critic_fake = disc(fake)
            l1 = l1_loss(fake, high_res)
            perceptual = vgg_loss(fake, high_res)
            adversarial = -torch.mean(critic_fake)
            loss_gen = l1 + 0.006 * perceptual + 1e-3 * adversarial

        opt_gen.zero_grad()
        g_scaler.scale(loss_gen).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        loop.set_postfix(loss_disc=float(loss_disc.item()), loss_gen=float(loss_gen.item()))


def main():
    with open("configs/esrgan_train.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = MyImageFolder(root_dir=cfg["data_root"], high_res=cfg["high_res"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=True,
    )

    gen = Generator(
        in_channels=3,
        num_channels=cfg.get("num_channels", 64),
        num_blocks=cfg.get("num_blocks", 23),
    ).to(device)
    disc = Discriminator(in_channels=3).to(device)

    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=cfg["learning_rate"], betas=(0.0, 0.9))
    opt_disc = optim.Adam(disc.parameters(), lr=cfg["learning_rate"], betas=(0.0, 0.9))

    l1_loss = nn.L1Loss()
    vgg_loss = VGGLoss(device=device)

    g_scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    d_scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    if cfg.get("load_model", False):
        load_checkpoint(cfg["checkpoint_gen"], gen, opt_gen, cfg["learning_rate"], device=device)
        load_checkpoint(cfg["checkpoint_disc"], disc, opt_disc, cfg["learning_rate"], device=device)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    for epoch in range(cfg["num_epochs"]):
        print(f"Epoch [{epoch + 1}/{cfg['num_epochs']}]")
        train_fn(
            loader=loader,
            disc=disc,
            gen=gen,
            opt_disc=opt_disc,
            opt_gen=opt_gen,
            l1_loss=l1_loss,
            vgg_loss=vgg_loss,
            g_scaler=g_scaler,
            d_scaler=d_scaler,
            device=device,
            lambda_gp=cfg.get("lambda_gp", 10.0),
        )

        if cfg.get("save_model", True):
            save_checkpoint(
                gen,
                opt_gen,
                os.path.join(cfg["checkpoint_dir"], f"gen_epoch_{epoch + 1}.pth.tar"),
            )
            save_checkpoint(
                disc,
                opt_disc,
                os.path.join(cfg["checkpoint_dir"], f"disc_epoch_{epoch + 1}.pth.tar"),
            )


if __name__ == "__main__":
    main()