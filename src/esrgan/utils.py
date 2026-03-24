import os
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from .dataset import build_transforms


def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    print(f"=> Saving checkpoint to {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer=None, lr=None, device="cpu"):
    print(f"=> Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr


def plot_examples(low_res_folder, output_folder, gen, device="cpu", high_res=416):
    os.makedirs(output_folder, exist_ok=True)
    _, _, _, test_transform = build_transforms(high_res)

    files = os.listdir(low_res_folder)
    gen.eval()

    for file_name in files:
        image_path = os.path.join(low_res_folder, file_name)
        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            upscaled_img = gen(
                test_transform(image=np.asarray(image))["image"]
                .unsqueeze(0)
                .to(device)
            )

        save_image(upscaled_img, os.path.join(output_folder, file_name))

    gen.train()