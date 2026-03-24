import os
import time
import yaml
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from model import Generator
from utils import load_checkpoint
from dataset import build_transforms


def run_inference(input_dir, output_dir, checkpoint_path, high_res=416):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, _, test_transform = build_transforms(high_res)

    gen = Generator(in_channels=3).to(device)
    opt_dummy = torch.optim.Adam(gen.parameters(), lr=1e-4)
    load_checkpoint(checkpoint_path, gen, opt_dummy, lr=1e-4, device=device)

    os.makedirs(output_dir, exist_ok=True)
    gen.eval()

    for file_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, file_name)
        if not os.path.isfile(image_path):
            continue

        image = Image.open(image_path).convert("RGB")

        start_time = time.time()
        with torch.no_grad():
            upscaled_img = gen(
                test_transform(image=np.asarray(image))["image"]
                .unsqueeze(0)
                .to(device)
            )
        elapsed_time = time.time() - start_time
        print(f"{file_name}: ESRGAN elapsed time = {elapsed_time:.4f} seconds")

        save_image(upscaled_img, os.path.join(output_dir, file_name))


if __name__ == "__main__":
    with open("configs/esrgan_train.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_inference(
        input_dir=cfg["test_dir"],
        output_dir=cfg["output_dir"],
        checkpoint_path=cfg["checkpoint_gen"],
        high_res=cfg["high_res"],
    )