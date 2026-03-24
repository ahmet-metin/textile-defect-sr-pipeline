import torch
import torch.nn as nn
from torchvision.models import vgg19


class VGGLoss(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.vgg = vgg19(weights="DEFAULT").features[:35].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.loss = nn.MSELoss()

    def forward(self, input_tensor, target_tensor):
        vgg_input_features = self.vgg(input_tensor)
        vgg_target_features = self.vgg(target_tensor)
        return self.loss(vgg_input_features, vgg_target_features)


def gradient_penalty(critic, real, fake, device):
    batch_size, c, h, w = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1), device=device).repeat(1, c, h, w)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    return torch.mean((gradient_norm - 1) ** 2)