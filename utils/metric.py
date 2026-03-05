import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


def calculate_distribution_metrics(real_images, generated_images, device="cuda"):
    """
    real_images: Tensor of shape (B, 1, 32, 32), floats in [0, 1] or [-1, 1]
    generated_images: Tensor of shape (B, 1, 32, 32), floats in [0, 1] or [-1, 1]
    """

    real_images = real_images.to(device)
    generated_images = generated_images.to(device)

    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=50).to(device)

    def prepare_for_inception(img_tensor):
        img_tensor = img_tensor.clamp(0, 1)

        img_tensor = (img_tensor * 255).to(torch.uint8)

        img_tensor = img_tensor.repeat(1, 3, 1, 1)
        return img_tensor

    real_prepared = prepare_for_inception(real_images)
    fake_prepared = prepare_for_inception(generated_images)

    fid.update(real_prepared, real=True)
    kid.update(real_prepared, real=True)

    fid.update(fake_prepared, real=False)
    kid.update(fake_prepared, real=False)

    fid_score = fid.compute()
    kid_mean, kid_std = kid.compute()

    print(f"FID Score: {fid_score.item():.4f} (Lower is better)")
    print(f"KID Score: {kid_mean.item():.4f} ± {kid_std.item():.4f} (Lower is better)")

    return fid_score.item(), kid_mean.item()
