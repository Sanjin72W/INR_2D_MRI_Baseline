import os
import torch
import numpy as np

import matplotlib.pyplot as plt


def get_image_from_prediction(pred_flat, H, W):
    img_complex = torch.view_as_complex(pred_flat.view(H, W, 2))
    return img_complex


def save_epoch_results_as_png(model, sample, device, epoch, save_dir, supervision_mode="image"):
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch + 1}")
    os.makedirs(epoch_dir, exist_ok=True)
    model.eval()
    coords = sample['coords'].to(device)
    with torch.no_grad():
        pred_flat = model(coords)
    H, W = sample['gt_img'].shape
    if supervision_mode in ['kspace', 'kspace_csm', 'image']:
        pred_img_complex = get_image_from_prediction(pred_flat, H, W)
        pred_real = pred_img_complex.real.detach().cpu().numpy()
        pred_imag = pred_img_complex.imag.detach().cpu().numpy()
        pred_mag = torch.abs(pred_img_complex).detach().cpu().numpy()

        plt.imsave(os.path.join(epoch_dir, "pred_real.png"), pred_real, cmap="gray")
        plt.imsave(os.path.join(epoch_dir, "pred_imag.png"), pred_imag, cmap="gray")
        plt.imsave(os.path.join(epoch_dir, "pred_mag.png"), pred_mag, cmap="gray")

        print(f"Saved epoch {epoch + 1} predictions in {epoch_dir}")
    elif supervision_mode in ['pred_full_kspae', 'pred_loss_kspace']:
        fake_pred_img_complex = get_image_from_prediction(pred_flat, H, W)
        pred_img_complex = torch.fft.ifft2(fake_pred_img_complex)
        pred_real = pred_img_complex.real.detach().cpu().numpy()
        pred_imag = pred_img_complex.imag.detach().cpu().numpy()
        pred_mag = torch.abs(pred_img_complex).detach().cpu().numpy()

        plt.imsave(os.path.join(epoch_dir, "pred_real.png"), pred_real, cmap="gray")
        plt.imsave(os.path.join(epoch_dir, "pred_imag.png"), pred_imag, cmap="gray")
        plt.imsave(os.path.join(epoch_dir, "pred_mag.png"), pred_mag, cmap="gray")

        print(f"Saved epoch {epoch + 1} predictions in {epoch_dir}")

    else:
        raise ValueError("Unsupport Prediction_mode")


def save_best_image(model, sample, device, psnr, ssim, save_dir, supervision_mode="image"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    coords = sample['coords'].to(device)
    with torch.no_grad():
        pred_flat = model(coords)
    H, W = sample['gt_img'].shape
    if supervision_mode in ['kspace', 'kspace_csm', 'image']:
        pred_img_complex = get_image_from_prediction(pred_flat, H, W)
        pred_real = pred_img_complex.real.detach().cpu().numpy()
        pred_imag = pred_img_complex.imag.detach().cpu().numpy()
        pred_mag = torch.abs(pred_img_complex).detach().cpu().numpy()

        best_dir = save_dir
        plt.imsave(os.path.join(best_dir, "best_pred_real.png"), pred_real, cmap="gray")
        plt.imsave(os.path.join(best_dir, "best_pred_imag.png"), pred_imag, cmap="gray")
        plt.imsave(os.path.join(best_dir, "best_pred_mag.png"), pred_mag, cmap="gray")

    elif supervision_mode in ['pred_full_kspae', 'pred_loss_kspace']:
        fake_pred_img_complex = get_image_from_prediction(pred_flat, H, W)
        pred_img_complex = torch.fft.ifft2(fake_pred_img_complex)
        pred_real = pred_img_complex.real.detach().cpu().numpy()
        pred_imag = pred_img_complex.imag.detach().cpu().numpy()
        pred_mag = torch.abs(pred_img_complex).detach().cpu().numpy()

        best_dir = save_dir
        plt.imsave(os.path.join(best_dir, "pred_real.png"), pred_real, cmap="gray")
        plt.imsave(os.path.join(best_dir, "pred_imag.png"), pred_imag, cmap="gray")
        plt.imsave(os.path.join(best_dir, "pred_mag.png"), pred_mag, cmap="gray")

    metrics_filepath = os.path.join(best_dir, "best_metrics.txt")
    with open(metrics_filepath, "w") as f:
        f.write(f"PSNR: {psnr:.2f}\nSSIM: {ssim:.4f}\n")
    #print(f"Saved best prediction images in {best_dir}")



def plot_loss_curve(train_loss_history, loss_curve_path):
    plt.figure()
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Saved loss curve to {loss_curve_path}")