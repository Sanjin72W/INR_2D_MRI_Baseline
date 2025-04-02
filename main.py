import os
import yaml
import torch
import numpy as np

from model import Fullmodel
from train import train_epoch_image
from dataset import MRIDataset
from torch.utils.data import Subset, DataLoader
from visualize import save_epoch_results_as_png, save_best_image, plot_loss_curve

def main():
    with open("config.yaml","r") as f:
        config = yaml.safe_load(f)

    gpu_id = config['gpu_id']
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"Running on GPU {gpu_id}: {gpu_name}")
    else:
        device = torch.device('cpu')
        print("Running on CPU")

    train_dataset = MRIDataset(config["dataset_path"], split='train')
    one_image_subset = Subset(train_dataset, [0])
    train_loader = DataLoader(one_image_subset, batch_size=1, shuffle=False)

    model = Fullmodel(
        encoding_mode=config["encoder"]["encoding_mode"],
        in_features=config["encoder"]["in_features"],
        out_features=config["encoder"]["out_features"],
        coordinate_scales=config["encoder"]["coordinate_scales"],
        mlp_hidden_features=config["mlp"]["mlp_hidden_features"],
        mlp_hidden_layers=config["mlp"]["mlp_hidden_layers"],
        omega_0=config["mlp"]["omega_0"],
        activation=config["mlp"]["activation"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    train_loss_history = []
    train_psnr_history = []
    result_dir = "{}_{}_{}_B_{}_{}".format(
        config['prediction_mode'],
        config['mlp']['activation'],
        config['encoder']['encoding_mode'],
        "TV" if config["use_tv"] else "NoTV",
        "penalty" if config["use_penalty"] else "No_penalty",
        )

    print("Result dir:", result_dir)

    best_metrics_path = os.path.join(result_dir, "best_metrics.txt")

    if os.path.exists(best_metrics_path):
        try:
            with open(best_metrics_path, "r") as f:
                lines = f.readlines()
            best_psnr, best_ssim = 0, 0
            for line in lines:
                line = line.strip()
                if line.startswith("PSNR:"):
                    best_psnr = float(line.split("PSNR:")[-1].strip())
                elif line.startswith("SSIM:"):
                    best_ssim = float(line.split("SSIM:")[-1].strip())
        except Exception as e:
            print("读取best_metrics.txt出错，初始化best_psnr和best_ssim为0:", e)
            best_psnr, best_ssim = 0, 0
    else:
        best_psnr, best_ssim = 0, 0

    print("historial Best PSNR:", best_psnr, "Best SSIM:", best_ssim)

    print("Start Training!")
    for epoch in range(config["epochs"]):
        if config["supervision_mode"] in ['kspace', 'kspace_csm', 'image']:
            loss, psnr, ssim, nse = train_epoch_image(
                model, train_loader, optimizer, device,
                supervision_mode=config["supervision_mode"],
                lambda_tv=config["lambda_tv"]
            )
        else:
            raise ValueError("Unsupport Prediction_mode")

        train_loss_history.append(loss)
        train_psnr_history.append(psnr)
        sample = train_dataset[0]

        if (epoch + 1) % config.get("save_interval", 1000) == 0:
            save_epoch_results_as_png(model, sample, device, epoch, result_dir,
                                      supervision_mode=config["supervision_mode"])
            print(
                f"Epoch {epoch + 1}/{config['epochs']}: Loss={loss:.4e}, PSNR={psnr:.2f}, SSIM={ssim:.4f}, NSE={nse:.4f}")
        if psnr > best_psnr or ssim > best_ssim:
            best_psnr = max(best_psnr, psnr)
            best_ssim = max(best_ssim, ssim)
            save_best_image(model, sample, device, psnr, ssim, result_dir, supervision_mode=config["supervision_mode"])
        scheduler.step()

    loss_curve_path = os.path.join(result_dir, "loss_curve.png")
    plot_loss_curve(train_loss_history, loss_curve_path)

    np.savez(os.path.join(config['result_dir'], "train_psnr_history.npz"), train_psnr_history=train_psnr_history)

if __name__ == "__main__":
    main()
