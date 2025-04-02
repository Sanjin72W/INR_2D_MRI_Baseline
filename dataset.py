import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

def normalize01(img):
    """
    Normalize the image to the range [0, 1].
    """
    if not isinstance(img, np.ndarray):
        try:
            img = img.cpu().numpy()
        except AttributeError:
            img = np.array(img)

    if len(img.shape) == 3:
        nimg = img.shape[0]
    else:
        nimg = 1
        r, c = img.shape
        img = np.reshape(img, (nimg, r, c))

    img2 = np.empty(img.shape, dtype=np.float32)

    for i in range(nimg):
        min_val = img[i].min()
        max_val = img[i].max()
        if max_val > min_val:
            img2[i] = (img[i] - min_val) / (max_val - min_val)
        else:
            img2[i] = 0.0

    return np.squeeze(img2)

def vis_data(sample):
    real_part = np.real(sample)
    imag_part = np.imag(sample)
    magnitude = np.abs(sample)

    rows, cols = real_part.shape
    y_indices, x_indices = np.indices((rows, cols))  # y_indices, x_indices 均为 shape (256,232)

    x_flat = x_indices.flatten()
    y_flat = y_indices.flatten()
    real_flat = real_part.flatten()
    imag_flat = imag_part.flatten()
    mag_flat = magnitude.flatten()

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x_flat, y_flat, c=real_flat, cmap='viridis', marker='o', s=10)
    plt.title(f'Sample - Real Part')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    plt.colorbar(sc)
    plt.savefig("Real.png")

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x_flat, y_flat, c=imag_flat, cmap='viridis', marker='o', s=10)
    plt.title(f'Sample - Imaginary Part')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    plt.colorbar(sc)
    plt.show()
    plt.savefig("Imag.png")

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x_flat, y_flat, c=mag_flat, cmap='viridis', marker='o', s=10)
    plt.title(f'Sample - Magnitude')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    plt.colorbar(sc)
    plt.savefig("Mag.png")

class MRIDataset(Dataset):
    def __init__(self, file_path, split='train'):
        super(MRIDataset, self).__init__()
        self.file_path = file_path
        self.split = split
        self.data = h5py.File(file_path, 'r')
        self.org_data = self.data['trnOrg'][:]      #(N, H, W)      (360, 256, 232)
        self.mask_data = self.data['trnMask'][:]    #(N, H, W)      (360, 256, 232)
        self.csm_data = self.data['trnCsm'][:]      #(N, C, H, W)   (360, 12, 360, 232)
        self.num_samples = self.org_data[0]
        self.H = self.org_data.shape[1]
        self.W = self.org_data.shape[2]
        self.C = self.csm_data.shape[1]

        xs = np.linspace(-1, 1, self.W)
        ys = np.linspace(-1, 1, self.H)

        grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
        coords = np.stack([grid_x, grid_y], axis=-1)
        self.coords = torch.tensor(coords, dtype=torch.float32).view(-1, 2)

    def __getitem__(self, idx):
        org_np = self.org_data[idx]
        mask_np = self.mask_data[idx]
        csm_np = self.csm_data[idx]

        # vis_data(self.org_data[idx])

        org = torch.from_numpy(org_np)
        mask = torch.from_numpy(mask_np).float()
        csm = torch.from_numpy(csm_np).to(torch.complex64)

        if org.ndim == 2:
            org_expand = org.unsqueeze(0).repeat(csm.shape[0], 1, 1)

        full_kspace = torch.fft.fft2(org)
        masked_kspace = full_kspace * mask

        inverse_masked_kspace = full_kspace * (1 - mask)
        ###########
        plt.imshow(normalize01(np.abs(full_kspace)), cmap=plt.cm.gray, clim=(0.0, 0.8))
        plt.axis('off')
        plt.show()
        plt.savefig('full_kspace.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(normalize01(np.abs(masked_kspace)), cmap=plt.cm.gray, clim=(0.0, 0.8))
        plt.axis('off')
        plt.show()
        plt.savefig('masked_kspace.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(normalize01(np.abs(inverse_masked_kspace)), cmap=plt.cm.gray, clim=(0.0, 0.8))
        plt.axis('off')
        plt.show()
        plt.savefig('inverse_masked_kspace.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        csm_org = org_expand * csm

        full_csm_kspace = torch.fft.fft2(csm_org)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0).repeat(csm.shape[0], 1, 1)
        masked_csm_kspace = full_csm_kspace * mask

        # coil_images = torch.fft.ifft2(masked_csm_kspace) * csm

        # numerator = torch.sum(torch.conj(csm) * coil_images, dim=0)
        # denom = torch.sum(torch.abs(csm) ** 2, dim=0) + 1e-8  # 加上小常数避免除0
        # down_sampling_image = numerator / denom

        sample = {
            'coords': self.coords,
            'gt_full_kspace': full_kspace,
            'gt_loss_kspace': masked_kspace,
            'gt_full_csm_kspace': full_csm_kspace,
            'gt_loss_csm_kspace': masked_csm_kspace,
            'mask': mask,
            'gt_img': org,
            'gt_csm': csm,
        }
        return sample

    def __len__(self):
        return self.num_samples
