import os

import random
import argparse
from pathlib import Path
import json
import itertools
import time
import logging
import math

import elasticdeform
import accelerate
import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
import SimpleITK as sitk
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers import ControlNetModel
from diffusers.utils.torch_utils import is_compiled_module
from transformers import CLIPVisionModelWithProjection

from diffusers import StableDiffusionControlNetPipeline

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio

from scipy.ndimage import zoom
from skimage import transform

PSNR = PeakSignalNoiseRatio(data_range=1.0)
SSIM = StructuralSimilarityIndexMeasure(data_range=1.0)

print("PyTorch version:", torch.__version__)
print("Is CUDA available:", torch.cuda.is_available())
print("cuDNN version:", torch.backends.cudnn.version())
print("Is cuDNN enabled:", torch.backends.cudnn.enabled)

import wandb

logger = get_logger(__name__)

def log_validation(
    args, accelerator, weight_dtype, step, checkpoint_path
):
    weight_dtype = torch.float16
    logger.info("Running validation... ")

    val_dataset = MyDataset(args.val_data_json_file, size=args.resolution, displacement_rate=args.displacement_rate, random_y_squeeze_rate=args.random_y_squeeze_rate, random_rotation_degree=args.random_rotation_degree, random_shift_range=args.random_shift_range, image_root_path=args.data_root_path, clip_image_processor_path=args.image_encoder_path)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )

    controlnet = ControlNetModel.from_pretrained(os.path.join(checkpoint_path, "controlnet"))
    controlnet = controlnet.to(accelerator.device, dtype=weight_dtype)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(accelerator.device, dtype=weight_dtype)
    
    noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(os.path.join(checkpoint_path, 'image_encoder')).to(accelerator.device, dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet").to(accelerator.device, dtype=weight_dtype)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=weight_dtype,
        scheduler=noise_scheduler,
    )
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    val_psnr = 0.0
    val_ssim = 0.0

    val_psnr_cropped = 0.0
    val_ssim_cropped = 0.0

    for v_i, val_example in enumerate(tqdm(val_dataloader)):
        image = val_example["images"]
        control_image = val_example["control_images"]
        clip_image = val_example["clip_images"].to(accelerator.device, dtype=weight_dtype)
        with torch.no_grad():
            encoder_hidden_states = image_encoder(clip_image).last_hidden_state

        with torch.autocast("cuda"):
            result_image = pipeline(
                prompt_embeds=encoder_hidden_states,
                image=control_image,
                num_inference_steps=100,
                guidance_scale=0.0,
                generator=generator,
                output_type="np",
            ).images

        image = image * 0.5 + 0.5
        image = image.cpu().float()
        control_image = control_image * 0.5 + 0.5
        control_image = control_image.cpu().float()
        result_image = np.transpose(result_image, (0, 3, 1, 2))
        result_image = torch.from_numpy(result_image).float()
        val_psnr += PSNR(result_image, image)
        val_ssim += SSIM(result_image, image)

        for i in range(image.shape[0]):
            # Find non-zero pixel indices in the first channel
            non_zero_indices = torch.nonzero(control_image[i, 0])

            # Get bounding box of non-zero region
            min_row, min_col = torch.min(non_zero_indices, dim=0)[0]
            max_row, max_col = torch.max(non_zero_indices, dim=0)[0]
            result_image_cropped = result_image[i:i+1].clone()
            result_image_cropped = result_image_cropped[:, :, min_row:max_row+1, min_col:max_col+1]
            image_cropped = image[i:i+1, :, min_row:max_row+1, min_col:max_col+1]
            val_psnr_cropped += PSNR(result_image_cropped, image_cropped)
            val_ssim_cropped += SSIM(result_image_cropped, image_cropped)

    val_psnr /= len(val_dataloader)
    val_ssim /= len(val_dataloader)

    val_psnr_cropped /= len(val_dataset)
    val_ssim_cropped /= len(val_dataset)

    logger.info(f"Validation PSNR: {val_psnr}, Cropped: {val_psnr_cropped}")
    logger.info(f"Validation SSIM: {val_ssim}, Cropped: {val_ssim_cropped}")
    
    wandb.log({"val_psnr": val_psnr, "val_ssim": val_ssim, "step": step, "cropped_val_psnr": val_psnr_cropped, "cropped_val_ssim": val_ssim_cropped})

    logger.info("Run plot validation...")

    val_dataset = ValDataset(args.plot_data_json_file, size=args.resolution, image_root_path=args.data_root_path, clip_image_processor_path=args.image_encoder_path)
    fig, axs = plt.subplots(len(val_dataset), 8, figsize=(40, 5 * len(val_dataset)))

    for v_i, val_example in enumerate(tqdm(val_dataset)):
        image = val_example["image"]
        control_image = val_example["control_image"]
        gt_seg_image = val_example["gt_seg_image"]
        control_image = control_image.unsqueeze(0)
        clip_image = val_example["clip_image"].to(accelerator.device, dtype=weight_dtype)
        with torch.no_grad():
            encoder_hidden_states = image_encoder(clip_image).last_hidden_state

        with torch.autocast("cuda"):
            result_image = pipeline(
                prompt_embeds=encoder_hidden_states,
                image=control_image,
                num_inference_steps=100,
                guidance_scale=0.0,
                generator=generator,
                output_type="np",
            ).images[0]

        channel_names = val_example["channel_names"]
        # Plot context images and generated images for each channel
        for ch in range(3):
            vmax = max(image[:,:,ch].max(), result_image[:,:,ch].max())
            axs[v_i, ch].imshow(image[:,:,ch], cmap="gray", vmin=0, vmax=vmax)
            axs[v_i, ch].set_title("Context Image " + channel_names[ch])
            axs[v_i, ch].axis("off")

            axs[v_i, ch+4].imshow(result_image[:,:,ch], cmap="gray", vmin=0, vmax=vmax)
            axs[v_i, ch+4].set_title("Generated Image " + channel_names[ch])
            axs[v_i, ch+4].axis("off")

        axs[v_i, 3].imshow(control_image[0,0], cmap="gray", vmin=-1, vmax=1)
        axs[v_i, 3].set_title("Structure Image")
        axs[v_i, 3].axis("off")

        axs[v_i, 7].imshow(gt_seg_image, cmap="viridis", vmin=0, vmax=3)
        axs[v_i, 7].set_title("Ground Truth Segmentation")
        axs[v_i, 7].axis("off")

    os.makedirs(os.path.join(args.output_dir, "validation_plots"), exist_ok=True)
                 
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "validation_plots", f"validation_{step}.png"), dpi=300)
    plt.close()
    del pipeline
    del controlnet

    return val_psnr, val_ssim, val_psnr_cropped, val_ssim_cropped

class ValDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, size=512, image_root_path="", clip_image_processor_path=None):
        super().__init__()

        self.size = size
        self.image_root_path = image_root_path

        train_min_max_file = os.path.join(self.image_root_path, "train_min_max.json")
        with open(train_min_max_file, "r") as f:
            train_min_max = json.load(f)

        self.b50_min = train_min_max["B50"]["prostate_min"]
        self.b50_max = train_min_max["B50"]["prostate_max"]
        self.b400_min = train_min_max["B400"]["prostate_min"]
        self.b400_max = train_min_max["B400"]["prostate_max"]
        self.b800_min = train_min_max["B800"]["prostate_min"]
        self.b800_max = train_min_max["B800"]["prostate_max"]
        self.t2_min = train_min_max["T2"]["prostate_min"]
        self.t2_max = train_min_max["T2"]["prostate_max"]

        self.data = []
        with open(json_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        if clip_image_processor_path is None:
            self.clip_image_processor = CLIPImageProcessor()
        else:
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(clip_image_processor_path)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        image_file = item["image"]
        crop_coor = item["crop_coor"]

        b50_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.image_root_path, 'B50', image_file)))
        b400_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.image_root_path, 'B400', image_file)))
        b800_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.image_root_path, 'B800', image_file)))
        control_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.image_root_path, 'T2', image_file))).astype(np.float32)
        gt_seg_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.image_root_path, 'PZTZ', image_file)))

        # global min-max normalization
        b50_image = (b50_image - self.b50_min) / (self.b50_max - self.b50_min)
        b400_image = (b400_image - self.b400_min) / (self.b400_max - self.b400_min)
        b800_image = (b800_image - self.b800_min) / (self.b800_max - self.b800_min)

        # clip 0.0 ~ 1.0
        b50_image = np.clip(b50_image, 0, 1)
        b400_image = np.clip(b400_image, 0, 1)
        b800_image = np.clip(b800_image, 0, 1)
        
        image = np.stack([b50_image, b400_image, b800_image], axis=-1)

        channel_names = ["B50", "B400", "B800"]
        image = image[crop_coor['y_start']:crop_coor['y_start']+crop_coor['y_size'], crop_coor['x_start']:crop_coor['x_start']+crop_coor['x_size']]
        control_image = control_image[crop_coor['y_start']:crop_coor['y_start']+crop_coor['y_size'], crop_coor['x_start']:crop_coor['x_start']+crop_coor['x_size']]
        gt_seg_image = gt_seg_image[crop_coor['y_start']:crop_coor['y_start']+crop_coor['y_size'], crop_coor['x_start']:crop_coor['x_start']+crop_coor['x_size']]
        
        # Define target size
        target_size = (self.size, self.size)

        # Create an empty array of zeros with the target size
        image_padded = np.zeros((target_size[0], target_size[1], image.shape[2]), dtype=np.float32)
        control_image_padded = np.zeros((target_size[0], target_size[1]), dtype=np.float32)
        gt_seg_image_padded = np.zeros((target_size[0], target_size[1]), dtype=np.int32)
        # Calculate padding offsets
        y_offset = (target_size[0] - image.shape[0]) // 2
        x_offset = (target_size[1] - image.shape[1]) // 2
        # Place the cropped image in the center of the padded image
        image_padded[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1], :] = image
        control_image_padded[y_offset:y_offset+control_image.shape[0], x_offset:x_offset+control_image.shape[1]] = control_image
        gt_seg_image_padded[y_offset:y_offset+gt_seg_image.shape[0], x_offset:x_offset+gt_seg_image.shape[1]] = gt_seg_image

        # Apply min-max normalization to each channel
        control_image = (control_image_padded - control_image_padded.min()) / (control_image_padded.max() - control_image_padded.min())
        image = image_padded
        control_image = control_image[..., np.newaxis]
        control_image = self.transform(control_image)

        clip_image = self.clip_image_processor(images=image, return_tensors="pt", do_rescale=False).pixel_values

        return {
            "channel_names": channel_names,
            "image": image,
            "control_image": control_image,
            "clip_image": clip_image,
            "gt_seg_image": gt_seg_image_padded
        }
    def __len__(self):
        return len(self.data)

# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, size=512, displacement_rate=32, random_y_squeeze_rate=0.05, random_rotation_degree=5, random_shift_range=2, i_drop_rate=0.0, image_root_path="", clip_image_processor_path=None, isTrain=False):
        super().__init__()

        self.size = size
        self.displacement_rate = displacement_rate
        self.random_y_squeeze_rate = random_y_squeeze_rate
        self.random_rotation_degree = random_rotation_degree
        self.random_shift_range = random_shift_range
        self.i_drop_rate = i_drop_rate
        self.image_root_path = image_root_path
        self.isTrain = isTrain

        train_min_max_file = os.path.join(self.image_root_path, "train_min_max.json")
        with open(train_min_max_file, "r") as f:
            train_min_max = json.load(f)

        self.b50_min = train_min_max["B50"]["prostate_min"]
        self.b50_max = train_min_max["B50"]["prostate_max"]
        self.b400_min = train_min_max["B400"]["prostate_min"]
        self.b400_max = train_min_max["B400"]["prostate_max"]
        self.b800_min = train_min_max["B800"]["prostate_min"]
        self.b800_max = train_min_max["B800"]["prostate_max"]
        self.t2_min = train_min_max["T2"]["prostate_min"]
        self.t2_max = train_min_max["T2"]["prostate_max"]

        self.data = []
        with open(json_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        if clip_image_processor_path is None:
            self.clip_image_processor = CLIPImageProcessor()
        else:
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(clip_image_processor_path)
        
    def __getitem__(self, idx):
        item = self.data[idx]

        image_file = item["image"]
        distortion_pivot = item["distortion_pivot"]
        crop_coor = item["crop_coor"]
        
        # read image
        raw_b50_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.image_root_path, 'B50', image_file))).astype(np.float32)
        raw_b400_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.image_root_path, 'B400', image_file))).astype(np.float32)
        raw_b800_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.image_root_path, 'B800', image_file))).astype(np.float32)
        control_raw_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.image_root_path, 'T2', image_file))).astype(np.float32)
        gt_seg_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.image_root_path, 'PZTZ', image_file)))

        # global min-max normalization
        raw_b50_image = (raw_b50_image - self.b50_min) / (self.b50_max - self.b50_min)
        raw_b400_image = (raw_b400_image - self.b400_min) / (self.b400_max - self.b400_min)
        raw_b800_image = (raw_b800_image - self.b800_min) / (self.b800_max - self.b800_min)

        # clip 0.0 ~ 1.0
        raw_b50_image = np.clip(raw_b50_image, 0, 1)
        raw_b400_image = np.clip(raw_b400_image, 0, 1)
        raw_b800_image = np.clip(raw_b800_image, 0, 1)

        image = np.stack([raw_b50_image, raw_b400_image, raw_b800_image], axis=-1)
        image = image[crop_coor['y_start']:crop_coor['y_start']+crop_coor['y_size'], crop_coor['x_start']:crop_coor['x_start']+crop_coor['x_size']]
        control_image = control_raw_image[crop_coor['y_start']:crop_coor['y_start']+crop_coor['y_size'], crop_coor['x_start']:crop_coor['x_start']+crop_coor['x_size']]
        gt_seg_image = gt_seg_image[crop_coor['y_start']:crop_coor['y_start']+crop_coor['y_size'], crop_coor['x_start']:crop_coor['x_start']+crop_coor['x_size']]
        
        # Define target size
        target_size = (self.size, self.size)

        # Create an empty array of zeros with the target size
        image_padded = np.zeros((target_size[0], target_size[1], image.shape[2]), dtype=np.float32)
        control_image_padded = np.zeros((target_size[0], target_size[1]), dtype=np.float32)
        gt_seg_image_padded = np.zeros((target_size[0], target_size[1]), dtype=np.int32)
        # Calculate padding offsets
        y_offset = (target_size[0] - image.shape[0]) // 2
        x_offset = (target_size[1] - image.shape[1]) // 2
        # Place the cropped image in the center of the padded image
        image_padded[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1], :] = image
        control_image_padded[y_offset:y_offset+control_image.shape[0], x_offset:x_offset+control_image.shape[1]] = control_image
        gt_seg_image_padded[y_offset:y_offset+gt_seg_image.shape[0], x_offset:x_offset+gt_seg_image.shape[1]] = gt_seg_image

        control_image = (control_image_padded - control_image_padded.min()) / (control_image_padded.max() - control_image_padded.min())
        
        image = image_padded

        control_image = control_image[..., np.newaxis]
        image = self.transform(image)
        control_image = self.transform(control_image)
        gt_seg_image = torch.from_numpy(gt_seg_image_padded)

        if np.random.rand() > 0.5:
            displacement = self.get_random_displacement(distortion_pivot, raw_b50_image)
            b50_deformed = elasticdeform.deform_grid(raw_b50_image, displacement, order=3)
            b400_deformed = elasticdeform.deform_grid(raw_b400_image, displacement, order=3)
            b800_deformed = elasticdeform.deform_grid(raw_b800_image, displacement, order=3)
            upscaled_displacement = self.upscale_displacement(displacement, raw_b50_image.shape)
            jacobian_det = self.compute_jacobian(upscaled_displacement)

            b50_adjusted = self.adjust_pixel_values(raw_b50_image, b50_deformed, jacobian_det)
            b400_adjusted = self.adjust_pixel_values(raw_b400_image, b400_deformed, jacobian_det)
            b800_adjusted = self.adjust_pixel_values(raw_b800_image, b800_deformed, jacobian_det)

            image_deformed = np.stack([b50_adjusted, b400_adjusted, b800_adjusted], axis=-1)
            image_deformed = np.clip(image_deformed, 0, 1)

            image_deformed = self.random_squeeze_y_axis(image_deformed)
            image_deformed = self.random_rotation(image_deformed)
            x_shift = random.randint(-self.random_shift_range, self.random_shift_range)
            y_shift = random.randint(-self.random_shift_range, self.random_shift_range)
            image_deformed = np.roll(image_deformed, (x_shift, y_shift), axis=(1, 0))

        else:
            image_deformed = np.stack([raw_b50_image, raw_b400_image, raw_b800_image], axis=-1)
            
        image_deformed = image_deformed[crop_coor['y_start']:crop_coor['y_start']+crop_coor['y_size'], crop_coor['x_start']:crop_coor['x_start']+crop_coor['x_size']]
        image_deformed_padded = np.zeros((target_size[0], target_size[1], image_deformed.shape[2]), dtype=image_deformed.dtype)
        image_deformed_padded[y_offset:y_offset+image_deformed.shape[0], x_offset:x_offset+image_deformed.shape[1], :] = image_deformed

        image_deformed = image_deformed_padded

        clip_image = self.clip_image_processor(images=image_deformed, return_tensors="pt", do_rescale=False).pixel_values
        
        return {
            "image": image,
            "clip_image": clip_image,
            "control_image": control_image,
            "gt_seg_image": gt_seg_image,
        }

    def __len__(self):
        return len(self.data)

    def downsample_point(self, point, arr):
        original_size = np.array(arr.shape)
        target_size = original_size // self.displacement_rate
        return [int(round(p * t / o)) for p, o, t in zip(point, original_size, target_size)]
    
    def get_random_displacement(self, points, arr):
        points_downsampled = []
        for point in points:
            points_downsampled.append(self.downsample_point(point, arr))
        points_downsampled = np.array(points_downsampled)

        center_point = points_downsampled[4]

        min_x = points_downsampled[0][1]
        max_x = points_downsampled[2][1]
        min_y = points_downsampled[0][0]
        max_y = points_downsampled[6][0]

        displacement_size_y = arr.shape[0] // self.displacement_rate
        displacement_size_x = arr.shape[1] // self.displacement_rate
        displacement = np.zeros((2, displacement_size_y, displacement_size_x))
        for i in range(displacement_size_y):
            for j in range(displacement_size_x):
                if (i,j) == (center_point[0], center_point[1]):
                    displacement[0, i, j] = np.random.randn() * 10  # randn: expansion+compression, rand: expansion only
                    displacement[1, i, j] = np.random.randn() * 10
                elif (min_y < i < max_y) and (min_x < j < max_x):
                    displacement[0, i, j] = np.random.randn() * 10
                    displacement[1, i, j] = np.random.randn() * 10

        return displacement

    def upscale_displacement(self, disp, target_shape, order=3):
        factor = target_shape[0] / disp.shape[1]  # assumes square image and displacement grid
        upscaled_disp = np.zeros((2, *target_shape))
        upscaled_disp[0] = zoom(disp[0], factor, order=order)
        upscaled_disp[1] = zoom(disp[1], factor, order=order)
        return upscaled_disp

    def compute_jacobian(self, disp):
        # Compute spatial gradients of displacement field
        grad_y_x = np.gradient(disp[0], axis=1)  # ∂(y-disp)/∂x
        grad_y_y = np.gradient(disp[0], axis=0)  # ∂(y-disp)/∂y
        grad_x_x = np.gradient(disp[1], axis=1)  # ∂(x-disp)/∂x
        grad_x_y = np.gradient(disp[1], axis=0)  # ∂(x-disp)/∂y

        # Jacobian determinant: J = (1 + ∂y/∂y)(1 + ∂x/∂x) - (∂y/∂x)(∂x/∂y)
        jacobian = (1 + grad_y_y) * (1 + grad_x_x) - (grad_y_x * grad_x_y)
        return jacobian

    def adjust_pixel_values(self, original_image, deformed_image, jacobian_det):
        return deformed_image * jacobian_det
    
    def random_squeeze_y_axis(self, image):
        """
        Randomly scales the image along the Y-axis while keeping the X and Z axes unchanged.

        Parameters:
            image (numpy.ndarray): 3D image array with shape (height, width, channels).
            scale_range (tuple): Tuple with min and max scale factors for the Y-axis.

        Returns:
            numpy.ndarray: Scaled image.
        """
        # Ensure image is a 3D numpy array
        if image.ndim != 3:
            raise ValueError("Input image must be a 3D numpy array.")

        # Randomly choose a scale factor for the Y-axis
        scale_y = np.random.uniform(1-self.random_y_squeeze_rate, 1+self.random_y_squeeze_rate)
        
        # Define scale factors for X, Y, Z axes
        scale_factors = [scale_y, 1, 1]
        
        # Rescale the image
        image_rescaled = transform.rescale(image, scale_factors, mode='reflect', anti_aliasing=True)
        
        return image_rescaled

    def random_rotation(self, image):
        # Randomly choose an angle for rotation
        max_angle = self.random_rotation_degree
        angle = np.random.uniform(-max_angle, max_angle)
        
        # Rotate the image
        image_rotated = transform.rotate(image, angle, mode='reflect')
        
        return image_rotated


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    control_images = torch.stack([example["control_image"] for example in data])
    gt_seg_images = torch.stack([example["gt_seg_image"] for example in data])  

    return {
        "images": images,
        "clip_images": clip_images,
        "control_images": control_images,
        "gt_seg_images": gt_seg_images
    }
        
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--val_data_json_file",
        type=str,
        default=None,
        required=True,
        help="Validation data",
    )
    parser.add_argument(
        "--plot_data_json_file",
        type=str,
        default=None,
        required=True,
        help="Plot data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ipadapter-dec+controlnet",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--displacement_rate",
        type=int,
        default=32,
        help=(
            "The rate for displacement"
        ),
    )
    parser.add_argument(
        "--random_y_squeeze_rate",
        type=float,
        default=0.05,
        help=(
            "The rate for random y squeeze"
        ),
    )
    parser.add_argument(
        "--random_rotation_degree",
        type=int,
        default=5,
        help=(
            "The degree for random rotation"
        ),
    )
    parser.add_argument(
        "--random_shift_range",
        type=int,
        default=2,
        help=(
            "The range for random shift"
        ),
    )
    parser.add_argument(
        "--image_condition_drop_rate",
        type=float,
        default=0.1,
        help=(
            "The rate for image condition drop"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    wandb.init(project="DWIDistortion")

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", cross_attention_dim=1280, ignore_mismatched_sizes=True, low_cpu_mem_usage=False)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    unet.conv_out.requires_grad_(True)

    pretrained_state_dict = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    ).state_dict()

    # Enable gradients for randomly initialized layers
    # (layers whose shapes changed due to ignore_mismatched_sizes=True)
    random_initialized_layers = []
    for name, param in unet.named_parameters():
        if param.shape != pretrained_state_dict[name].shape:
            param.requires_grad = True
            random_initialized_layers.append(name)
    print("Random initialized layers: ", random_initialized_layers)
    # Free up memory by deleting pretrained_state_dict
    del pretrained_state_dict

    vae.requires_grad_(False)
    image_encoder.requires_grad_(True)

    controlnet = ControlNetModel.from_unet(unet, conditioning_channels=1)
    controlnet.train()
    unet.train()
    vae.eval()
    image_encoder.train()

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:

                models[0].save_pretrained(os.path.join(output_dir, "unet"))
                weights.pop()

                models[1].save_pretrained(os.path.join(output_dir, "controlnet"))
                weights.pop()

                models[2].save_pretrained(os.path.join(output_dir, "image_encoder"))
                weights.pop()

        def load_model_hook(models, input_dir):

            model = models.pop()
            load_image_encoder = CLIPVisionModelWithProjection.from_pretrained(os.path.join(input_dir, "image_encoder"))
            model.load_state_dict(load_image_encoder.state_dict())
            del load_image_encoder

            model = models.pop()
            load_controlnet = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
            model.register_to_config(**load_controlnet.config)
            model.load_state_dict(load_controlnet.state_dict())
            del load_controlnet

            model = models.pop()
            load_unet = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.load_state_dict(load_unet.state_dict())
            del load_unet
            
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    params_to_opt = itertools.chain(
    image_encoder.parameters(),
    unet.conv_out.parameters(),
    (param for name, param in unet.named_parameters() if name in random_initialized_layers),  # randomly initialized layers
    controlnet.parameters()
    )
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_json_file, size=args.resolution, displacement_rate=args.displacement_rate, random_y_squeeze_rate=args.random_y_squeeze_rate, random_rotation_degree=args.random_rotation_degree, random_shift_range=args.random_shift_range, image_root_path=args.data_root_path, clip_image_processor_path=args.image_encoder_path, isTrain=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Prepare everything with our `accelerator`.
    unet, controlnet, image_encoder, optimizer, train_dataloader = accelerator.prepare(unet, controlnet, image_encoder, optimizer, train_dataloader)
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    train_losses = []
    train_noise_losses = []
    val_psnrs = []
    val_ssims = []

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    for epoch in range(first_epoch, args.num_train_epochs):
        begin = time.perf_counter()

        epoch_train_loss = 0.0
        epoch_train_noise_loss = 0.0

        for step, batch in enumerate(train_dataloader):

            if accelerator.sync_gradients:
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-last")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        if args.val_data_json_file is not None and args.plot_data_json_file is not None:
                            val_psnr, val_ssim, val_psnr_cropped, val_ssim_cropped = log_validation(args, accelerator, weight_dtype, global_step, save_path)

                            if global_step!=0 and (val_psnr_cropped+val_ssim_cropped) > (max(val_psnrs)+max(val_ssims)):
                                save_path = os.path.join(args.output_dir, f"checkpoint-best-{global_step}")
                                accelerator.save_state(save_path)
                                logger.info(f"Saved state to {save_path}")

                            val_psnrs.append(float(val_psnr_cropped))
                            val_ssims.append(float(val_ssim_cropped))

            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(unet, controlnet, image_encoder):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long().to(accelerator.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(accelerator.device, dtype=weight_dtype)
                
                encoder_hidden_states = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).last_hidden_state
            
                controlnet_image = batch["control_images"].to(accelerator.device, dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                noise_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=noisy_latents.dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=noisy_latents.dtype),
                    return_dict=False,
                )[0]

                noise_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                timesteps[timesteps == 0] = 1

                loss = noise_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                avg_noise_loss = accelerator.gather(noise_loss.repeat(args.train_batch_size)).mean().item()

                # accumulate the average loss for each batch
                epoch_train_loss += avg_loss
                epoch_train_noise_loss += avg_noise_loss

                # Backward pass and optimization for main model
                optimizer.zero_grad()
                accelerator.backward(loss, retain_graph=True)
                optimizer.step()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"Loss: {avg_loss:.3f}")
                global_step += 1

            if global_step >= args.max_train_steps:
                break

            begin = time.perf_counter()
        epoch_train_loss /= len(train_dataloader)
        epoch_train_noise_loss /= len(train_dataloader)

        wandb.log({
            "train_loss": epoch_train_loss,
            "train_noise_loss": epoch_train_noise_loss,
            "step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        train_losses.append(epoch_train_loss)
        train_noise_losses.append(epoch_train_noise_loss)

        losses_data = {
            "train_losses": train_losses,
            "train_noise_losses": train_noise_losses,
            "val_psnrs": val_psnrs,
            "val_ssims": val_ssims,
        }
        with open(os.path.join(args.output_dir, "losses.json"), "w") as f:
            json.dump(losses_data, f)

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        unet = unwrap_model(unet)
        unet.save_pretrained(os.path.join(save_path, "unet"))
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(os.path.join(save_path, "controlnet"))
        image_encoder = unwrap_model(image_encoder)
        image_encoder.save_pretrained(os.path.join(save_path, "image_encoder"))
        logger.info(f"Saved final model to {save_path}")

    accelerator.end_training()
                
if __name__ == "__main__":
    main()    