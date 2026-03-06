import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torchvision.transforms import Resize
import matplotlib as mpl
#mpl.use('TkAgg')  # or whatever other backend that you want
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, ListedColormap, BoundaryNorm

#sys.path.append("/home/intern/user/jack/data/quick_nowcasting_folder/")
#from nowcasting.utils import rainfall_to_pixel
#from nowcasting.helpers.vis import get_cmap


############ Shut print()'s mouth up ok? ############
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


############ Radar Reflectivity Calculation #############
def pixel_to_dBZ(img):
    """

    Parameters
    ----------
    img : np.ndarray or float

    Returns
    -------

    """
    return img * 70.0 - 10.0

def pixel_to_dBZ_nonlinear(img):
    ashift=35.0
    afact=4.0
    atan_dBZ_min = -1.482
    atan_dBZ_max = 1.412
    tan_pix = np.tan((img*(atan_dBZ_max-atan_dBZ_min)) + atan_dBZ_min)
    return tan_pix*afact + ashift

def dBZ_to_pixel(dBZ_img):
    """

    Parameters
    ----------
    dBZ_img : np.ndarray

    Returns
    -------

    """
    return np.clip((dBZ_img + 10.0) / 70.0, a_min=0.0, a_max=1.0)

def dBZ_to_pixel_nonlinear(dBz_img):
    ashift=35.0
    afact=4.0
    atan_dBZ_min = -1.482
    atan_dBZ_max = 1.412
    atan_dBZ = np.arctan((dBz_img - ashift)/afact)
    return (atan_dBZ - atan_dBZ_min)/(atan_dBZ_max-atan_dBZ_min)

def linpix_to_nonlinpix(linpix_img):
    return dBZ_to_pixel_nonlinear(pixel_to_dBZ(linpix_img))

def nonlinpix_to_linpix(nonlin_img):
    return(dBZ_to_pixel(pixel_to_dBZ_nonlinear(nonlin_img)))

def pixel_to_rainfall(img, a=58.53, b=1.56, lin=True):
    """Convert the pixel values to real rainfall intensity

    Parameters
    ----------
    img : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    rainfall_intensity : np.ndarray
    """
    if lin:
        dBZ = pixel_to_dBZ(img)
    else:
        dBZ = pixel_to_dBZ_nonlinear(img)
    dBR = (dBZ - 10.0 * np.log10(a)) / b
    rainfall_intensity = np.power(10, dBR / 10.0)
    return rainfall_intensity

def rainfall_to_pixel(rainfall_intensity, a=58.53, b=1.56, lin=True):
    """Convert the rainfall intensity to pixel values

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    pixel_vals : np.ndarray
    """
    dBR = np.log10(rainfall_intensity) * 10.0
    # dBZ = 10b log(R) +10log(a)
    dBZ = dBR * b + 10.0 * np.log10(a)
    if lin:
        pixel_vals = dBZ_to_pixel(dBZ)
    else:
        pixel_vals = dBZ_to_pixel_nonlinear(dBZ)
    return pixel_vals

def dBZ_to_rainfall(dBZ, a=58.53, b=1.56):
    return np.power(10, (dBZ - 10 * np.log10(a))/(10*b))

def rainfall_to_dBZ(rainfall, a=58.53, b=1.56):
    return 10*np.log10(a) + 10*b*np.log10(rainfall)

def dBR_to_rainfall(dBR):
    return 10**(dBR/10)

def rainfall_to_dBR(rainfall):
    rainfall[rainfall < 1] = 1
    return np.nan_to_num(np.log10(rainfall) * 10.0, posinf=0, neginf=0)

def dBZ_normalize(dBZ):
    return np.clip(dBZ/60, a_min=0, a_max=1)

def rainfall_normalize(rainfall):
    return np.clip(rainfall/60, a_min=0, a_max=1)


####################### DL #######################
def warmup_lambda(warmup_steps, min_lr_ratio=0.1):
    def ret_lambda(epoch):
        if epoch <= warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * epoch / warmup_steps
        else:
            return 1.0
    return ret_lambda

def linear_warmup_cosine_decay_lr_scheduler(optimizer, peak_learning_rate=0.001, warmup_percentage=0.1, warmup_min_lr_ratio=0.05, min_lr_ratio=0.01, total_num_steps=None):

    if warmup_percentage > 0:
        print("using linear warmup...")
        print(f"warmup percentage: {warmup_percentage}%\n")
        warmup_iter = int(np.round(warmup_percentage * total_num_steps))
        warmup_scheduler = LambdaLR(optimizer,
                                    lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                            min_lr_ratio=warmup_min_lr_ratio))
        cosine_scheduler = CosineAnnealingLR(optimizer,
                                            T_max=(total_num_steps - warmup_iter),
                                            eta_min=min_lr_ratio * peak_learning_rate)
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                    milestones=[warmup_iter])
    else:
        lr_scheduler = CosineAnnealingLR(optimizer,
                                        T_max=total_num_steps,
                                        eta_min=min_lr_ratio * peak_learning_rate)
    return lr_scheduler

def tfpn_concat(y_actual, y_pred, threshold, return_verbose=False):
    if threshold > 1:
        threshold /= 255.
    t = torch.where(y_actual < threshold, 0, 1)
    p = torch.where(y_pred < threshold, 0, 1)
    
    #### single-batch version ####
    hit = torch.sum((t * p),axis=(1,2))
    miss = torch.sum((t * (1-p)),axis=(1,2))
    false_alarm = torch.sum(((1-t) * p),axis=(1,2))
    corr_reject = torch.sum(((1-t) * (1-p)),axis=(1,2))
    
    all_in_one = torch.stack((hit, miss, false_alarm, corr_reject), dim=1)              # (frames, 4)
    # print(f"shape of all_in_one: {all_in_one.shape}")

    if return_verbose:
        return hit, miss, false_alarm, corr_reject#, all_in_one
    else:
        return all_in_one

def tfpn_concat_frame(y_actual, y_pred, threshold, return_verbose=False):
    if threshold > 1:
        threshold /= 255.
    t = torch.where(y_actual < threshold, 0, 1)
    p = torch.where(y_pred < threshold, 0, 1)
    
    #### single-batch version ####
    hit = torch.sum((t * p))
    miss = torch.sum((t * (1-p)))
    false_alarm = torch.sum(((1-t) * p))
    corr_reject = torch.sum(((1-t) * (1-p)))
    
    all_in_one = torch.stack((hit, miss, false_alarm, corr_reject))              # (frames, 4)
    # print(f"shape of all_in_one: {all_in_one.shape}")

    if return_verbose:
        return hit, miss, false_alarm, corr_reject#, all_in_one
    else:
        return all_in_one

class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=0.00005, 
                 balancing_weights = (1, 1, 2, 5, 10, 10, 10, 30, 30, 30),
                 rainfall_thresholds = [0.5, 2, 5, 10, 15, 20, 30, 50, 100]):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.balancing_weight = balancing_weights
        self.rainfall_threshold = rainfall_thresholds
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight

    def forward(self, input, target, mask=None):            # input: prediction | target: ground truth

        balancing_weights = self.balancing_weight

        weights = torch.ones_like(input) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in np.array(self.rainfall_threshold)]

        # print(f"device of input: {input.device}")
        # print(f"device of target: {target.device}")
        # print(f"device of balancing_weights: {balancing_weights.device}")
        # print(f"device of weights: {weights.device}")
        # print(f"device of thresholds: {thresholds.device}")

        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        
        if mask is not None:
            weights = weights * mask.float()
        
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum(weights * ((input-target)**2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input-target))), (2, 3, 4))

        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight*torch.mean(mse) + self.mae_weight*torch.mean(mae))


def plot_radar_sequence(cond=None,
                        target=None,
                        pred=None,
                        save_path=None,
                        one_for_each=False):
    '''
    Input shape for all cond, target and pred
    => (B, T, C, H, W)
    '''
    
    #dBz
    levels = [
    -32768,
    10, 15, 20, 24, 28, 32,
    34, 38, 41, 44, 47, 50,
    53, 56, 58, 60, 62
    ]
    levels = dBZ_to_pixel(np.array(levels))*255.
    cmap = ListedColormap([
        '#FFFFFF00', '#08C5F5', '#0091F3', '#3898FF', '#008243', '#00A433',
        '#00D100', '#01F508', '#77FF00', '#E0D100', '#FFDC01', '#EEB200',
        '#F08100', '#F00101', '#E20200', '#B40466', '#ED02F0'
    ])

    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    vmin = vmax = None
    
    if not one_for_each:
        rows = 3
        cond_len = len(np.squeeze(cond.float().detach().cpu().numpy()[0]))
        target_len = len(np.squeeze(target.float().detach().cpu().numpy()[0]))
        frames = cond_len * 2
        fig, axs = plt.subplots(rows, frames, figsize=(20, 6))
        plot_stride = (target_len // cond_len) // 2
        
        ## Input ##
        for i in range(frames):
            if i < cond_len:
                axs[0,i].imshow(np.squeeze(cond.float().detach().cpu().numpy()[0])[i]*255,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
            else:
                axs[0,i].axis('off')
            axs[0, i].set_xticks([])
            axs[0, i].set_yticks([])
    
        ## Target ##
        for frame in range(frames):
            if target_len // cond_len > 2:
                axs[1,frame].imshow(np.squeeze(target.float().detach().cpu().numpy()[0])[::plot_stride][frame]*255,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
            else:
                axs[1,frame].imshow(np.squeeze(target.float().detach().cpu().numpy()[0])[frame]*255,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
            axs[1, frame].set_xticks([])
            axs[1, frame].set_yticks([])
            
        ## Prediction ##
        for frame in range(frames):
            if target_len // cond_len > 2:
                axs[2,frame].imshow(np.squeeze(pred.float().detach().cpu().numpy()[0])[::plot_stride][frame]*255,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
            else:
                axs[2,frame].imshow(np.squeeze(pred.float().detach().cpu().numpy()[0])[frame]*255,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
            
            axs[2, frame].set_xticks([])
            axs[2, frame].set_yticks([])
        
        axs[0, 0].set_ylabel('Input', rotation=90, labelpad=30, va='center', fontsize=12)
        axs[1, 0].set_ylabel('Observation', rotation=90, labelpad=30, va='center', fontsize=12)
        axs[2, 0].set_ylabel('Prediction', rotation=90, labelpad=30, va='center', fontsize=12)
    
    else:
        rows = 1
        frames = 3
        fig, axs = plt.subplots(rows, frames, figsize=(20, 6))
        
        assert len(np.squeeze(cond.float().detach().cpu().numpy()[0]).shape) == 2
        axs[0].imshow(np.squeeze(cond.float().detach().cpu().numpy()[0])*255,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
        axs[1].imshow(np.squeeze(target.float().detach().cpu().numpy()[0])*255,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
        axs[2].imshow(np.squeeze(pred.float().detach().cpu().numpy()[0])*255,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
        
        axs[0].set_title('Input', fontsize=12)
        axs[1].set_title('Target', fontsize=12)
        axs[2].set_title('Prediction', fontsize=12)
        
        for i in range(3):
            axs[i].set_xticks([])
            axs[i].set_yticks([])
    
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def spatial_resize(input_frames=None, input_size=(480,480), output_size=(240,240)):
    """Using torch.transforms.Resize to do quick up/down sampling
    
    Inputs
    -----------------------------
    input_frames: pytorch tensor (fp32)
                  size: (B, T, C, H_in, W_in)
    input_size: tuple
                (H_in, W_in)
    output_size: tuple
                 (H_out, W_out)
    
    ========================================
    Return:
    -----------------------------
    output_frames: pytorch tensor (fp32)
                   size: (B, T, C, H_out, W_out)
    
    """
    assert len(input_frames.shape) == 5
    H_in, W_in = input_size[0], input_size[1]
    H_out, W_out = output_size[0], output_size[1]

    resize = Resize((H_out, W_out))
    output_frames= input_frames.view(-1, 1, H_in, W_in)
    output_frames = resize(output_frames)
    output_frames = output_frames.view(input_frames.shape[0], input_frames.shape[1], 1, H_out, W_out)

    return output_frames

def spatial_resize_frame(input_frames=None, input_size=(480,480), output_size=(240,240)):
    """Using torch.transforms.Resize to do quick up/down sampling
    
    Inputs
    -----------------------------
    input_frames: pytorch tensor (fp32)
                  size: (B, C, H_in, W_in)
    input_size: tuple
                (H_in, W_in)
    output_size: tuple
                 (H_out, W_out)
    
    ========================================
    Return:
    -----------------------------
    output_frames: pytorch tensor (fp32)
                   size: (B, C, H_out, W_out)
    
    """
    assert len(input_frames.shape) == 4
    H_in, W_in = input_size[0], input_size[1]
    H_out, W_out = output_size[0], output_size[1]

    resize = Resize((H_out, W_out))
    output_frames = resize(input_frames)

    return output_frames
