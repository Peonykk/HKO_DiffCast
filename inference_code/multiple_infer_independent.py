import time
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from omegaconf import OmegaConf
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

from get_realtime_radar import quick_sample_realtime
file_basedir = os.path.dirname(__file__)

def int_list(arg):
    return list(map(int, arg.split(',')))
def str_list(arg):
    return arg.split(',')


## Command line arguments
#--------------------------------------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="1", help="gpu device id", type=str)
    parser.add_argument('--ddim_steps', default=250, help="smaller, faster but lower quality", type=int)
    parser.add_argument('--member_id', default=[0,1], help="member ids, length means number of forecasts to be gen using current gpu device", type=str_list)
    parser.add_argument('--test_datetime', default="202411201000", help="forecast basetime", type=str)
    parser.add_argument('--clip_single', default=1, help="1 will enable max-clipping", type=int)
    parser.add_argument('--clip_dbz', default=56, help="Clip max dbz to 56 if clip_single is enabled", type=int)
    parser.add_argument('--ckpt_path', required=True, help= 'Path of the checkpoint file to be used',type=str)
    parser.add_argument('--version_name', required=True, help='Name of the model using for inference', type=str)
    parser.add_argument('--freq', required=True, help='timestep between frames used in the model', type=str)
    parser.add_argument('--scale_factor', default= 1, help='scaling up the prediction reflectivity', type=float)
    
    return parser
    
parser = get_parser()
args = parser.parse_args()

gpu = args.gpu
clip_single_pred = args.clip_single
clip_dbz = args.clip_dbz
member_i_all = args.member_id

os.environ["CUDA_VISIBLE_DEVICES"] = gpu



## Load model weights
#--------------------------------------------------------------------------------------
import torch

load_dict = torch.load(args.ckpt_path, map_location='cpu')
model_state_dict = load_dict['model_state_dict']


device = torch.device("cuda:0")
ddim_steps = args.ddim_steps
freq = args.freq
scale_factor = args.scale_factor

print(f"loading model weights...")

from diffcast.models.diffcast import get_model
from diffcast.models.vmlstm_B import VMRNN
from diffcast.models.functions import *

deterministic_device = torch.device('cuda:0')
diffusion_device = torch.device('cuda:0')
ctxnet_device = torch.device('cuda:0')
deterministic_model = VMRNN(img_size=480, patch_size=8,
                             in_chans=1, embed_dim=512,
                             depths=[10], drop_rate=0,
                             attn_drop_rate=0, drop_path_rate=0.1)

Model = get_model(img_channels=1,
                  dim = 128,
                  dim_mults = (1,2,4,8,16),
                  T_in = 5, 
                  T_out = 10,
                  timesteps = 1000,           # number of steps
                  sampling_timesteps = ddim_steps,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
                  VSB_depth = [10],
                  diffusion_device = diffusion_device,
                  deterministic_device = deterministic_device,
                  ctxnet_device = ctxnet_device,
                  auto_normalize = False
                  )
Model.load_backbone(deterministic_model)
Model.load_state_dict(model_state_dict)
Model.to(device)
print(f"model weights loaded!!")



current_dt = pd.to_datetime(args.test_datetime, format='%Y%m%d%H%M')




with torch.no_grad():
    
    ## Get realtime input radar sequence
    #--------------------------------------------------------------------------------------
    valid_batch, sample_datetimes = quick_sample_realtime(base_time=current_dt, nonlin2lin=True)
    
    seq_t = valid_batch.transpose((1, 0, 2, 3, 4)) / 255. # => (B, T=5, 1, H, W)

    seq_in= seq_t[:, :5]
    seq_in = torch.Tensor(seq_in).to(device)

    base_output_dir = os.path.join(file_basedir, args.version_name, "output")
    label_datetimes = pd.date_range(start=sample_datetimes[0][-1] + pd.Timedelta(freq),
                                    periods=10,
                                    freq=freq)
    basetime = sample_datetimes[0][5-1]
    basetime_readable = basetime.strftime('%Y%m%d%H%M')
    
    
    ## Generate members' forecast in series
    #--------------------------------------------------------------------------------------
    for member_i in member_i_all:
    
        t0 = time.time()
    
        prob_ref_save_dir = os.path.join(base_output_dir, basetime_readable[0:4], basetime_readable[4:6], basetime_readable[6:8], basetime_readable[8:], f'member{member_i}')
        if not os.path.exists(prob_ref_save_dir):
            os.makedirs(prob_ref_save_dir, exist_ok=True)
        
        print(f"\nGenerating ensemble member {member_i}...")
        prob_p, deter_pred, residuals = Model.sample(frames_in=seq_in, T_out=10)
        prob_p, deter_pred = prob_p.squeeze(), deter_pred.squeeze()
        prob_p = prob_p * scale_factor
        
        ###### Clipping radar reflectivity ######
        if clip_single_pred==1:
            print(f"Using clipping with maxclip dBZ: {clip_dbz}")
            prob_p_tempout=torch.clamp(prob_p, min=0, max=(clip_dbz+10.0)/70.0)      # 56 => 0.9428
        else:
            prob_p_tempout=torch.clone(prob_p)
        
        ###### Saving radar reflectivity ######
        npy_save_path = prob_ref_save_dir + f'/temporary_member{member_i}'
        np.save(npy_save_path, prob_p_tempout.detach().cpu().numpy().astype(np.float32))
        

        t1=time.time()
        print(f"Member {member_i} inference time: {((t1-t0)/60):.1f}mins")

print("Done!!")







