### Codes here are adapted from https://github.com/DeminYu98/DiffCast/blob/main/diffcast.py
import os
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

from diffcast.models.functions import *


# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# helpers functions
def exists(x): #check if the value exist
    return x is not None

def default(val, d):  #return d as a default value if val does not exist
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs): #data passage without modifications
    return t

def cycle(dl): #for creating infinite loop in a data loader for epochs
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num): # check if the number has integer square root
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor): #grouping numbers
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image): # convert image to a specific format
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions
def normalize_to_neg_one_to_one(img): # to centre around 0
    return img * 2 - 1

def unnormalize_to_zero_to_one(t): # convert back to the original value by previous function
    return (t + 1) * 0.5


# small helper modules
class Residual(nn.Module): # return fn(x)+x # input neural network  and implement the residual
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None): # scaling up pixels for higher resolution 
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None): # scaling down pixels for zooming out
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module): #normalization layer, root mean squared normalization
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1)) #learnable parameter g

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module): # Pre-normalization wrapper: apply normalization before running a fucntion, before applying a layer
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds # the time/position encoder
class SinusoidalPosEmb(nn.Module): # position encoding, create sinusoidal embeddings for perverving timesteps 
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1) # scaling factor, control how fast the frequency changes
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) # gives decreasing frequancies
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module): # flexible alternatives/ random position encoders, just an alternative
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random) # if is_random is false, weights can be learnt during training, if true, the weights are fixed

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8): # simple neural network to process weather data
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1) # 3*3 conv
        self.norm = nn.GroupNorm(groups, dim_out) # group normalization
        self.act = nn.SiLU() # signmoid activation

    def forward(self, x, scale_shift = None): #forward method
        x = self.proj(x) #convolution
        x = self.norm(x) #group normalization

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift #adjust with time information, for time/ condition modulation

        x = self.act(x) #activation
        return x

class ResnetBlock(nn.Module): # smarter processor, residual block with time embedding conditioning 
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential( #time embedding goes through MLP and split into scale and shift parameters
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module): #long range dependencies, linear for computation efficiency
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False) # QKV = Query, Key, Value

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2) # a value between 0-1 for weighting the attention value, is found with comparing real pictures
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v) # einsum operations for matrix multiplications

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out) 

class Attention(nn.Module): # standard attention module, like transformer: to give weight to features so important feature is traced
    def __init__(self, dim, heads = 4, dim_head = 32): # scaled dot-product attention
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out) #for small feature output
    
    
    
class TemporalAttention(nn.Module): # attention depend on choice of time, for long-range attention
    """A Temporal Attention block for Temporal Attention Unit"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.activation = nn.GELU()                          # GELU
        self.spatial_gating_unit = TemporalAttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x
    

class TemporalAttentionModule(nn.Module): #large kernel attention with squeeze-and-excitation
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False), # expansion
            nn.Sigmoid()
        )

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        f_x = self.conv1(attn)         # 1x1 conv
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        return se_atten * f_x * u
    
    
class ConvGRUCell(nn.Module): # for spatial-temporal memory: moving memory into next frame
    def __init__(self, input_dim, hidden_dim, kernel_size, n_layer=1):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super().__init__()
        self.padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.cur_states = [None for _ in range(n_layer)]
        self.n_layer = n_layer
        self.conv_gates = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=input_dim + hidden_dim if i == 0 else hidden_dim * 2,
                    out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
                    kernel_size=kernel_size,
                    padding=self.padding,
                )
                for i in range(n_layer)
            ]
        )

        self.conv_cans = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=input_dim + hidden_dim if i == 0 else hidden_dim * 2,
                    out_channels=self.hidden_dim,  # for candidate neural memory
                    kernel_size=kernel_size,
                    padding=self.padding,
                )
                for i in range(n_layer)
            ]
        )

    def init_hidden(self, batch_shape, device): #initialize hidden state to zeroes
        b, _, h, w = batch_shape
        for i in range(self.n_layer):
            self.cur_states[i] = torch.zeros((b, self.hidden_dim, h, w), device=device)

    def step_forward(self, input_tensor, index): # process one timestep through one ConvGRU layer
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        h_cur = self.cur_states[index]
        assert h_cur is not None
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates[index](combined)

        reset_gate, update_gate = torch.split(torch.sigmoid(combined_conv), self.hidden_dim, dim=1)
        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_cans[index](combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        self.cur_states[index] = h_next
        return h_next
    
    def forward(self, input_tensor): # process through all ConvGRU layers sequentially
        for i in range(self.n_layer):
            input_tensor = self.step_forward(input_tensor, i) # at layer 0 = input + h0, layer 1 = output of layer 0 + h1
        return input_tensor


# model
class ContextNet(nn.Module): # temporal encoder that process radar sequences ad create context features
    def __init__(
        self,
        dim,    # must be same as Unet
        dim_mults=(1, 2, 4, 8),     # must be same as Unet
        channels = 1,  ## change from 1 to number of features (5)
    ):
        super().__init__()
        self.channels = channels 
        self.dim = dim
        self.dim_mults = dim_mults
        # self.channels = T_in * (2 * input_channels)
        self.init_conv = nn.Conv2d(channels, dim, 7, padding = 3) # extract inital features from raw radar

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:])) # calculating dimensions at each level and output dimension pairs of each block
        
        self.downs = nn.ModuleList([])
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) -1 )
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_in), # processes spatial features
                        ConvGRUCell(dim_in, dim_in, 3, n_layer=1), # maintain tenporal memory at the feature
                        Downsample(dim_in, dim_out) if not is_last else nn.Identity() # reduce spatial resolution
                    ]
                )
            )

    def init_state(self, shape, device): # to reset all ConvGRU memories before processing a new sequence
        for i, ml in enumerate(self.downs): # different levels will have different resolutions pretain local and large scale details respectively
            temp_shape = list(shape)
            temp_shape[-2] //= 2 ** i
            temp_shape[-1] //= 2 ** i
            ml[1].init_hidden(temp_shape, device)
            
    def forward(self, x):
        x = self.init_conv(x)
        context = []
        for i, (resnet, conv, downsample) in enumerate(self.downs):
            x = resnet(x) # process spatial feature
            x = conv(x) # update temperal memory
            context.append(x) # save for later use
            x = downsample(x) # reduced resolution
        return context # list all four different feature maps at different scales
    
    def scan_ctx(self, frames): # process a sequence of frames through time
        b, t, c, h, w = frames.shape
        state_shape = (b, c, h, w)
        self.init_state(state_shape, frames.device)
        local_ctx = None
        globla_ctx = None
        
        for i in range(t): # process each frame sequentially
            globla_ctx = self.forward(frames[:,i])
            if i == 5:                                                  # in_len = 5 / 10 
                local_ctx = [h.clone() for h in globla_ctx] # save context at frame 5 to prevent GRU fading details
        return globla_ctx, local_ctx 
        
        
class Unet(nn.Module): # Unet is a generator that take noisy future frames and past frames, receives temporal guidance from ContextNet
    def __init__( # recognizing the diffusion step and output denoised and refined future frames
        self, # working in a 5-frame chunk that is autoregressive 
        dim,
        T_in,
        dim_mults=(1, 2, 4, 8),
        resnet_block_groups = 8,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16 
        # input_channels = 5 
    ):
        super().__init__()

        # determine dimensions
        self.channels = T_in * 2    # concat past 5 radars and noisy targets
        # self.channels = T_in * (2 * input_channels)                # T_in * 2 since we will concat past_radar and x_t, where x_t is noisy_target, both have shape (B,T,C,H,W)
        input_channels = self.channels

        init_dim = dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.frag_idx_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([]) # first read resnet block takes in dim_in*2 for U-Net features and context features
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out): # temporal guidance to to diffusion model at multiple scales
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in * 2, dim_in, time_emb_dim = time_dim * 2), # skipping connections saves features at mutliple processing stages
                block_klass(dim_in, dim_in, time_emb_dim = time_dim * 2),
                Residual(PreNorm(dim_in, TemporalAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1] # bottleneck: highest dimension
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim * 2)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim))) # standard attention
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim * 2)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)): # decoder
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim * 2),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim * 2),
                Residual(PreNorm(dim_out, TemporalAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))


        self.out_dim = T_in

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim * 2)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, cond=None, ctx=None, idx=None):
        
        # x: (b, t, c, h, w)       => noisy target radar (B, T=5, C=1, H=480, W=480)
        # cond: (b, t, c, h, w)    => past radar (B, T=5, C=1, H=480, W=480)
        # time: (b, )              => forward process timestamp t
        # ctx:                     => guidance info from ConvGRU
        # idx:                     => fragment index. Currently, input 5 past frames, output (5 + 5) pred frames autoregressively, so idx is either 0/1 (since 2 fragments)
        
        x = rearrange(x, 'b t c h w -> b (t c) h w')             # (B, T=5, C=1, H=480, W=480) => (B, C=5, H=480, W=480)
        if exists(cond):
            cond = rearrange(cond, 'b t c h w -> b (t c) h w')   # (B, T=5, C=1, H=480, W=480) => (B, C=5, H=480, W=480)
        
        cond = default(cond, lambda: torch.zeros_like(x))        # if cond not exist, then just create a zero tensor, and become unconditional (Not the case of us !!!)
        x = torch.cat((cond, x), dim = 1)                        # => (B, C=10, H=480, W=480)

        x = self.init_conv(x)                                    # (B, C=10, H=480, W=480) => (B, C=dim, H=480, W=480)
        r = x.clone()                                            # clone for skip connection later

        t = self.time_mlp(time)                                  # Embed forward process timestamp
        f_idx = self.frag_idx_mlp(idx)                           # Embed fragment index
        t = torch.cat((t, f_idx), dim = 1)

        h = []                                                   # To store intermediate state in self.downs for skip connection later
        
        ## Encoder
        for idx, (block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(torch.cat((x, ctx[idx]),dim=1), t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)
        
        ## Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        ## Decoder
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.out_dim)
        return x


# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        ctx_net,
        *,
        diffusion_device = 'cuda:3',
        ctxnet_device = 'cuda:1',
        deterministic_device = 'cuda:2',
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        VSB_depth = [4,4,4,4],       # VMRNN's model hyperparameter
        two_stage_training = False,
    ):
        super().__init__()
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.ctx_net = ctx_net
        self.VSB_depth = VSB_depth
        
        ### sharding ###
        self.diffusion_device = diffusion_device
        self.ctxnet_device = ctxnet_device
        self.deterministic_device = deterministic_device
        self.model.to(self.diffusion_device)
        self.ctx_net.to(torch.device(self.ctxnet_device))
        
        ### Two stage training? ###
        self.two_stage_training = two_stage_training


        #self.channels = self.model.channels
        
        
        ### Vanilla DDPM => pred_noise; Enhanced => pred_v
        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32).to(self.diffusion_device))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        ## offset noise strength - in blogpost, they claimed 0.1 was ideal
        self.offset_noise_strength = offset_noise_strength

        ## derive loss weight
        # snr - signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)

        ## Clipping signal-to-noise ratio => https://arxiv.org/abs/2303.09556
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)
        
        
        ## Vanilla DDPM using noise (epsilon) as training objective, it is later suggested to train using v here to achieve more stable training
        ## Phyiscal intuition of this v is the 'rate of change' of noise added, find more in "Progressive Distillation for Fast Sampling of Diffusion Models"
        ## https://arxiv.org/abs/2202.00512 
        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))
            

    @property
    def device(self):
        return self.betas.device
    
    def load_backbone(self, backbone_net):
        self.backbone_net = backbone_net
        self.backbone_net.to(self.deterministic_device)
        
        ## Freezing all parameters in deterministic model if you want to do two-stage training
        if self.two_stage_training:
            self.backbone_net.eval()
            for param in self.backbone_net.parameters():
                param.requires_grad = False
        else:
            self.backbone_net.train()
            
    
    #### The following variable names are straight forward, you can simply write the formula out if you find it hard to read
    #--------------------------------------------------------------------------------------------------------------------------------
    # beta ( b_t )                                            => beta at t according to your beta scheduling
    # alpha ( a_t )                                           => 1 - b_t
    # alphas_cumprod ( bar(a_t) )                             => a_1 * a_2 * a_3 * ... * a_t
    # alphas_cumprod_prev ( bar(a_{t-1}) )                    => a_1 * a_2 * a_3 * ... * a_{t-1}
    # sqrt_alphas_cumprod ( sqrt(bar(a_t)) )                  => sqrt( a_1 * a_2 * a_3 * ... * a_t )
    # sqrt_one_minus_alphas_cumprod ( sqrt(1-bar(a_t)) )      => obvious
    # sqrt_recip_alphas_cumprod ( sqrt(1/bar(a_t)) )          => obvious
    # sqrt_recipm1_alphas_cumprod ( sqrt( [1/bar(a_t)]-1 ) )  => obvious
    # log_one_minus_alphas_cumprod                            => Not used here
    
    # posterior_mean (mu_t in Inverse Process)
    #
    #
    
    
    ## Forward Process: x_t = sqrt(bar(a_t)) * x_0 + sqrt(1-bar(a_t)) * epsilon
    
    
    # Make 'x0' as subject in Forward Process formula
    # x_0 = sqrt( 1/bar(a_t) ) * x_t - sqrt( [1/bar(a_t)]-1 ) * epsilon
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # Make 'epsilon' as subject in Forward Process formula
    # epsilon = [sqrt(1/bar(a_t)) * x_t - x_0] / sqrt( [1/bar(a_t)]-1 )
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    # 
    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )
    
    # 
    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    #register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
    #register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
    #register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
    # b_t * sqrt(bar(a_t))
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, cond=None, ctx=None, idx=None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, cond=cond, ctx=ctx, idx=idx)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, cond=None, ctx=None, idx=None, clip_denoised = True):
        preds = self.model_predictions(x, t, cond=cond, ctx=ctx, idx=idx,)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, cond=None, ctx=None, idx=None,):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, cond=cond, ctx=ctx, idx=idx, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None, ctx=None, idx=None, return_all_timesteps = False):
        batch, device = shape[0], cond.device if cond is not None else self.device

        frames_pred = torch.randn(shape, device = device)
        imgs = [frames_pred]

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            frames_pred, _ = self.p_sample(frames_pred, t, cond=cond, ctx=ctx, idx=idx)
            imgs.append(frames_pred)

        ret = frames_pred if not return_all_timesteps else torch.stack(imgs, dim = 1)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, cond=None, ctx=None, idx=None, return_all_timesteps = False):
        batch, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        device = cond.device if cond is not None else self.device
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        frames_pred = torch.randn(shape, device = device)
        imgs = [frames_pred]

        disable_bar = True if sampling_timesteps < 500 else False
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', disable=disable_bar):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            
            pred_noise, x_start, *_ = self.model_predictions(frames_pred, time_cond, cond=cond, ctx=ctx, idx=idx, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                frames_pred = x_start
                imgs.append(frames_pred)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(frames_pred)

            frames_pred = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(frames_pred)

        ret = frames_pred if not return_all_timesteps else torch.stack(imgs, dim = 1)

        return ret

    @torch.no_grad()
    def sample(self, frames_in, T_out, return_all_timesteps = False, disable_pbar=False):
        B, T_in, c, h, w = frames_in.shape
        device = self.device
        
        
        d_out = model_forward_single_layer(self.backbone_net, frames_in.to(self.deterministic_device), T_out, self.VSB_depth)     # todo
        deterministic_out = torch.stack(d_out).permute(1, 0, 2, 3, 4).contiguous()                # todo
        backbone_output = deterministic_out[:, T_in - 1:].to(self.diffusion_device)
        

        global_ctx, local_ctx = self.ctx_net.scan_ctx(torch.cat((frames_in, backbone_output), dim=1).to(self.ctxnet_device))
        global_ctx = [g.to(self.diffusion_device) for g in global_ctx]
        local_ctx = [l.to(self.diffusion_device) for l in local_ctx]

        # frames_in = rearrange(frames_in, 'b t c h w -> b (t c) h w')
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        
        frames_pred = []
        ys = []
        
        pre_frag = frames_in
        pre_mu = None
        
        for frag_idx in tqdm(range(T_out // T_in), desc="sampling frags:", disable=disable_pbar):
            
            mu = backbone_output[:, frag_idx * T_in : (frag_idx + 1) * T_in]

            # Two strategies for channel condition
            # cond = pre_frag
            cond = pre_frag - pre_mu if  pre_mu is not None else torch.zeros_like(pre_frag)

            y = sample_fn(
                (B, T_in, c, h, w), 
                cond=cond, 
                ctx=global_ctx if frag_idx > 0 else local_ctx,
                idx=torch.full((B,), frag_idx, device = device, dtype = torch.long), 
                return_all_timesteps = return_all_timesteps
                )

            frag_pred = y + mu
            
            frames_pred.append(
                frag_pred  # if frag_idx > 0 else mu
                )
            ys.append(y)
            
            pre_frag = frag_pred
            pre_mu = mu
        
        
        frames_pred = torch.cat(frames_pred, dim=1)
        frames_pred = frames_pred.clamp(0,1)
        ys = torch.cat(ys, dim=1)
        
        return frames_pred, backbone_output, ys
    
    
    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        
    """
    def predict(self, frames_in,  compute_loss=False, **kwargs):
        T_out = default(kwargs.get('T_out'), 10)
        pred, mu, y = self.sample(frames_in=frames_in, T_out=T_out)
        if compute_loss:
            raise NotImplementedError("We are sorry that we do not support training process for now because of business limitation ")
        return pred, None
    """
    
    
    def p_losses(self, x_start, t, cond, noise=None, offset_noise_strength=None, ctx=None, idx=None, pixelwise_loss_weight=None):
        
        #print("stop 1")
        b, _, c, h, w = x_start.shape
        
        ## Generate noise
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        ## offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
        ## But here we didnt use it
        #--------------------------------------------------------------
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)
        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        
        ## Forward Process
        #--------------------------------------------------------------
        x_noisy = self.q_sample(x_start = x_start, t = t, noise = noise)
        
        
        
        ## Predicting noise / velocity
        #--------------------------------------------------------------
        # denoising network inputs: x, time, cond=None, ctx=None, idx=None
        model_out = self.model(x_noisy, t, cond=cond, ctx=ctx, idx=idx)
        
        
        
        '''This may enhance image quality, but we didnt use it here
        ######## Enable self-conditioning ########
        # if doing self-conditioning, 50% of chance, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.denoise_predictions(x, t).pred_x_start
                x_self_cond.detach_()
        # predict and take gradient step
        #model_out = self.model(x, t, x_self_cond)
        model_out = self.denoise(x, t, x_self_cond)
        '''
        
        ## We chose pred_v as objective
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        ## Loss
        ####### min_snr clipped  =>  https://arxiv.org/abs/2303.09556 #######
        p_loss = F.mse_loss(model_out, target, reduction = 'none')
        p_loss = reduce(p_loss, 'b ... -> b', 'mean')
        p_loss = p_loss * extract(self.loss_weight, t, p_loss.shape)
        p_loss = p_loss.mean()

        
        return model_out, p_loss

    
    def forward(self, frames_in, frames_gt, **kwargs):
    
        T_out = default(kwargs.get('T_out'), frames_gt.shape[1])                                                 # T_out=target_len
        B, T_in, c, h, w = frames_in.shape
        # device = self.device

        
        ## Deterministic Model
        #--------------------------------------------------------------
        # frames_in (past_radar) => (B, T=5, C=1, H=480, W=480)
        # Since the deterministic model used is RNN based, it is suggested to calc loss like LOSS{ concat(past_radar, target_radar)[1:], deterministic_out }
        #     => i.e. feeding 1st frame in past_radar => model output next frame => therefore model is also aims to learn to predict past_radar[1:5]
        #--------------------------------------------------------------
        d_out = model_forward_single_layer(self.backbone_net, frames_in.to(self.deterministic_device), T_out, self.VSB_depth)
        deterministic_out = torch.stack(d_out).permute(1, 0, 2, 3, 4).contiguous()              # (B, T=14, C=1, H=480, W=480)
        deterministic_output = deterministic_out[:, T_in - 1:].to(self.diffusion_device)        # (B, T=10, C=1, H=480, W=480)
        
        deterministic_loss = nn.MSELoss()(deterministic_out.to(self.diffusion_device), torch.cat((frames_in, frames_gt), 1)[:, 1:])
        

        ## ConvGRU to scan over concat(past_radar, deterministic_output)
        #--------------------------------------------------------------
        # The author didnt mention in his paper: ConvGRU here is so hard to capture long-term dependency (forgetting info in first few frames), 
        #       so he slice out the guidance info when ConvGRU scan upto input_len (return as local_ctx), 
        #       therefore you will see in next part, instead global_ctx (fullscan) of local_ctx (scan over input_frames) is used as ctx when frag_idx is 0
        #--------------------------------------------------------------
        global_ctx, local_ctx = self.ctx_net.scan_ctx(torch.cat((frames_in, deterministic_output), dim=1).to(self.ctxnet_device))
        global_ctx = [g.to(self.diffusion_device) for g in global_ctx]
        local_ctx = [l.to(self.diffusion_device) for l in local_ctx]

        
        full_res = torch.cat((torch.zeros_like(frames_in), (frames_gt - deterministic_output)), dim=1)
        
        t = torch.randint(0, self.num_timesteps, (B,), device=self.diffusion_device).long()
        
        
        diff_loss = 0.
        
        for frag_idx in range(T_out//T_in):
        
            res_cond = full_res[:, frag_idx*T_in:(frag_idx+1)*T_in]
            res_target = full_res[:, (frag_idx+1)*T_in:(frag_idx+2)*T_in]
            
            ## p_losses inputs: x_start (x_0), t, cond, noise=None, offset_noise_strength=None, ctx=None, idx=None
            ## Forward Process (x_0 -> x_t) will be done inside self.p_losses()
            epsilon, epsilon_loss = self.p_losses(x_start=res_target,
                                                  t=t, 
                                                  cond=res_cond, 
                                                  ctx=global_ctx if frag_idx > 0 else local_ctx,
                                                  idx=torch.full((B,), frag_idx, device=self.diffusion_device, dtype=torch.long),)
            diff_loss += epsilon_loss / (T_out//T_in)
            
        alpha = torch.tensor(0.5)
        loss = (1-alpha) * deterministic_loss + alpha * diff_loss
        
        return  deterministic_loss, diff_loss, loss
        

def get_model(
    img_channels=1,  ##change to default to number of features 
    dim = 64,
    dim_mults = (1,2,4,8),
    T_in = 5, 
    T_out = 20,
    timesteps = 1000, 
    sampling_timesteps = 250, 
    VSB_depth = [4,4,4,4],
    diffusion_device = 'cuda:3', 
    ctxnet_device = 'cuda:1', 
    deterministic_device = 'cuda:2', 
    two_stage_training = False,
    **kwargs
):
    
    unet = Unet(
        dim = dim,
        T_in=T_in,
        dim_mults = dim_mults
    )
    unet.to(diffusion_device)

    context_net = ContextNet(
        dim = dim,
        dim_mults=dim_mults,
        channels=img_channels,
    )
    context_net.to(ctxnet_device)
    
    diffusion = GaussianDiffusion(
        model = unet,
        ctx_net = context_net,
        timesteps = timesteps,
        sampling_timesteps = sampling_timesteps,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        offset_noise_strength = 0.,
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        diffusion_device = diffusion_device,
        ctxnet_device = ctxnet_device,
        deterministic_device = deterministic_device,
        VSB_depth = VSB_depth,
        two_stage_training = two_stage_training,
    )
        
    return diffusion