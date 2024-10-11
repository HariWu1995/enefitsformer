"""
Paper: ETSformer: Exponential Smoothing Transformers for Time-series Forecasting
Source: https://github.com/lucidrains/ETSformer-pytorch
"""
from typing import Callable, Any
from functools import partial
from collections import namedtuple

import math
import numpy as np

import torch
from torch import nn

import einops
from einops.layers.torch import Rearrange
from scipy.fftpack import next_fast_len


# Define constants
PI = math.pi
EPS = 1e-7
Intermediates = namedtuple('Intermediates', 'growth_latents, periodicity_latents, level_output')


# Define functions
def exists(val):
    return val is not None


def fourier_extrapolate(signal: torch.Tensor, start: int, end: int):
    device = signal.device
    fhat = torch.fft.fft(signal)
    fhat_len = fhat.shape[-1]
    time = torch.linspace(start, end-1, end-start, device=device, dtype=torch.complex64)
    freqs = torch.linspace(0, fhat_len-1, fhat_len, device=device, dtype=torch.complex64)
    res = fhat[..., None, :] * (1.j * 2 * PI * freqs[..., None, :] * time[..., :, None] / fhat_len).exp() / fhat_len
    return res.sum(dim=-1).real


def conv1d_fft(values, weights, v_dim = -2, w_dim = -1):
    # Algorithm 3 in paper

    N = values.shape[v_dim]
    M = weights.shape[w_dim]

    fast_len = next_fast_len(N+M-1)

    f_v = torch.fft.rfft(values, n=fast_len, dim=v_dim)
    f_w = torch.fft.rfft(weights, n=fast_len, dim=w_dim)

    f_vw = f_v * einops.rearrange(f_w.conj(), '... -> ... 1')
    out = torch.fft.irfft(f_vw, fast_len, dim=v_dim)
    out = out.roll(-1, dims=(v_dim,))

    indices = torch.arange(start=fast_len-N, end=fast_len, dtype=torch.long, device=values.device)
    out = out.index_select(v_dim, indices)
    return out


# Define sub-modules
def FeatureExtraction(time_features: int, model_dim: int, kernel_size: int = 3, dropout: float = 0.):
    return nn.Sequential(
        Rearrange('b n d -> b d n'),    # swap axes to apply conv on feature dimension
        nn.Conv1d(time_features, model_dim, kernel_size=kernel_size, padding=kernel_size//2),
        nn.Dropout(dropout),
        Rearrange('b d n -> b n d'),    # return to original shape
    )


def FeedForward(dim: int, mult: int = 4, dropout: float = 0.):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.Sigmoid(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim),
        nn.Dropout(dropout),
    )


class FeedForwardBlock(nn.Module):

    def __init__(self, *, dim, **kwargs):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.feedforward = FeedForward(dim, **kwargs)
        self.post_norm = nn.LayerNorm(dim)
        # self.batch_norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.feedforward(x) + x # residual skip-connection
        x = self.post_norm(x)
        # x = self.batch_norm(x.swapaxes(1,2)).swapaxes(1,2)
        return x


class MHESA(nn.Module):
    """
    Multi-Head Exponential Smoothing Attention
    """
    def __init__(self, *, dim: int, heads: int = 8, dropout: float = 0.):
        super().__init__()
        self.heads = heads
        self.init_state = nn.Parameter(torch.randn(heads, dim // heads))

        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.randn(heads))

        self.project_in = nn.Linear(dim, dim)
        self.project_out = nn.Linear(dim, dim)

    def naive_attention(self, x, weights):
        n, h = x.shape[-2], self.heads

        # in appendix A.1 - Algorithm 2
        arange = torch.arange(n, device = x.device)

        weights = einops.repeat(weights, '... l -> ... t l', t = n)
        indices = einops.repeat(arange, 'l -> h t l', h = h, t = n)

        indices = (indices - einops.rearrange(arange + 1, 't -> 1 t 1')) % n
        weights = weights.gather(-1, indices)
        weights = self.dropout(weights)

        # causal
        weights = weights.tril()

        # multiply
        output = torch.einsum('b h n d, h m n -> b h m d', x, weights)
        return output

    def forward(self, x, naive: bool = False):
        b, n, d = x.shape
        h, device = self.heads, x.device

        # linear-projection input
        x = self.project_in(x)

        # split-out heads
        x = einops.rearrange(x, 'b n (h d) -> b h n d', h=h)

        # temporal differentiation
        x = torch.cat([einops.repeat(self.init_state, 'h d -> b h 1 d', b=b), x], dim=-2)
        x = x[:, :, 1:] - x[:, :, :-1]

        # prepare exponential alpha
        alpha = self.alpha.sigmoid()
        alpha = einops.rearrange(alpha, 'h -> h 1')

        # arange == powers
        arange = torch.arange(n, device=device)
        weights = alpha * (1 - alpha) ** torch.flip(arange, dims=(0,))

        output = self.naive_attention(x, weights) if naive \
                      else conv1d_fft(x, weights)

        # get initial state contribution
        init_weight = (1 - alpha) ** (arange + 1)
        init_output = einops.rearrange(     init_weight, 'h n -> h n 1') \
                    * einops.rearrange(self.init_state , 'h d -> h 1 d')

        output = output + init_output

        # merge heads
        output = einops.rearrange(output, 'b h n d -> b n (h d)')
        return self.project_out(output)


class FrequencyAttention(nn.Module):
    
    def __init__(self, *, K: int = 4, dropout: float = 0.):
        super().__init__()
        self.K = K
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        freqs = torch.fft.rfft(x, dim=1)

        # get amplitudes
        amp = freqs.abs()
        amp = self.dropout(amp)

        # top-k amplitudes - for periodicity, branded as attention
        topk_amp, _ = amp.topk(k=self.K, dim=1, sorted=True)

        # mask-out all freqs with lower amplitudes than the lowest value of the top-k above
        topk_freqs = freqs.masked_fill(amp < topk_amp[:, -1:], 0.+0.j)

        # inverse fft
        x_hat = torch.fft.irfft(topk_freqs, n=x.shape[1], dim=1)
        return x_hat


class Level(nn.Module):

    def __init__(self, time_features: int, model_dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([0.]))
        self.to_growth = nn.Linear(model_dim, time_features)
        self.to_period = nn.Linear(model_dim, time_features)

    def forward(self, x, latent_growth, latent_period):
        # following equation in appendix A.2
        n, device = x.shape[1], x.device
        alpha = self.alpha.sigmoid()

        arange = torch.arange(n, device=device)
        powers = torch.flip(arange, dims=(0,))

        # es-Attention for raw time-series signal with periodicity terms (from Frequency Attention) subtracted out
        periodicity = self.to_period(latent_period)
        periodicity_attented_weights = alpha * (1 - alpha) ** powers
        periodicity_term = conv1d_fft(x - periodicity, periodicity_attented_weights)

        # auxiliary term
        growth = self.to_growth(latent_growth)
        growth_smoothing_weights = (1 - alpha) ** powers
        growth_term = conv1d_fft(growth, growth_smoothing_weights)

        return periodicity_term + growth_term


class LevelStack(nn.Module):

    def forward(self, x, num_steps_forecast):
        return einops.repeat(x[:, -1], 'b d -> b n d', n=num_steps_forecast)


class GrowthDampening(nn.Module):

    def __init__(self, dim, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.dampen_factor = nn.Parameter(torch.randn(heads))

    def forward(self, growth, *, num_steps_forecast):
        device, h = growth.device, self.heads
        dampen_factor = self.dampen_factor.sigmoid()

        # like level stack, it takes the last growth for forecasting
        last_growth = growth[:, -1]
        last_growth = einops.rearrange(last_growth, 'b l (h d) -> b l 1 h d', h=h)

        # prepare dampening factors per head and the powers
        dampen_factor = einops.rearrange(dampen_factor, 'h -> 1 1 1 h 1')
        powers = (torch.arange(num_steps_forecast, device=device) + 1)
        powers = einops.rearrange(powers, 'n -> 1 1 n 1 1')

        # following Eq(2) in the paper
        dampened_growth = last_growth * (dampen_factor ** powers).cumsum(dim=2)
        return einops.rearrange(dampened_growth, 'b l n h d -> b l n (h d)')


# Define model
class ETSFormer(nn.Module):

    def __init__(self, *, model_dim: int, time_features: int = 1, 
                        num_classes: int = 1, feature_weights = None, 
                          embed_dim: int = 10, embed_kernel_size: int = 3, 
                         num_layers: int = 2, num_heads: int = 8, 
                               topK: int = 4, dropout: float = 0., ):

        super().__init__()
        assert model_dim % num_heads == 0, \
            'model dimension must be divisible by number of heads'
        self.model_dim = model_dim
        self.time_features = time_features
        self.feature_weights = feature_weights
        if self.feature_weights is not None:
            self.feature_weights = torch.Tensor(self.feature_weights)
            assert self.feature_weights.shape[0] == time_features, \
                f"Shape of class weights (= {self.feature_weights.shape[0]}) must match number of features (= {time_features})"
            
        self.extractor = FeatureExtraction(self.time_features, model_dim, kernel_size=embed_kernel_size, dropout=dropout)
        self.encoders = nn.ModuleList([])

        for idx in range(num_layers):
            is_last_layer = idx == (num_layers - 1)
            self.encoders.append(
                nn.ModuleList([
                                            FrequencyAttention(K = topK, dropout = dropout),
                               MHESA(dim = model_dim, heads = num_heads, dropout = dropout),
                    FeedForwardBlock(dim = model_dim) if not is_last_layer else None,
                         Level(model_dim = model_dim, time_features = self.time_features),
                ])
            )

        self.growth_dampening_module = GrowthDampening(dim=model_dim, heads=num_heads)
        self.latents_to_time_features = nn.Linear(in_features=model_dim, out_features=self.time_features)
        self.level_stack = LevelStack()

    def forward(self, x, *, forecast_horizon: int = 0, return_latents: bool = False):

        is_single_feature = (x.ndim == 2)
        if is_single_feature:
            x = einops.rearrange(x, 'b n -> b n 1')
        z = self.extractor(x)

        latent_growths = []
        latent_periods = []

        for freq_attn, mhes_attn, ff_block, level in self.encoders:
            latent_period = freq_attn(z)
            z = z - latent_period

            latent_growth = mhes_attn(z)
            z = z - latent_growth

            if exists(ff_block):
                z = ff_block(z)

            x = level(x, latent_growth, latent_period)

            latent_growths.append(latent_growth)
            latent_periods.append(latent_period)

        latent_growths = torch.stack(latent_growths, dim=-2)
        latent_periods = torch.stack(latent_periods, dim=-2)

        latents = Intermediates(latent_growths, latent_periods, x)

        if forecast_horizon == 0:
            return latents

        # Decoder
        latent_periods = einops.rearrange(latent_periods, 'b n l d -> b l d n')
        extrapolated_periods = fourier_extrapolate(latent_periods, x.shape[1], x.shape[1] + forecast_horizon)
        extrapolated_periods = einops.rearrange(extrapolated_periods, 'b l d n -> b l n d')

        dampened_growths = self.growth_dampening_module(latent_growths, num_steps_forecast=forecast_horizon)
        level = self.level_stack(x, num_steps_forecast=forecast_horizon)

        aggregated_latents = dampened_growths.sum(dim=1) + extrapolated_periods.sum(dim=1)
        forecasted = level + self.latents_to_time_features(aggregated_latents)

        if is_single_feature:
            forecasted = einops.rearrange(forecasted, 'b n 1 -> b n')

        if return_latents:
            return forecasted, latents

        return forecasted


# Define wrapper
class MultiheadLayerNorm(nn.Module):

    def __init__(self, dim, heads = 1, eps: float = None):
        super().__init__()
        self.eps = eps if eps is not None else EPS
        self.g = nn.Parameter(torch.ones(heads, 1, dim))
        self.b = nn.Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        std = torch.var(x, dim=-1, keepdim=True, unbiased=False).sqrt()
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class ClassificationWrapper(nn.Module):

    def __init__(self, *, transformer: nn.Module, 
                          num_classes: int = 10, dropout: float = 0., 
                            num_heads: int = 16, dim_head: int = 32,
                    level_kernel_size: int = 3,
                   growth_kernel_size: int = 3,
                   period_kernel_size: int = 3, multilabel: bool = False):

        super().__init__()
        assert isinstance(transformer, ETSFormer)
        self.transformer = transformer
        time_features = transformer.time_features
        model_dim = transformer.model_dim
        inner_dim = dim_head * num_heads
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.queries = nn.Parameter(torch.randn(num_heads, dim_head))
        self.multilabel = multilabel

        self.growth_to_kv = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Conv1d(model_dim, inner_dim * 2, growth_kernel_size, bias = False, padding = growth_kernel_size // 2),
            Rearrange('... (kv h d) n -> ... (kv h) n d', kv = 2, h = num_heads),
            MultiheadLayerNorm(dim_head, heads = 2 * num_heads),
        )

        self.period_to_kv = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Conv1d(model_dim, inner_dim * 2, period_kernel_size, bias = False, padding = period_kernel_size // 2),
            Rearrange('... (kv h d) n -> ... (kv h) n d', kv = 2, h = num_heads),
            MultiheadLayerNorm(dim_head, heads = 2 * num_heads),
        )

        self.level_to_kv = nn.Sequential(
            Rearrange('b n t -> b t n'),
            nn.Conv1d(time_features, inner_dim * 2, level_kernel_size, bias = False, padding = level_kernel_size // 2),
            Rearrange('b (kv h d) n -> b (kv h) n d', kv = 2, h = num_heads),
            MultiheadLayerNorm(dim_head, heads = 2 * num_heads),
        )

        self.to_out = nn.Linear(inner_dim, model_dim)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, num_classes)
        )

    def forward(self, time_series, ignore_transformer: bool = False):

        if not ignore_transformer:
            time_series = self.transformer(time_series)

        latent_growths, latent_periods, output_levels = time_series
        latent_growths = latent_growths.mean(dim=-2)
        latent_periods = latent_periods.mean(dim=-2)

        # Convert
        output_levels  =  self.level_to_kv(output_levels)
        latent_growths = self.growth_to_kv(latent_growths)
        latent_periods = self.period_to_kv(latent_periods)

        # queries, key, values
        q = self.queries * self.scale
        kvs = torch.cat([latent_growths, latent_periods, output_levels], dim=-2)
        k, v = kvs.chunk(2, dim=1)

        # cross-attention pooling
        sim = torch.einsum('h d, b h j d -> b h j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b h j, b h j d -> b h d', attn, v)
        out = einops.rearrange(out, 'b ... -> b (...)')

        # project to logits
        out = self.to_out(out)
        out = self.to_logits(out)

        return out

