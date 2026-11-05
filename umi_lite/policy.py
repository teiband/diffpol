"""Compact CNN-based Diffusion Policy (Chi et al., RSS 2023), sized for small GPUs/CPUs.

  image encoder : ResNet-18 with GroupNorm (or a ~0.3M-param 'small' CNN for CPU)
  noise model   : 1-D temporal conditional U-Net with FiLM conditioning
  diffusion     : DDPM training (100 steps, cosine schedule), DDIM sampling (default 10 steps)
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- normalizer
class Normalizer(nn.Module):
    """Per-dimension min/max -> [-1, 1]. Stored as buffers so it is saved with the model."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("scale", torch.ones(dim))
        self.register_buffer("offset", torch.zeros(dim))

    def fit(self, lo, hi):
        lo, hi = torch.as_tensor(lo).float(), torch.as_tensor(hi).float()
        rng = hi - lo
        ok = rng > 1e-6
        scale = torch.where(ok, 2.0 / rng.clamp(min=1e-6), torch.ones_like(rng))
        offset = torch.where(ok, -1.0 - scale * lo, -(hi + lo) / 2)
        self.scale.copy_(scale)
        self.offset.copy_(offset)

    def forward(self, x):
        return x * self.scale + self.offset

    def inverse(self, x):
        return (x - self.offset) / self.scale


# ----------------------------------------------------------------------------- vision
def resnet18_gn():
    import torchvision
    net = torchvision.models.resnet18(weights=None,
                                      norm_layer=lambda c: nn.GroupNorm(max(c // 16, 1), c))
    net.fc = nn.Identity()
    return net, 512


class SmallCNN(nn.Module):
    """Tiny encoder for CPU-only training; spatial-softmax keypoints keep positional info."""

    def __init__(self, n_kp=32):
        super().__init__()
        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.Mish())
        self.net = nn.Sequential(blk(3, 32, 2), blk(32, 64, 2), blk(64, 64, 1), blk(64, 128, 2),
                                 nn.Conv2d(128, n_kp, 1))
        self.out_dim = n_kp * 2

    def forward(self, x):
        h = self.net(x)                                   # B,K,H,W
        B, K, H, W = h.shape
        p = F.softmax(h.flatten(2), -1).view(B, K, H, W)
        ys = torch.linspace(-1, 1, H, device=x.device)
        xs = torch.linspace(-1, 1, W, device=x.device)
        ex = (p.sum(2) * xs).sum(-1)
        ey = (p.sum(3) * ys).sum(-1)
        return torch.cat([ex, ey], -1)


class ImageEncoder(nn.Module):
    def __init__(self, kind="resnet18", crop=88):
        super().__init__()
        if kind == "resnet18":
            self.net, self.out_dim = resnet18_gn()
        elif kind == "small":
            self.net = SmallCNN()
            self.out_dim = self.net.out_dim
        else:
            raise ValueError(kind)
        self.crop = crop

    def preprocess(self, rgb_uint8, train):
        """(N,H,W,3) uint8 -> (N,3,crop,crop) float, random crop + light colour jitter in training"""
        x = rgb_uint8.permute(0, 3, 1, 2).float() / 255.0
        N, _, H, W = x.shape
        c = self.crop
        if train:
            i = torch.randint(0, H - c + 1, (1,)).item()
            j = torch.randint(0, W - c + 1, (1,)).item()
            x = x[:, :, i:i + c, j:j + c]
            b = 1 + 0.2 * (torch.rand(N, 1, 1, 1, device=x.device) - 0.5)
            ctr = 1 + 0.2 * (torch.rand(N, 1, 1, 1, device=x.device) - 0.5)
            m = x.mean((1, 2, 3), keepdim=True)
            x = ((x - m) * ctr + m * b).clamp(0, 1)
        else:
            i, j = (H - c) // 2, (W - c) // 2
            x = x[:, :, i:i + c, j:j + c]
        return (x - 0.5) / 0.5

    def forward(self, rgb_uint8, train=False):
        return self.net(self.preprocess(rgb_uint8, train))


# ----------------------------------------------------------------------------- 1-D U-Net
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        e = t.float()[:, None] * f[None]
        return torch.cat([e.sin(), e.cos()], -1)


def conv_block(i, o, k, g):
    return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.GroupNorm(g, o), nn.Mish())


class CondResBlock(nn.Module):
    def __init__(self, i, o, cond_dim, k=5, g=8):
        super().__init__()
        self.b1, self.b2 = conv_block(i, o, k, g), conv_block(o, o, k, g)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * o))
        self.res = nn.Conv1d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x, c):
        h = self.b1(x)
        scale, bias = self.film(c).unsqueeze(-1).chunk(2, dim=1)
        h = self.b2(h * scale + bias)
        return h + self.res(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, action_dim, cond_dim, down_dims=(64, 128, 256), k=5, g=8, t_dim=64):
        super().__init__()
        self.t_enc = nn.Sequential(SinusoidalPosEmb(t_dim), nn.Linear(t_dim, 4 * t_dim), nn.Mish(),
                                   nn.Linear(4 * t_dim, t_dim))
        cd = t_dim + cond_dim
        dims = [action_dim, *down_dims]
        io = list(zip(dims[:-1], dims[1:]))
        self.down = nn.ModuleList()
        for n, (a, b) in enumerate(io):
            last = n == len(io) - 1
            self.down.append(nn.ModuleList([CondResBlock(a, b, cd, k, g), CondResBlock(b, b, cd, k, g),
                                            nn.Conv1d(b, b, 3, 2, 1) if not last else nn.Identity()]))
        m = down_dims[-1]
        self.mid = nn.ModuleList([CondResBlock(m, m, cd, k, g), CondResBlock(m, m, cd, k, g)])
        self.up = nn.ModuleList()
        for a, b in reversed(io[1:]):
            self.up.append(nn.ModuleList([CondResBlock(2 * b, a, cd, k, g), CondResBlock(a, a, cd, k, g),
                                          nn.ConvTranspose1d(a, a, 4, 2, 1)]))
        self.final = nn.Sequential(conv_block(down_dims[0], down_dims[0], k, g),
                                   nn.Conv1d(down_dims[0], action_dim, 1))

    def forward(self, x, t, cond):
        """x (B,T,A) noisy actions, t (B,) diffusion step, cond (B,C) -> (B,T,A) predicted noise"""
        x = x.transpose(1, 2)
        c = torch.cat([self.t_enc(t), cond], -1)
        skips = []
        for r1, r2, down in self.down:
            x = r2(r1(x, c), c)
            skips.append(x)
            x = down(x)
        for r in self.mid:
            x = r(x, c)
        for r1, r2, up in self.up:
            x = torch.cat([x, skips.pop()], 1)
            x = up(r2(r1(x, c), c))
        return self.final(x).transpose(1, 2)


# ----------------------------------------------------------------------------- diffusion
class NoiseScheduler(nn.Module):
    def __init__(self, n_train=100):
        super().__init__()
        s = 0.008
        t = torch.linspace(0, n_train, n_train + 1) / n_train
        ac = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
        self.register_buffer("ac", torch.cumprod(1 - betas, 0))
        self.n_train = n_train

    def add_noise(self, x0, noise, t):
        a = self.ac[t].view(-1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def ddim_sample(self, eps_fn, shape, n_steps, device):
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.n_train - 1, 0, n_steps, device=device).round().long()
        for i, t in enumerate(ts):
            a_t = self.ac[t]
            a_prev = self.ac[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=device)
            eps = eps_fn(x, t.expand(shape[0]))
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x


# ----------------------------------------------------------------------------- policy
class DiffusionPolicy(nn.Module):
    def __init__(self, lowdim_dim, action_dim, obs_horizon=2, pred_horizon=16, encoder="resnet18",
                 crop=88, down_dims=(64, 128, 256), n_train_steps=100, n_infer_steps=10):
        super().__init__()
        self.cfg = dict(lowdim_dim=lowdim_dim, action_dim=action_dim, obs_horizon=obs_horizon,
                        pred_horizon=pred_horizon, encoder=encoder, crop=crop,
                        down_dims=tuple(down_dims), n_train_steps=n_train_steps,
                        n_infer_steps=n_infer_steps)
        self.To, self.Tp, self.A = obs_horizon, pred_horizon, action_dim
        self.n_infer = n_infer_steps
        self.encoder = ImageEncoder(encoder, crop)
        self.norm_low = Normalizer(lowdim_dim)
        self.norm_act = Normalizer(action_dim)
        cond_dim = obs_horizon * (self.encoder.out_dim + lowdim_dim)
        self.unet = ConditionalUnet1D(action_dim, cond_dim, down_dims)
        self.sched = NoiseScheduler(n_train_steps)

    def condition(self, rgb, lowdim, train=False):
        B = rgb.shape[0]
        f = self.encoder(rgb.flatten(0, 1), train).view(B, self.To, -1)
        return torch.cat([f, self.norm_low(lowdim)], -1).flatten(1)

    def loss(self, batch):
        cond = self.condition(batch["rgb"], batch["lowdim"], train=True)
        x0 = self.norm_act(batch["action"])
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.sched.n_train, (x0.shape[0],), device=x0.device)
        pred = self.unet(self.sched.add_noise(x0, noise, t), t, cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def predict(self, rgb, lowdim):
        """rgb (B,To,H,W,3) uint8, lowdim (B,To,D) raw -> (B,Tp,A) raw actions"""
        cond = self.condition(rgb, lowdim)
        x = self.sched.ddim_sample(lambda x, t: self.unet(x, t, cond),
                                   (rgb.shape[0], self.Tp, self.A), self.n_infer, rgb.device)
        return self.norm_act.inverse(x)

    @staticmethod
    def load(path, device="cpu", use_ema=True):
        ck = torch.load(path, map_location=device, weights_only=False)
        p = DiffusionPolicy(**ck["cfg"])
        p.load_state_dict(ck["ema"] if use_ema and ck.get("ema") else ck["model"])
        return p.to(device).eval(), ck


class EMA:
    def __init__(self, model, max_decay=0.999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.max_decay, self.step = max_decay, 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.max_decay, (1 + self.step) / (10 + self.step))
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)
