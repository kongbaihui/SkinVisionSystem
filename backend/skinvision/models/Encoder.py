import math
import importlib
import warnings
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat
from timm.layers import DropPath
from backend.skinvision.models.selective_scan import selective_scan_torch

selective_scan_fn = None
selective_scan_ref = None
try:
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore",
			message=r"`torch\.cuda\.amp\.custom_(fwd|bwd)\(args\.\.\.\)` is deprecated.*",
			category=FutureWarning,
			module=r"mamba_ssm\..*",
		)
		from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except Exception:
	pass

selective_scan_fn_v1 = None
selective_scan_ref_v1 = None
try:
	selective_scan_module = importlib.import_module("selective_scan")
	selective_scan_fn_v1 = selective_scan_module.selective_scan_fn
	selective_scan_ref_v1 = selective_scan_module.selective_scan_ref
except Exception:
	pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class PatchEmbed2D(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
		super().__init__()
		if isinstance(patch_size, int):
			patch_size = (patch_size, patch_size)
		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
		self.norm = norm_layer(embed_dim) if norm_layer is not None else None

	def forward(self, x):
		x = self.proj(x).permute(0, 2, 3, 1)
		if self.norm is not None:
			x = self.norm(x)
		return x


class PatchMerging2D(nn.Module):
	def __init__(self, dim, norm_layer=nn.LayerNorm):
		super().__init__()
		self.dim = dim
		self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
		self.norm = norm_layer(4 * dim)

	def forward(self, x):
		batch_size, height, width, channels = x.shape
		shape_fix = [-1, -1]
		if (width % 2 != 0) or (height % 2 != 0):
			print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
			shape_fix[0] = height // 2
			shape_fix[1] = width // 2

		x0 = x[:, 0::2, 0::2, :]
		x1 = x[:, 1::2, 0::2, :]
		x2 = x[:, 0::2, 1::2, :]
		x3 = x[:, 1::2, 1::2, :]

		if shape_fix[0] > 0:
			x0 = x0[:, : shape_fix[0], : shape_fix[1], :]
			x1 = x1[:, : shape_fix[0], : shape_fix[1], :]
			x2 = x2[:, : shape_fix[0], : shape_fix[1], :]
			x3 = x3[:, : shape_fix[0], : shape_fix[1], :]

		x = torch.cat([x0, x1, x2, x3], -1)
		x = x.view(batch_size, height // 2, width // 2, 4 * channels)
		x = self.norm(x)
		x = self.reduction(x)
		return x


class SS2D(nn.Module):
	def __init__(
		self,
		d_model,
		d_state=16,
		d_conv=3,
		expand=2,
		dt_rank="auto",
		dt_min=0.001,
		dt_max=0.1,
		dt_init="random",
		dt_scale=1.0,
		dt_init_floor=1e-4,
		dropout=0.0,
		conv_bias=True,
		bias=False,
		device=None,
		dtype=None,
		**kwargs,
	):
		factory_kwargs = {"device": device, "dtype": dtype}
		super().__init__()
		self.d_model = d_model
		self.d_state = d_state
		self.d_conv = d_conv
		self.expand = expand
		self.d_inner = int(self.expand * self.d_model)
		self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

		self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
		self.conv2d = nn.Conv2d(
			in_channels=self.d_inner,
			out_channels=self.d_inner,
			groups=self.d_inner,
			bias=conv_bias,
			kernel_size=d_conv,
			padding=(d_conv - 1) // 2,
			**factory_kwargs,
		)
		self.act = nn.SiLU()

		self.x_proj = (
			nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
			nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
			nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
			nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
		)
		self.x_proj_weight = nn.Parameter(torch.stack([tensor.weight for tensor in self.x_proj], dim=0))
		del self.x_proj

		self.dt_projs = (
			self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
			self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
			self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
			self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
		)
		self.dt_projs_weight = nn.Parameter(torch.stack([tensor.weight for tensor in self.dt_projs], dim=0))
		self.dt_projs_bias = nn.Parameter(torch.stack([tensor.bias for tensor in self.dt_projs], dim=0))
		del self.dt_projs

		self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
		self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

		self.forward_core = self.forward_corev0

		self.out_norm = nn.LayerNorm(self.d_inner)
		self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
		self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

	@staticmethod
	def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
		dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
		dt_init_std = dt_rank ** -0.5 * dt_scale
		if dt_init == "constant":
			nn.init.constant_(dt_proj.weight, dt_init_std)
		elif dt_init == "random":
			nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
		else:
			raise NotImplementedError

		dt = torch.exp(
			torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
			+ math.log(dt_min)
		).clamp(min=dt_init_floor)
		inv_dt = dt + torch.log(-torch.expm1(-dt))
		with torch.no_grad():
			dt_proj.bias.copy_(inv_dt)
		dt_proj.bias._no_reinit = True
		return dt_proj

	@staticmethod
	def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
		tensor = repeat(
			torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
			"n -> d n",
			d=d_inner,
		).contiguous()
		tensor = torch.log(tensor)
		if copies > 1:
			tensor = repeat(tensor, "d n -> r d n", r=copies)
			if merge:
				tensor = tensor.flatten(0, 1)
		tensor = nn.Parameter(tensor)
		tensor._no_weight_decay = True
		return tensor

	@staticmethod
	def D_init(d_inner, copies=1, device=None, merge=True):
		tensor = torch.ones(d_inner, device=device)
		if copies > 1:
			tensor = repeat(tensor, "n1 -> r n1", r=copies)
			if merge:
				tensor = tensor.flatten(0, 1)
		tensor = nn.Parameter(tensor)
		tensor._no_weight_decay = True
		return tensor

	def forward_corev0(self, x: torch.Tensor):
		# Decide from the input device, not merely whether a CUDA extension imports.
		self.selective_scan = selective_scan_torch
		if x.is_cuda and selective_scan_fn is not None:
			self.selective_scan = selective_scan_fn
		elif x.is_cuda and selective_scan_fn_v1 is not None:
			return self.forward_corev1(x)

		batch_size, channels, height, width = x.shape
		length = height * width
		groups = 4

		x_hwwh = torch.stack(
			[x.view(batch_size, -1, length), torch.transpose(x, dim0=2, dim1=3).contiguous().view(batch_size, -1, length)],
			dim=1,
		).view(batch_size, 2, -1, length)
		xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

		x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(batch_size, groups, -1, length), self.x_proj_weight)
		dts, bs_term, cs_term = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
		dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(batch_size, groups, -1, length), self.dt_projs_weight)

		xs = xs.float().view(batch_size, -1, length)
		dts = dts.contiguous().float().view(batch_size, -1, length)
		bs_term = bs_term.float().view(batch_size, groups, -1, length)
		cs_term = cs_term.float().view(batch_size, groups, -1, length)
		ds = self.Ds.float().view(-1)
		a_logs = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
		dt_projs_bias = self.dt_projs_bias.float().view(-1)

		out_y = self.selective_scan(
			xs,
			dts,
			a_logs,
			bs_term,
			cs_term,
			ds,
			z=None,
			delta_bias=dt_projs_bias,
			delta_softplus=True,
			return_last_state=False,
		).view(batch_size, groups, -1, length)
		inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(batch_size, 2, -1, length)
		wh_y = torch.transpose(out_y[:, 1].view(batch_size, -1, width, height), dim0=2, dim1=3).contiguous().view(batch_size, -1, length)
		invwh_y = torch.transpose(inv_y[:, 1].view(batch_size, -1, width, height), dim0=2, dim1=3).contiguous().view(batch_size, -1, length)
		return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

	def forward_corev1(self, x: torch.Tensor):
		if selective_scan_fn_v1 is None:
			raise ImportError("selective_scan_fn_v1 is unavailable. Please install selective_scan.")
		self.selective_scan = selective_scan_fn_v1

		batch_size, channels, height, width = x.shape
		length = height * width
		groups = 4

		x_hwwh = torch.stack(
			[x.view(batch_size, -1, length), torch.transpose(x, dim0=2, dim1=3).contiguous().view(batch_size, -1, length)],
			dim=1,
		).view(batch_size, 2, -1, length)
		xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

		x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(batch_size, groups, -1, length), self.x_proj_weight)
		dts, bs_term, cs_term = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
		dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(batch_size, groups, -1, length), self.dt_projs_weight)

		xs = xs.float().view(batch_size, -1, length)
		dts = dts.contiguous().float().view(batch_size, -1, length)
		bs_term = bs_term.float().view(batch_size, groups, -1, length)
		cs_term = cs_term.float().view(batch_size, groups, -1, length)
		ds = self.Ds.float().view(-1)
		a_logs = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
		dt_projs_bias = self.dt_projs_bias.float().view(-1)

		out_y = self.selective_scan(
			xs,
			dts,
			a_logs,
			bs_term,
			cs_term,
			ds,
			delta_bias=dt_projs_bias,
			delta_softplus=True,
		).view(batch_size, groups, -1, length)
		inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(batch_size, 2, -1, length)
		wh_y = torch.transpose(out_y[:, 1].view(batch_size, -1, width, height), dim0=2, dim1=3).contiguous().view(batch_size, -1, length)
		invwh_y = torch.transpose(inv_y[:, 1].view(batch_size, -1, width, height), dim0=2, dim1=3).contiguous().view(batch_size, -1, length)
		return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

	def forward(self, x: torch.Tensor, **kwargs):
		batch_size, height, width, channels = x.shape
		x = x.to(self.in_proj.weight.device)
		xz = self.in_proj(x)
		x, z = xz.chunk(2, dim=-1)

		x = x.permute(0, 3, 1, 2).contiguous()
		x = self.act(self.conv2d(x))
		y1, y2, y3, y4 = self.forward_core(x)
		y = y1 + y2 + y3 + y4
		y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(batch_size, height, width, -1)
		y = self.out_norm(y)
		y = y * F.silu(z)
		out = self.out_proj(y)
		if self.dropout is not None:
			out = self.dropout(out)
		return out


class MEB(nn.Module):
	def __init__(self, input_output_dim, d_state=16, d_conv=3, expand=2):
		super().__init__()
		self.input_output_dim = input_output_dim
		self.norm = nn.LayerNorm(input_output_dim)
		self.ss2d_1 = SS2D(d_model=input_output_dim // 4, d_state=d_state, d_conv=d_conv, expand=expand)
		self.ss2d_2 = SS2D(d_model=input_output_dim // 4, d_state=d_state, d_conv=d_conv, expand=expand)
		self.ss2d_3 = SS2D(d_model=input_output_dim // 4, d_state=d_state, d_conv=d_conv, expand=expand)
		self.ss2d_4 = SS2D(d_model=input_output_dim // 4, d_state=d_state, d_conv=d_conv, expand=expand)
		self.proj = nn.Linear(input_output_dim, input_output_dim)
		self.skip_scale = nn.Parameter(torch.ones(1))

	def forward(self, x):
		if x.dtype == torch.float16:
			x = x.type(torch.float32)
		batch_size, channels = x.shape[:2]
		assert channels == self.input_output_dim
		img_dims = x.shape[2:]
		x_flat = x.reshape(batch_size, *img_dims, channels)
		x_norm = self.norm(x_flat)
		x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=-1)
		x_ss2d1 = self.ss2d_1(x1) + self.skip_scale * x1
		x_ss2d2 = self.ss2d_2(x2) + self.skip_scale * x2
		x_ss2d3 = self.ss2d_3(x3) + self.skip_scale * x3
		x_ss2d4 = self.ss2d_4(x4) + self.skip_scale * x4
		x_ss2d = torch.cat([x_ss2d1, x_ss2d2, x_ss2d3, x_ss2d4], dim=-1)
		x_ss2d = self.norm(x_ss2d)
		x_ss2d = self.proj(x_ss2d)
		return x_ss2d.view(batch_size, self.input_output_dim, *img_dims)


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
	batch_size, height, width, num_channels = x.size()
	channels_per_group = num_channels // groups
	x = x.view(batch_size, height, width, groups, channels_per_group)
	x = torch.transpose(x, 3, 4).contiguous()
	return x.view(batch_size, height, width, -1)


class LLM(nn.Module):
	def __init__(
		self,
		input_dim: int,
		output_dim: int,
		drop_path: float = 0,
		norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
		attn_drop_rate: float = 0,
		d_state: int = 16,
		**kwargs,
	):
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.norm = nn.LayerNorm(input_dim)
		self.ln_1 = norm_layer(input_dim // 2)
		self.self_attention = MEB(input_output_dim=input_dim // 2, d_state=d_state, d_conv=3, expand=2)
		self.drop_path = DropPath(drop_path)
		self.proj = nn.Linear(input_dim, output_dim)
		self.conv33conv33conv11 = nn.Sequential(
			nn.BatchNorm2d(input_dim // 2),
			nn.Conv2d(in_channels=input_dim // 2, out_channels=input_dim // 2, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(input_dim // 2),
			nn.ReLU(),
			nn.Conv2d(in_channels=input_dim // 2, out_channels=input_dim // 2, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(input_dim // 2),
			nn.ReLU(),
			nn.Conv2d(in_channels=input_dim // 2, out_channels=input_dim // 2, kernel_size=1, stride=1),
			nn.ReLU(),
		)

	def forward(self, input: torch.Tensor):
		input = input.permute(0, 3, 1, 2).contiguous()
		batch_size, channels = input.shape[:2]
		img_dims = input.shape[2:]
		input = input.permute(0, 2, 3, 1).contiguous()
		input_left, input_right = input.chunk(2, dim=-1)
		x = self.ln_1(input_right)
		x = x.permute(0, 3, 1, 2).contiguous()
		x = self.self_attention(x)
		x = self.drop_path(x)
		input_left = input_left.permute(0, 3, 1, 2).contiguous()
		input_left = self.conv33conv33conv11(input_left)
		input_left = input_left.permute(0, 2, 3, 1).contiguous()
		x = x.permute(0, 2, 3, 1).contiguous()
		output = torch.cat((input_left, x), dim=-1)
		output = channel_shuffle(output, groups=2)
		out = output + input
		out = self.norm(out)
		out = self.proj(out)
		output_dim = self.output_dim * img_dims[0] * img_dims[1]
		out = out.view(batch_size, *img_dims, output_dim // (img_dims[0] * img_dims[1]))
		return out


class VSSLayer(nn.Module):
	def __init__(
		self,
		dim,
		depth,
		attn_drop=0.0,
		drop_path=0.0,
		norm_layer=nn.LayerNorm,
		downsample=None,
		use_checkpoint=False,
		d_state=16,
		**kwargs,
	):
		super().__init__()
		self.dim = dim
		self.use_checkpoint = use_checkpoint
		self.blocks = nn.ModuleList(
			[
				LLM(
					input_dim=dim,
					output_dim=dim,
					drop_path=drop_path[index] if isinstance(drop_path, list) else drop_path,
					norm_layer=norm_layer,
					attn_drop_rate=attn_drop,
					d_state=d_state,
				)
				for index in range(depth)
			]
		)
		self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

	def forward(self, x):
		for blk in self.blocks:
			if self.use_checkpoint:
				x = checkpoint.checkpoint(blk, x)
			else:
				x = blk(x)
		if self.downsample is not None:
			x = self.downsample(x)
		return x


class Encoder(nn.Module):
	def __init__(
		self,
		patch_size=4,
		in_chans=3,
		depths=[2, 2, 9],
		dims=[96, 192, 384, 768],
		d_state=16,
		drop_rate=0.0,
		attn_drop_rate=0.0,
		drop_path_rate=0.1,
		norm_layer=nn.LayerNorm,
		use_checkpoint=False,
	):
		super().__init__()
		self.num_layers = len(depths)
		if isinstance(dims, int):
			dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
		self.embed_dim = dims[0]
		self.num_features = dims[-1]
		self.dims = dims
		self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim, norm_layer=norm_layer)
		self.pos_drop = nn.Dropout(p=drop_rate)
		dpr = [value.item() for value in torch.linspace(0, drop_path_rate, sum(depths))]
		self.layers = nn.ModuleList()
		for i_layer in range(self.num_layers):
			layer = VSSLayer(
				dim=dims[i_layer],
				depth=depths[i_layer],
				d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
				drop=drop_rate,
				attn_drop=attn_drop_rate,
				drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
				norm_layer=norm_layer,
				downsample=PatchMerging2D,
				use_checkpoint=use_checkpoint,
			)
			self.layers.append(layer)

	def forward(self, x):
		skip_list = []
		x = self.patch_embed(x)
		x = self.pos_drop(x)
		for layer in self.layers:
			skip_list.append(x)
			x = layer(x)
		x1 = skip_list[0]
		x2 = skip_list[1]
		x3 = skip_list[2]
		x4 = x
		return x1, x2, x3, x4