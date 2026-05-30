import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import math
from torch.nn.parameter import Parameter
from _quan_base import _Conv2dQ, Qmodes, _LinearQ, _ActQ


__all__ = ['Conv2dQ', 'LinearQ', 'ActQ',
           'LearnableBias', 'StatsWeightQuantizer',
           'StrictLsqRowQuantizer', 'StrictLsqFeatureQuantizer', 'StrictLsqImageQuantizer',
           'StrictLinearQ', 'StrictHeadQLinear', 'StrictConv2dQ']


class FunQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        # indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        indicate_middle = 1.0 - indicate_small - indicate_big  # Thanks to @haolibai
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
            -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        # The following operation can make sure that alpha is always greater than zero in any case and can also
        # suppress the update speed of alpha. (Personal understanding)
        # grad_alpha.clamp_(-alpha.item(), alpha.item())  # FYI
        return grad_weight, grad_alpha, None, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class Conv2dQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=8, mode=Qmodes.kernel_wise, **kwargs):
        super(Conv2dQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, mode=mode)
        self.act = ActQ(in_features=in_channels, nbits_a=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() * 2)
            self.init_state.fill_(1)
        """  
        Implementation according to paper. 
        Feels wrong ...
        When we initialize the alpha as a big number (e.g., self.weight.abs().max() * 2), 
        the clamp function can be skipped.
        Then we get w_q = w / alpha * alpha = w, and $\frac{\partial w_q}{\partial \alpha} = 0$
        As a result, I don't think the pseudo-code in the paper echoes the formula.
       
        Please see jupyter/STE_LSQ.ipynb fo detailed comparison.
        """
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.alpha, g).abs().clamp(min=1e-6)
        # print(alpha.shape)
        # print(self.weight.shape)
        alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        x = self.act(x)
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        # wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LinearQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, **kwargs):
        super(LinearQ, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, mode=Qmodes.kernel_wise)
        self.act = ActQ(in_features=in_features, nbits_a=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g).abs().clamp(min=1e-6)
        alpha = alpha.unsqueeze(1)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        x = self.act(x)
        # w = self.weight / alpha
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2:
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.linear(x, w_q, self.bias)


class ActQ(_ActQ):
    def __init__(self, in_features, nbits_a=4, mode=Qmodes.kernel_wise, **kwargs):
        super(ActQ, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
        # print(self.alpha.shape, self.zero_point.shape)
    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            # The init alpha for activation is very very important as the experimental results shows.
            # Please select a init_rate for activation.
            # self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * self.init_rate)
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                Qn = 0
                Qp = 2 ** self.nbits - 1
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.zero_point.data.copy_(self.zero_point.data * 0.9 + 0.1 * (torch.min(x.detach()) - self.alpha.data * Qn))
            self.init_state.fill_(1)

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
        alpha = grad_scale(self.alpha, g).abs().clamp(min=1e-6)
        zero_point = grad_scale(zero_point, g)
        # x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        if len(x.shape)==2:
            alpha = alpha.unsqueeze(0)
            zero_point = zero_point.unsqueeze(0)
        elif len(x.shape)==4:
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = round_pass((x / alpha + zero_point).clamp(Qn, Qp))
        x = (x - zero_point) * alpha

        return x


class LearnableBias(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_dim), requires_grad=True)

    def forward(self, x):
        return x + self.bias


def _strict_clip_eps(x):
    return torch.where(x > 1e-5, x, x.new_tensor(1e-5))


class StatsWeightQuantizer(nn.Module):
    """Strict symmetric weight quantizer that explicitly rounds to integer grid
    and dequantizes back via scale * int.
    """
    def __init__(self, num_bits):
        super().__init__()
        self.num_bits = num_bits
        self.register_buffer('last_scale', torch.tensor(0.))
        self.last_int = None

    def forward(self, weight):
        qmax = float(2 ** (self.num_bits - 1) - 1)
        qmin = -float(2 ** (self.num_bits - 1))
        if weight.ndim == 2:
            scale = 2 * weight.detach().abs().mean(dim=1, keepdim=True) / math.sqrt(qmax)
        elif weight.ndim == 4:
            scale = 2 * weight.detach().abs().mean(dim=(1, 2, 3), keepdim=True) / math.sqrt(qmax)
        else:
            raise ValueError(f"Unsupported weight shape for StatsWeightQuantizer: {tuple(weight.shape)}")
        scale = scale.detach().clamp(min=1e-5)
        w_int = round_pass((weight / scale).clamp(qmin, qmax))
        self.last_int = w_int.detach()
        self.last_scale = scale.detach()
        return w_int * scale


class _StrictLsqBase(nn.Module):
    def __init__(self, bit=4, all_positive=False, learnable=True):
        super().__init__()
        self.bit = bit
        self.all_positive = all_positive
        self.learnable = learnable
        self.register_parameter('s', None)
        self.initialized = False
        self.last_int = None
        self.register_buffer('last_scale', torch.tensor(0.))

        if all_positive:
            self.thd_neg = 0
            self.thd_pos = 1 if bit == 1 else 2 ** bit - 1
        else:
            self.thd_neg = -1 if bit == 1 else -2 ** (bit - 1)
            self.thd_pos = 1 if bit == 1 else 2 ** (bit - 1) - 1

    def _ensure_param(self, init_val):
        init_val = init_val.detach()
        if self.s is None:
            self.s = nn.Parameter(init_val.clone(), requires_grad=self.learnable)
        elif self.s.shape != init_val.shape:
            raise RuntimeError(f"Quantizer scale shape changed from {tuple(self.s.shape)} to {tuple(init_val.shape)}")
        if not self.initialized:
            self.s.data.copy_(init_val)
            self.initialized = True

    def _quant_dequant(self, x, scale, grad_scale_factor):
        scale = grad_scale(_strict_clip_eps(scale), grad_scale_factor)
        x_int = round_pass((x / scale).clamp(self.thd_neg, self.thd_pos))
        self.last_int = x_int.detach()
        self.last_scale = scale.detach()
        return x_int * scale


class StrictLsqRowQuantizer(_StrictLsqBase):
    """Row-wise LSQ-style fake quant, but with explicit int grid then dequant.
    For [B, N, C] => one scale per token row N; for [B, H, N, D] => one scale per N.
    """
    def forward(self, x):
        if x.ndim == 3:
            init_val = 2 * x.detach().abs().mean(dim=-1).mean(dim=0) / math.sqrt(self.thd_pos)
            self._ensure_param(init_val)
            scale = self.s.view(1, -1, 1)
            grad = 1.0 / math.sqrt(self.thd_pos * x.shape[0] * x.shape[-1])
            return self._quant_dequant(x, scale, grad)
        if x.ndim == 4:
            init_val = 2 * x.detach().abs().mean(dim=-1).mean(dim=0).mean(dim=0) / math.sqrt(self.thd_pos)
            self._ensure_param(init_val)
            scale = self.s.view(1, 1, -1, 1)
            grad = 1.0 / math.sqrt(self.thd_pos * x.shape[0] * x.shape[1] * x.shape[-1])
            return self._quant_dequant(x, scale, grad)
        raise ValueError(f"StrictLsqRowQuantizer only supports 3D/4D tensors, got {x.ndim}D")


class StrictLsqFeatureQuantizer(_StrictLsqBase):
    """Feature-wise LSQ-style fake quant with explicit int/dequant path.
    For [B, C] -> scale on C; [B, N, C] -> scale on C; [B, H, N, D] -> scale on D.
    """
    def forward(self, x):
        if x.ndim == 2:
            init_val = 2 * x.detach().abs().mean(dim=0) / math.sqrt(self.thd_pos)
            self._ensure_param(init_val)
            scale = self.s.view(1, -1)
            grad = 1.0 / math.sqrt(self.thd_pos * x.shape[0])
            return self._quant_dequant(x, scale, grad)
        if x.ndim == 3:
            init_val = 2 * x.detach().abs().mean(dim=0).mean(dim=0) / math.sqrt(self.thd_pos)
            self._ensure_param(init_val)
            scale = self.s.view(1, 1, -1)
            grad = 1.0 / math.sqrt(self.thd_pos * x.shape[0] * x.shape[1])
            return self._quant_dequant(x, scale, grad)
        if x.ndim == 4:
            init_val = 2 * x.detach().abs().mean(dim=0).mean(dim=0).mean(dim=0) / math.sqrt(self.thd_pos)
            self._ensure_param(init_val)
            scale = self.s.view(1, 1, 1, -1)
            grad = 1.0 / math.sqrt(self.thd_pos * x.shape[0] * x.shape[1] * x.shape[2])
            return self._quant_dequant(x, scale, grad)
        raise ValueError(f"StrictLsqFeatureQuantizer only supports 2D/3D/4D tensors, got {x.ndim}D")


class StrictLsqImageQuantizer(_StrictLsqBase):
    """Image/channel-wise activation quantizer for patch embedding input."""
    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"StrictLsqImageQuantizer expects 4D tensor, got {x.ndim}D")
        init_val = 2 * x.detach().abs().mean(dim=-1).mean(dim=-1).mean(dim=0) / math.sqrt(self.thd_pos)
        self._ensure_param(init_val)
        scale = self.s.view(1, -1, 1, 1)
        grad = 1.0 / math.sqrt(self.thd_pos * x.shape[0] * x.shape[2] * x.shape[3])
        return self._quant_dequant(x, scale, grad)


class StrictLinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, nbits_a=4,
                 input_quant='row', use_learnable_bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_bits = nbits_w
        self.act_bits = nbits_a
        self.weight_quant = StatsWeightQuantizer(num_bits=nbits_w)
        self.input_quant = StrictLsqRowQuantizer(bit=nbits_a, all_positive=False, learnable=True) if input_quant == 'row' else \
            StrictLsqFeatureQuantizer(bit=nbits_a, all_positive=False, learnable=True)
        self.move_b4 = LearnableBias(in_features) if use_learnable_bias else nn.Identity()
        self.move_aft = LearnableBias(in_features) if use_learnable_bias else nn.Identity()

    def forward(self, x):
        w_q = self.weight_quant(self.weight)
        x = self.move_b4(x)
        x = self.input_quant(x)
        x = self.move_aft(x)
        return F.linear(x, w_q, self.bias)


class StrictHeadQLinear(StrictLinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=8, nbits_a=8):
        super().__init__(in_features, out_features, bias=bias, nbits_w=nbits_w, nbits_a=nbits_a,
                         input_quant='feature', use_learnable_bias=True)


class StrictConv2dQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=8, mode=Qmodes.kernel_wise, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                         nbits=nbits_w, mode=mode)
        self.weight_quant = StatsWeightQuantizer(num_bits=nbits_w)
        self.act = StrictLsqImageQuantizer(bit=nbits_w, all_positive=False, learnable=True)

    def forward(self, x):
        w_q = self.weight_quant(self.weight)
        x = self.act(x)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
