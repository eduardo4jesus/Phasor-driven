import logging
from typing import Tuple, Union

import torch
from torch import Tensor, autograd
from torch.nn.modules.utils import _pair
from torch.nn.functional import conv2d

__torch_conv2d__ = conv2d


class Conv2dRFFTFunction(autograd.Function):
    @classmethod
    def _transform_input(cls, x: Tensor, size: Tuple[int]) -> Tensor:
        return torch.fft.rfftn(x, s=size)

    @classmethod
    def _transform_kernel(cls, x: Tensor, size: Tuple[int]) -> Tensor:
        return torch.fft.rfftn(x, s=size)

    @classmethod
    def _inv_transform(cls, x: Tensor, size: Tuple[int]) -> Tensor:
        return torch.fft.irfftn(x, s=size)

    @classmethod
    def _spectral_operation(cls, x: Tensor, w: Tensor) -> Tensor:
        a, b = x.real, x.imag
        c, d = w.real, w.imag

        return (a * c - b * d) + (b * c + a * d) * 1j

    @classmethod
    def conv2d(
        cls,
        input: Tensor,
        weight: Tensor,
        bias: Tensor = None,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
    ) -> Tensor:
        stride = stride if type(stride) is tuple else _pair(stride)
        padding = padding if type(padding) is tuple else _pair(padding)
        dilation = dilation if type(dilation) is tuple else _pair(dilation)

        # Restrictions of our model
        fit_constraints = (
            input.size(-1) == input.size(-2)
            and weight.size(-1) == weight.size(-2)
            and input.size(-1) >= weight.size(-1)
            and padding[0] == padding[1]
            and stride == _pair(1)
            and dilation == _pair(1)
            and groups == 1
        )

        method = cls.__name__ if fit_constraints else "built-in conv2d"
        # logging.debug(
        #     f"Applying {method}"
        #     f"("
        #     f"input_shape={input.shape} "
        #     f"weights_shape={weight.shape} "
        #     f"bias_shape={bias.shape if bias is not None else None} "
        #     f"stride={stride} "
        #     f"padding={padding} "
        #     f"dilation={dilation} "
        #     f"groups={groups}"
        #     ")"
        # )

        if not fit_constraints:
            return __torch_conv2d__(
                input,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
            )

        return cls.apply(input, weight, bias, stride, padding, dilation, groups)

    @classmethod
    def forward(
        cls,
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        stride: Union[int, Tuple[int]],
        padding: Union[int, Tuple[int]],
        dilation: Union[int, Tuple[int]],
        groups: int,
    ):
        # logging.debug(f"Function `forward` called from {cls.__name__}")
        stride = stride if type(stride) is tuple else _pair(stride)
        padding = padding if type(padding) is tuple else _pair(padding)
        dilation = dilation if type(dilation) is tuple else _pair(dilation)

        # Restrictions of our model
        fit_constraints = (
            input.size(-1) == input.size(-2)
            and weight.size(-1) == weight.size(-2)
            and input.size(-1) >= weight.size(-1)
            and padding[0] == padding[1]
            and stride == _pair(1)
            and dilation == _pair(1)
        )
        assert fit_constraints

        x = torch.nn.functional.pad(input, (*padding, *padding)).unsqueeze(1)
        w = weight.unsqueeze(0)

        size_transform = x.size()[-2:]  # Transform Size
        X = cls._transform_input(x, size=size_transform)
        del x

        # Given that Conv2d actually does correlation, we conjugate the
        # kernel values w. This is equivalent to rotate the kernel by 180
        # degrees in the spatial domain.
        W = cls._transform_kernel(w, size=size_transform).conj()
        del w

        Y = cls._spectral_operation(X, W).sum(2)
        del X, W

        if ctx:
            ctx.save_for_backward(input, weight, bias)
            ctx.size_transform = size_transform
            ctx.padding = padding

        """
            Y has shape (B, F, C, *size_transform)
            Using the linearity property as trick to reduce the number of inverse
            transforms. The accumulation of the activation is computed on the spectral
            domain instead of the spatial domain, thus reducing the ammount of
            computation done by the inverse transform.
            """
        y = cls._inv_transform(Y, size=size_transform)

        size_y = size_transform[0] - weight.size(-1) + 1
        y = y[..., :size_y, :size_y]
        del Y

        if bias is not None:
            # The commented line below should save some memory.
            # However, causes the following error:
            # Output 0 of Conv2dFFTFunctionBackward is a view and is being
            # modified inplace.
            # y += bias.reshape(-1, 1, 1)
            y = y + bias.reshape(-1, 1, 1)

        return y

    @classmethod
    def backward(cls, ctx, grad_output):
        # logging.debug(f"Function `backward` called from `{cls.__name__}`")
        input, weight, bias = ctx.saved_tensors
        size_transform = ctx.size_transform
        padding = ctx.padding

        D_OUTPUT = cls._transform_input(grad_output.unsqueeze(2), size=size_transform)

        d_input = d_w = d_bias = None

        if ctx.needs_input_grad[0]:  # derivative w.r.t input.
            # NOTE: The forward pass operates correlation instead of convolution.
            # The derivative of the correlation is a convolution and vice-versa.
            # Hence, here we DO NOT need to conjugate the one of the arguments in
            # the frequency domain.
            W = cls._transform_kernel(weight.unsqueeze(0), size=size_transform)
            D_INPUT = cls._spectral_operation(D_OUTPUT, W).sum(1)
            del W

            d_input = cls._inv_transform(D_INPUT, size=size_transform)

            assert padding[0] == padding[1]
            assert size_transform[0] == size_transform[1]
            i, j = (0 + padding[0]), (size_transform[0] - padding[0])
            d_input = d_input[..., i:j, i:j]

        if ctx.needs_input_grad[1]:  # derivative w.r.t weights.
            # NOTE: The forward pass operates correlation instead of convolution.
            # The derivative of the correlation w.r.t weights follows the same
            # operation as the forward pass. Hence, we need to conjugate the second
            # argument (kernel) for the convolution here, i.e the `input`.
            x = torch.nn.functional.pad(input, (*padding, *padding)).unsqueeze(1)
            X = cls._transform_kernel(x, size=size_transform)

            D_W = cls._spectral_operation(X, D_OUTPUT.conj()).sum(0)
            d_w = cls._inv_transform(D_W, size=size_transform)

            k = weight.size(-1)
            d_w = d_w[..., :k, :k]

        if bias is not None and ctx.needs_input_grad[2]:
            d_bias = grad_output.sum([0, 2, 3])

        # logging.debug(f"D_INPUT.shape: {d_input.shape}")
        # logging.debug(d_input)

        # logging.debug(f"d_w.shape: {d_w.shape}")
        # logging.debug(d_w)

        return d_input, d_w, d_bias, None, None, None, None


class Conv2dRFFTPhasorFunction(Conv2dRFFTFunction):
    @classmethod
    def _spectral_operation(cls, x: Tensor, w: Tensor):
        a, b = torch.abs(x), torch.angle(x)
        c, d = torch.abs(w), torch.angle(w)

        return a * c * torch.exp((b + d) * 1j)
