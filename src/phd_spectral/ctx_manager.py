from torch import autograd
from torch.nn import functional as F


class OverrideConv2d(object):
    """Replaces only conv2d operation by the given function."""

    def __init__(self, new_function: autograd.Function):
        assert type(new_function) in [autograd.function.FunctionMeta, type(None)]
        self._fn = F.conv2d
        self._new_fn = new_function

    def __enter__(self):
        F.__dict__[self._fn.__name__] = (
            self._new_fn.conv2d if self._new_fn is not None else self._fn
        )

    def __exit__(self, *args):
        F.__dict__[self._fn.__name__] = self._fn
