import torch


class NaNObserver(torch.overrides.TorchFunctionMode):
    def __init__(
        self,
        inf_only: bool = False,
        nan_only: bool = False,
        ignore_functions: list[str] = [],
    ):
        """For tracking intermediate tensors and check for NaNs/Infs.

        Example usage:
        with NaNObserver():
            output = model(input_ids, attention_mask)
        """
        super().__init__()
        self.ignore_functions = ignore_functions
        self.inf_only = inf_only
        self.nan_only = nan_only

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func.__name__ in self.ignore_functions:
            return func(*args, **kwargs)

        for idx, arg in enumerate(args):
            if (
                self.nan_only
                and isinstance(arg, torch.Tensor)
                and torch.isnan(arg).any()
            ):
                raise ValueError(
                    f"NaN detected in the argument {idx} of function {func.__name__}"
                )
            elif (
                self.inf_only
                and isinstance(arg, torch.Tensor)
                and torch.isinf(arg).any()
            ):
                raise ValueError(
                    f"Inf detected in the argument {idx} of function {func.__name__}"
                )

        for key, value in kwargs.items():
            if (
                self.nan_only
                and isinstance(value, torch.Tensor)
                and torch.isnan(value).any()
            ):
                raise ValueError(
                    f"NaN detected in the argument {key} of function {func.__name__}"
                )
            elif (
                self.inf_only
                and isinstance(value, torch.Tensor)
                and torch.isinf(value).any()
            ):
                raise ValueError(
                    f"Inf detected in the argument {key} of function {func.__name__}"
                )

        output = func(*args, **kwargs)

        if (
            self.nan_only
            and isinstance(output, torch.Tensor)
            and torch.isnan(output).any()
        ):
            raise ValueError(
                f"NaN detected in the output of function: {func.__name__}"
            )
        elif (
            self.inf_only
            and isinstance(output, torch.Tensor)
            and torch.isinf(output).any()
        ):
            raise ValueError(
                f"Inf detected in the output of function: {func.__name__}"
            )
        return output
