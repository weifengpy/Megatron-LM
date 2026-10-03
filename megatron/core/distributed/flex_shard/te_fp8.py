# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""TransformerEngine blockwise FP8 weights for FlexShard's FP8 parameter all-gather.

FlexShard keeps each weight's local shard in bf16. Its blockwise FP8 placement quantizes each
rank's 128-row block rows before the all-gather and builds the gathered weight from the FP8 data
and scales. With TransformerEngine's own weight quantizer for the blockwise recipe, the gathered
bytes and scales equal TransformerEngine quantizing the full bf16 weight, since a 128 x 128 block
never straddles ranks and the scales are padded only along their columns. TransformerEngine's
layers then use the gathered ``Float8BlockwiseQTensor`` as is, instead of quantizing the weight
themselves.
"""

from typing import Tuple

import torch

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
    from transformer_engine.pytorch.quantization import get_fp8_te_dtype
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
        Float8BlockwiseQTensor,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

BLOCK_SIZE = 128


def is_blockwise_fp8_weight(module: torch.nn.Module, name: str, param: torch.nn.Parameter) -> bool:
    """Whether ``module.<name>`` is a weight TransformerEngine quantizes with 128 x 128 blocks:
    the 2D weights of its linear layers (``weight``, or ``weight<i>`` of a grouped linear), with
    both dims multiples of 128."""
    return (
        isinstance(module, TransformerEngineBaseModule)
        and (name == "weight" or (name.startswith("weight") and name[len("weight") :].isdigit()))
        and param.dim() == 2
        and param.shape[0] % BLOCK_SIZE == 0
        and param.shape[1] % BLOCK_SIZE == 0
    )


class TEBlockwiseFp8Weights:
    """flex_shard's ``BlockwiseFp8Quantizer`` and ``BlockwiseFp8WeightFactory`` for
    TransformerEngine's blockwise recipe (``Float8BlockScaling``).

    ``quantize`` uses the recipe's weight quantizer, row-wise only, on a rank's block rows. The
    factory builds a ``Float8BlockwiseQTensor`` over the gathered buffers, without column-wise
    data: TransformerEngine derives it from the row-wise data when backward needs it, and
    ``drop_columnwise`` frees it once the weights change.
    """

    def __init__(self, recipe) -> None:
        qparams = recipe.fp8_quant_fwd_weight
        kwargs = dict(
            fp8_dtype=get_fp8_te_dtype(recipe, fprop_tensor=True),
            amax_epsilon=qparams.amax_epsilon,
            force_pow_2_scales=qparams.power_2_scale,
            block_scaling_dim=recipe.w_block_scaling_dim,
        )
        assert kwargs["block_scaling_dim"] == 2, (
            "FlexShard's FP8 parameter all-gather needs 2D (128 x 128) weight blocks, got "
            f"block_scaling_dim={kwargs['block_scaling_dim']}"
        )
        self._rowwise_quantizer = Float8BlockQuantizer(rowwise=True, columnwise=False, **kwargs)
        self.weight_quantizer = Float8BlockQuantizer(rowwise=True, columnwise=True, **kwargs)
        # A stable callable: placements compare weight factories by identity.
        self.weight_factory = self._weight

    def scale_cols(self, in_dim: int, block_size: int) -> int:
        """Scales per block row: TransformerEngine pads them to a multiple of 4."""
        assert block_size == BLOCK_SIZE, f"expected {BLOCK_SIZE}-row blocks, got {block_size}"
        return self._rowwise_quantizer.get_scale_shape((block_size, in_dim), columnwise=False)[1]

    def quantize(
        self, weight: torch.Tensor, block_size: int, fp8_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a rank's block rows as TransformerEngine quantizes the whole weight."""
        assert block_size == BLOCK_SIZE, f"expected {BLOCK_SIZE}-row blocks, got {block_size}"
        quantized = self._rowwise_quantizer(weight)
        return quantized._rowwise_data, quantized._rowwise_scale_inv

    def _weight(
        self,
        fp8_data: torch.Tensor,
        recip_scale: torch.Tensor,
        block_size: int,
        *,
        orig_dtype: torch.dtype,
        requires_grad: bool,
    ) -> torch.Tensor:
        return Float8BlockwiseQTensor(
            shape=fp8_data.shape,
            dtype=orig_dtype,
            fp8_dtype=self.weight_quantizer.dtype,
            rowwise_data=fp8_data.view(torch.uint8),
            rowwise_scale_inv=recip_scale,
            columnwise_data=None,
            columnwise_scale_inv=None,
            quantizer=self.weight_quantizer,
            is_2D_scaled=True,
            requires_grad=requires_grad,
        )


def drop_columnwise(param: torch.Tensor) -> None:
    """Free the column-wise data TransformerEngine derived from a gathered FP8 weight for
    backward. Its row-wise data views FlexShard's gathered buffer, which the next all-gather
    refills with the updated weights, so the derived copy must not outlive them."""
    if HAVE_TE and isinstance(param, Float8BlockwiseQTensor) and param._columnwise_data is not None:
        param.update_usage(rowwise_usage=True, columnwise_usage=False)
