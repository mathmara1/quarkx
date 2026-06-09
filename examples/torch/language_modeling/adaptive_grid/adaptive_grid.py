"""
Per-block adaptive grid selection for OCP MX formats, using Quark's primitives.

This is a research prototype that implements the idea explored in:
  - Cook et al., "Adaptive Block-Scaled Data Types" (IF4), arXiv:2603.28765
  - Egiazarian et al., "Grid Games", arXiv:2605.12327
  - The user's own validation library at C:/Users/mathmara/adaptive-mxfp6

For each block of `block_size` weights, we quantize-dequantize under several
candidate MX element formats (e.g., MXFP6_e2m3 and MXFP6_e3m2), measure the
per-block MSE under each, and select whichever format minimizes error for
that specific block. The output is the reconstructed BF16 tensor with each
block represented in its winning format.

This is fake-quantization: we simulate the round-trip in BF16 so accuracy
evaluations work end-to-end without any new hardware kernels. The bit-level
storage representation (which would need a per-block format selector tag in
real deployment, e.g., 1 bit per block for 2 candidates) is irrelevant for
the accuracy claim — that's a hardware-feasibility concern.

The implementation deliberately uses Quark's existing
`fake_quantize_fp4_fp6_per_group_with_scale` primitive so this prototype
inherits any correctness fixes / hardware optimizations that land in Quark
without modification.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
    fake_quantize_fp4_fp6_per_group_with_scale,
)
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.utils import even_round, reshape_to_blocks


@dataclass
class AdaptiveQuantResult:
    """
    Output of `adaptive_quantize_dequantize`.

    Attributes:
        recon: Reconstructed tensor, same shape as input.
        choices: int8 tensor of shape (..., n_blocks). Value at index i is
            the position in `formats` of the winning format for block i.
        per_block_mse: float tensor of shape (..., n_blocks). MSE of the
            winning format for each block (== min across formats).
    """

    recon: torch.Tensor
    choices: torch.Tensor
    per_block_mse: torch.Tensor

    def format_distribution(self, formats: list[Dtype]) -> dict[str, float]:
        """Fraction of blocks that chose each format. For diagnostics."""
        flat = self.choices.flatten()
        total = flat.numel()
        return {
            fmt.value: float((flat == i).sum().item()) / total
            for i, fmt in enumerate(formats)
        }


def _fake_quantize_one_format(
    x: torch.Tensor, fmt: Dtype, block_size: int, axis: int
) -> torch.Tensor:
    """
    Quark per-group fake-quantize with absmax-RNE scale (the OCP MX default).

    Wraps `fake_quantize_fp4_fp6_per_group_with_scale` with the scale-
    computation step inlined (so callers don't need to set up a full
    PerBlockMXObserver Module just to run a tensor through).
    """
    blocks = reshape_to_blocks(x, block_size, axis)
    amax = torch.max(torch.abs(blocks), dim=-1)[0]
    scale = even_round(amax, fmt)
    return fake_quantize_fp4_fp6_per_group_with_scale(
        x, scale, axis, block_size, fmt.value
    )


def adaptive_quantize_dequantize(
    x: torch.Tensor,
    formats: list[Dtype],
    block_size: int = 32,
    axis: int = -1,
) -> AdaptiveQuantResult:
    """
    Quantize each block of `x` using the candidate format that minimizes
    per-block MSE.

    Args:
        x: Input tensor. The `axis` dimension must be divisible by `block_size`.
        formats: List of Quark `Dtype` enum values to choose between.
            Typical: `[Dtype.fp6_e2m3, Dtype.fp6_e3m2]`.
        block_size: MX block size. OCP spec uses 32.
        axis: Dimension along which blocks are formed. Default -1 (last dim),
            which matches the contraction axis of `F.linear`.

    Returns:
        AdaptiveQuantResult with the reconstructed tensor, per-block format
        choices, and the per-block MSE of the winning format.
    """
    if not formats:
        raise ValueError("formats must contain at least one Dtype")
    # The MSE reshape below assumes blocks are formed along the last dim.
    # If anyone needs a different axis, we'd need to transpose `stacked` and
    # `x` before reshaping. Punt for the prototype — Linear weights use -1.
    if axis not in (-1, x.dim() - 1):
        raise NotImplementedError(
            f"adaptive_quantize_dequantize prototype only supports axis=-1, got {axis}"
        )

    *lead, n = x.shape
    if n % block_size != 0:
        raise ValueError(
            f"Last dim ({n}) must be divisible by block_size ({block_size}). "
            "Padding is not implemented in this prototype."
        )

    # Per-format reconstruction. Each entry has shape == x.shape.
    recons = [_fake_quantize_one_format(x, fmt, block_size, axis) for fmt in formats]

    # Stack to (n_formats, ...x.shape). All ops below are vectorized over n_formats.
    stacked = torch.stack(recons, dim=0)

    # Per-element squared error, then per-block mean.
    # sqerr shape: (n_formats, ..., n)
    sqerr = (x.unsqueeze(0) - stacked).pow(2)
    # Reshape last dim into (n_blocks, block_size) so we can mean across blocks.
    n_blocks = n // block_size
    sqerr = sqerr.reshape(len(formats), *lead, n_blocks, block_size)
    mse_per_block = sqerr.mean(dim=-1)  # (n_formats, ..., n_blocks)

    # Argmin over the format dim gives, for each block, the index of the
    # winning format in `formats`.
    choices = mse_per_block.argmin(dim=0)  # (..., n_blocks), dtype int64

    # Gather the winning reconstruction per block.
    # Reshape stacked to (n_formats, ..., n_blocks, block_size) then gather
    # along dim 0 using `choices` expanded to broadcast across the block dim.
    stacked_blocks = stacked.reshape(len(formats), *lead, n_blocks, block_size)
    gather_idx = choices.unsqueeze(0).unsqueeze(-1).expand(1, *choices.shape, block_size)
    chosen_blocks = torch.gather(stacked_blocks, dim=0, index=gather_idx).squeeze(0)
    recon = chosen_blocks.reshape(*lead, n)

    # Per-block MSE of the winning format. Same shape as `choices`.
    chosen_mse = torch.gather(mse_per_block, dim=0, index=choices.unsqueeze(0)).squeeze(0)

    return AdaptiveQuantResult(
        recon=recon,
        choices=choices.to(torch.int8),
        per_block_mse=chosen_mse,
    )


def quantize_model_weights_inplace(
    model: nn.Module,
    formats: list[Dtype],
    block_size: int = 32,
    exclude_name_substrings: tuple[str, ...] = ("lm_head", "embed"),
) -> dict[str, dict[str, float]]:
    """
    Walk `model`, replace each nn.Linear's weight tensor with its adaptively-
    quantized reconstruction. Activations stay BF16 (weight-only PTQ).

    Args:
        model: Any nn.Module. Typically an HF AutoModelForCausalLM.
        formats: Candidate Dtypes (e.g., `[Dtype.fp6_e2m3, Dtype.fp6_e3m2]`).
        block_size: MX block size (32 per OCP spec).
        exclude_name_substrings: Linear layers whose qualified name contains
            any of these substrings are left in BF16. Standard PTQ practice
            for `lm_head` and token embeddings.

    Returns:
        Per-layer dict mapping qualified name -> format-distribution dict
        (what fraction of blocks chose each format). Useful for diagnostics.
    """
    summary: dict[str, dict[str, float]] = {}
    targets: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in exclude_name_substrings):
            continue
        if module.in_features % block_size != 0:
            # Skip layers whose input dim isn't block-aligned. Same policy
            # as the user's adaptive-mxfp6 library.
            continue
        targets.append((name, module))

    for name, module in targets:
        w = module.weight.data
        original_dtype = w.dtype
        # Quark's primitive expects float32 or higher precision internally.
        result = adaptive_quantize_dequantize(
            w.to(torch.float32), formats, block_size=block_size, axis=-1
        )
        module.weight.data.copy_(result.recon.to(original_dtype))
        summary[name] = result.format_distribution(formats)

    return summary


# Convenience: short string -> Dtype mapping for CLI / config-file use.
FORMAT_NAME_TO_DTYPE: dict[str, Dtype] = {
    "fp6_e2m3": Dtype.fp6_e2m3,
    "fp6_e3m2": Dtype.fp6_e3m2,
    "fp4": Dtype.fp4,
}


def parse_formats(names: list[str]) -> list[Dtype]:
    """Convert a list of format-name strings to Dtype enums. Raises on unknown names."""
    out = []
    for n in names:
        if n not in FORMAT_NAME_TO_DTYPE:
            raise ValueError(
                f"Unknown format {n!r}. Supported: {sorted(FORMAT_NAME_TO_DTYPE)}"
            )
        out.append(FORMAT_NAME_TO_DTYPE[n])
    return out
