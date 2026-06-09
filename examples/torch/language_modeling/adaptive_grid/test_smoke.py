"""
Laptop smoke test for the new mxfp_adaptive scheme registration.

What this tests (without GPU or the C++ kernel actually running):
  - Imports for AdaptiveMXSpec, PerBlockMXAdaptiveObserver, AdaptiveStaticFakeQuantize
    all resolve.
  - AdaptiveMXSpec validates its `formats` field (rejects empty / unknown).
  - AdaptiveMXSpec.to_quantization_spec() produces a QTensorConfig with the
    expected fields (adaptive_formats populated, observer_cls set to
    PerBlockMXAdaptiveObserver, etc.).
  - PerBlockMXAdaptiveObserver can be instantiated from such a QTensorConfig.
  - get_fake_quantize() routes to AdaptiveStaticFakeQuantize when
    adaptive_formats is set, falls through to StaticScaledFakeQuantize
    otherwise.
  - Module-level scheme registration in quantize_quark.py runs without error
    and the new scheme is reachable via LLMTemplate.get_supported_schemes().

What this does NOT test (needs Linux + C++ toolchain — run on RunPod):
  - Actually calling fake_quantize_to_low_precision_fp (Quark's C++ op).
  - End-to-end model quantize + eval.

Run with:
    python test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "[ok]" if condition else "[FAIL]"
    print(f"  {mark} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        sys.exit(1)


def main() -> None:
    section("imports")
    from quark.torch.quantization.config.config import (  # noqa: E402
        AdaptiveMXSpec,
        OCP_MXFP6E2M3Spec,
        QLayerConfig,
        QTensorConfig,
    )
    from quark.torch.quantization.config.type import Dtype  # noqa: E402
    from quark.torch.quantization.observer import (  # noqa: E402
        OBSERVER_CLASSES,
        PerBlockMXAdaptiveObserver,
        PerBlockMXObserver,
    )
    from quark.torch.quantization.tensor_quantize import (  # noqa: E402
        AdaptiveStaticFakeQuantize,
        ScaledFakeQuantize,
        StaticScaledFakeQuantize,
    )
    check("AdaptiveMXSpec importable", True)
    check("PerBlockMXAdaptiveObserver importable", True)
    check("AdaptiveStaticFakeQuantize importable", True)
    check(
        "PerBlockMXAdaptiveObserver in OBSERVER_CLASSES",
        PerBlockMXAdaptiveObserver in OBSERVER_CLASSES,
        f"OBSERVER_CLASSES has {len(OBSERVER_CLASSES)} entries",
    )

    section("AdaptiveMXSpec field validation")
    # Empty formats list — should reject
    try:
        AdaptiveMXSpec(formats=[])
    except ValueError as e:
        check("empty formats rejected", "must contain at least one" in str(e))
    else:
        check("empty formats rejected", False, "no exception raised")

    # Unknown format — should reject
    try:
        AdaptiveMXSpec(formats=["fp6_e2m3", "nonsense_format"])
    except ValueError as e:
        check("unknown format rejected", "nonsense_format" in str(e) or "unsupported" in str(e).lower())
    else:
        check("unknown format rejected", False, "no exception raised")

    # Valid spec
    spec = AdaptiveMXSpec(formats=["fp6_e2m3", "fp6_e3m2"], ch_axis=-1)
    check("valid spec constructs", spec is not None)

    section("Spec -> QTensorConfig conversion")
    qspec = spec.to_quantization_spec()
    check("returns QTensorConfig", isinstance(qspec, QTensorConfig))
    check(
        "dtype is one of the formats",
        qspec.dtype in (Dtype.fp6_e2m3, Dtype.fp6_e3m2),
        f"got dtype={qspec.dtype}",
    )
    check(
        "adaptive_formats populated",
        qspec.adaptive_formats == ["fp6_e2m3", "fp6_e3m2"],
        f"got {qspec.adaptive_formats}",
    )
    check(
        "observer_cls is PerBlockMXAdaptiveObserver",
        qspec.observer_cls is PerBlockMXAdaptiveObserver,
        f"got {qspec.observer_cls}",
    )
    check("ch_axis preserved", qspec.ch_axis == -1)
    check("group_size = 32 (OCP)", qspec.group_size == 32)
    check("scale_format = e8m0", qspec.scale_format == "e8m0")

    section("compare against equivalent OCP_MXFP6E2M3Spec (sanity)")
    ref_qspec = OCP_MXFP6E2M3Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
    check(
        "same group_size as OCP_MXFP6E2M3Spec",
        qspec.group_size == ref_qspec.group_size,
    )
    check(
        "same scale_format as OCP_MXFP6E2M3Spec",
        qspec.scale_format == ref_qspec.scale_format,
    )
    check(
        "different observer class (adaptive vs PerBlockMX)",
        qspec.observer_cls is not ref_qspec.observer_cls,
    )

    section("observer construction (no forward yet)")
    obs = PerBlockMXAdaptiveObserver(qspec=qspec)
    check("observer instantiates", obs is not None)
    check(
        "adaptive_format_names captured",
        obs.adaptive_format_names == ["fp6_e2m3", "fp6_e3m2"],
    )
    check(
        "adaptive_dtypes resolved",
        all(isinstance(d, Dtype) for d in obs.adaptive_dtypes) and len(obs.adaptive_dtypes) == 2,
    )
    check("adaptive_recon initially None", obs.adaptive_recon is None)
    check("adaptive_choices initially None", obs.adaptive_choices is None)

    # Constructing the observer for an unrelated QTensorConfig (no adaptive_formats)
    # should fail with a clear message.
    try:
        bad_qspec = ref_qspec  # standard MXFP6 spec, no adaptive_formats
        PerBlockMXAdaptiveObserver(qspec=bad_qspec)
    except ValueError as e:
        check(
            "rejects qspec without adaptive_formats",
            "adaptive_formats" in str(e),
            "ValueError raised correctly",
        )
    else:
        check("rejects qspec without adaptive_formats", False, "no exception")

    section("get_fake_quantize dispatch")
    fake_quantizer = ScaledFakeQuantize.get_fake_quantize(quant_spec=qspec)
    check(
        "adaptive spec routes to AdaptiveStaticFakeQuantize",
        isinstance(fake_quantizer, AdaptiveStaticFakeQuantize),
        f"got {type(fake_quantizer).__name__}",
    )
    # The reference path should still work — non-adaptive spec routes to StaticScaledFakeQuantize.
    ref_quantizer = ScaledFakeQuantize.get_fake_quantize(quant_spec=ref_qspec)
    check(
        "non-adaptive spec routes to StaticScaledFakeQuantize",
        isinstance(ref_quantizer, StaticScaledFakeQuantize)
        and not isinstance(ref_quantizer, AdaptiveStaticFakeQuantize),
        f"got {type(ref_quantizer).__name__}",
    )

    section("dynamic adaptive (currently unsupported) is rejected")
    dyn_spec = AdaptiveMXSpec(formats=["fp6_e2m3", "fp6_e3m2"], is_dynamic=True)
    dyn_qspec = dyn_spec.to_quantization_spec()
    try:
        ScaledFakeQuantize.get_fake_quantize(quant_spec=dyn_qspec)
    except NotImplementedError as e:
        check("is_dynamic=True raises NotImplementedError", "static" in str(e).lower())
    else:
        check("is_dynamic=True raises NotImplementedError", False, "no exception raised")

    section("QLayerConfig wrap + scheme registration via LLMTemplate")
    from quark.torch import LLMTemplate  # noqa: E402

    layer_config = QLayerConfig(weight=qspec)
    test_name = "mxfp_adaptive_smoke_test"
    # Unregister first in case of prior run leaving state.
    try:
        LLMTemplate.unregister_scheme(test_name)
    except Exception:
        pass
    LLMTemplate.register_scheme(test_name, config=layer_config)
    check(
        f"{test_name!r} appears in supported schemes",
        test_name in LLMTemplate.get_supported_schemes(),
    )
    LLMTemplate.unregister_scheme(test_name)
    check(
        f"{test_name!r} removable after unregister",
        test_name not in LLMTemplate.get_supported_schemes(),
    )

    section("module-level registration in quantize_quark.py")
    # Verify the bundled scheme name "mxfp_adaptive_e2m3_e3m2" gets registered
    # when quantize_quark.py is imported. Use importlib so we don't run main().
    import importlib.util  # noqa: E402

    qq_path = Path(__file__).resolve().parent.parent / "llm_ptq" / "quantize_quark.py"
    spec_qq = importlib.util.spec_from_file_location("quantize_quark", qq_path)
    assert spec_qq is not None and spec_qq.loader is not None
    qq_module = importlib.util.module_from_spec(spec_qq)
    # quantize_quark.py registers schemes at module-import time.
    spec_qq.loader.exec_module(qq_module)
    check(
        "'mxfp_adaptive_e2m3_e3m2' registered after import",
        "mxfp_adaptive_e2m3_e3m2" in LLMTemplate.get_supported_schemes(),
    )

    print("\nAll smoke tests passed.")
    print("Note: actual C++ quantize call (fake_quantize_to_low_precision_fp)")
    print("was NOT exercised. Run on RunPod (Linux + ROCm) for end-to-end validation.")


if __name__ == "__main__":
    main()
