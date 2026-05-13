#!/usr/bin/env python3
"""算子验证脚本 — 对比框架实现 (Model) 与生成实现 (ModelNew) 的输出一致性。

多 shape 模式下：每个 shape 独立 try/except，全部跑完后落盘 verify_result.json。
策略 A：passed < total 即整体判失败（exit 1），同时失败清单记录在 JSON 的 `failures` 字段。

用法:
    python verify.py --op_name <算子名> [--verify_dir <验证目录>] [--timeout <超时秒数>]
"""
import argparse
import gc
import json
import os
import sys
import traceback


ERROR_MSG_LIMIT = 2000

# 精度判定常量（dtype 无关）
MAX_ERROR_CAP = 0.1
REQUIRED_MATCHED_RATIO = 0.9


class AccuracyError(AssertionError):
    """精度判定失败异常，附带结构化 metrics 便于下游统计。"""

    def __init__(self, message, metrics):
        super().__init__(message)
        self.metrics = metrics


def truncate_error(msg: str, limit: int = ERROR_MSG_LIMIT) -> str:
    if msg is None:
        return ""
    if len(msg) <= limit:
        return msg
    half = limit // 2
    return f"{msg[:half]}\n... [truncated {len(msg) - limit} chars] ...\n{msg[-half:]}"


def describe_input(inputs):
    """输入列表的结构化描述（用于 JSON）。"""
    try:
        import torch
    except Exception:
        torch = None
    descs = []
    for x in inputs:
        if torch is not None and isinstance(x, torch.Tensor):
            descs.append({
                "type": "tensor",
                "shape": list(x.shape),
                "dtype": str(x.dtype),
            })
        else:
            try:
                val = x if isinstance(x, (int, float, bool, str)) else repr(x)
            except Exception:
                val = "<unrepr>"
            descs.append({"type": "scalar", "value": val})
    return descs


def cleanup_npu_memory():
    try:
        import torch
        import torch_npu  # noqa: F401
        torch.npu.empty_cache()
    except Exception:
        pass
    gc.collect()


def get_limits(data_type):
    """根据数据类型返回精度判定的三元组 (small_value_threshold, small_value_error, rel_threshold)。

    参考 NPU Benchmark 精度对比方法：
    - small_value_threshold：判定元素是否落在"小值域"的阈值
    - small_value_error：小值域元素的绝对误差上限
    - rel_threshold：正常值域元素的相对误差上限，同时也是 MERE 的判定阈值

    阈值表：
    | 数据类型      | small_value_threshold | small_value_error | rel_threshold |
    |--------------|-----------------------|-------------------|---------------|
    | FLOAT16      | 2^{-11}               | 2^{-16}           | 2^{-10}       |
    | BFLOAT16     | 2^{-8}                | 2^{-16}           | 2^{-7}        |
    | FLOAT32      | 2^{-14}               | 2^{-30}           | 2^{-13}       |
    | HiFloat32    | 2^{-12}               | 2^{-28}           | 2^{-11}       |
    | FLOAT8 E4M3  | 2^{-4}                | 2^{-6}            | 2^{-3}        |
    | FLOAT8 E5M2  | 2^{-3}                | 2^{-5}            | 2^{-2}        |

    由于 torch.dtype 中没有直接定义 HiFloat32，可通过字符串传入 "hifloat32" 获取对应阈值。
    """  # noqa: E501
    import torch

    # 字符串映射（用于 HiFloat32 或其他自定义类型）
    str_to_limits = {
        "float16":     (2**(-11), 2**(-16), 2**(-10)),
        "bfloat16":    (2**(-8),  2**(-16), 2**(-7)),
        "float32":     (2**(-14), 2**(-30), 2**(-13)),
        "hifloat32":   (2**(-12), 2**(-28), 2**(-11)),
        "float8_e4m3": (2**(-4),  2**(-6),  2**(-3)),
        "float8_e5m2": (2**(-3),  2**(-5),  2**(-2)),
        "fp8_e4m3":    (2**(-4),  2**(-6),  2**(-3)),
        "fp8_e5m2":    (2**(-3),  2**(-5),  2**(-2)),
    }
    if isinstance(data_type, str):
        return str_to_limits.get(data_type.lower(), (2**(-14), 2**(-30), 2**(-13)))

    # torch.dtype 映射
    dtype_limits_map = {
        torch.float16:  (2**(-11), 2**(-16), 2**(-10)),
        torch.bfloat16: (2**(-8),  2**(-16), 2**(-7)),
        torch.float32:  (2**(-14), 2**(-30), 2**(-13)),
    }

    float8_e4m3 = getattr(torch, 'float8_e4m3fn', None) or getattr(torch, 'float8_e4m3', None)
    if float8_e4m3 is not None:
        dtype_limits_map[float8_e4m3] = (2**(-4), 2**(-6), 2**(-3))

    float8_e5m2 = getattr(torch, 'float8_e5m2fn', None) or getattr(torch, 'float8_e5m2', None)
    if float8_e5m2 is not None:
        dtype_limits_map[float8_e5m2] = (2**(-3), 2**(-5), 2**(-2))

    return dtype_limits_map.get(data_type, (2**(-14), 2**(-30), 2**(-13)))


def resolve_input_provider(torch_module):
    """解析任务文件的输入提供方式。"""
    if hasattr(torch_module, "get_input_groups"):
        groups = torch_module.get_input_groups()
        return groups, len(groups)
    elif hasattr(torch_module, "get_inputs"):
        return [torch_module.get_inputs()], 1
    else:
        raise AttributeError(
            f"模块必须提供 get_inputs() 或 get_input_groups() 方法"
        )


def compare(fw_out, impl_out, data_type):
    """对比框架输出和实现输出"""
    import torch
    fw_flat = fw_out.flatten().detach().cpu()
    impl_flat = impl_out.flatten()
    if isinstance(impl_flat, torch.Tensor):
        impl_flat = impl_flat.detach().cpu()
    else:
        impl_flat = torch.tensor(impl_flat, dtype=fw_flat.dtype)

    size = fw_flat.numel()

    if fw_flat.shape != impl_flat.shape:
        raise AssertionError(
            f"验证失败，输出形状不一致: framework={fw_flat.shape}, impl={impl_flat.shape}"
        )

    fw_nan_mask = torch.isnan(fw_flat)
    impl_nan_mask = torch.isnan(impl_flat)
    if not torch.equal(fw_nan_mask, impl_nan_mask):
        fw_nan_count = fw_nan_mask.sum().item()
        impl_nan_count = impl_nan_mask.sum().item()
        raise AssertionError(
            f"验证失败，NaN 位置不匹配: Framework={fw_nan_count}/{size}, "
            f"Implementation={impl_nan_count}/{size}"
        )

    fw_inf_mask = torch.isinf(fw_flat)
    impl_inf_mask = torch.isinf(impl_flat)
    if not torch.equal(fw_inf_mask, impl_inf_mask):
        fw_inf_count = fw_inf_mask.sum().item()
        impl_inf_count = impl_inf_mask.sum().item()
        raise AssertionError(
            f"验证失败，Inf 位置不匹配: Framework={fw_inf_count}/{size}, "
            f"Implementation={impl_inf_count}/{size}"
        )
    if fw_inf_mask.any():
        if not torch.equal(
            torch.sign(fw_flat[fw_inf_mask]),
            torch.sign(impl_flat[impl_inf_mask]),
        ):
            raise AssertionError("验证失败，Inf 符号不匹配")

    finite_mask = torch.isfinite(fw_flat) & torch.isfinite(impl_flat)
    finite_count = finite_mask.sum().item()
    if finite_count == 0:
        print("警告: 所有值都是非有限值，跳过精度检查")
        return

    fw_finite = fw_flat[finite_mask]
    impl_finite = impl_flat[finite_mask]

    if fw_finite.dtype == torch.bool:
        if not torch.equal(fw_finite, impl_finite):
            raise AssertionError(f"验证失败，布尔值不匹配: dtype={data_type}")
        return

    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.to(fw_finite.dtype)

    # 执行 NPU Benchmark 精度验证
    _check_accuracy_npu_benchmark(fw_finite, impl_finite, data_type)


def _check_accuracy_npu_benchmark(golden, actual, data_type):
    """执行 NPU Benchmark 精度验证（分类 + 三项判定）。

    元素级 matched 定义：
    - |golden| < small_value_threshold（小值域）：|diff| <= small_value_error
    - 否则（正常值域）：|diff| / (|golden| + 1e-7) <= rel_threshold

    通过条件（三项 AND）：
    1. max(|diff|) <= MAX_ERROR_CAP（0.1，dtype 无关的绝对误差上限）
    2. matched_ratio = sum(matched) / total_finite >= REQUIRED_MATCHED_RATIO（0.98）
    3. MERE < rel_threshold（对所有 finite 元素计算相对误差再取均值，
       分母统一用 |golden| + 1e-7 防除零）

    Args:
        golden: 参考输出（金标准）
        actual: 被测实现输出
        data_type: 数据类型，用于获取对应的阈值三元组

    Raises:
        AccuracyError: 当精度验证未通过时，异常的 metrics 属性携带结构化指标
    """
    import torch

    # 统一升 float32，避免低精度 dtype 自身误差污染计算
    golden_f = golden.float()
    actual_f = actual.float()

    sv_thr, sv_err, rel_thr = get_limits(data_type)

    abs_diff = (actual_f - golden_f).abs()
    abs_golden = golden_f.abs()

    # 分桶
    small_mask = abs_golden < sv_thr
    normal_mask = ~small_mask

    # 元素级 matched
    small_ok = abs_diff <= sv_err
    rel_err = abs_diff / (abs_golden + 1e-7)
    normal_ok = rel_err <= rel_thr
    matched_mask = torch.where(small_mask, small_ok, normal_ok)

    total_finite = matched_mask.numel()
    matched_count = int(matched_mask.sum().item())
    matched_ratio = matched_count / total_finite if total_finite > 0 else 1.0
    max_abs_diff = abs_diff.max().item() if total_finite > 0 else 0.0

    # MERE：对所有 finite 元素计算相对误差再取均值（分母统一 |golden| + 1e-7 防除零）
    normal_count = int(normal_mask.sum().item())
    if total_finite > 0:
        MERE = rel_err.mean().item()
        mere_ok = MERE < rel_thr
    else:
        MERE = None
        mere_ok = True

    cap_ok = max_abs_diff <= MAX_ERROR_CAP
    ratio_ok = matched_ratio >= REQUIRED_MATCHED_RATIO
    is_pass = cap_ok and ratio_ok and mere_ok

    if is_pass:
        return

    metrics = {
        "matched_ratio": matched_ratio,
        "max_abs_diff": max_abs_diff,
        "MERE": MERE,
        "rel_threshold": rel_thr,
        "small_value_threshold": sv_thr,
        "small_value_error": sv_err,
        "max_error_cap": MAX_ERROR_CAP,
        "required_matched_ratio": REQUIRED_MATCHED_RATIO,
        "total_finite": total_finite,
        "matched_count": matched_count,
        "small_count": int(small_mask.sum().item()),
        "normal_count": normal_count,
        "checks": {
            "max_error_cap": cap_ok,
            "required_matched_ratio": ratio_ok,
            "MERE": mere_ok,
        },
    }

    # 失败摘要 + 前 N 个 unmatched 位置（按所属桶注明判定标准）
    unmatched_mask = ~matched_mask
    unmatched_indices = torch.where(unmatched_mask)[0]
    num_to_show = min(10, len(unmatched_indices))

    mere_str = f"{MERE:.6e}" if MERE is not None else "n/a"
    error_msg = (
        f"验证失败 dtype={data_type}: "
        f"max_abs_diff={max_abs_diff:.6e} (cap={MAX_ERROR_CAP}, ok={cap_ok}), "
        f"matched_ratio={matched_ratio:.6f} (req>={REQUIRED_MATCHED_RATIO}, ok={ratio_ok}), "
        f"MERE={mere_str} (rel_thr={rel_thr:.6e}, ok={mere_ok}); "
        f"small_count={metrics['small_count']}, normal_count={normal_count}\n"
    )
    if num_to_show > 0:
        error_msg += f"前 {num_to_show} 个未通过的位置:\n"
        for i in range(num_to_show):
            idx = unmatched_indices[i].item()
            if small_mask[idx].item():
                error_msg += (
                    f"  位置[{idx}] (小值域): framework={golden[idx]:.6e}, "
                    f"impl={actual[idx]:.6e}, |diff|={abs_diff[idx]:.6e} "
                    f"(允许<={sv_err:.6e})\n"
                )
            else:
                error_msg += (
                    f"  位置[{idx}] (正常域): framework={golden[idx]:.6e}, "
                    f"impl={actual[idx]:.6e}, 相对误差={rel_err[idx]:.6e} "
                    f"(允许<={rel_thr:.6e})\n"
                )
    raise AccuracyError(error_msg, metrics)


def run_single_case(
    framework_model,
    impl_model,
    inputs,
    device,
    case_idx,
    total_cases
):
    """验证单组输入。失败时抛出 AssertionError。"""
    import torch

    print(f"  测试第 {case_idx}/{total_cases} 组输入...", file=sys.stderr)

    inputs_for_impl = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]
    inputs_for_framework = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]

    with torch.no_grad():
        impl_output = impl_model(*inputs_for_impl)
        framework_output = framework_model(*inputs_for_framework)

    if not isinstance(framework_output, (list, tuple)):
        framework_output = [framework_output]
    if not isinstance(impl_output, (list, tuple)):
        impl_output = [impl_output]

    if len(framework_output) != len(impl_output):
        raise AssertionError(
            f"[用例 {case_idx}/{total_cases}] 输出数量不一致: "
            f"framework={len(framework_output)}, impl={len(impl_output)}"
        )

    for i, (fw_out, impl_out) in enumerate(zip(framework_output, impl_output)):
        if fw_out is None or impl_out is None:
            raise AssertionError(
                f"[用例 {case_idx}/{total_cases}] 输出 {i} 为 None: "
                f"framework={fw_out is None}, impl={impl_out is None}"
            )
        if isinstance(fw_out, torch.Tensor) and isinstance(impl_out, torch.Tensor):
            try:
                data_type = fw_out.dtype
                compare(fw_out, impl_out, data_type)
            except AccuracyError as e:
                raise AccuracyError(
                    f"[用例 {case_idx}/{total_cases}] {str(e)}", e.metrics
                ) from e
            except AssertionError as e:
                raise AssertionError(f"[用例 {case_idx}/{total_cases}] {str(e)}") from e


def verify_implementations(op_name, verify_dir, triton_impl_name="triton_ascend_impl", output_path=None):
    """验证框架实现和生成实现的结果一致性。

    每个 shape 独立 try/except，全部跑完后写 verify_result.json。

    Returns:
        (passed_cases, total_cases)
    """
    import torch
    import torch_npu  # noqa: F401

    sys.path.insert(0, verify_dir)

    torch_module = __import__(f"{op_name}_torch")
    impl_module = __import__(f"{op_name}_{triton_impl_name}")

    FrameworkModel = torch_module.Model
    ModelNew = impl_module.ModelNew
    get_init_inputs = torch_module.get_init_inputs

    # 在获取输入之前设置种子，确保随机生成的输入可复现
    torch.manual_seed(0)
    torch.npu.manual_seed(0)

    input_groups, total_cases = resolve_input_provider(torch_module)

    device = torch.device("npu")

    failures = []
    passed_cases = 0

    for case_idx, inputs in enumerate(input_groups, start=1):
        input_desc = describe_input(inputs)
        framework_model = None
        impl_model = None
        try:
            init_params = get_init_inputs()
            torch.manual_seed(0)
            torch.npu.manual_seed(0)
            framework_model = FrameworkModel(*init_params).to(device)

            torch.manual_seed(0)
            torch.npu.manual_seed(0)
            impl_model = ModelNew(*init_params).to(device)

            run_single_case(
                framework_model, impl_model, inputs, device, case_idx, total_cases
            )
            passed_cases += 1
        except Exception as e:
            err_detail = traceback.format_exc()
            print(f"  [用例 {case_idx}/{total_cases}] 失败: {type(e).__name__}: {e}", file=sys.stderr)
            failure_entry = {
                "case_idx": case_idx,
                "input_desc": input_desc,
                "error_type": type(e).__name__,
                "error_msg": truncate_error(err_detail),
            }
            if isinstance(e, AccuracyError):
                failure_entry["metrics"] = e.metrics
            failures.append(failure_entry)
        finally:
            del framework_model
            del impl_model
            cleanup_npu_memory()

    failed_cases = total_cases - passed_cases

    # 落盘 verify_result.json
    if output_path is None:
        output_path = os.path.join(verify_dir, "verify_result.json")
    result = {
        "op_name": op_name,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "failures": failures,
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"验证结果已保存到: {output_path}", file=sys.stderr)
    except Exception as e:
        print(f"警告: 无法写入 verify_result.json: {e}", file=sys.stderr)

    if failed_cases == 0:
        print(f"验证成功：共 {total_cases} 组测试用例全部通过")
    else:
        print(
            f"验证失败：{passed_cases}/{total_cases} 组通过，"
            f"{failed_cases} 组失败（详见 {output_path}）",
            file=sys.stderr,
        )

    return passed_cases, total_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="算子验证脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument(
        "--verify_dir", default=".",
        help="验证目录，包含 {op_name}_torch.py 和 {op_name}_triton_ascend_impl.py（默认当前目录）",
    )
    parser.add_argument("--timeout", type=int, default=900, help="超时秒数（默认 900，已忽略：当前为同进程模式）")
    parser.add_argument(
        "--triton_impl_name", default="triton_ascend_impl",
        help="Triton 实现模块名（不含 op_name 前缀，默认 triton_ascend_impl）",
    )
    parser.add_argument(
        "--output", default=None,
        help="验证结果 JSON 输出路径（默认 {verify_dir}/verify_result.json）",
    )
    args = parser.parse_args()

    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        print(f"错误: 验证目录不存在: {verify_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        passed, total = verify_implementations(
            args.op_name, verify_dir, args.triton_impl_name, args.output
        )
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0 if passed == total and total > 0 else 1)