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
import subprocess
import traceback


ERROR_MSG_LIMIT = 2000


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


def get_limit(data_type):
    """根据数据类型获取精度阈值 - 使用 2 的幂次方阈值（与 NPU Benchmark 标准一致）
    参考文档: 精度对比方法.md
    数据类型: FLOAT16, BFLOAT16, FLOAT32, HiFloat32, FLOAT8 E4M3, FLOAT8 E5M2
    判定标准: MERE < threshold 且 MARE < 10 * threshold


    阈值表:
    | 数据类型      | 阈值 (2^n)      | 十进制值       |
    |--------------|----------------|---------------|
    | FLOAT16      | 2^{-10}        | 0.0009765625  |
    | BFLOAT16     | 2^{-7}         | 0.0078125     |
    | FLOAT32      | 2^{-13}        | 0.0001220703  |
    | HiFloat32    | 2^{-11}        | 0.0004882812  |
    | FLOAT8 E4M3  | 2^{-3}         | 0.125         |
    | FLOAT8 E5M2  | 2^{-2}         | 0.25          |

    由于 torch.dtype 中没有直接定义 HiFloat32，可通过字符串传入 "hifloat32" 获取对应阈值。
    """  # noqa: E501


    import torch

    # 支持字符串类型（用于 HiFloat32 或其他自定义类型）

    if isinstance(data_type, str):
        str_to_threshold = {
            "float16": 2**(-10),
            "bfloat16": 2**(-7),
            "float32": 2**(-13),
            "hifloat32": 2**(-11),
            "float8_e4m3": 2**(-3),
            "float8_e5m2": 2**(-2),
            "fp8_e4m3": 2**(-3),
            "fp8_e5m2": 2**(-2),
        }
        return str_to_threshold.get(data_type.lower(), 2**(-13))

    # torch.dtype 类型映射
    dtype_threshold_map = {
        torch.float16: 2**(-10),    # FLOAT16
        torch.bfloat16: 2**(-7),    # BFLOAT16
        torch.float32: 2**(-13),    # FLOAT32
    }

    # 安全获取 FP8 类型（PyTorch 2.0+ 支持）
    # FLOAT8 E4M3: 2^{-3}
    float8_e4m3 = getattr(torch, 'float8_e4m3fn', None) or getattr(torch, 'float8_e4m3', None)
    if float8_e4m3 is not None:
        dtype_threshold_map[float8_e4m3] = 2**(-3)

    # FLOAT8 E5M2: 2^{-2}
    float8_e5m2 = getattr(torch, 'float8_e5m2fn', None) or getattr(torch, 'float8_e5m2', None)
    if float8_e5m2 is not None:
        dtype_threshold_map[float8_e5m2] = 2**(-2)

    return dtype_threshold_map.get(data_type, 2**(-13))


def get_small_value_threshold(data_type):
    """获取小值域阈值 (Small Value Threshold)。

    当 |golden| < threshold 时，采用小值域通过标准评估精度。

    阈值表:
    | 数据类型      | 小值域阈值 (2^n)  | 十进制值       |
    |--------------|------------------|---------------|
    | FLOAT16      | 2^{-11}          | 4.8828125e-4  |
    | BFLOAT16     | 2^{-8}           | 0.00390625    |
    | FLOAT32      | 2^{-14}          | 6.1035156e-5  |
    | HiFloat32    | 2^{-12}          | 2.4414062e-4  |
    | FLOAT8 E4M3  | 2^{-4}           | 0.0625        |
    | FLOAT8 E5M2  | 2^{-3}           | 0.125         |
    """
    import torch

    if isinstance(data_type, str):
        str_to_threshold = {
            "float16": 2**(-11),
            "bfloat16": 2**(-8),
            "float32": 2**(-14),
            "hifloat32": 2**(-12),
            "float8_e4m3": 2**(-4),
            "float8_e5m2": 2**(-3),
            "fp8_e4m3": 2**(-4),
            "fp8_e5m2": 2**(-3),
        }
        return str_to_threshold.get(data_type.lower(), 2**(-14))

    dtype_threshold_map = {
        torch.float16: 2**(-11),
        torch.bfloat16: 2**(-8),
        torch.float32: 2**(-14),
    }

    float8_e4m3 = getattr(torch, 'float8_e4m3fn', None) or getattr(torch, 'float8_e4m3', None)
    if float8_e4m3 is not None:
        dtype_threshold_map[float8_e4m3] = 2**(-4)

    float8_e5m2 = getattr(torch, 'float8_e5m2fn', None) or getattr(torch, 'float8_e5m2', None)
    if float8_e5m2 is not None:
        dtype_threshold_map[float8_e5m2] = 2**(-3)

    return dtype_threshold_map.get(data_type, 2**(-14))


def get_small_value_error(data_type):
    """获取小值域 error 指标。

    当 |golden| < small_value_threshold 时，若 |actual - golden| > error 则计为错误。

    阈值表:
    | 数据类型      | 小值域 error (2^n) | 十进制值       |
    |--------------|-------------------|---------------|
    | FLOAT16      | 2^{-16}           | 1.5258789e-5  |
    | BFLOAT16     | 2^{-16}           | 1.5258789e-5  |
    | FLOAT32      | 2^{-30}           | 9.3132257e-10 |
    | HiFloat32    | 2^{-28}           | 3.7252903e-9  |
    | FLOAT8 E4M3  | 2^{-6}            | 0.015625      |
    | FLOAT8 E5M2  | 2^{-5}            | 0.03125       |
    """
    import torch

    if isinstance(data_type, str):
        str_to_error = {
            "float16": 2**(-16),
            "bfloat16": 2**(-16),
            "float32": 2**(-30),
            "hifloat32": 2**(-28),
            "float8_e4m3": 2**(-6),
            "float8_e5m2": 2**(-5),
            "fp8_e4m3": 2**(-6),
            "fp8_e5m2": 2**(-5),
        }
        return str_to_error.get(data_type.lower(), 2**(-30))

    dtype_error_map = {
        torch.float16: 2**(-16),
        torch.bfloat16: 2**(-16),
        torch.float32: 2**(-30),
    }

    float8_e4m3 = getattr(torch, 'float8_e4m3fn', None) or getattr(torch, 'float8_e4m3', None)
    if float8_e4m3 is not None:
        dtype_error_map[float8_e4m3] = 2**(-6)

    float8_e5m2 = getattr(torch, 'float8_e5m2fn', None) or getattr(torch, 'float8_e5m2', None)
    if float8_e5m2 is not None:
        dtype_error_map[float8_e5m2] = 2**(-5)

    return dtype_error_map.get(data_type, 2**(-30))


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
    print(f"      总元素数: {size}", file=sys.stderr)

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
    fw_nan_count = fw_nan_mask.sum().item()
    if fw_nan_count > 0:
        print(f"      NaN 检查通过: NaN数量={fw_nan_count}", file=sys.stderr)

    fw_inf_mask = torch.isinf(fw_flat)
    impl_inf_mask = torch.isinf(impl_flat)
    if not torch.equal(fw_inf_mask, impl_inf_mask):
        fw_inf_count = fw_inf_mask.sum().item()
        impl_inf_count = impl_inf_mask.sum().item()
        raise AssertionError(
            f"验证失败，Inf 位置不匹配: Framework={fw_inf_count}/{size}, "
            f"Implementation={impl_inf_count}/{size}"
        )
    fw_inf_count = fw_inf_mask.sum().item()
    if fw_inf_count > 0:
        print(f"      Inf 检查通过: Inf数量={fw_inf_count}", file=sys.stderr)

    if fw_inf_mask.any():
        if not torch.equal(
            torch.sign(fw_flat[fw_inf_mask]),
            torch.sign(impl_flat[impl_inf_mask]),
        ):
            raise AssertionError("验证失败，Inf 符号不匹配")

    finite_mask = torch.isfinite(fw_flat) & torch.isfinite(impl_flat)
    finite_count = finite_mask.sum().item()

    if finite_count == 0:
        print("      警告: 所有值都是非有限值，跳过精度检查", file=sys.stderr)
        return

    print(f"      有限值数量: {finite_count}", file=sys.stderr)

    fw_finite = fw_flat[finite_mask]
    impl_finite = impl_flat[finite_mask]

    if fw_finite.dtype == torch.bool:
        if not torch.equal(fw_finite, impl_finite):
            raise AssertionError(f"验证失败，布尔值不匹配: dtype={data_type}")
        print(f"      布尔值检查通过", file=sys.stderr)
        return

    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.to(fw_finite.dtype)
        print(f"      dtype转换: impl -> {fw_finite.dtype}", file=sys.stderr)

    # 执行 NPU Benchmark 精度验证
    _check_accuracy_npu_benchmark(fw_finite, impl_finite, data_type)


def _check_accuracy_npu_benchmark(golden, actual, data_type):
    """执行 NPU Benchmark 精度验证（单标杆比对）。

    验证两个张量的数值一致性：
    - 计算 MERE（平均相对误差）和 MARE（最大相对误差）
    - 使用 2 的幂次方作为阈值
    - 判定标准：
      - 若所有 golden 都落在小值域（|golden| < small_value_threshold），仅检查小值域通过标准
      - 若所有 golden 都不在小值域，检查 MERE < threshold 且 MARE < 10 * threshold
      - 若混合情况，将输出分割成小值部分和正常部分分别评估：
        - 小值部分：检查小值域通过标准
        - 正常部分：仅对非小值计算 MERE/MARE 并检查常规精度标准
        - 两部分都通过才算整体通过

    小值域通过标准：
    - ErrorCount = sum(I(|golden| < small_value_threshold and |actual - golden| > error))
    - 通过条件：ErrorCount <= 2

    Args:
        golden: 参考输出（金标准）
        actual: 被测实现输出
        data_type: 数据类型，用于获取对应的阈值

    Raises:
        AssertionError: 当精度验证未通过时
    """
    import torch

    # 统一转换为 float32 进行计算
    golden_f = golden.float()
    actual_f = actual.float()

    threshold = get_limit(data_type)
    diff = (actual_f - golden_f).abs()

    # 小值域通过标准
    small_value_threshold = get_small_value_threshold(data_type)
    small_value_error = get_small_value_error(data_type)
    small_value_mask = golden_f.abs() < small_value_threshold

    # 判定标准：
    # - 若所有 golden 都落在小值域，仅检查小值域通过标准
    # - 若所有 golden 都不在小值域，检查常规精度标准
    # - 若混合情况，将输出分割成小值部分和正常部分分别评估
    has_small_value = small_value_mask.any().item()
    has_normal_value = (~small_value_mask).any().item()

    is_pass = True
    normal_MERE = None
    normal_MARE = None

    total_elements = golden_f.numel()
    small_count = small_value_mask.sum().item()
    normal_count = total_elements - small_count

    print(f"    [精度检查] 总元素数={total_elements}, 小值域元素数={small_count}, "
          f"正常值域元素数={normal_count}", file=sys.stderr)

    if has_small_value:
        small_value_errors = diff[small_value_mask]
        error_count = (small_value_errors > small_value_error).sum().item()
        small_value_pass = error_count <= 2
        is_pass = is_pass and small_value_pass
        print(f"    [小值域检查] threshold={small_value_threshold:.6e}, "
              f"error_limit={small_value_error:.6e}, ErrorCount={error_count}, "
              f"通过={small_value_pass}", file=sys.stderr)

    if has_normal_value:
        # 正常部分：仅对非小值计算相对误差
        normal_golden = golden_f[~small_value_mask]
        normal_actual = actual_f[~small_value_mask]
        normal_diff = diff[~small_value_mask]
        normal_denom = normal_golden.abs() + 1e-7
        normal_relative_error = normal_diff / normal_denom
        normal_MERE = normal_relative_error.mean().item()
        normal_MARE = normal_relative_error.max().item()
        normal_pass = (normal_MERE < threshold) and (normal_MARE < 10 * threshold)
        is_pass = is_pass and normal_pass
        print(f"    [正常值域检查] MERE={normal_MERE:.6e}, MARE={normal_MARE:.6e}, "
              f"threshold={threshold}, 通过={normal_pass}", file=sys.stderr)

    if not is_pass:
        error_msg = f"验证失败，输出不一致: dtype={data_type}, threshold={threshold}\n"

        if has_small_value and not small_value_pass:
            error_msg += (
                f"小值域未通过: small_value_threshold={small_value_threshold:.6e}, "
                f"small_value_error={small_value_error:.6e}, ErrorCount={error_count}\n"
            )

        if has_normal_value and not normal_pass:
            error_msg += (
                f"正常值域未通过: MERE={normal_MERE:.6e}, MARE={normal_MARE:.6e}\n"
            )
            # 收集正常值域中超出阈值的样本
            mismatch_mask = normal_relative_error > threshold
            mismatch_indices = torch.where(mismatch_mask)[0]
            num_to_show = min(10, len(mismatch_indices))
            if len(mismatch_indices) > 0:
                error_msg += f"前 {num_to_show} 个超出阈值的值:\n"
                for i in range(num_to_show):
                    idx = mismatch_indices[i].item()
                    error_msg += (
                        f"  位置[{idx}]: framework={normal_golden[idx]:.6e}, "
                        f"impl={normal_actual[idx]:.6e}, "
                        f"相对误差={normal_relative_error[idx]:.6e}\n"
                    )
        raise AssertionError(error_msg)

    print(f"    [精度检查] 通过", file=sys.stderr)


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
    print(f"    输入描述: {describe_input(inputs)}", file=sys.stderr)

    inputs_for_impl = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]
    inputs_for_framework = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]

    with torch.no_grad():
        print(f"    执行框架模型...", file=sys.stderr)
        framework_output = framework_model(*inputs_for_framework)
        print(f"    执行实现模型...", file=sys.stderr)
        impl_output = impl_model(*inputs_for_impl)

    if not isinstance(framework_output, (list, tuple)):
        framework_output = [framework_output]
    if not isinstance(impl_output, (list, tuple)):
        impl_output = [impl_output]

    print(f"    输出数量: framework={len(framework_output)}, impl={len(impl_output)}", file=sys.stderr)

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
            print(f"    比对输出 {i}: shape={list(fw_out.shape)}, dtype={fw_out.dtype}", file=sys.stderr)
            try:
                data_type = fw_out.dtype
                compare(fw_out, impl_out, data_type)
            except AssertionError as e:
                raise AssertionError(f"[用例 {case_idx}/{total_cases}] {str(e)}") from e
        else:
            print(f"    输出 {i} 非 Tensor，跳过精度比对", file=sys.stderr)


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

    print(f"=" * 60, file=sys.stderr)
    print(f"开始验证算子: {op_name}", file=sys.stderr)
    print(f"总测试用例数: {total_cases}", file=sys.stderr)
    print(f"=" * 60, file=sys.stderr)

    for case_idx, inputs in enumerate(input_groups, start=1):
        print(f"\n{'-' * 50}", file=sys.stderr)
        print(f"[用例 {case_idx}/{total_cases}] 开始执行", file=sys.stderr)

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
            print(f"[用例 {case_idx}/{total_cases}] 通过", file=sys.stderr)

        except Exception as e:
            err_detail = traceback.format_exc()
            print(f"[用例 {case_idx}/{total_cases}] 失败: {type(e).__name__}: {e}", file=sys.stderr)
            failures.append({
                "case_idx": case_idx,
                "input_desc": input_desc,
                "error_type": type(e).__name__,
                "error_msg": truncate_error(err_detail),
            })

        finally:
            del framework_model
            del impl_model
            cleanup_npu_memory()

    failed_cases = total_cases - passed_cases
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"验证完成: {passed_cases}/{total_cases} 通过, {failed_cases} 失败", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

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
    parser.add_argument("--timeout", type=int, default=900, help="超时秒数（默认 900）")
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

    # 策略 A：passed < total → exit 1
    sys.exit(0 if passed == total and total > 0 else 1)