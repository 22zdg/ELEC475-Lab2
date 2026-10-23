"""CUDA-synchronized timing, parameter counts and forward operation estimates."""
import time
import functools
import copy
from typing import Dict, Tuple, Optional, Any, Callable
import torch
import torch.nn as nn

class Timer:
    """
    Measure wall-clock time with CUDA synchronization.
    Usable as a context manager or decorator.
    """
    def __init__(self, name: str = "Operation", verbose: bool = False):
        self.name = name
        self.verbose = verbose
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        if self.verbose:
            print(f"[{self.name}] Elapsed Time: {self.elapsed:.4f} seconds")

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class ModelProfiler:
    """
    Count parameters and estimate forward MACs with THOP or fallback hooks.
    Reported FLOPs equal twice MACs; some operators are uncounted.
    """
    def __init__(self, model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 64, 64), device: str = "cpu"):
        self.model = model
        self.input_size = input_size
        self.device = torch.device(device)
        self.total_params, self.trainable_params = self._count_parameters()
        self.macs, self.flops = self._compute_flops()

    def _count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    def _compute_flops(self) -> Tuple[int, int]:
        """Profile a copy of the model, using THOP when available."""
        # Profiling must not change BatchNorm state, dropout mode, or checkpoints.
        working_model = copy.deepcopy(self.model).to(self.device).eval()
        try:
            import thop
            dummy = torch.randn(*self.input_size, device=self.device)
            with torch.no_grad():
                macs, _ = thop.profile(working_model, inputs=(dummy,), verbose=False)
            self.backend = "THOP (FLOPs = 2 x reported MACs; some operators are uncounted)"
            return int(macs), int(2 * macs)
        except Exception:
            pass

        # Fallback forward hooks
        total_macs = 0
        hooks = []

        def conv2d_hook(m: nn.Conv2d, inp: Any, out: torch.Tensor):
            nonlocal total_macs
            batch_size = out.shape[0]
            out_c, out_h, out_w = out.shape[1], out.shape[2], out.shape[3]
            in_c = m.in_channels // m.groups
            k_h, k_w = m.kernel_size
            macs_per_item = out_c * out_h * out_w * in_c * k_h * k_w
            total_macs += batch_size * macs_per_item

        def linear_hook(m: nn.Linear, inp: Any, out: torch.Tensor):
            nonlocal total_macs
            batch_size = out.shape[0]
            total_macs += batch_size * m.in_features * m.out_features

        def pool_hook(m: nn.Module, inp: Any, out: torch.Tensor):
            nonlocal total_macs
            batch_size = out.shape[0]
            if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
                k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
                total_macs += batch_size * out.numel() // batch_size * (k[0] * k[1])

        # Register hooks
        for mod in working_model.modules():
            if isinstance(mod, nn.Conv2d):
                hooks.append(mod.register_forward_hook(conv2d_hook))
            elif isinstance(mod, nn.Linear):
                hooks.append(mod.register_forward_hook(linear_hook))
            elif isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d)):
                hooks.append(mod.register_forward_hook(pool_hook))

        dummy = torch.randn(*self.input_size, device=self.device)
        try:
            with torch.no_grad():
                working_model(dummy)
        finally:
            for h in hooks:
                h.remove()
        self.backend = "Conv/Linear/Pool hooks (BN, LRN, activations uncounted)"

        flops = total_macs * 2
        return int(total_macs), int(flops)

    def get_params_millions(self) -> float:
        return self.total_params / 1e6

    def get_macs_millions(self) -> float:
        return self.macs / 1e6

    def get_flops_millions(self) -> float:
        return self.flops / 1e6

    def get_model_size_mb(self) -> float:
        """Estimated memory size for float32 weights."""
        return (self.total_params * 4) / (1024 * 1024)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "params_m": self.get_params_millions(),
            "macs": self.macs,
            "macs_m": self.get_macs_millions(),
            "flops": self.flops,
            "flops_m": self.get_flops_millions(),
            "size_mb": self.get_model_size_mb(),
            "backend": self.backend,
        }

    def print_summary(self, model_name: str = "Model"):
        print("=" * 60)
        print(f"Computational Complexity Profile: {model_name}")
        print(f"Input Tensor Shape   : {self.input_size}")
        print(f"Counting Method      : {self.backend}")
        print("-" * 60)
        print(f"Total Parameters     : {self.total_params:,} ({self.get_params_millions():.4f} M)")
        print(f"Trainable Parameters : {self.trainable_params:,}")
        print(f"Multiply-Accumulates : {self.macs:,} ({self.get_macs_millions():.2f} MMACs)")
        print(f"Floating Point Ops   : {self.flops:,} ({self.get_flops_millions():.2f} MFLOPs)")
        print(f"Weight Size (FP32)   : {self.get_model_size_mb():.2f} MB")
        print("=" * 60)


def benchmark_inference(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 64, 64),
    device: str = "cpu",
    reps: int = 100,
    warmup: int = 10
) -> Tuple[float, float]:
    """
    Measures inference latency and throughput with warm-up and CUDA synchronization.
    Returns:
        (median_latency_ms, throughput_imgs_per_sec)
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    dummy = torch.randn(*input_size, device=dev)
    batch_size = input_size[0]

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()

    # Timing runs
    timings = []
    with torch.no_grad():
        for _ in range(reps):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # to ms

    timings.sort()
    median_latency_ms = timings[len(timings) // 2]
    throughput = (batch_size / (median_latency_ms / 1000.0))
    return median_latency_ms, throughput


def compare_model_complexities(
    models_dict: Dict[str, nn.Module],
    input_size: Tuple[int, ...] = (1, 3, 64, 64),
    device: str = "cpu"
) -> str:
    """Return a Markdown table of model size, operation estimates and latency."""
    headers = ["Model Architecture", "Params (M)", "MACs (M)", "FLOPs (M)", "Latency (ms)", "Throughput (img/s)"]
    rows = []

    for name, model in models_dict.items():
        profiler = ModelProfiler(model, input_size=input_size, device=device)
        latency, throughput = benchmark_inference(model, input_size=input_size, device=device, reps=50, warmup=10)
        rows.append([
            name,
            f"{profiler.get_params_millions():.3f}",
            f"{profiler.get_macs_millions():.2f}",
            f"{profiler.get_flops_millions():.2f}",
            f"{latency:.2f}",
            f"{throughput:.1f}"
        ])

    col_widths = [max(len(str(val)) for val in col) for col in zip(headers, *rows)]
    col_widths = [max(w, len(h)) for w, h in zip(col_widths, headers)]

    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    sep_str    = "-|-".join("-" * w for w in col_widths)
    row_strs   = [" | ".join(f"{val:<{w}}" for val, w in zip(r, col_widths)) for r in rows]

    table_md = f"| {header_str} |\n| {sep_str} |\n" + "\n".join(f"| {r} |" for r in row_strs)
    return table_md
