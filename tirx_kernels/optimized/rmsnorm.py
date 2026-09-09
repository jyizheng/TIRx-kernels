# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Single-read RMSNorm (FP16, ``eps = 1e-6``), a faster schedule for ``basic/rmsnorm``.

Same contract as ``tirx_kernels.basic.rmsnorm``: ``output = input * rsqrt(mean(input^2) + eps)
* weight`` over the last axis of an FP16 ``(batch, hidden)`` tensor, validated against
``flashinfer.norm.rmsnorm``. The schedule differs from the original kernel in three ways:

* **one global read per row.** Every thread loads its 16-byte input and weight slices
  once, keeps them packed in registers (``vec`` x 4 ``b32`` words each) across the
  reduction, and produces the output from those registers. The original stages the row
  in FP32 shared memory and reads it back; FlashInfer's CUDA kernel reads the input twice.
* **one CTA per row, or many rows per CTA.** The grid is ``ceil(batch / rows_per_cta)``
  instead of a 152-CTA persistent loop, so large batches fill the machine. For
  ``hidden_size <= 256`` a row is owned by a ``hidden_size / 8``-lane slice of a warp and
  eight warps pack ``rows_per_cta`` rows; larger rows use ``hidden_size / (8 * vec)``
  threads, with ``vec`` chosen from the batch size: many one-vector threads when the
  batch is too small to fill the SMs, fewer eight-vector threads when it is
  bandwidth-bound (see ``_geometry``).
* **one shared-memory exchange.** The warp butterfly reduces each warp, lane 0 publishes
  one float, and after a single ``bar.sync`` every thread sums the ``warps`` partials
  itself, so there is no second barrier and no serial second-stage warp.
"""

import math
from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.runner import bench

KERNEL_META = {
    "name": "rmsnorm_opt",
    "category": "optimized",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "f2e04400e330fb2debe0bf8730d9424a1d37927f",
            },
            "import": "flashinfer",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}

EPS = 1e-6
VEC = 8  # fp16 elements per 16-byte load
SM_COUNT = 148  # B200/B300 SM count; only steers the threads-per-row choice
MIN_THREADS_PER_ROW = 128
MAX_THREADS_PER_ROW = 512
WARPS_PER_CTA_SMALL = 8  # CTA size when several rows share one CTA
FULL_MASK = 0xFFFFFFFF

# Same matrix as basic/rmsnorm so the two compare row for row.
CONFIGS = [
    {"hidden_size": hs, "batch_size": bs, "label": f"hs{hs}_bs{bs}"}
    for hs in [128, 4096, 5120, 8192]
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 4113]
]


def _geometry(hidden_size: int, batch_size: int, max_threads_per_row: int | None = None):
    """Trace-time schedule: (vecs_per_thread, threads_per_row, rows_per_cta, warps).

    Rows of at most 256 elements share a CTA (one ``n_vec``-lane warp slice per row).
    Larger rows get one CTA each, and the thread count per row follows the batch:
    a small batch cannot fill the machine with CTAs, so it wants many threads per
    row with one 16-byte vector each (latency-bound); a large batch is bandwidth-
    bound and wants fewer, fatter threads (up to eight vectors each) so that more
    CTAs are resident and the reduction touches fewer warps. The target is
    ``SM_COUNT * 512 / batch`` threads, clamped to [128, 512], realized by the
    32-aligned split of ``n_vec`` nearest to it in log2 distance. Measured on
    hidden 8192: 512 threads at batch <= 128, 128 threads at batch >= 512.
    """
    if hidden_size % VEC != 0:
        raise ValueError(f"hidden_size={hidden_size} must be a multiple of {VEC}")
    n_vec = hidden_size // VEC
    if n_vec <= 32:
        if 32 % n_vec != 0:
            raise ValueError(f"hidden_size={hidden_size}: {n_vec} lanes must divide a warp")
        rows_per_warp = 32 // n_vec
        return 1, n_vec, rows_per_warp * WARPS_PER_CTA_SMALL, WARPS_PER_CTA_SMALL
    candidates = [
        (n_vec // vec, vec)
        for vec in (1, 2, 4, 8)
        if n_vec % vec == 0 and (n_vec // vec) % 32 == 0 and n_vec // vec <= 1024
    ]
    if not candidates:
        raise ValueError(f"hidden_size={hidden_size} has no 32-aligned split into <= 1024 threads")
    if max_threads_per_row is None:
        target = max(MIN_THREADS_PER_ROW, min(MAX_THREADS_PER_ROW, SM_COUNT * 512 // batch_size))
    else:
        target = max_threads_per_row
    threads, vec = min(candidates, key=lambda c: (abs(math.log2(c[0] / target)), -c[0]))
    return vec, threads, 1, threads // 32


def _shfl_c(width: int) -> int:
    # shfl.sync.bfly c operand: segment mask (32 - width) << 8 | clamp 0x1f
    return ((32 - width) << 8) | 0x1F


def make_kernel(
    hidden_size: int,
    batch_size: int,
    *,
    max_threads_per_row: int | None = None,
    late_weight: bool = False,
):
    vec, tpr, rows_per_cta, warps = _geometry(hidden_size, batch_size, max_threads_per_row)
    nthreads = warps * 32
    grid = (batch_size + rows_per_cta - 1) // rows_per_cta
    sub_warp = rows_per_cta > 1  # a row is a tpr-lane slice of a warp
    tpr_shift = tpr.bit_length() - 1
    inv_hidden = 1.0 / hidden_size

    @K.kernel(warps=warps, arch="sm_100a", grid=grid)
    def rmsnorm_opt(inp: K.gptr[K.f16], wgt: K.gptr[K.f16], out: K.gptr[K.f16]):
        cta = K.cta_id()
        tid = K.thread_id()
        lane = K.lane_id()
        warp = K.warp_id()
        if sub_warp:
            row = K.local_scalar("int32", init=cta * rows_per_cta + (tid >> tpr_shift))
            tx = tid & (tpr - 1)
        else:
            row = K.local_scalar("int32", init=cta)
            tx = tid
            red = K.smem_pool().alloc([warps], K.f32)

        def body():
            # ---- one read of this thread's input and weight slices --------------
            xb = K.alloc_local([vec * 4], "uint32")  # packed f16x2 input words
            wb = K.alloc_local([vec * 4], "uint32")  # packed f16x2 weight words
            col = K.local_scalar("int32", init=tx * VEC)  # first element of slice 0
            base = K.local_scalar("int32", init=row * hidden_size)
            for k in range(vec):
                off = k * tpr * VEC  # slice k is tpr*VEC elements further along the row
                K.ptx.ld.global_.nc.v4.b32(
                    xb[4 * k],
                    xb[4 * k + 1],
                    xb[4 * k + 2],
                    xb[4 * k + 3],
                    inp.ptr_to([K.Cast("int64", base + col + off)]),
                )

            def load_weight():
                for k in range(vec):
                    off = k * tpr * VEC
                    K.ptx.ld.global_.nc.v4.b32(
                        wb[4 * k],
                        wb[4 * k + 1],
                        wb[4 * k + 2],
                        wb[4 * k + 3],
                        wgt.ptr_to([K.Cast("int64", col + off)]),
                    )

            if not late_weight:
                load_weight()

            # ---- sum of squares from the packed registers ----------------------
            acc = K.local_scalar("float32", init=K.float32(0.0))
            pair = K.alloc_local([2], "float32")
            for w in range(vec * 4):
                K.idioms.cast_f16x2_to_f32x2(pair, 0, xb[w])
                K.ptx.fma.rn.f32(acc, pair[0], pair[0], acc)
                K.ptx.fma.rn.f32(acc, pair[1], pair[1], acc)

            # ---- reduce: warp butterfly (width tpr for sub-warp rows) ------------
            width = tpr if sub_warp else 32
            peer = K.local_scalar("uint32")
            delta = width // 2
            while delta >= 1:
                K.ptx.shfl_sync.bfly.b32(
                    peer,
                    K.reinterpret("uint32", acc),
                    K.uint32(delta),
                    K.uint32(_shfl_c(width)),
                    K.uint32(FULL_MASK),
                )
                K.assign(acc, acc + K.reinterpret("float32", peer))
                delta //= 2
            if not sub_warp and warps > 1:
                # one exchange: each warp publishes its total, every thread sums them
                with K.If(lane == 0), K.Then():
                    K.ptx.st.shared.f32(red.ptr_to([warp]), acc)
                K.ptx.bar.sync(K.uint32(0), K.uint32(nthreads))
                part = K.local_scalar("float32")
                K.assign(acc, K.float32(0.0))
                for i in range(warps):
                    K.ptx.ld.shared.f32(part, red.ptr_to([i]))
                    K.assign(acc, acc + part)

            rms = K.local_scalar("float32")
            K.ptx.rsqrt.approx.ftz.f32(rms, acc * K.float32(inv_hidden) + K.float32(EPS))
            if late_weight:
                load_weight()

            # ---- scale, apply the weight, pack, store --------------------------
            xf = K.alloc_local([2], "float32")
            wf = K.alloc_local([2], "float32")
            ob = K.alloc_local([vec * 4], "uint32")
            for w in range(vec * 4):
                K.idioms.cast_f16x2_to_f32x2(xf, 0, xb[w])
                K.idioms.cast_f16x2_to_f32x2(wf, 0, wb[w])
                K.ptx.cvt.rn.f16x2.f32(ob[w], (xf[1] * rms) * wf[1], (xf[0] * rms) * wf[0])
            for k in range(vec):
                off = k * tpr * VEC
                K.ptx.st.global_.v4.b32(
                    out.ptr_to([K.Cast("int64", base + col + off)]),
                    ob[4 * k],
                    ob[4 * k + 1],
                    ob[4 * k + 2],
                    ob[4 * k + 3],
                )

        if sub_warp and batch_size % rows_per_cta != 0:
            with K.If(row < batch_size), K.Then():
                body()
        else:
            body()

    return rmsnorm_opt


def get_kernel(hidden_size, batch_size, **kwargs):
    return make_kernel(hidden_size, batch_size, **kwargs).func


def prepare_data(batch_size, dim):
    import torch

    torch.manual_seed(42)
    return (
        torch.randn(batch_size, dim, dtype=torch.float16, device="cuda"),
        torch.randn(dim, dtype=torch.float16, device="cuda"),
    )


def _kernel_args(input_data, weights, output):
    return input_data.view(-1), weights, output.view(-1)


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.basic import rmsnorm as basic
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {
        "config": dict(kwargs),
        "executable": compile_kernel(get_kernel(**kwargs)),
        # The original kernel is timed as an in-repo reference; it must be compiled
        # here because the GPU stage may not compile.
        "basic_executable": compile_kernel(basic.get_kernel(kwargs["hidden_size"])),
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(hidden_size, batch_size, **kwargs):
    """Compile, run, and verify against flashinfer.norm.rmsnorm (same arbiter as basic/rmsnorm)."""
    import torch

    from tirx_kernels.runner import compile_kernel

    input_data, weights = prepare_data(batch_size, hidden_size)
    ex = compile_kernel(get_kernel(hidden_size, batch_size))
    output = torch.empty((batch_size, hidden_size), dtype=torch.float16, device="cuda")
    ex(*_kernel_args(input_data, weights, output))
    torch.cuda.synchronize()

    import flashinfer

    ref = torch.empty_like(output)
    flashinfer.norm.rmsnorm(input_data, weights, EPS, enable_pdl=False, out=ref)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs):
    """Time the kernel against flashinfer and against the original basic/rmsnorm TIRx kernel."""
    import torch

    config = dict(prepared["config"])
    hidden_size = config.pop("hidden_size")
    batch_size = config.pop("batch_size")
    ex = prepared["executable"]

    input_data, weights = prepare_data(batch_size, hidden_size)
    output = torch.empty((batch_size, hidden_size), dtype=torch.float16, device="cuda")
    args = _kernel_args(input_data, weights, output)
    funcs = {"tirx": lambda: ex(*args)}

    def build_flashinfer():
        import flashinfer

        out_fi = torch.empty_like(output)
        return lambda: flashinfer.norm.rmsnorm(
            input_data, weights, EPS, enable_pdl=False, out=out_fi
        )

    def build_basic():
        from tirx_kernels.basic import rmsnorm as basic

        ex_basic = prepared["basic_executable"]
        out_basic = torch.empty_like(output)
        basic_args = basic._kernel_args(input_data, weights, out_basic)
        return lambda: ex_basic(*basic_args)

    return bench(
        funcs,
        references={"flashinfer": build_flashinfer, "tirx_rmsnorm": build_basic},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
        **kwargs,
    )


def run_bench(
    hidden_size,
    batch_size,
    *,
    warmup=None,
    repeat=None,
    timer=None,
    rounds=1,
    cooldown_s=1.0,
    **kwargs,
):
    prepared = prepare_bench(hidden_size=hidden_size, batch_size=batch_size, **kwargs)
    return prepared.run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )
