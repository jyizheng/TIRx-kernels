# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved SM100a KDA backward for the packed Kimi K3 8x1024 H=96 workload.

The supported contract is the prepared-tensor contract of
``fla.ops.kda.chunk_bwd.chunk_kda_bwd`` with B=1, T=8192 packed as eight
1024-token sequences, Hqk=Hv=96, K=V=128 and chunk_size=64 -- the official
``kda-bwd-packed-1024x8-h96`` row.  Inputs are the saved L2-normalized q/k, v,
the activated update gate beta, the saved Aqk/Akk interaction matrices, the
chunk-local base-2 cumulative gate g, the per-sequence K-first initial state,
the upstream do and dht, and cu_seqlens.  The kernel writes dq, dk, dv, db, dg
and dh0; dA and dbias are absent from this contract.

The selected kernel is the ``fused-zfold-pipeline`` frontier member of the
2026-09-08 KDA-backward evolution run ``ablation-kda-bwd-2``.  It is a single
fused persistent kernel: one 12-warp CTA per SM runs pass 1 (forward over
64-token chunks, recomputing the chunk states h_t in TMEM and storing bf16
snapshots plus a 2^g cache) for a host-scheduled set of (sequence, head)
chains, then pass 2 (backward over chunks) for its own chains.  Per-chain
readiness is published with an epoch-stamped gpu-scope release flag, which
removes the pass-1 launch and most of the wave-quantization tail of the
768-chain workload on 152 SMs.  Pass 2 folds the W/U/Vn/dw chain into a single
TMEM intermediate ``Z = vb - h^T kbg`` consumed as a bf16 tile by both
``Vn^T = Z Akk^T`` and ``dAs = dv2 Z^T``, so the W accumulator and the dw MMA
and readout round trip disappear.

Numerical notes.  All MMAs are ``kind::f16`` bf16 x bf16 -> fp32, the same
class the FLA reference feeds ``tl.dot``; no tf32 or fp8 path is used.  Two
choices are lower precision than the reference and are deliberate:
``2^g`` is cached to global memory in bf16 and the inverse gate is recovered
with ``rcp.approx.ftz.f32`` instead of the reference's fp32
``exp2(gn - g)``.  Measured on the official row this keeps the maximum
normalized RMS error ratio at 5.959e-3 against the FLA reference, inside the
tightest per-output limit of 8e-3, but the margin is only about 1.34x.
"""

import ctypes
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

D = 128
CHUNK = 64
MMA_SS = "tcgen05.mma.cta_group::1.kind::f16"
TMA_LD = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
TMA_ST = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
TMA_PREFETCH = "cp.async.bulk.prefetch.tensor.3d.L2.global.tile"
TC_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TC_ST32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
TC_LD8 = "tcgen05.ld.sync.aligned.32x32b.x8.b32"
TC_LD4 = "tcgen05.ld.sync.aligned.32x32b.x4.b32"
TC_ST16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
WAIT_LD = "tcgen05.wait::ld.sync.aligned"
WAIT_ST = "tcgen05.wait::st.sync.aligned"
FENCE_ASYNC = "fence.proxy.async.shared::cta"
BULK_COMMIT = "cp.async.bulk.commit_group"
BULK_WAIT_READ = "cp.async.bulk.wait_group.read"
BULK_WAIT = "cp.async.bulk.wait_group"
TC_FENCE_BEFORE = "tcgen05.fence::before_thread_sync"
TC_FENCE_AFTER = "tcgen05.fence::after_thread_sync"


def idesc(M, N, *, ta=0, tb=0, na=0, nb=0):
    """Dense tcgen05 instruction descriptor: bf16 x bf16 -> f32 (PTX ISA table 45)."""
    return (
        (1 << 4) | (1 << 7) | (1 << 10) | (na << 13) | (nb << 14) | (ta << 15) | (tb << 16)
        | ((N >> 3) << 17) | ((M >> 4) << 24)
    )



def stage_units(tile):
    """16-byte units between consecutive stages of a staged KTile."""
    t = tile
    return (t._phys(t._coord(1, 0, 0)) - t._phys(t._coord(0, 0, 0))) * t.bits // 8 // 16


class Operand:
    """A tcgen05 smem operand pre-encoded at trace-time stage 0 (plus per-stage 16B offset)."""

    def __init__(self, view, major, units=0):
        self.desc, self.off = view.encode(major=major, mma_k=16)
        extent = view.cols if major == "k" else view.rows
        self.n_k = extent // 16
        self.units = units

    def base(self, stage=None):
        if stage is None or self.units == 0:
            return K.local_scalar("uint64", init=self.desc.value)
        return K.local_scalar("uint64", init=self.desc.value + K.Cast("uint64", stage) * K.uint64(self.units))


def mma_issue(tm, dcol, a, b, idesc, accumulate, a_stage=None, b_stage=None, n_k=None):
    """One k-chain of tcgen05.mma from the calling (single, elected) thread."""
    n_k = n_k or b.n_k
    a_base = a.base(a_stage)
    b_base = b.base(b_stage)
    for kp in range(n_k):
        K.ptx[MMA_SS](
            K.Cast("uint32", tm[0] + dcol),
            a_base + K.uint64(a.off(kp)),
            b_base + K.uint64(b.off(kp)),
            K.uint32(idesc),
            K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0),
            K.ptx.pred(1 if (accumulate or kp > 0) else 0),
        )





TM_H = 0
TM_W = 128
TM_U = 192
TMEM_COLS_H = 256

KV_BYTES = 2 * CHUNK * D * 2
AKK_BYTES = CHUNK * CHUNK * 2





class AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def encode_tensor_map(tensor, dtype: str, dims, strides_bytes, box, swizzle=3, l2promo=2):
    """cuTensorMapEncodeTiled through TVM's runtime function (dims innermost first)."""
    import tvm

    desc = AlignedTensorMap()
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    rank = len(dims)
    assert len(strides_bytes) == rank - 1 and len(box) == rank
    encode(
        desc.ptr, dtype, rank, ctypes.c_void_p(int(tensor.data_ptr())),
        *[int(d) for d in dims], *[int(s) for s in strides_bytes], *[int(b) for b in box],
        *([1] * rank), 0, swizzle, l2promo, 0,
    )
    return desc


def token_map(tensor, T, H, inner, box_inner, box_rows=CHUNK, swizzle=3):
    """[T, H, inner] tensor viewed as dims (inner, T, H): coordinates (d0, token, head)."""
    esz = tensor.element_size()
    dtype = {2: "bfloat16", 4: "float32"}[esz]
    return encode_tensor_map(tensor, dtype, (inner, T, H), (esz * inner * H, esz * inner),
                             (box_inner, box_rows, 1), swizzle=swizzle)


def state_map(tensor, n_states):
    """[n_states, 128, 128] bf16 states viewed as dims (128 v, 128 k, n): coordinates (v0, 0, idx)."""
    return encode_tensor_map(tensor, "bfloat16", (D, D, n_states), (2 * D, 2 * D * D), (64, D, 1))









UNITS_PER_STAGE = 512
SBO_UNITS = 64


class Op:
    """A tcgen05 matrix-descriptor operand: trace-time stage `base` of the single staged pool.

    Every operand is the pool's base descriptor (encoded once for stage 0 with the 64-row
    column-atom stride) plus a compile-time immediate: the stage offset in 16 B units and, for
    128-row tiles, the larger LBO field.  The issuing warp therefore keeps one 64-bit register
    instead of one encoded descriptor per operand.
    """

    LBO_BASE = 64 * 8

    def __init__(self, base_desc, base, rows, kdim, major):
        self.rows = rows
        self.major = major
        self.n_k = kdim // 16
        ldo = rows * 8
        assert ldo >= self.LBO_BASE
        self._bd = base_desc
        self._imm = base * UNITS_PER_STAGE + ((ldo - self.LBO_BASE) << 16)

    def off(self, kp):
        if self.major == "k":
            return (kp % 4) * 2 + (kp // 4) * self.rows * 8
        return kp * 128

    def desc(self, kp, units=None):
        """Descriptor of k-tile `kp`: one immediate on the base so nothing per-operand is worth hoisting.

        Materializing `base + operand_offset` per operand lets ptxas hoist ~30 live 64-bit values out of
        the chunk loop; at the issue warp's 24 registers they spill and every MMA reloads its descriptor
        from local memory.  Folding the stage and k-step offsets into a single immediate keeps the only
        loop-invariant the base descriptor itself.
        """
        d = self._bd[0] + K.uint64(self._imm + self.off(kp))
        if units is not None:
            d = d + units
        return d


def mma_chain(tm, dcol, a, b, idesc_val, accumulate, a_units=None, b_units=None):
    """One k-chain of tcgen05.mma from the calling (single, elected) thread."""
    n_k = b.n_k
    assert a.n_k == n_k, (a.n_k, n_k)
    for kp in range(n_k):
        K.ptx[MMA_SS](
            K.Cast("uint32", tm[0] + dcol),
            a.desc(kp, a_units),
            b.desc(kp, b_units),
            K.uint32(idesc_val),
            K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0),
            K.ptx.pred(1 if (accumulate or kp > 0) else 0),
        )





TM_DH = 0
S1, S2, S3, S4, S6, S5 = 128, 192, 256, 320, 384, 448
TMEM_COLS_B = 512
DO_BYTES = CHUNK * D * 2
H_BYTES = D * D * 2
AQK_BYTES = CHUNK * CHUNK * 2


SCHED_MAXP2, SCHED_MAXP1 = 8, 12
SCHED_STRIDE = 2 + SCHED_MAXP2 + SCHED_MAXP1


def make_fused_kernel(H: int, static_grid=None):
    HK = H * D
    HK64 = K.int64(HK)
    QKVE_BYTES = 4 * CHUNK * D * 2









    T1, T2, T3, T5, T6, DHB = 0, 2, 4, 8, 10, 12
    DV2, ZT, DVB, DAM = 6, 8, 6, 9
    PB0, PB1 = 12, 14
    ST_Q, ST_K, ST_V, ST_G = 12, 14, 16, 8


    S_DO, S_H, S_AQK, S_AKK = 18, 20, 24, 25
    IN_BYTES = 3 * CHUNK * D * 2 + CHUNK * 8 * 2
    EG_BYTES = CHUNK * D * 2

    @K.kernel(warps=12, arch="sm_100a", min_blocks_per_sm=1,
              grid="num_ctas" if static_grid is None else static_grid)
    def kda_bwd_fused(
        q: K.gptr[K.bf16],
        k: K.gptr[K.bf16],
        v: K.gptr[K.bf16],
        beta: K.gptr[K.bf16],
        aqk: K.gptr[K.bf16],
        akk: K.gptr[K.bf16],
        g: K.gptr[K.f32],
        egcache: K.gptr[K.bf16],
        do: K.gptr[K.bf16],
        dht: K.gptr[K.f32],
        h0: K.gptr[K.f32],
        hsnap: K.gptr[K.bf16],
        cu_seqlens: K.gptr[K.i64],
        flags: K.gptr[K.i32],
        sched: K.gptr[K.i32],
        dq: K.gptr[K.f32],
        dk: K.gptr[K.f32],
        dv: K.gptr[K.bf16],
        db: K.gptr[K.f32],
        dg: K.gptr[K.f32],
        dh0: K.gptr[K.f32],
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        g_map: K.TensorMap,
        eg_map: K.TensorMap,
        beta_map: K.TensorMap,
        do_map: K.TensorMap,
        aqk_map: K.TensorMap,
        akk_map: K.TensorMap,
        h_map: K.TensorMap,
        scale: K.f32,
        num_seqs: K.i32,
        num_ctas: K.i32,
        epoch: K.i32,
    ):
        for buf in (q, k, v, beta, aqk, akk, g, do, hsnap, egcache):
            K.keep_alive(buf.data)
        num_work = num_seqs * K.int32(H)



        # Host-built schedule: per CTA [n_p2, n_p1, p2 chains..., p1 chains...].
        cta = K.local_scalar("int32", init=K.Cast("int32", K.cta_id()))
        sbase = K.local_scalar("int32", init=cta * K.int32(SCHED_STRIDE))
        n_p2 = K.local_scalar("int32")
        K.ptx.ld.global_.s32(n_p2, sched.ptr_to([sbase]))
        n_p1 = K.local_scalar("int32")
        K.ptx.ld.global_.s32(n_p1, sched.ptr_to([sbase + K.int32(1)]))

        def p2_chain(i):
            c = K.local_scalar("int32")
            K.ptx.ld.global_.s32(c, sched.ptr_to([sbase + K.int32(2) + i]))
            return c

        def p1_chain(i):
            c = K.local_scalar("int32")
            K.ptx.ld.global_.s32(c, sched.ptr_to([sbase + K.int32(2 + SCHED_MAXP2) + i]))
            return c

        sp = K.specialize()
        cg = sp.role("cg", warps=list(range(8)), regs=208)
        auxg = sp.warpgroup("aux", warps=[8, 9, 10, 11], regs=88)
        loader = sp.role("loader", warps=[8], group=auxg)
        mma = sp.role("mma", warps=[9], group=auxg)
        idle = sp.role("idle", warps=[10, 11], group=auxg)

        smem = K.smem_pool()
        s_tmem = smem.alloc((4,), K.i32, align=16)
        b_in_full = K.TMABar(smem, 1); b_in_full.init(1)
        b_eg_full = K.TMABar(smem, 1); b_eg_full.init(1)
        b_mid_free = K.MBarrier(smem, 1); b_mid_free.init(256)
        b_do_full = K.TMABar(smem, 1); b_do_full.init(1)
        b_h_full = K.TMABar(smem, 1); b_h_full.init(1)
        b_aqk_full = K.TMABar(smem, 1); b_aqk_full.init(1)
        b_akk_full = K.TMABar(smem, 2); b_akk_full.init(1)
        b_do_empty = K.TCGen05Bar(smem, 1); b_do_empty.init(1)
        b_h_free = K.MBarrier(smem, 1); b_h_free.init(256)
        b_aqk_empty = K.TCGen05Bar(smem, 1); b_aqk_empty.init(1)
        b_akk_empty = K.TCGen05Bar(smem, 2); b_akk_empty.init(1)
        mb_names = ["t_early", "dhb_ready", "zT_ready", "vnT_ready", "dv2T_ready",
                    "dAqk_tile_ready", "dAm_ready", "X_ready", "intra_ready", "dv_epi_done"]
        MB = {}
        for nm in mb_names:
            MB[nm] = K.MBarrier(smem, 1)
            MB[nm].init(256)
        b_dg0_ready = K.MBarrier(smem, 1); b_dg0_ready.init(256)
        b_aqk_masked = K.MBarrier(smem, 1); b_aqk_masked.init(64)

        p_kv = K.Pipeline(smem, 2, full="tma", empty="mbar", init_empty=256)
        p_akk1 = K.Pipeline(smem, 1, full="tma", empty="tcgen05")
        p_tiles = K.Pipeline(smem, 1, full="mbar", empty="tcgen05", init_full=256)
        p_hs = K.Pipeline(smem, 1, full="mbar", empty="mbar", init_full=256, init_empty=9)
        p_w = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=256)
        p_vn = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=256)
        p_g = K.Pipeline(smem, 2, full="tma", empty="mbar", init_empty=256)
        tc_names = ["Z_done", "Vn_done", "dv2_done", "dAqk_done", "dk_done",
                    "dAs_done", "dvb_done", "X_done", "Y_done",
                    "dq2_done", "dkt_done", "chunk_done"]
        TC = {}
        for nm in tc_names:
            TC[nm] = K.TCGen05Bar(smem, 1)
            TC[nm].init(1)

        TT = smem.alloc((27, 64, 64), K.bf16, swizzle=K.SW128B)
        s_beta = smem.alloc((64,), K.f32, align=16)
        s_beta_in = smem.alloc((CHUNK, 8), K.bf16, align=128)
        s_dgk = smem.alloc((2, 128), K.f32, align=16)
        s_cs = smem.alloc((128,), K.f32, align=16)
        s_beta_g = smem.alloc((2, CHUNK, 8), K.bf16, align=128)
        s_beta1 = smem.alloc((2, CHUNK), K.f32, align=16)

        with K.If(K.thread_id() == 0), K.Then():
            K.ptx.st.shared.s32(K.address_of(s_tmem[1]), K.int32(0))
            K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()
        with K.If(K.warp_id() == 8), K.Then():
            K.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                K.address_of(s_tmem[0]), K.uint32(TMEM_COLS_B))
        K.cuda.cta_sync()

        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def make_phaser():
            """Sequential IKET ranges for one role: phase(name) ends the current range and starts the next."""
            tok = K.alloc_local([1], "uint32")
            K.assign(tok[0], K.cuda.iket.sentinel_token("idle"))

            def phase(name):
                K.cuda.iket.range_end(tok[0])
                K.assign(tok[0], K.cuda.iket.range_start(name))

            def phase_end():
                K.cuda.iket.range_end(tok[0])
                K.assign(tok[0], K.cuda.iket.sentinel_token("idle"))

            return phase, phase_end

        def tmem_preamble():
            tmv = K.alloc_local([1], "int32")
            K.ptx.ld.volatile.shared.s32(tmv[0], K.address_of(s_tmem[0]))
            return tmv

        def pack_bf16x2(dst, lo, hi):
            K.ptx.cvt.rn.bf16x2.f32(dst, hi, lo)

        def work_coords(work):
            # Exact selected workload: chain = sequence * 96 + head.
            seq = K.local_scalar("int32", init=work // K.int32(96))
            head = K.local_scalar("int32", init=work - seq * K.int32(96))
            bos = K.local_scalar("int64", init=K.Cast("int64", seq) * K.int64(1024))
            seq_len = K.local_scalar("int32", init=K.int32(1024))
            nch = K.local_scalar("int32", init=K.int32(16))
            return seq, head, bos, seq_len, nch

        def chunk_base(seq):
            cb = K.local_scalar("int32", init=K.int32(0))
            with K.serial(seq) as i:
                cs = K.alloc_local([2], "int64")
                K.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([i]))
                K.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([i + 1]))
                K.assign(cb, cb + ((K.Cast("int32", cs[1] - cs[0]) + K.int32(CHUNK - 1)) >> 6))
            return cb








        # KV 0:8, Akk 8, snapshot 9:13, gate ring 13:21, derived tiles 21:27.
        P1_KV, P1_AKK, P1_HS, P1_G, P1_KG, P1_KBG, P1_VB = 0, 8, 9, 13, 21, 23, 25
        G_BYTES = CHUNK * D * 4 + CHUNK * 8 * 2
        TM_H, TM_W, TM_U = 0, 128, 192

        def bf16_bits_to_f32(u16val):
            return K.reinterpret("float32", K.Cast("uint32", u16val) << K.uint32(16))

        def p1_compute():
            tm = tmem_preamble()
            wr = K.warp_id_in_role()
            lane = K.lane_id()
            wg = K.local_scalar("int32", init=wr >> 2)
            quad = K.local_scalar("int32", init=wr & 3)
            x = K.local_scalar("int32", init=quad * 32 + lane)
            row0 = K.local_scalar("int32", init=wg * 32)
            x64 = K.Cast("int64", x)
            xs = K.local_scalar("int32", init=x >> 6)
            xr = K.local_scalar("int32", init=x & 63)
            xg = K.local_scalar("int32", init=x >> 5)
            xgc = K.local_scalar("int32", init=(x & 31) * 2)

            def tmem_at(col):
                return K.Cast("uint32", tm[0] + col + (quad << 21))

            st_kv = K.PipelineState(2, phase=0)
            st_te = K.PipelineState(1, phase=1)
            st_hs = K.PipelineState(1, phase=1)
            st_g = K.PipelineState(2, phase=0)
            st_w = K.PipelineState(1, phase=0)
            st_vn = K.PipelineState(1, phase=0)
            gv = K.alloc_local([32], "float32")
            kk = K.alloc_local([32], "float32")
            vv = K.alloc_local([32], "float32")
            bb = K.alloc_local([32], "float32")
            acc = K.alloc_local([64], "float32")
            wds = K.alloc_local([32], "uint32")
            gn = K.local_scalar("float32")
            egn = K.local_scalar("float32")
            eg = K.local_scalar("float32")
            egng = K.local_scalar("float32")
            bu = K.local_scalar("uint16")
            ku = K.local_scalar("uint16")
            vu = K.local_scalar("uint16")
            phase, phase_end = make_phaser()
            tid_all = K.local_scalar("int32", init=wr * 32 + lane)
            with K.serial(n_p1) as it:
                chain = K.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                head64 = K.Cast("int64", head)
                gcol = K.local_scalar("int64", init=head64 * K.int64(D) + x64)
                def h_c0():
                    rows = K.int32(CHUNK)
                    p_g.full.wait(st_g.stage, st_g.phase)
                    gst = K.local_scalar("int32", init=P1_G + st_g.stage * K.int32(4) + xg)
                    with K.If(lane < K.int32(8)), K.Then():
                        btok = wr * K.int32(8) + lane
                        K.ptx.ld.shared.u16(bu, s_beta_g.ptr_to([st_g.stage, btok, head & K.int32(7)]))
                        K.ptx.st.shared.f32(K.address_of(s_beta1[st_g.stage, btok]), bf16_bits_to_f32(bu))
                    for i in range(32):
                        K.ptx.ld.shared.f32(gv[i], TT[gst].ptr_to(row0 + i, xgc))
                    K.ptx.ld.shared.f32(gn, TT[gst].ptr_to(rows - K.int32(1), xgc))
                    K.ptx.bar.sync(K.uint32(1), K.uint32(256))
                    for u in range(8):
                        K.ptx["ld.shared.v4.f32"](bb[4 * u], bb[4 * u + 1], bb[4 * u + 2], bb[4 * u + 3],
                                                  K.address_of(s_beta1[st_g.stage, row0 + 4 * u]))
                    K.ptx[FENCE_ASYNC]()
                    p_g.empty.arrive(st_g.stage)
                    st_g.advance()

                phase("h-c0")
                h_c0()
                with K.serial(nch) as n:
                    rows = K.int32(CHUNK)
                    tok0 = K.local_scalar("int64", init=bos + K.Cast("int64", n * K.int32(CHUNK)))
                    K.ptx.ex2.approx.ftz.f32(egn, gn)
                    phase("hw-kv")
                    p_kv.full.wait(st_kv.stage, st_kv.phase)
                    phase("h-kv")
                    kst = K.local_scalar("int32", init=P1_KV + st_kv.stage * K.int32(4) + xs * K.int32(2))
                    for i in range(32):
                        K.ptx.ld.shared.u16(ku, TT[kst].ptr_to(row0 + i, xr))
                        K.ptx.ld.shared.u16(vu, TT[kst + K.int32(1)].ptr_to(row0 + i, xr))
                        K.assign(kk[i], bf16_bits_to_f32(ku))
                        K.assign(vv[i], bf16_bits_to_f32(vu))
                    K.ptx[FENCE_ASYNC]()
                    p_kv.empty.arrive(st_kv.stage)
                    st_kv.advance()
                    phase("hw-tiles")
                    p_tiles.empty.wait(0, st_te.phase)
                    st_te.advance()
                    K.ptx[TC_FENCE_AFTER]()
                    K.ptx[FENCE_ASYNC]()
                    phase("h-tiles")
                    for u in range(4):
                        wkg = K.alloc_local([4], "uint32")
                        wkbg = K.alloc_local([4], "uint32")
                        wvb = K.alloc_local([4], "uint32")
                        vals = K.alloc_local([24], "float32")
                        for e in range(8):
                            i = 8 * u + e
                            K.ptx.ex2.approx.ftz.f32(eg, gv[i])
                            K.ptx.ex2.approx.ftz.f32(egng, gn - gv[i])
                            K.ptx.cvt.rn.bf16.f32(bu, eg)
                            egidx = (tok0 + K.Cast("int64", row0 + K.int32(i))) * HK64 + gcol
                            K.ptx["st.global.L1::no_allocate.b16"](egcache.ptr_to([egidx]), bu)
                            K.assign(vals[e], kk[i] * egng)
                            K.assign(vals[8 + e], kk[i] * bb[i] * eg)
                            K.assign(vals[16 + e], vv[i] * bb[i])
                        for p in range(4):
                            pack_bf16x2(wkg[p], vals[2 * p], vals[2 * p + 1])
                            pack_bf16x2(wkbg[p], vals[8 + 2 * p], vals[8 + 2 * p + 1])
                            pack_bf16x2(wvb[p], vals[16 + 2 * p], vals[16 + 2 * p + 1])
                        col = row0 + 8 * u
                        K.ptx["st.shared.v4.b32"](TT[P1_KG + xs].ptr_to(xr, col), wkg[0], wkg[1], wkg[2], wkg[3])
                        K.ptx["st.shared.v4.b32"](TT[P1_KBG + xs].ptr_to(xr, col), wkbg[0], wkbg[1], wkbg[2], wkbg[3])
                        K.ptx["st.shared.v4.b32"](TT[P1_VB + xs].ptr_to(xr, col), wvb[0], wvb[1], wvb[2], wvb[3])

                    K.ptx[FENCE_ASYNC]()
                    p_tiles.full.arrive(0)
                    phase("hw-hs")
                    p_hs.empty.wait(st_hs.stage, st_hs.phase)
                    K.ptx[TC_FENCE_AFTER]()
                    phase("h-decay")
                    hc0 = wg * 64
                    hsst = K.local_scalar("int32", init=P1_HS + st_hs.stage * K.int32(4) + wg * K.int32(2) + xs)
                    with K.If(n == K.int32(0)):
                        with K.Then():
                            h0base = ((K.Cast("int64", seq) * K.int64(H) + head64) * K.int64(D) + x64) * K.int64(D) \
                                + K.Cast("int64", hc0)
                            for m in range(8):
                                K.ptx["ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.f32"](
                                    *(acc[8 * m + i] for i in range(8)),
                                    h0.ptr_to([h0base + K.int64(8 * m)]))
                        with K.Else():
                            K.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_H + hc0))
                            K.ptx[TC_LD32](*(acc[32 + i] for i in range(32)), tmem_at(TM_H + hc0 + 32))
                            K.ptx[WAIT_LD]()
                    for p in range(32):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(8):
                        K.ptx["st.shared.v4.b32"](TT[hsst].ptr_to(xr, 8 * u),
                                                  wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                    for p in range(32):
                        dpair = K.local_scalar("uint64")
                        K.ptx["mul.rn.f32x2"](dpair, K.cuda.make_float2(acc[2 * p], acc[2 * p + 1]),
                                              K.cuda.make_float2(egn, egn))
                        K.assign(acc[2 * p], K.cuda.float2_x(dpair))
                        K.assign(acc[2 * p + 1], K.cuda.float2_y(dpair))
                    K.ptx[TC_ST32](tmem_at(TM_H + hc0), *(acc[i] for i in range(32)))
                    K.ptx[TC_ST32](tmem_at(TM_H + hc0 + 32), *(acc[32 + i] for i in range(32)))
                    K.ptx[WAIT_ST]()
                    K.ptx[TC_FENCE_BEFORE]()
                    K.ptx[FENCE_ASYNC]()
                    p_hs.full.arrive(st_hs.stage)
                    phase("hw-W")
                    p_w.full.wait(0, st_w.phase)
                    st_w.advance()
                    K.ptx[TC_FENCE_AFTER]()
                    K.ptx[FENCE_ASYNC]()
                    phase("h-wT")
                    K.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_W + wg * 32))
                    K.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(4):
                        K.ptx["st.shared.v4.b32"](TT[P1_KBG + xs].ptr_to(xr, row0 + 8 * u),
                                                  wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                    K.ptx[TC_FENCE_BEFORE]()
                    K.ptx[FENCE_ASYNC]()
                    p_w.empty.arrive(0)
                    phase("h-c0")
                    with K.If(n + K.int32(1) < nch), K.Then():
                        h_c0()
                    phase("hw-Vn")
                    p_vn.full.wait(0, st_vn.phase)
                    st_vn.advance()
                    K.ptx[TC_FENCE_AFTER]()
                    K.ptx[FENCE_ASYNC]()
                    phase("h-vnT")
                    K.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_U + wg * 32))
                    K.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(4):
                        K.ptx["st.shared.v4.b32"](TT[P1_VB + xs].ptr_to(xr, row0 + 8 * u),
                                                  wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                    K.ptx[TC_FENCE_BEFORE]()
                    K.ptx[FENCE_ASYNC]()
                    p_vn.empty.arrive(0)

                    with K.If(elected()), K.Then():
                        p_hs.empty.arrive(st_hs.stage)
                    st_hs.advance()
                    phase_end()

            p_tiles.empty.wait(0, st_te.phase)
            K.ptx[TC_FENCE_AFTER]()

        def p1_mma():
            tm = tmem_preamble()



            bd1 = K.alloc_local([1], "uint64")
            zq1 = K.alloc_local([1], "int32")
            op_kbg_k = Op(bd1, P1_KBG, 128, 64, "k")
            op_vb_k = Op(bd1, P1_VB, 128, 64, "k")
            op_kg_k = Op(bd1, P1_KG, 128, 64, "k")
            op_akk1_k = Op(bd1, P1_AKK, 64, 64, "k")
            op_hs_mn = Op(bd1, P1_HS, 128, 128, "mn")
            op_w_mn = Op(bd1, P1_KBG, 128, 128, "mn")
            ID_M128N64 = idesc(128, 64)
            ID_VN = idesc(128, 64, ta=1, tb=1, nb=1)
            ID_HUPD = idesc(128, 128)
            st_tiles = K.PipelineState(1, phase=0)
            st_akk = K.PipelineState(1, phase=0)
            st_hs = K.PipelineState(1, phase=0)
            st_w = K.PipelineState(1, phase=0)
            st_vn = K.PipelineState(1, phase=0)
            mphase, mphase_end = make_phaser()
            with K.serial(n_p1) as it:
                chain = K.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                with K.serial(nch) as n:
                    K.ptx.ld.volatile.shared.s32(zq1[0], K.address_of(s_tmem[1]))
                    K.cuda.tcgen05.encode_matrix_descriptor(
                        K.address_of(bd1[0]), TT[zq1[0]].ptr_to(0, 0), ldo=Op.LBO_BASE, sdo=SBO_UNITS,
                        swizzle=K.SW128B.value)
                    mphase("hmw-tiles")
                    p_tiles.full.wait(0, st_tiles.phase)
                    p_akk1.full.wait(st_akk.stage, st_akk.phase)
                    K.ptx[TC_FENCE_AFTER]()
                    akk_u = K.local_scalar("uint64", init=K.Cast("uint64", st_akk.stage) * K.uint64(UNITS_PER_STAGE))
                    mphase("hm-WU")
                    with K.If(elected()), K.Then():

                        mma_chain(tm, TM_W, op_kbg_k, op_akk1_k, ID_M128N64, False, b_units=akk_u)
                        p_w.full.arrive(0)

                        mma_chain(tm, TM_U, op_vb_k, op_akk1_k, ID_M128N64, False, b_units=akk_u)
                        p_akk1.empty.arrive(st_akk.stage)
                    st_akk.advance()
                    mphase("hmw-wT")
                    p_w.empty.wait(0, st_w.phase)
                    st_w.advance()
                    mphase("hmw-hs")
                    p_hs.full.wait(st_hs.stage, st_hs.phase)
                    K.ptx[TC_FENCE_AFTER]()
                    hs_u = K.local_scalar("uint64", init=K.Cast("uint64", st_hs.stage) * K.uint64(4 * UNITS_PER_STAGE))
                    mphase("hm-Vn")
                    with K.If(elected()), K.Then():

                        mma_chain(tm, TM_U, op_hs_mn, op_w_mn, ID_VN, True, a_units=hs_u)
                        p_vn.full.arrive(0)
                    st_hs.advance()
                    mphase("hmw-vnT")
                    p_vn.empty.wait(0, st_vn.phase)
                    st_vn.advance()
                    K.ptx[TC_FENCE_AFTER]()
                    mphase("hm-hupd")
                    with K.If(elected()), K.Then():

                        mma_chain(tm, TM_H, op_kg_k, op_vb_k, ID_HUPD, True)
                        p_tiles.empty.arrive(0)
                    st_tiles.advance()
                    mphase_end()

        def p1_loader():
            st_kv = K.PipelineState(2, phase=1)
            st_akk = K.PipelineState(1, phase=1)
            st_g = K.PipelineState(2, phase=1)
            with K.serial(n_p1) as it:
                chain = K.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                bos32 = K.local_scalar("int32", init=K.Cast("int32", bos))
                head8 = K.local_scalar("int32", init=head >> K.int32(3))
                with K.serial(nch) as n:
                    tok0 = bos32 + n * K.int32(CHUNK)
                    p_g.empty.wait(st_g.stage, st_g.phase)
                    with K.If(elected()), K.Then():
                        p_g.full.arrive(st_g.stage, tx_count=G_BYTES)
                        mbg = K.cuda.cvta_generic_to_shared(p_g.full.ptr_to([st_g.stage]))
                        for j in range(4):
                            K.ptx[TMA_LD](TT[P1_G + st_g.stage * K.int32(4) + K.int32(j)].ptr_to(0, 0),
                                          K.address_of(g_map), K.int32(32 * j), tok0, head, mbg)
                        K.ptx[TMA_LD](s_beta_g.ptr_to([st_g.stage, 0, 0]), K.address_of(beta_map),
                                      K.int32(0), tok0, head8, mbg)
                    st_g.advance()
                    p_kv.empty.wait(st_kv.stage, st_kv.phase)
                    with K.If(elected()), K.Then():
                        p_kv.full.arrive(st_kv.stage, tx_count=KV_BYTES)
                        mb = K.cuda.cvta_generic_to_shared(p_kv.full.ptr_to([st_kv.stage]))
                        for tmap, half in ((k_map, 0), (v_map, 1)):
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[P1_KV + st_kv.stage * K.int32(4) + K.int32((d0 // 64) * 2 + half)].ptr_to(0, 0),
                                              K.address_of(tmap), K.int32(d0), tok0, head, mb)
                        with K.If(n + K.int32(1) < nch), K.Then():
                            for tmap in (k_map, v_map):
                                for d0 in (0, 64):
                                    K.ptx[TMA_PREFETCH](K.address_of(tmap), K.int32(d0), tok0 + K.int32(CHUNK), head)
                            for d0 in (0, 32, 64, 96):
                                K.ptx[TMA_PREFETCH](K.address_of(g_map), K.int32(d0), tok0 + K.int32(CHUNK), head)
                            K.ptx[TMA_PREFETCH](K.address_of(akk_map), K.int32(0), tok0 + K.int32(CHUNK), head)
                    st_kv.advance()
                    p_akk1.empty.wait(st_akk.stage, st_akk.phase)
                    with K.If(elected()), K.Then():
                        p_akk1.full.arrive(st_akk.stage, tx_count=AQK_BYTES)
                        mb2 = K.cuda.cvta_generic_to_shared(p_akk1.full.ptr_to([st_akk.stage]))
                        K.ptx[TMA_LD](TT[P1_AKK + st_akk.stage].ptr_to(0, 0), K.address_of(akk_map), K.int32(0), tok0, head, mb2)
                    st_akk.advance()

        def p1_storer():
            st_hs = K.PipelineState(1, phase=0)
            with K.serial(n_p1) as it:
                chain = K.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                cb = chunk_base(seq)
                with K.serial(nch) as n:
                    p_hs.full.wait(st_hs.stage, st_hs.phase)
                    with K.If(elected()), K.Then():
                        K.ptx[FENCE_ASYNC]()
                        idx = (cb + n) * K.int32(H) + head
                        for d0 in (0, 64):
                            K.ptx[TMA_ST](K.address_of(h_map), K.int32(d0), K.int32(0), idx,
                                          TT[P1_HS + st_hs.stage * K.int32(4) + K.int32((d0 // 64) * 2)].ptr_to(0, 0))
                        K.ptx[BULK_COMMIT]()
                        K.ptx[BULK_WAIT_READ](0)
                        p_hs.empty.arrive(st_hs.stage)
                    st_hs.advance()



                with K.If(elected()), K.Then():
                    K.ptx[BULK_WAIT](0)
                    K.ptx["fence.proxy.async.global"]()
                    K.ptx["st.release.gpu.global.s32"](flags.ptr_to([chain]), epoch)




        with cg:
            p1_compute()
            K.ptx.bar.sync(K.uint32(5), K.uint32(384))
            tm = tmem_preamble()
            wr = K.warp_id_in_role()
            lane = K.lane_id()
            wg = K.local_scalar("int32", init=wr >> 2)
            quad = K.local_scalar("int32", init=wr & 3)
            x = K.local_scalar("int32", init=quad * 32 + lane)
            row0 = K.local_scalar("int32", init=wg * 32)
            x64 = K.Cast("int64", x)
            xs = K.local_scalar("int32", init=x >> 6)
            xr = K.local_scalar("int32", init=x & 63)
            tid_all = K.local_scalar("int32", init=wr * 32 + lane)
            phalf = K.local_scalar("int32", init=x & 1)
            pcol = K.local_scalar("int32", init=x & ~1)
            prow0 = K.local_scalar("int32", init=row0 + phalf * 16)
            ps = K.local_scalar("int32", init=pcol >> 6)
            pr = K.local_scalar("int32", init=pcol & 63)
            is_odd = phalf != K.int32(0)
            cyc = K.local_scalar("int32", init=K.int32(0))

            def tmem_at(col):
                return K.Cast("uint32", tm[0] + col + (quad << 21))

            def ld32(regs, col, base=0):
                K.ptx[TC_LD32](*(regs[base + i] for i in range(32)), tmem_at(col))

            def ld8(regs, col, base=0):
                K.ptx[TC_LD8](*(regs[base + i] for i in range(8)), tmem_at(col))

            def ld4(regs, col, base=0):
                K.ptx[TC_LD4](*(regs[base + i] for i in range(4)), tmem_at(col))

            def st_row(stage0, col0, words, wbase=0, nunits=4):
                """Write this thread's row x, columns [col0, col0 + 8*nunits) of the [128][64] tile at stage0/stage0+1."""
                for u in range(nunits):
                    K.ptx["st.shared.v4.b32"](TT[stage0 + xs].ptr_to(xr, col0 + 8 * u),
                                              words[wbase + 4 * u], words[wbase + 4 * u + 1],
                                              words[wbase + 4 * u + 2], words[wbase + 4 * u + 3])

            def st_pair_rows(stage0, words):
                """Pair layout: rows pcol and pcol+1, columns [prow0, prow0+16): words[0:8] row pcol, words[8:16] row pcol+1."""
                for r in range(2):
                    for u in range(2):
                        K.ptx["st.shared.v4.b32"](TT[stage0 + ps].ptr_to(pr + r, prow0 + 8 * u),
                                                  words[8 * r + 4 * u], words[8 * r + 4 * u + 1],
                                                  words[8 * r + 4 * u + 2], words[8 * r + 4 * u + 3])

            def bar_all():
                K.ptx.bar.sync(K.uint32(1), K.uint32(256))

            def bar_wg():
                K.ptx.bar.sync(K.uint32(2) + K.Cast("uint32", wg), K.uint32(128))

            def twait(nm):
                TC[nm].wait(0, cyc & K.int32(1))
                K.ptx[TC_FENCE_AFTER]()
                K.ptx["bar.warp.sync"](K.uint32(0xFFFFFFFF))

            def marrive(nm):
                K.ptx[TC_FENCE_BEFORE]()
                MB[nm].arrive(0)

            def lo(w):
                return K.reinterpret("float32", w << K.uint32(16))

            def hi(w):
                return K.reinterpret("float32", w & K.uint32(0xFFFF0000))

            def shfl_xor1(val):
                r = K.local_scalar("uint32")
                K.ptx.shfl_sync.bfly.b32(r, K.reinterpret("uint32", val), K.uint32(1), K.uint32(0x1F),
                                         K.uint32(0xFFFFFFFF))
                return K.reinterpret("float32", r)


            def q_ptr(c, col):
                return TT[ST_Q + (col >> 6)].ptr_to(c, col & 63)

            def k_ptr(c, col):
                return TT[ST_K + (col >> 6)].ptr_to(c, col & 63)

            def v_ptr(c, col):
                return TT[ST_V + (col >> 6)].ptr_to(c, col & 63)

            def e_ptr(c, col):
                return TT[ST_G + (col >> 6)].ptr_to(c, col & 63)

            def load_transpose_frag(base, frag):
                """Load this warp's 32x32 block as four groups of four 8x8 fragments."""
                col0 = (quad & K.int32(1)) * K.int32(32)
                tile = TT[base + xs]
                for rb in range(2):
                    for cb in range(2):
                        o = 4 * (2 * rb + cb)
                        K.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            frag[o], frag[o + 1], frag[o + 2], frag[o + 3],
                            tile.m8n8x4(row0 + K.int32(16 * rb), col0 + K.int32(16 * cb), lane),
                        )

            def store_transpose_frag(base, frag):
                """Transpose those fragments in place, turning [token,channel] into [channel,token]."""
                col0 = (quad & K.int32(1)) * K.int32(32)
                tile = TT[base + xs]
                mm = lane >> K.int32(3)
                jj = lane & K.int32(7)
                for rb in range(2):
                    for cb in range(2):
                        o = 4 * (2 * rb + cb)
                        # .trans swaps each 8x8 atom.  Swap the x4 atom positions too:
                        # source (rb, cb) becomes destination (cb, rb).
                        ptr = tile.ptr_to(
                            col0 + K.int32(16 * cb) + (mm >> K.int32(1)) * K.int32(8) + jj,
                            row0 + K.int32(16 * rb) + (mm & K.int32(1)) * K.int32(8),
                        )
                        K.ptx["stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"](
                            ptr, frag[o], frag[o + 1], frag[o + 2], frag[o + 3]
                        )


            egA = K.alloc_local([16], "float32")
            egB = K.alloc_local([16], "float32")
            enA = K.alloc_local([16], "float32")
            enB = K.alloc_local([16], "float32")
            egcw = K.alloc_local([16], "uint32")
            t4 = K.alloc_local([4], "float32")
            acc = K.alloc_local([64], "float32")
            wds = K.alloc_local([32], "uint32")
            dgv = K.alloc_local([32], "float32")
            gn = K.local_scalar("float32")
            egn = K.local_scalar("float32")
            dgk = K.local_scalar("float32")
            dgk_k = K.local_scalar("float32")
            t0 = K.local_scalar("float32")
            t1 = K.local_scalar("float32")
            u16 = K.local_scalar("uint16")
            u16b = K.local_scalar("uint16")

            def ex2(dst, val):
                K.ptx.ex2.approx.ftz.f32(dst, val)

            def rcp(dst, val):
                K.ptx.rcp.approx.ftz.f32(dst, val)

            def load_u16_pair(words, i, ptr):
                if i % 2 == 0:
                    K.ptx.ld.global_.nc.u16(u16, ptr)
                else:
                    K.ptx.ld.global_.nc.u16(u16b, ptr)
                    K.ptx.mov.b32(words[i >> 1], u16, u16b)

            def load_u16_pair_sh(words, i, ptr):
                if i % 2 == 0:
                    K.ptx.ld.shared.u16(u16, ptr)
                else:
                    K.ptx.ld.shared.u16(u16b, ptr)
                    K.ptx.mov.b32(words[i >> 1], u16, u16b)

            def shfl_xor1_u32(val):
                r = K.local_scalar("uint32")
                K.ptx.shfl_sync.bfly.b32(r, val, K.uint32(1), K.uint32(0x1F), K.uint32(0xFFFFFFFF))
                return r

            def gcol_ptr(tensor, i):
                """Global pointer to row (row0+i) of this thread's column (clamped to the last valid row)."""
                tokc = tok0 + K.Cast("int64", K.min(row0 + K.int32(i), rows - K.int32(1)))
                return tensor.ptr_to([tokc * HK64 + gcol])

            def s_beta_row(c):
                b = K.local_scalar("float32")
                K.ptx.ld.shared.f32(b, K.address_of(s_beta[c]))
                return b

            def wsel(cond, a, b):
                return K.Select(cond, a, b)

            phase, phase_end = make_phaser()

            with K.serial(n_p2) as i2:
                work = K.local_scalar("int32", init=p2_chain(i2))
                seq, head, bos, seq_len, nch = work_coords(work)
                head64 = K.Cast("int64", head)
                gcol = K.local_scalar("int64", init=head64 * K.int64(D) + x64)
                with K.serial(nch) as rn:
                    n = nch - K.int32(1) - rn
                    par = cyc & K.int32(1)



                    rows = K.int32(CHUNK)
                    last = K.int32(CHUNK - 1)
                    tok0 = K.local_scalar("int64", init=bos + K.Cast("int64", n * K.int32(CHUNK)))
                    x_base = K.local_scalar("int64", init=(tok0 + K.Cast("int64", row0)) * HK64 + gcol)

                    phase("w-in")
                    b_in_full.wait(0, par)
                    b_eg_full.wait(0, par)
                    phase("c0")
                    with K.If(lane < K.int32(8)), K.Then():
                        btok = wr * K.int32(8) + lane
                        K.ptx.ld.shared.u16(u16, s_beta_in.ptr_to([btok, head & K.int32(7)]))
                        K.ptx.st.shared.f32(K.address_of(s_beta[btok]), lo(K.Cast("uint32", u16)))
                    # Read the final gate while ST_G is still token-major: lanes
                    # then span adjacent channels and the access is conflict-free.
                    K.ptx.ld.shared.u16(u16, e_ptr(last, x))
                    K.assign(egn, lo(K.Cast("uint32", u16)))
                    qw = K.alloc_local([16], "uint32")
                    kw = K.alloc_local([16], "uint32")
                    vw = K.alloc_local([16], "uint32")
                    qc = K.alloc_local([16], "uint32")
                    kc = K.alloc_local([16], "uint32")
                    vc = K.alloc_local([16], "uint32")

                    # All four TMA tiles start as [64 tokens, 128 channels].  Matrix
                    # load/store transposes each 32x32 warp block in place, after
                    # which every channel owner can fetch its 32-token row with four
                    # conflict-free 16-byte loads.  The first CTA barrier protects
                    # off-diagonal blocks from being overwritten before their owner
                    # has loaded them; the second publishes all transposed blocks.
                    load_transpose_frag(ST_Q, qw)
                    load_transpose_frag(ST_K, kw)
                    load_transpose_frag(ST_V, vw)
                    load_transpose_frag(ST_G, wds)
                    bar_all()
                    store_transpose_frag(ST_Q, qw)
                    store_transpose_frag(ST_K, kw)
                    store_transpose_frag(ST_V, vw)
                    store_transpose_frag(ST_G, wds)
                    # Each warp reads exactly the 32x32 transposed block that
                    # it just produced; only its own lanes need to rendezvous.
                    K.ptx["bar.warp.sync"](K.uint32(0xFFFFFFFF))

                    for u in range(4):
                        K.ptx["ld.shared.v4.b32"](
                            qc[4 * u], qc[4 * u + 1], qc[4 * u + 2], qc[4 * u + 3],
                            TT[ST_Q + xs].ptr_to(xr, row0 + K.int32(8 * u)),
                        )
                        K.ptx["ld.shared.v4.b32"](
                            kc[4 * u], kc[4 * u + 1], kc[4 * u + 2], kc[4 * u + 3],
                            TT[ST_K + xs].ptr_to(xr, row0 + K.int32(8 * u)),
                        )
                        K.ptx["ld.shared.v4.b32"](
                            egcw[4 * u], egcw[4 * u + 1], egcw[4 * u + 2], egcw[4 * u + 3],
                            TT[ST_G + xs].ptr_to(xr, row0 + K.int32(8 * u)),
                        )
                        K.ptx["ld.shared.v4.b32"](
                            vc[4 * u], vc[4 * u + 1], vc[4 * u + 2], vc[4 * u + 3],
                            TT[ST_V + xs].ptr_to(xr, row0 + K.int32(8 * u)),
                        )
                    for i in range(16):
                        K.assign(egA[i], lo(wds[i])); K.assign(egB[i], hi(wds[i]))
                        rcp(enA[i], egA[i]); rcp(enB[i], egB[i])
                        matrix = i & 3
                        rb = i >> 3
                        brow = row0 + K.int32(16 * rb + 8 * (matrix & 1)) + (lane >> K.int32(2))
                        bta = s_beta_row(brow)
                        # Preserve each ldmatrix fragment's register layout so
                        # stmatrix.trans performs the derived-operand transpose.
                        pack_bf16x2(
                            wds[i],
                            lo(kw[i]) * egA[i] * bta,
                            hi(kw[i]) * egB[i] * bta,
                        )
                        pack_bf16x2(
                            qw[i],
                            lo(qw[i]) * egA[i] * scale,
                            hi(qw[i]) * egB[i] * scale,
                        )
                        pack_bf16x2(
                            kw[i],
                            lo(kw[i]) * enA[i],
                            hi(kw[i]) * enB[i],
                        )

                    phase("w-chunk")
                    TC["chunk_done"].wait(0, par ^ K.int32(1))
                    K.ptx[TC_FENCE_AFTER]()
                    K.ptx[FENCE_ASYNC]()
                    phase("c1")
                    store_transpose_frag(T1, qw)
                    store_transpose_frag(T2, kw)
                    store_transpose_frag(T3, wds)
                    K.ptx[FENCE_ASYNC]()
                    # Z = vb - h^T kbg starts from vb in fp32: this thread's channel row of
                    # the transposed v tile times beta per token goes straight into TMEM.
                    for half in range(2):
                        vb32 = K.alloc_local([16], "float32")
                        for p in range(8):
                            i = 16 * half + 2 * p
                            bpair = K.alloc_local([2], "float32")
                            K.ptx["ld.shared.v2.f32"](bpair[0], bpair[1], K.address_of(s_beta[row0 + i]))
                            K.assign(vb32[2 * p], lo(vc[i >> 1]) * bpair[0])
                            K.assign(vb32[2 * p + 1], hi(vc[i >> 1]) * bpair[1])
                        K.ptx[TC_ST16](tmem_at(S2 + wg * 32 + 16 * half), *(vb32[j] for j in range(16)))
                    K.ptx[WAIT_ST]()
                    marrive("t_early")
                    phase("c1c")

                    phase("w-h")
                    b_h_full.wait(0, par)
                    K.ptx[TC_FENCE_AFTER]()
                    phase("c2")
                    dgk2 = K.local_scalar("uint64", init=K.cuda.make_float2(K.float32(0.0), K.float32(0.0)))
                    hst = K.local_scalar("int32", init=wg * 2 + xs)
                    dbase = ((K.Cast("int64", seq) * K.int64(H) + head64) * K.int64(D) + x64) * K.int64(D) \
                        + K.Cast("int64", wg * 64)
                    with K.If(rn == K.int32(0)):
                        with K.Then():
                            for m in range(8):
                                K.ptx["ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.f32"](
                                    *(acc[8 * m + i] for i in range(8)),
                                    dht.ptr_to([dbase + K.int64(8 * m)]))
                        with K.Else():
                            ld32(acc, TM_DH + wg * 64)
                            ld32(acc, TM_DH + wg * 64 + 32, 32)
                            K.ptx[WAIT_LD]()
                    for half in range(2):
                        hc = wg * 64 + 32 * half
                        a0 = 32 * half


                        for p in range(16):
                            dpair = K.local_scalar("uint64")
                            K.ptx["mul.rn.f32x2"](dpair, K.cuda.make_float2(acc[a0 + 2 * p], acc[a0 + 2 * p + 1]),
                                                  K.cuda.make_float2(egn, egn))
                            K.assign(acc[a0 + 2 * p], K.cuda.float2_x(dpair))
                            K.assign(acc[a0 + 2 * p + 1], K.cuda.float2_y(dpair))
                        for u in range(4):
                            K.ptx["ld.shared.v4.b32"](wds[0], wds[1], wds[2], wds[3],
                                                      TT[S_H + hst].ptr_to(xr, 32 * half + 8 * u))
                            for p in range(4):
                                K.ptx["fma.rn.f32x2"](
                                    dgk2,
                                    K.cuda.make_float2(lo(wds[p]), hi(wds[p])),
                                    K.cuda.make_float2(acc[a0 + 8 * u + 2 * p], acc[a0 + 8 * u + 2 * p + 1]),
                                    dgk2,
                                )
                        for p in range(16):
                            pack_bf16x2(wds[p], acc[a0 + 2 * p], acc[a0 + 2 * p + 1])
                        for u in range(4):
                            K.ptx["st.shared.v4.b32"](TT[DHB + hst].ptr_to(xr, 32 * half + 8 * u),
                                                      wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                        K.ptx[TC_ST32](tmem_at(TM_DH + hc), *(acc[a0 + i] for i in range(32)))
                    K.assign(dgk, K.cuda.float2_x(dgk2) + K.cuda.float2_y(dgk2))
                    K.ptx[WAIT_ST]()
                    K.ptx[FENCE_ASYNC]()
                    marrive("dhb_ready")


                    def readout_to_tile(slot, stage0):
                        ld32(acc, slot + wg * 32)
                        K.ptx[WAIT_LD]()
                        for p in range(16):
                            pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                        st_row(stage0, row0, wds, 0, 4)
                        K.ptx[FENCE_ASYNC]()

                    phase("w-Z")
                    twait("Z_done")
                    phase("c3")
                    readout_to_tile(S2, ZT)
                    marrive("zT_ready")
                    phase("w-dv2")
                    twait("dv2_done")
                    phase("c5")
                    readout_to_tile(S3, DV2)
                    marrive("dv2T_ready")
                    phase("w-Vn")
                    twait("Vn_done")
                    phase("c4")
                    readout_to_tile(S2, T6)
                    marrive("vnT_ready")


                    def readout64(slot, stage, mask, scale_by=None, negate=False):
                        ld32(acc, slot + wg * 32)
                        K.ptx[WAIT_LD]()
                        cc = quad * 16 + lane
                        with K.If(lane < K.int32(16)), K.Then():
                            for p in range(16):
                                vv2 = []
                                for e in range(2):
                                    jj = row0 + 2 * p + e
                                    val = acc[2 * p + e]
                                    if scale_by is not None:
                                        val = val * scale_by
                                    if negate:
                                        val = K.float32(0.0) - val
                                    vv2.append(val if mask is None else K.Select(mask(cc, jj), val, K.float32(0.0)))
                                pack_bf16x2(wds[p], vv2[0], vv2[1])
                            for u in range(4):
                                K.ptx["st.shared.v4.b32"](TT[stage].ptr_to(cc, row0 + 8 * u),
                                                          wds[4 * u], wds[4 * u + 1], wds[4 * u + 2], wds[4 * u + 3])
                        K.ptx[FENCE_ASYNC]()

                    phase("w-dAs")
                    twait("dAs_done")
                    phase("c8")
                    # dAs lives in S4 (dq is issued later); its masked tile goes to stage 9.
                    readout64(S4, DAM, lambda cc, jj: jj < cc)
                    marrive("dAm_ready")
                    phase("w-dAqk")
                    twait("dAqk_done")
                    phase("c7")
                    # T5 (stage 8) held ZT until the dAs MMA consumed it.
                    readout64(S1, T5, lambda cc, jj: jj <= cc)
                    marrive("dAqk_tile_ready")
                    twait("dk_done")
                    phase("w-dvb")
                    twait("dvb_done")
                    phase("passA")

                    pbx = K.local_scalar("int32", init=K.int32(PB0) + wg * K.int32(PB1 - PB0))

                    def pass_a(full):
                        ld8(acc, S3 + wg * 32, 0)
                        K.ptx[WAIT_LD]()
                        for b in range(4):
                            ab = 8 * (b % 2)
                            if b < 3:
                                ld8(acc, S3 + wg * 32 + 8 * (b + 1), 8 * ((b + 1) % 2))
                            vq = K.alloc_local([4], "uint32")
                            K.ptx["ld.shared.v4.b32"](
                                vq[0], vq[1], vq[2], vq[3],
                                TT[ST_V + xs].ptr_to(xr, row0 + 8 * b),
                            )
                            dbp = K.alloc_local([8], "float32")
                            for e in range(8):
                                i = 8 * b + e
                                vv = lo(vq[e >> 1]) if e % 2 == 0 else hi(vq[e >> 1])
                                K.assign(dbp[e], acc[ab + e] * vv)

                                def st_dv(i=i, e=e, ab=ab):
                                    K.ptx.cvt.rn.bf16.f32(u16, acc[ab + e] * s_beta_row(row0 + i))
                                    K.ptx["st.global.L1::no_allocate.b16"](
                                        dv.ptr_to([x_base + K.int64(i * HK)]), u16)
                                if full:
                                    st_dv()
                                else:
                                    with K.If(row0 + K.int32(i) < rows), K.Then():
                                        st_dv()




                            for e in range(8):
                                i = 8 * b + e
                                K.ptx.st.shared.f32(TT[pbx + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane), dbp[e])
                            # bf16 dvb tile: operand of dwb = -h dvb and of dh -= T3 dvb^T.
                            dvw = K.alloc_local([4], "uint32")
                            for p in range(4):
                                pack_bf16x2(dvw[p], acc[ab + 2 * p], acc[ab + 2 * p + 1])
                            K.ptx["st.shared.v4.b32"](TT[DVB + xs].ptr_to(xr, row0 + 8 * b),
                                                      dvw[0], dvw[1], dvw[2], dvw[3])
                            if b < 3:
                                K.ptx[WAIT_LD]()

                    pass_a(True)
                    K.ptx[FENCE_ASYNC]()
                    marrive("dv_epi_done")
                    phase("w-X")
                    twait("X_done")
                    phase("c9")
                    readout64(S1, T6, None)
                    marrive("X_ready")
                    phase("dbv")
                    bar_wg()
                    tq = lane & K.int32(3)
                    ti = quad * 8 + (lane >> 2)
                    srow = (quad & K.int32(1)) * 32 + lane
                    dsum_v = K.local_scalar("float32", init=K.float32(0.0))
                    for u in range(8):
                        K.ptx["ld.shared.v4.f32"](t4[0], t4[1], t4[2], t4[3], TT[pbx + (quad >> 1)].ptr_to(srow, 8 * u))
                        K.assign(dsum_v, dsum_v + ((t4[0] + t4[1]) + (t4[2] + t4[3])))
                    # Stages 12-17 (partials, v) are dead now: let the loader bring the next chunk's q/k/v.
                    K.ptx[FENCE_ASYNC]()
                    b_mid_free.arrive(0)
                    phase("w-Y")
                    twait("Y_done")
                    phase("c10")
                    readout64(S2, T5 + 1, lambda cc, jj: jj < cc, negate=True)
                    marrive("intra_ready")


                    phase("w-epi")
                    twait("dq2_done")
                    phase("epi")
                    K.assign(dgk_k, K.float32(0.0))

                    def q_loads(b, base):
                        ld8(acc, S4 + wg * 32 + 8 * b, base)

                    def k_loads(b, base):
                        ld4(acc, S5 + wg * 32 + 4 * b, base + 0)
                        ld4(acc, S6 + wg * 32 + 4 * b, base + 4)
                        ld4(acc, S3 + wg * 32 + 4 * b, base + 8)

                    def epilogue(full):
                        assert full
                        pair0 = K.local_scalar("uint64")
                        pair1 = K.local_scalar("uint64")
                        pair2 = K.local_scalar("uint64")
                        pair3 = K.local_scalar("uint64")
                        pair4 = K.local_scalar("uint64")
                        pair5 = K.local_scalar("uint64")

                        q_loads(0, 0)
                        K.ptx[WAIT_LD]()
                        for b in range(4):
                            ab = 8 * (b % 2)
                            if b < 3:
                                q_loads(b + 1, 8 * ((b + 1) % 2))
                            for p in range(4):
                                i = 8 * b + 2 * p
                                # Prepare the inverse gates while q readout and
                                # stores are in flight, before the dkt wait.
                                # Their live range stays inside the epilogue.
                                rcp(enA[i >> 1], lo(egcw[i >> 1]))
                                rcp(enB[i >> 1], hi(egcw[i >> 1]))
                                K.ptx["mul.rn.f32x2"](
                                    pair0,
                                    K.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    K.cuda.make_float2(lo(egcw[i >> 1]) * scale, hi(egcw[i >> 1]) * scale),
                                )
                                K.ptx["st.global.L1::no_allocate.f32"](
                                    dq.ptr_to([x_base + K.int64(i * HK)]), K.cuda.float2_x(pair0))
                                K.ptx["st.global.L1::no_allocate.f32"](
                                    dq.ptr_to([x_base + K.int64((i + 1) * HK)]), K.cuda.float2_y(pair0))
                                K.ptx["mul.rn.f32x2"](pair1, K.cuda.make_float2(lo(qc[i >> 1]), hi(qc[i >> 1])), pair0)
                                K.assign(dgv[i], K.cuda.float2_x(pair1))
                                K.assign(dgv[i + 1], K.cuda.float2_y(pair1))
                            if b < 3:
                                K.ptx[WAIT_LD]()
                        twait("dkt_done")


                        dbx = 2 * wg
                        k_loads(0, 0)
                        K.ptx[WAIT_LD]()
                        for b in range(8):
                            ab = 12 * (b % 2)
                            if b < 7:
                                k_loads(b + 1, 12 * ((b + 1) % 2))
                            for p in range(2):
                                i = 4 * b + 2 * p
                                K.assign(pair0, K.cuda.make_float2(enA[i >> 1], enB[i >> 1]))
                                K.assign(pair1, K.cuda.make_float2(lo(kc[i >> 1]), hi(kc[i >> 1])))
                                K.ptx["add.rn.f32x2"](
                                    pair2,
                                    K.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    K.cuda.make_float2(acc[ab + 8 + 2 * p], acc[ab + 8 + 2 * p + 1]),
                                )
                                K.ptx["mul.rn.f32x2"](pair2, pair2, pair0)
                                K.ptx["mul.rn.f32x2"](
                                    pair3,
                                    K.cuda.make_float2(acc[ab + 4 + 2 * p], acc[ab + 4 + 2 * p + 1]),
                                    K.cuda.make_float2(lo(egcw[i >> 1]), hi(egcw[i >> 1])),
                                )
                                K.ptx["mul.rn.f32x2"](pair4, pair1, pair3)
                                K.ptx.st.shared.f32(
                                    TT[dbx + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane), K.cuda.float2_x(pair4))
                                K.ptx.st.shared.f32(
                                    TT[dbx + ((i + 1) >> 4)].ptr_to(4 * ((i + 1) & 15) + quad, 2 * lane),
                                    K.cuda.float2_y(pair4))
                                K.ptx["mul.rn.f32x2"](
                                    pair4,
                                    K.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    pair0,
                                )
                                K.ptx["mul.rn.f32x2"](pair5, pair1, pair4)
                                K.assign(dgk_k, dgk_k + K.cuda.float2_x(pair5) + K.cuda.float2_y(pair5))
                                K.ptx["mul.rn.f32x2"](
                                    pair3, pair3,
                                    K.cuda.make_float2(s_beta_row(row0 + i), s_beta_row(row0 + i + 1)),
                                )
                                K.ptx["add.rn.f32x2"](pair5, pair2, pair3)
                                K.ptx["st.global.L1::no_allocate.f32"](
                                    dk.ptr_to([x_base + K.int64(i * HK)]), K.cuda.float2_x(pair5))
                                K.ptx["st.global.L1::no_allocate.f32"](
                                    dk.ptr_to([x_base + K.int64((i + 1) * HK)]), K.cuda.float2_y(pair5))
                                K.ptx["sub.rn.f32x2"](pair3, pair3, pair2)
                                K.ptx["fma.rn.f32x2"](
                                    pair5, pair1, pair3, K.cuda.make_float2(dgv[i], dgv[i + 1]),
                                )
                                K.assign(dgv[i], K.cuda.float2_x(pair5))
                                K.assign(dgv[i + 1], K.cuda.float2_y(pair5))
                            if b < 7:
                                K.ptx[WAIT_LD]()

                        bar_wg()
                        dsum = K.local_scalar("float32", init=dsum_v)
                        for u in range(8):
                            K.ptx["ld.shared.v4.f32"](t4[0], t4[1], t4[2], t4[3], TT[dbx + (quad >> 1)].ptr_to(srow, 8 * u))
                            K.assign(dsum, dsum + ((t4[0] + t4[1]) + (t4[2] + t4[3])))
                        K.ptx[FENCE_ASYNC]()
                        b_h_free.arrive(0)
                        for s in (1, 2):
                            r = K.local_scalar("uint32")
                            K.ptx.shfl_sync.bfly.b32(r, K.reinterpret("uint32", dsum), K.uint32(s), K.uint32(0x1F), K.uint32(0xFFFFFFFF))
                            K.assign(dsum, dsum + K.reinterpret("float32", r))
                        with K.If(tq == K.int32(0)), K.Then():
                            K.ptx["st.global.L1::no_allocate.f32"](
                                db.ptr_to([(tok0 + K.Cast("int64", row0 + ti)) * K.int64(H) + head64]), dsum)

                    epilogue(True)
                    phase("cumsum")
                    # dg = local suffix sum + (per-channel constant C) [+ wg1 total for wg0].
                    # Each warpgroup publishes what the other needs, then one barrier.
                    for i in range(30, -1, -1):
                        K.assign(dgv[i], dgv[i] + dgv[i + 1])
                    K.ptx.st.shared.f32(K.address_of(s_dgk[wg, x]),
                                        dgk + dgk_k + K.Select(wg == K.int32(0), K.float32(0.0), dgv[0]))
                    b_dg0_ready.arrive(0)
                    b_dg0_ready.wait(0, cyc & K.int32(1))
                    K.ptx.ld.shared.f32(t0, K.address_of(s_dgk[K.int32(1) - wg, x]))
                    K.assign(t1, t0 + dgk + dgk_k)
                    for i in range(32):
                        K.assign(dgv[i], dgv[i] + t1)
                    for i in range(32):
                        K.ptx["st.global.L1::no_allocate.f32"](
                            dg.ptr_to([x_base + K.int64(i * HK)]), dgv[i])
                    phase_end()
                    K.assign(cyc, cyc + K.int32(1))

                phase("dh0")
                TC["chunk_done"].wait(0, (cyc & K.int32(1)) ^ K.int32(1))
                K.ptx[TC_FENCE_AFTER]()
                ld32(acc, TM_DH + wg * 64)
                ld32(acc, TM_DH + wg * 64 + 32, 32)
                K.ptx[WAIT_LD]()
                obase = ((K.Cast("int64", seq) * K.int64(H) + head64) * K.int64(D) + x64) * K.int64(D) \
                    + K.Cast("int64", wg * 64)
                for m in range(8):
                    K.ptx["st.global.L1::no_allocate.v8.f32"](
                        dh0.ptr_to([obase + K.int64(8 * m)]),
                        *(acc[8 * m + i] for i in range(8)))
                phase_end()

        with auxg:



            with mma:
                p1_mma()
                K.ptx.bar.sync(K.uint32(5), K.uint32(384))
                tm = tmem_preamble()
                cyc = K.local_scalar("int32", init=K.int32(0))

                def mwait(nm):
                    MB[nm].wait(0, cyc & K.int32(1))
                    K.ptx[TC_FENCE_AFTER]()

                mphase, mphase_end = make_phaser()

                bd = K.alloc_local([1], "uint64")
                zq = K.alloc_local([1], "int32")
                op_T1k = Op(bd, T1, 128, 64, "k")
                op_T1mn = Op(bd, T1, 128, 128, "mn")
                op_T2k = Op(bd, T2, 128, 64, "k")
                op_T2mn = Op(bd, T2, 128, 128, "mn")
                op_T3k = Op(bd, T3, 128, 64, "k")
                op_T3mn = Op(bd, T3, 128, 128, "mn")
                op_T5k = Op(bd, T5, 128, 64, "k")
                op_ZTk = Op(bd, ZT, 128, 64, "k")
                op_ZTmn = Op(bd, ZT, 128, 128, "mn")
                op_dAqk_k = Op(bd, T5, 64, 64, "k")
                op_dAkk_k = Op(bd, T5 + 1, 64, 64, "k")
                op_T6mn = Op(bd, T6, 128, 128, "mn")
                op_dAqk_mn = Op(bd, T5, 64, 64, "mn")
                op_dAm_k = Op(bd, DAM, 64, 64, "k")
                op_X_mn = Op(bd, T6, 64, 64, "mn")
                op_dAkk_mn = Op(bd, T5 + 1, 64, 64, "mn")
                op_DHBk = Op(bd, DHB, 128, 128, "k")
                op_DHBmn = Op(bd, DHB, 128, 128, "mn")
                op_DV2k = Op(bd, DV2, 128, 64, "k")
                op_DV2mn = Op(bd, DV2, 128, 128, "mn")
                op_DVBk = Op(bd, DVB, 128, 64, "k")
                op_DVBmn = Op(bd, DVB, 128, 128, "mn")
                op_do_k128 = Op(bd, S_DO, 64, 128, "k")
                op_do_mn64 = Op(bd, S_DO, 64, 64, "mn")
                op_h_k = Op(bd, S_H, 128, 128, "k")
                op_h_mn = Op(bd, S_H, 128, 128, "mn")
                op_aqk_mn = Op(bd, S_AQK, 64, 64, "mn")
                op_akk_k = Op(bd, S_AKK, 64, 64, "k")
                op_akk_mn = Op(bd, S_AKK, 64, 64, "mn")
                ID_128x64 = idesc(128, 64)
                ID_128x64_TATB_NB = idesc(128, 64, ta=1, tb=1, nb=1)
                ID_128x64_TATB = idesc(128, 64, ta=1, tb=1)
                ID_128x64_TB = idesc(128, 64, tb=1)
                ID_128x64_TB_NA = idesc(128, 64, tb=1, na=1)
                ID_128x128_TB = idesc(128, 128, tb=1)
                ID_128x128_NB = idesc(128, 128, nb=1)
                ID_128x128 = idesc(128, 128)
                ID_64x64_TB = idesc(64, 64, tb=1)
                ID_64x64_TATB = idesc(64, 64, ta=1, tb=1)
                ID_64x64 = idesc(64, 64)

                with K.serial(n_p2) as i2:
                    work = K.local_scalar("int32", init=p2_chain(i2))
                    seq, head, bos, seq_len, nch = work_coords(work)
                    with K.serial(nch) as rn:
                        par = cyc & K.int32(1)
                        K.ptx.ld.volatile.shared.s32(zq[0], K.address_of(s_tmem[1]))
                        K.cuda.tcgen05.encode_matrix_descriptor(
                            K.address_of(bd[0]), TT[zq[0]].ptr_to(0, 0), ldo=Op.LBO_BASE, sdo=SBO_UNITS,
                            swizzle=K.SW128B.value)
                        akk_u = K.local_scalar("uint64", init=K.Cast("uint64", par) * K.uint64(UNITS_PER_STAGE))
                        mphase("mw-early")
                        mwait("t_early")
                        b_akk_full.wait(par, (cyc >> 1) & K.int32(1))
                        b_h_full.wait(0, par)
                        K.ptx[TC_FENCE_AFTER]()
                        mphase("m-Z")
                        with K.If(elected()), K.Then():
                            # Z = vb - h^T kbg: vb was deposited in S2 by the compute warps.
                            mma_chain(tm, S2, op_h_mn, op_T3mn, ID_128x64_TATB_NB, True)
                            TC["Z_done"].arrive(0)
                        mphase("mw-aqk")
                        b_aqk_masked.wait(0, par)
                        b_do_full.wait(0, par)
                        K.ptx[TC_FENCE_AFTER]()
                        mphase("m-dvp")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S3, op_do_mn64, op_aqk_mn, ID_128x64_TATB, False)
                            b_aqk_empty.arrive(0)
                        mphase("mw-dhb")
                        mwait("dhb_ready")
                        mphase("m-dv2")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S3, op_DHBmn, op_T2mn, ID_128x64_TATB, True)
                            TC["dv2_done"].arrive(0)
                        mphase("mw-zT")
                        mwait("zT_ready")
                        mphase("m-Vn")
                        with K.If(elected()), K.Then():
                            # Vn^T = Z Akk^T
                            mma_chain(tm, S2, op_ZTk, op_akk_k, ID_128x64, False, b_units=akk_u)
                            TC["Vn_done"].arrive(0)
                        mphase("mw-dv2T")
                        mwait("dv2T_ready")
                        mphase("m-dAs")
                        with K.If(elected()), K.Then():
                            # dAs = dv2 Z^T, parked in S4 until its masked readout
                            mma_chain(tm, S4, op_DV2mn, op_ZTmn, ID_64x64_TATB, False)
                            TC["dAs_done"].arrive(0)
                            mma_chain(tm, S3, op_DV2k, op_akk_mn, ID_128x64_TB, False, b_units=akk_u)
                            TC["dvb_done"].arrive(0)
                        mphase("mw-vnT")
                        mwait("vnT_ready")
                        mphase("m-dAqk")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S1, op_do_k128, op_T6mn, ID_64x64_TB, False)
                            TC["dAqk_done"].arrive(0)
                            mma_chain(tm, S5, op_DHBk, op_T6mn, ID_128x64_TB, False)
                            TC["dk_done"].arrive(0)
                        mphase("mw-dAm")
                        mwait("dAm_ready")
                        mwait("dAqk_tile_ready")
                        mphase("m-X")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S1, op_dAm_k, op_akk_k, ID_64x64, False, b_units=akk_u)
                            TC["X_done"].arrive(0)
                            # dq = h do^T + T2 dAqk^T needs only the dAqk tile: finish it now so the
                            # epilogue q-part never waits; then dh += q do while pass A runs.
                            mma_chain(tm, S4, op_h_k, op_do_k128, ID_128x64, False)
                            mma_chain(tm, S4, op_T2k, op_dAqk_k, ID_128x64, True)
                            TC["dq2_done"].arrive(0)
                            mma_chain(tm, TM_DH, op_T1k, op_do_mn64, ID_128x128_TB, True)
                            b_do_empty.arrive(0)
                        mphase("mw-dvepi")
                        mwait("dv_epi_done")
                        mphase("m-dwb")
                        with K.If(elected()), K.Then():
                            # dwb = -h dvb from the bf16 dvb tile.
                            mma_chain(tm, S6, op_h_k, op_DVBmn, ID_128x64_TB_NA, False)
                        mphase("mw-X")
                        mwait("X_ready")
                        mphase("m-Y")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S2, op_akk_mn, op_X_mn, ID_64x64_TATB, False, a_units=akk_u)
                            TC["Y_done"].arrive(0)
                            b_akk_empty.arrive(par)
                        mphase("mw-intra")
                        mwait("intra_ready")
                        mphase("m-dk2")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S6, op_T2k, op_dAkk_k, ID_128x64, True)
                        mphase("m-dkt")
                        with K.If(elected()), K.Then():
                            mma_chain(tm, S3, op_T1k, op_dAqk_mn, ID_128x64_TB, False)
                            mma_chain(tm, S3, op_T3k, op_dAkk_mn, ID_128x64_TB, True)
                            TC["dkt_done"].arrive(0)
                            # dh -= T3 dvb^T is only needed by the next chunk's decay.
                            mma_chain(tm, TM_DH, op_T3k, op_DVBk, ID_128x128_NB, True)
                            TC["chunk_done"].arrive(0)
                        mphase_end()
                        K.assign(cyc, cyc + K.int32(1))




            with loader:
                with K.If(elected()), K.Then():
                    for m in (q_map, k_map, v_map, g_map, eg_map, beta_map, do_map, aqk_map, akk_map, h_map):
                        K.ptx.prefetch.tensormap(K.address_of(m))
                p1_loader()
                K.ptx.bar.sync(K.uint32(5), K.uint32(384))
                cyc = K.local_scalar("int32", init=K.int32(0))
                lphase, lphase_end = make_phaser()
                with K.serial(n_p2) as i2:
                    work = K.local_scalar("int32", init=p2_chain(i2))
                    seq, head, bos, seq_len, nch = work_coords(work)
                    bos32 = K.local_scalar("int32", init=K.Cast("int32", bos))
                    cb = chunk_base(seq)
                    head8 = K.local_scalar("int32", init=head >> K.int32(3))


                    lphase("lw-flag")
                    with K.If(elected()), K.Then():
                        fl = K.local_scalar("int32", init=K.int32(0))
                        with K.While(fl != epoch):
                            K.ptx.ld.acquire.gpu.global_.s32(fl, flags.ptr_to([work]))
                    K.ptx["bar.warp.sync"](K.uint32(0xFFFFFFFF))
                    K.ptx["fence.proxy.async.global"]()
                    lphase_end()
                    with K.serial(nch) as rn:
                        n = nch - K.int32(1) - rn
                        par = cyc & K.int32(1)
                        npar = par ^ K.int32(1)
                        tok0 = bos32 + n * K.int32(CHUNK)
                        hidx = (cb + n) * K.int32(H) + head

                        lphase("lw-mid")
                        b_mid_free.wait(0, npar)
                        lphase("l-issue")
                        with K.If(elected()), K.Then():
                            b_in_full.arrive(0, tx_count=IN_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_in_full.ptr_to([0]))
                            K.ptx[TMA_LD](s_beta_in.ptr_to([0, 0]), K.address_of(beta_map), K.int32(0), tok0, head8, mb)
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[ST_Q + d0 // 64].ptr_to(0, 0), K.address_of(q_map), K.int32(d0), tok0, head, mb)
                                K.ptx[TMA_LD](TT[ST_K + d0 // 64].ptr_to(0, 0), K.address_of(k_map), K.int32(d0), tok0, head, mb)
                                K.ptx[TMA_LD](TT[ST_V + d0 // 64].ptr_to(0, 0), K.address_of(v_map), K.int32(d0), tok0, head, mb)
                        lphase("lw-chunk")
                        TC["chunk_done"].wait(0, npar)
                        lphase("l-issue-eg")
                        with K.If(elected()), K.Then():
                            b_eg_full.arrive(0, tx_count=EG_BYTES)
                            mbe = K.cuda.cvta_generic_to_shared(b_eg_full.ptr_to([0]))
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[ST_G + d0 // 64].ptr_to(0, 0), K.address_of(eg_map), K.int32(d0), tok0, head, mbe)
                        b_do_empty.wait(0, npar)
                        with K.If(elected()), K.Then():
                            b_do_full.arrive(0, tx_count=DO_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_do_full.ptr_to([0]))
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[S_DO + d0 // 64].ptr_to(0, 0), K.address_of(do_map), K.int32(d0), tok0, head, mb)
                        b_h_free.wait(0, npar)
                        with K.If(elected()), K.Then():
                            b_h_full.arrive(0, tx_count=H_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_h_full.ptr_to([0]))
                            for d0 in (0, 64):
                                K.ptx[TMA_LD](TT[S_H + (d0 // 64) * 2].ptr_to(0, 0), K.address_of(h_map), K.int32(d0), K.int32(0), hidx, mb)
                        b_aqk_empty.wait(0, npar)
                        with K.If(elected()), K.Then():
                            b_aqk_full.arrive(0, tx_count=AQK_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_aqk_full.ptr_to([0]))
                            K.ptx[TMA_LD](TT[S_AQK].ptr_to(0, 0), K.address_of(aqk_map), K.int32(0), tok0, head, mb)
                        b_akk_empty.wait(par, ((cyc >> 1) & K.int32(1)) ^ K.int32(1))
                        with K.If(elected()), K.Then():
                            b_akk_full.arrive(par, tx_count=AQK_BYTES)
                            mb = K.cuda.cvta_generic_to_shared(b_akk_full.ptr_to([par]))
                            K.ptx[TMA_LD](TT[S_AKK + par].ptr_to(0, 0), K.address_of(akk_map), K.int32(0), tok0, head, mb)
                            with K.If(n == K.int32(0)), K.Then():
                                with K.If(i2 + K.int32(1) < n_p2), K.Then():
                                    nxt = p2_chain(i2 + K.int32(1))
                                    seq2, head2, bos2, seq_len2, nch2 = work_coords(nxt)
                                    tokn = K.Cast("int32", bos2) + (nch2 - K.int32(1)) * K.int32(CHUNK)
                                    hidn = (chunk_base(seq2) + nch2 - K.int32(1)) * K.int32(H) + head2
                                    for tmap in (q_map, k_map, v_map, do_map, eg_map):
                                        for d0 in (0, 64):
                                            K.ptx[TMA_PREFETCH](K.address_of(tmap), K.int32(d0), tokn, head2)
                                    K.ptx[TMA_PREFETCH](K.address_of(aqk_map), K.int32(0), tokn, head2)
                                    K.ptx[TMA_PREFETCH](K.address_of(akk_map), K.int32(0), tokn, head2)
                                    for d0 in (0, 64):
                                        K.ptx[TMA_PREFETCH](K.address_of(h_map), K.int32(d0), K.int32(0), hidn)
                            with K.If(n > K.int32(0)), K.Then():
                                tokp = tok0 - K.int32(CHUNK)
                                for tmap in (q_map, k_map, v_map, do_map):
                                    for d0 in (0, 64):
                                        K.ptx[TMA_PREFETCH](K.address_of(tmap), K.int32(d0), tokp, head)
                                for d0 in (0, 64):
                                    K.ptx[TMA_PREFETCH](K.address_of(eg_map), K.int32(d0), tokp, head)
                                K.ptx[TMA_PREFETCH](K.address_of(aqk_map), K.int32(0), tokp, head)
                                K.ptx[TMA_PREFETCH](K.address_of(akk_map), K.int32(0), tokp, head)
                                for d0 in (0, 64):
                                    K.ptx[TMA_PREFETCH](K.address_of(h_map), K.int32(d0), K.int32(0), hidx - K.int32(H))
                        lphase_end()
                        K.assign(cyc, cyc + K.int32(1))





            with idle:
                with K.If(K.warp_id_in_role() == K.int32(0)), K.Then():
                    p1_storer()
                K.ptx.bar.sync(K.uint32(5), K.uint32(384))
                cyc = K.local_scalar("int32", init=K.int32(0))
                rowc = K.local_scalar("int32", init=K.warp_id_in_role() * K.int32(32) + K.lane_id())
                with K.serial(n_p2) as i2:
                    work = K.local_scalar("int32", init=p2_chain(i2))
                    seq, head, bos, seq_len, nch = work_coords(work)
                    with K.serial(nch) as rn:
                        par = cyc & K.int32(1)
                        b_aqk_full.wait(0, par)
                        # Preserve the lower triangle of the eight diagonal
                        # 8x8 atoms with one matrix load/store per warp.  This
                        # replaces seven predicated scalar-store sites whose
                        # sparse lanes generated 2--4-way bank conflicts.
                        diag = K.alloc_local([4], "uint32")
                        dmat = K.lane_id() >> K.int32(3)
                        dblk = K.warp_id_in_role() * K.int32(4) + dmat
                        dptr = TT[S_AQK].ptr_to(
                            dblk * K.int32(8) + (K.lane_id() & K.int32(7)),
                            dblk * K.int32(8),
                        )
                        K.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            diag[0], diag[1], diag[2], diag[3], dptr
                        )
                        drow = K.lane_id() >> K.int32(2)
                        dcol = (K.lane_id() & K.int32(3)) * K.int32(2)
                        dmask = K.Select(
                            dcol > drow,
                            K.uint32(0),
                            K.Select(dcol == drow, K.uint32(0x0000FFFF), K.uint32(0xFFFFFFFF)),
                        )
                        for e in range(4):
                            K.assign(diag[e], diag[e] & dmask)
                        K.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            dptr, diag[0], diag[1], diag[2], diag[3]
                        )
                        for u in range(1, 8):
                            with K.If(K.int32(8 * u) > rowc), K.Then():
                                K.ptx["st.shared.v4.b32"](TT[S_AQK].ptr_to(rowc, 8 * u),
                                                          K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0))
                        K.ptx[FENCE_ASYNC]()
                        b_aqk_masked.arrive(0)
                        K.assign(cyc, cyc + K.int32(1))

        K.cuda.cta_sync()
        with K.If(K.warp_id() == 8), K.Then():
            K.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
            K.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                K.Cast("uint32", K.local_scalar("int32", init=tmem_preamble()[0])), K.uint32(TMEM_COLS_B))

    return kda_bwd_fused

SCHEDULE_CLASSES = [(40, 6, 2), (80, 5, 5), (32, 4, 9)]


def build_schedule(num_chains, num_ctas, classes):
    """Per-CTA (pass-2 chains, pass-1 chains) lists.

    classes == "legacy": base = num_chains // num_ctas; the `extra` lowest CTAs own base+1 pass-2 chains
    (strided) and run pass 1 for their first chain only; the remaining CTAs absorb the other heavy
    chains' pass-1 work first, then their own.  Otherwise classes is a list of (count, n_p2, n_p1):
    pass-2 chains are dealt in rounds; each CTA runs pass 1 for its own first n_p1 chains and, if it has
    spare pass-1 capacity, for other CTAs' later chains first (so those are published early).
    """
    if classes == "legacy":
        base = num_chains // num_ctas
        extra = num_chains - base * num_ctas
        p2 = [[c + r * num_ctas for r in range(base + (1 if c < extra else 0))] for c in range(num_ctas)]
        n_light = num_ctas - extra
        extra1 = max(extra, 1)
        heavy_items = [(t % extra1) + (t // extra1 + 1) * num_ctas for t in range(extra * base)]
        p1 = []
        for c in range(num_ctas):
            if c < extra:
                p1.append([c])
            else:
                lidx = c - extra
                p1.append([heavy_items[t] for t in range(lidx, extra * base, n_light)]
                          + [c + r * num_ctas for r in range(base)])
        return p2, p1
    cta_p2, cta_p1 = [], []
    for count, a, b in classes:
        cta_p2 += [a] * count
        cta_p1 += [b] * count
    assert len(cta_p2) == num_ctas and sum(cta_p2) == num_chains and sum(cta_p1) == num_chains, \
        (len(cta_p2), sum(cta_p2), sum(cta_p1))
    p2 = [[] for _ in range(num_ctas)]
    chain = 0
    for r in range(max(cta_p2)):
        for c in range(num_ctas):
            if cta_p2[c] > r:
                p2[c].append(chain)
                chain += 1
    leftover = []
    p1 = []
    for c in range(num_ctas):
        take = min(cta_p1[c], len(p2[c]))
        p1.append(p2[c][:take])
        leftover += p2[c][take:]
    li = 0
    for c in range(num_ctas):
        spare = cta_p1[c] - len(p1[c])
        if spare > 0:
            p1[c] = leftover[li:li + spare] + p1[c]
            li += spare
    assert li == len(leftover), (li, len(leftover))
    return p2, p1


def build_schedule_tensor(num_chains, num_ctas, classes, dev):
    p2, p1 = build_schedule(num_chains, num_ctas, classes)
    assert sorted(c for l in p2 for c in l) == list(range(num_chains))
    assert sorted(c for l in p1 for c in l) == list(range(num_chains))
    table = torch.full((num_ctas, SCHED_STRIDE), -1, dtype=torch.int32)
    for c in range(num_ctas):
        assert len(p2[c]) <= SCHED_MAXP2 and len(p1[c]) <= SCHED_MAXP1, (len(p2[c]), len(p1[c]))
        table[c, 0] = len(p2[c])
        table[c, 1] = len(p1[c])
        for i, ch in enumerate(p2[c]):
            table[c, 2 + i] = ch
        for i, ch in enumerate(p1[c]):
            table[c, 2 + SCHED_MAXP2 + i] = ch
    return table.reshape(-1).to(dev)



# ----------------------------------------------------------------------------
# Public kernel identity and supported configurations
# ----------------------------------------------------------------------------

KERNEL_META = {
    "name": "agent_evolved_kda_backward_b1_t8192_h96",
    "category": "agent_evolved",
    "runtime_cuda_archs": ["sm_100a"],
    "reference_requirements": (
        {
            "package": "flash-linear-attention",
            "git": {
                "url": "https://github.com/fla-org/flash-linear-attention.git",
                "commit": "9c8e42e762fce087c27b673af4922795d9edb85e",
            },
            "import": "fla",
        },
    ),
    "provenance": {
        "generator": "hmz",
        "run": "ablation-kda-bwd-2",
        "selected_version": "fused-zfold-pipeline",
    },
}

_PACKED_SEQ_LENS = (1024,) * 8
_SUPPORTED_SEQ_LENS = {_PACKED_SEQ_LENS}

# The schedule classes below deal 768 chains over exactly 152 CTAs, which is the
# B200 SM count this member was tuned on.  Any other CTA count falls back to the
# generic strided schedule the same builder implements.
_TUNED_NUM_CTAS = 152


@dataclass(frozen=True, slots=True)
class KDABackwardConfig:
    label: str
    num_heads: int
    seq_lens: tuple[int, ...]
    seed: int = 0
    scale: float = 1.0 / math.sqrt(D)

    def validate(self) -> None:
        if self.num_heads != 96:
            raise ValueError(f"num_heads must be 96, got {self.num_heads}")
        if self.seq_lens not in _SUPPORTED_SEQ_LENS:
            raise ValueError(f"unsupported KDA sequence layout {self.seq_lens}")
        if sum(self.seq_lens) != 8192:
            raise ValueError(f"total tokens must be 8192, got {sum(self.seq_lens)}")
        if not math.isclose(self.scale, 1.0 / math.sqrt(D), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"scale must be 1/sqrt({D}), got {self.scale}")

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def num_seqs(self) -> int:
        return len(self.seq_lens)

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @property
    def num_chains(self) -> int:
        return self.num_seqs * self.num_heads


CONFIGS = [
    {
        "label": "h96_packed_1024x8",
        "num_heads": 96,
        "seq_lens": _PACKED_SEQ_LENS,
        "seed": 2858210371,
    }
]


def _cfg(**kwargs: Any) -> KDABackwardConfig:
    names = {field.name for field in fields(KDABackwardConfig)}
    values = {name: value for name, value in kwargs.items() if name in names}
    if "seq_lens" in values:
        values["seq_lens"] = tuple(int(length) for length in values["seq_lens"])
    values.setdefault("label", "custom")
    cfg = KDABackwardConfig(**values)
    cfg.validate()
    return cfg


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    return make_fused_kernel(cfg.num_heads).func


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------


def _randn(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    device: torch.device,
    generator: torch.Generator,
    scale: float,
) -> torch.Tensor:
    return (torch.randn(shape, dtype=torch.float32, device=device, generator=generator) * scale).to(
        dtype
    )


def _l2_normalize_bf16(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor.float()
    return (values * torch.rsqrt(values.square().sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)


def _saved_interaction_matrices(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Aqk/Akk exactly as FLA's varlen intra path saves them for the backward."""
    with _native_fla_backend(), torch.no_grad():
        from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra

        outputs = chunk_kda_fwd_intra(
            q=case["q"],
            k=case["k"],
            v=case["v"],
            gk=case["g"],
            beta=case["beta"],
            scale=case["scale"],
            cu_seqlens=case["cu_seqlens"],
            chunk_size=CHUNK,
        )
    return outputs[4], outputs[5]


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = torch.device(kwargs.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved KDA backward")

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    token_shape = (1, cfg.total_tokens, cfg.num_heads, D)
    state_shape = (cfg.num_seqs, cfg.num_heads, D, D)
    q = _l2_normalize_bf16(
        _randn(token_shape, torch.bfloat16, device=device, generator=generator, scale=0.5)
    )
    k = _l2_normalize_bf16(
        _randn(token_shape, torch.bfloat16, device=device, generator=generator, scale=0.5)
    )
    v = _randn(token_shape, torch.bfloat16, device=device, generator=generator, scale=0.5)
    beta = torch.sigmoid(
        _randn(
            (1, cfg.total_tokens, cfg.num_heads),
            torch.float32,
            device=device,
            generator=generator,
            scale=0.5,
        )
    ).to(torch.bfloat16)
    gate_increments = -(
        0.01
        + 0.04
        * torch.rand(token_shape, dtype=torch.float32, device=device, generator=generator)
    )
    offsets = [0]
    for length in cfg.seq_lens:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device=device)

    with _native_fla_backend():
        from fla.ops.utils import chunk_local_cumsum

        g = chunk_local_cumsum(gate_increments, chunk_size=CHUNK, cu_seqlens=cu_seqlens)

    case: dict[str, Any] = {
        "config": cfg,
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "g": g,
        "scale": cfg.scale,
        "cu_seqlens": cu_seqlens,
        "initial_state": _randn(
            state_shape, torch.float32, device=device, generator=generator, scale=0.01
        ),
        "do": _randn(token_shape, torch.bfloat16, device=device, generator=generator, scale=0.25),
        "dht": _randn(
            state_shape, torch.float32, device=device, generator=generator, scale=0.01
        ),
    }
    case["Aqk"], case["Akk"] = _saved_interaction_matrices(case)

    case["dq"] = torch.empty(token_shape, dtype=torch.float32, device=device)
    case["dk"] = torch.empty(token_shape, dtype=torch.float32, device=device)
    case["dv"] = torch.empty_like(v)
    case["db"] = torch.empty(beta.shape, dtype=torch.float32, device=device)
    case["dg"] = torch.empty(token_shape, dtype=torch.float32, device=device)
    case["dh0"] = torch.empty(state_shape, dtype=torch.float32, device=device)

    # Kernel-owned scratch: bf16 chunk-state snapshots, the bf16 2^g cache, and
    # the epoch-stamped per-chain readiness flags.
    num_chunks_max = (cfg.total_tokens + CHUNK - 1) // CHUNK + cfg.num_seqs
    case["hsnap"] = torch.empty(
        (num_chunks_max, cfg.num_heads, D, D), dtype=torch.bfloat16, device=device
    )
    case["egcache"] = torch.empty_like(q)
    case["flags"] = torch.zeros((cfg.num_chains,), dtype=torch.int32, device=device)

    num_ctas = min(
        torch.cuda.get_device_properties(device).multi_processor_count, cfg.num_chains
    )
    classes = SCHEDULE_CLASSES if num_ctas == _TUNED_NUM_CTAS else "legacy"
    case["num_ctas"] = num_ctas
    case["sched"] = build_schedule_tensor(cfg.num_chains, num_ctas, classes, device)

    T, H = cfg.total_tokens, cfg.num_heads
    case["tensor_maps"] = {
        "q": token_map(q, T, H, D, 64),
        "k": token_map(k, T, H, D, 64),
        "v": token_map(v, T, H, D, 64),
        "g": token_map(g, T, H, D, 32),
        "do": token_map(case["do"], T, H, D, 64),
        "eg": token_map(case["egcache"], T, H, D, 64),
        "beta": token_map(beta, T, H // 8, 8, 8, swizzle=0),
        "aqk": token_map(case["Aqk"], T, H, CHUNK, CHUNK),
        "akk": token_map(case["Akk"], T, H, CHUNK, CHUNK),
        "h": state_map(case["hsnap"], num_chunks_max * H),
    }
    # The readiness flags are epoch stamped instead of reset, so every launch
    # must pass a strictly increasing epoch.
    case["epoch"] = 0
    return case


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    cfg: KDABackwardConfig = case["config"]
    maps = case["tensor_maps"]
    for name in (
        "q", "k", "v", "beta", "Aqk", "Akk", "g", "initial_state", "do", "dht",
        "dq", "dk", "dv", "db", "dg", "dh0",
    ):
        if not case[name].is_contiguous():
            raise AssertionError(f"{name} must be contiguous")
    return (
        case["q"].view(-1),
        case["k"].view(-1),
        case["v"].view(-1),
        case["beta"].view(-1),
        case["Aqk"].view(-1),
        case["Akk"].view(-1),
        case["g"].view(-1),
        case["egcache"].view(-1),
        case["do"].view(-1),
        case["dht"].view(-1),
        case["initial_state"].view(-1),
        case["hsnap"].view(-1),
        case["cu_seqlens"],
        case["flags"],
        case["sched"],
        case["dq"].view(-1),
        case["dk"].view(-1),
        case["dv"].view(-1),
        case["db"].view(-1),
        case["dg"].view(-1),
        case["dh0"].view(-1),
        maps["q"].ptr,
        maps["k"].ptr,
        maps["v"].ptr,
        maps["g"].ptr,
        maps["eg"].ptr,
        maps["beta"].ptr,
        maps["do"].ptr,
        maps["aqk"].ptr,
        maps["akk"].ptr,
        maps["h"].ptr,
        cfg.scale,
        cfg.num_seqs,
        case["num_ctas"],
    )


def _launcher(executable, case: dict[str, Any]):
    args = _tirx_args(case)

    def launch() -> None:
        case["epoch"] += 1
        executable(*args, case["epoch"])

    launch._keep_alive = args
    return launch


# ----------------------------------------------------------------------------
# Correctness
# ----------------------------------------------------------------------------

_OUTPUT_NAMES = ("dq", "dk", "dv", "db", "dg", "dh0")
# FLA's test_kda.py normalized RMS error-ratio limits, in return order.
_RMS_LIMITS = (8e-3, 8e-3, 8e-3, 2e-2, 2e-2, 8e-3)


def _assert_supported_arch() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved KDA backward")
    capability = torch.cuda.get_device_capability()
    runtime_arch = f"sm_{capability[0]}{capability[1]}a"
    if runtime_arch not in KERNEL_META["runtime_cuda_archs"]:
        raise SkipTest(
            "agent-evolved KDA backward requires one of "
            f"{KERNEL_META['runtime_cuda_archs']}, got {runtime_arch}"
        )


@contextmanager
def _native_fla_backend():
    values = {"FLA_FLASH_KDA": "0", "FLA_TILELANG": "0"}
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_fla_reference(case: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    with _native_fla_backend(), torch.no_grad():
        from fla.ops.kda.chunk_bwd import chunk_kda_bwd

        dq, dk, dv, db, dg, dh0, dA, dbias = chunk_kda_bwd(
            q=case["q"],
            k=case["k"],
            v=case["v"],
            beta=case["beta"],
            Aqk=case["Aqk"],
            Akk=case["Akk"],
            g=case["g"],
            initial_state=case["initial_state"],
            do=case["do"],
            dht=case["dht"],
            scale=case["scale"],
            chunk_size=CHUNK,
            cu_seqlens=case["cu_seqlens"],
        )
    if dA is not None or dbias is not None:
        raise AssertionError("prepared-gate chunk_kda_bwd unexpectedly returned dA/dbias")
    return dq, dk, dv, db, dg, dh0


def check_correctness(outputs: dict[str, Any], **kwargs: Any) -> None:
    _cfg(**kwargs)
    first, actual, reference = outputs["first"], outputs["actual"], outputs["reference"]
    for name, tensor in zip(_OUTPUT_NAMES, actual):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} contains non-finite values")
    for name, one, two in zip(_OUTPUT_NAMES, first, actual):
        if not torch.equal(one, two):
            max_abs = float((one.float() - two.float()).abs().max())
            raise AssertionError(
                f"identical launches are not exactly repeatable for {name}; max abs diff={max_abs}"
            )
    for name, got, want, limit in zip(_OUTPUT_NAMES, actual, reference, _RMS_LIMITS):
        torch.testing.assert_close(got, want, atol=1e-1, rtol=1e-1, msg=lambda m, n=name: f"{n}: {m}")
        diff_rms = torch.sqrt(torch.mean((got.float() - want.float()).square()))
        reference_rms = torch.sqrt(torch.mean(want.float().square()))
        rms_ratio = float(diff_rms / (reference_rms + 1e-8))
        if rms_ratio >= limit:
            raise AssertionError(
                f"{name} normalized RMS error ratio {rms_ratio:.6e} must be below {limit:.0e}"
            )


def _clone_outputs(case: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    return tuple(case[name].clone() for name in _OUTPUT_NAMES)


def _poison_outputs(case: dict[str, Any], value: float) -> None:
    for name in _OUTPUT_NAMES:
        case[name].fill_(value)


def run_test(**kwargs: Any) -> None:
    _assert_supported_arch()
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
    launch = _launcher(executable, case)
    _poison_outputs(case, float("nan"))
    launch()
    torch.cuda.synchronize()
    first = _clone_outputs(case)
    _poison_outputs(case, 42.0)
    launch()
    torch.cuda.synchronize()
    actual = _clone_outputs(case)
    reference = _run_fla_reference(case)
    torch.cuda.synchronize()
    check_correctness({"first": first, "actual": actual, "reference": reference}, **kwargs)


# ----------------------------------------------------------------------------
# Benchmarking
# ----------------------------------------------------------------------------


def prepare_bench(**kwargs: Any):
    """Trace and compile before bench-suite assigns a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    _assert_supported_arch()
    config = dict(prepared["config"])
    config.update(kwargs)
    rounds = config.pop("rounds", 5)
    cooldown_s = config.pop("cooldown_s", 1.0)
    case = prepare_data(**config)
    launch = _launcher(prepared["executable"], case)
    launch()
    torch.cuda.synchronize()

    def _fla_builder():
        return lambda: _run_fla_reference(case)

    from tirx_kernels.runner import bench

    return bench(
        {"tirx": launch},
        references={"fla_chunk_kda_bwd": _fla_builder},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(**config)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = [
    "CONFIGS",
    "KERNEL_META",
    "check_correctness",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
]
