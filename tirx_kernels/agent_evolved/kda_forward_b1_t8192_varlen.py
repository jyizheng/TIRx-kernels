# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved SM100a packed-varlen KDA forward kernel (B=1, T=8192, K=V=128).

This canonical module exposes the six measured KDA-internal prefill workloads:
H is 64 or 96, and each head count is crossed with a single fixed 8192-token
sequence, a mixed packed pack of ``[1300, 547, 2048, 963, 271, 3063]``, and a
uniform pack of eight 1024-token sequences.  Inputs are BF16 q/k/v/g/beta with
FP32 gate parameters and a per-sequence FP32 ``initial_state``; there is no
final-state output.  The registry metadata records the evolution run and the
selected candidate.

The device body is preserved from the selected result.  One work item is a
(sequence, head) pair: the recurrent state is seeded from
``initial_state[seq, head]`` (v-major, matching FLA's ``state_v_first=True``)
and never crosses a ``cu_seqlens`` boundary.  64-token chunks stream through a
single fused persistent kernel that keeps the transposed state in TMEM.  The
host side compiles one specialization per workload class -- single sequence
(optionally split across two CTAs), uniform pack, and general varlen pack --
and every choice is derived from the config rather than from the environment.
Evolution history and measurement artifacts remain in the run tree rather than
becoming a second source of truth here.
"""
import ctypes
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K
import tvm

D_HEAD = 128
CHUNK = 64
LOG2E = 1.4426950408889634
NEG5LOG2E = -5.0 * LOG2E
TMEM_COLS = 512

# TMEM column map (f32 columns; 128 lanes each)
TM_S = 0  # S^T fp32 running state              [128 lanes v x 128 cols k]
TM_ST = 128  # S^T bf16 A-operand copy              (64 cols)
TM_U = 192  # U^T fp32 accumulator (G3 + G4)       (64 cols)
TM_UB = 256  # U^T bf16 A-operand copy              (32 cols)
TM_O = 288  # O^T fp32 accumulator (G5 + G6)       (64 cols)
TM_V1 = 352  # V1 = S~'^T X0^T fp32 accumulator     (64 cols; bf16 copy goes to TM_UB)
TM_D0 = 416  # G1_0: [Akk;Aqk] columns 0..31        (32 cols; chunk parity 1 uses TM_D0 + 64 = 480)
TM_D1 = 448  # G1_1: [Akk;Aqk] columns 32..63       (32 cols)

# tcgen05 instruction descriptors (kind::f16, bf16 x bf16 -> f32, dense)
ID_G1 = 0x08080490  # M128 N32  K-major A, K-major B
ID_G1B = 0x04080490  # M64 N32  K-major A, K-major B (G1_1: J1 is a 64-row tile)
ID_G23 = 0x08108490  # M128 N64  MN-major A (smem), K-major B
ID_G56 = 0x08100490  # M128 N64  tmem A, K-major B
ID_G5F = 0x08200490  # M128 N128 tmem A, K-major B
ID_G4B = 0x08104490  # M128 N64  tmem A, K-major B, negate B (G4'': U^T -= V1b TB^T)
ID_G7 = 0x08210490  # M128 N128 tmem A, MN-major B

MMA_SS = "tcgen05.mma.cta_group::1.kind::f16"
TMA_LD = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
TMA_ST = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
TMA_PREFETCH = "cp.async.bulk.prefetch.tensor.3d.L2.global.tile"
TC_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TC_ST32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
TC_LD256 = "tcgen05.ld.sync.aligned.16x256b.x8.b32"
TC_ST16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
WAIT_LD = "tcgen05.wait::ld.sync.aligned"
WAIT_ST = "tcgen05.wait::st.sync.aligned"
STM_X4T = "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
MMA_K8 = "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
MMA_K16 = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
LDM_X1 = "ldmatrix.sync.aligned.m8n8.x1.shared.b16"
LDM_X1T = "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16"
LDM_X4 = "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
LDM_X4T = "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
STM_X1 = "stmatrix.sync.aligned.m8n8.x1.shared.b16"
STM_X4 = "stmatrix.sync.aligned.m8n8.x4.shared.b16"
FENCE_ASYNC = "fence.proxy.async.shared::cta"
TMAP_REL = "fence.proxy.tensormap::generic.release.gpu"
TMAP_ACQ = "fence.proxy.tensormap::generic.acquire.gpu"
BULK_COMMIT = "cp.async.bulk.commit_group"
BULK_WAIT_READ = "cp.async.bulk.wait_group.read"
BULK_WAIT = "cp.async.bulk.wait_group"
BULK_PREFETCH = "cp.async.bulk.prefetch.L2.global"
LDG_V4 = "ld.global.L1::no_allocate.v4.b32"

BAR_INV, BAR_INV_N = 2, 128  # cg0 inverse stages
BAR_PREP, BAR_PREP_N = 3, 256  # prep gate-scan exchange (8 warps)
BAR_CG1, BAR_CG1_N = 4, 128  # cg1 tmem dealloc rendezvous
BAR_INV2, BAR_INV2_N = 5, 64  # cg0 32->64 merge (warps 0,1)

STAGE_ROWS = 3 * CHUNK  # k rows 0..63, q rows 64..127, g rows 128..191
STAGE_UNITS = STAGE_ROWS * D_HEAD * 2 // 16  # one stage in 16-byte descriptor units
KG_UNITS = CHUNK * D_HEAD * 2 // 16
J1_UNITS = CHUNK * D_HEAD * 2 // 16
STAGE_BYTES = STAGE_ROWS * D_HEAD * 2
KQ_BYTES = 2 * CHUNK * D_HEAD * 2  # TMA bytes per stage: k and q rows only (g comes from global)
V_BYTES = CHUNK * D_HEAD * 2


# Warpgroup register split for the fixed-shape two-CTA sequence split
# (prep, cg0, cg1); the five 128-thread columns must sum to 480.
SPLIT_REGS = (120, 112, 80)

def make_kernel(
    H: int, num_ctas: int, nseq: int, dyn: bool, iket: bool = False, split: int = 0,
    ntot_chunks: int = 0, unif: bool = False, seg: int = 0,
):
    """Trace the kernel for one (H, grid, scheduler) specialization.

    ``dyn`` turns on the atomic work-claim ring. It only pays off when packed
    sequences can differ in length, so a single-sequence workload keeps the
    cheaper static ``work = cta_id + k * num_ctas`` stride.
    """
    HK = H * D_HEAD
    # ``split`` > 0 turns on the fixed-shape sequence split: the single packed
    # sequence is cut at chunk ``split`` into two pieces that run on two CTAs.
    # The second piece re-derives the recurrent state over the first piece's
    # chunks with every emitting path switched off (no q operand tiles, no
    # intra-chunk attention block, no output MMAs, no output store), so its
    # prefix costs about 0.82 of an emitting chunk.  Balancing
    # ``max(A, r*A + ntot - A)`` is what shortens the makespan.
    assert not split or nseq == 1
    # A/B profiling of v140 found this issue order beneficial only for the
    # H96 mixed-varlen specialization.  Keep the incumbent schedule in every
    # other build so a local win does not regress the aggregate.
    # ``unif`` says every packed sequence has the same length and that length is
    # a whole number of chunks.  setup() decides it from the actual cu_seqlens
    # rather than from the sequence count, so an N-sequence input whose lengths
    # happen to differ takes the general path instead of silently assuming an
    # even split of the token axis.  Only a uniform pack may skip the per-item
    # cu_seqlens load, the LPT rank, the tail clamps and the per-CTA output
    # descriptor: all four are correct exactly when every chunk is a full
    # 64-row box that cannot cross a sequence boundary.
    # Pass-1 issue form, chosen per compiled specialization.  Every form
    # computes the same two fma.rn.f32 operations per channel; they differ only
    # in how many instructions carry them, so all three are bit-identical and
    # the choice is pure scheduling:
    #   0  scalar throughout (the incumbent)
    #   2  one fma.rn.f32x2 for the tanh argument and one for the decay, with
    #      the chunk-long cumsum left scalar so the 32 G registers stay free of
    #      pairing constraints
    #   1  form 2 plus add.rn.f32x2 for the cumsum, which does constrain them
    # Form 2 removes four instructions per token per lane and form 1 six, but
    # ptxas reacts very differently per specialization: measured against v152 on
    # an idle B200, form 1 is -1.6% on H96 fixed yet +14.4% on the H64 fixed
    # split build, while form 2 is -2.2%/-1.6% on the two fixed shapes and
    # +1.0% on H96 mixed.  Unmeasured specializations keep the incumbent.
    _cls = "fixed" if nseq == 1 else ("uniform" if unif else "varlen")
    PASS1_FORM = {
        (96, "fixed"): 2,
        (96, "uniform"): 1,
        (64, "fixed"): 2,
        (64, "varlen"): 2,
    }.get((H, _cls), 0)
    _pair_key = f"h{H}{'f' if _cls == 'fixed' else ('u' if _cls == 'uniform' else 'm')}"
    # The post-v154 retune rejects fixed/mixed builds but repeatedly favors
    # both uniform specializations, whose different pass-1 schedule lets ptxas
    # benefit from the shorter issue form.
    PAIR_INV = _pair_key in ("h96u", "h64u")
    uniform_pack = bool(unif) and nseq > 1
    varlen_pack = nseq > 1 and not uniform_pack   # partial tails, arbitrary bounds
    whole_boxes = nseq == 1 or uniform_pack       # every chunk is a full 64-row box
    state_pf_lead = min(seg // CHUNK, 4) if (uniform_pack and H == 64) else 1
    fuse_g5_v1 = H == 64 and (varlen_pack or bool(split))
    # One-stage side-pipeline cursors return to their initial parity after an
    # even state-only prefix, so dead Aqk/output handshakes may be omitted.
    elide_even_prefix_sidepipes = bool(split and split % 2 == 0)
    # Fused B rows are [X0; Q0], so its low/high N halves land in the
    # physical regions normally named O/V1.  Give those regions semantic
    # aliases for the rest of this specialization.
    v1_col = TM_O if fuse_g5_v1 else TM_V1
    o_col = TM_V1 if fuse_g5_v1 else TM_O
    assert not uniform_pack or (seg > 0 and seg % CHUNK == 0)
    reorder_h96_mixed = H == 96 and varlen_pack

    def phase_list(npre, nch):
        """Chunk loops of one work item: the state-only prefix, then the emitting run."""
        return ((True, npre), (False, nch)) if split else ((False, nch),)

    @K.kernel(warps=20, arch="sm_100a", min_blocks_per_sm=1, grid=num_ctas)
    def kda_fwd(
        q: K.gptr[K.bf16],
        k: K.gptr[K.bf16],
        v: K.gptr[K.bf16],
        g: K.gptr[K.bf16],
        beta: K.gptr[K.bf16],
        a_log: K.gptr[K.f32],
        dt_bias: K.gptr[K.f32],
        o: K.gptr[K.bf16],
        istate: K.gptr[K.f32],
        cu: K.gptr[K.i64],
        desc_ws: K.gptr[K.i8],
        wctr: K.gptr[K.u32],
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        g_map: K.TensorMap,
        o_map: K.TensorMap,
        scale: K.f32,
        num_work: K.i32,
    ):
        # q/k/v/g/o are only reached through the tensor maps.
        K.keep_alive(q.data)
        K.keep_alive(k.data)
        K.keep_alive(v.data)
        K.keep_alive(g.data)
        K.keep_alive(o.data)

        sp = K.specialize()
        # setmaxnreg redistributes the LAUNCH allocation (96 regs x 640 threads = 61440), not the
        # 64K file: the role targets must sum to 480 per 128-thread column (Kern validates against 65536).
        # Fixed shapes derive every cg1 cursor and have the smallest epilogue
        # footprint.  Reinvest the next cg1 tier in the two prep warpgroups;
        # packed shapes retain v77's allocation.
        fixed_wide_prep = nseq == 1 and H == 64
        _rp, _rc0, _rc1 = (
            (120, 112, 80) if fixed_wide_prep else (112, 112, 96)
        )
        if split:
            _rp, _rc0, _rc1 = SPLIT_REGS
        # The five warpgroups must sum to 480 per 128-thread column, so a
        # register handed to one role is taken from another.  cg0 owns the
        # hierarchical inverse -- the most register-hungry block in the kernel,
        # and the second-busiest role after prep (73% vs 77% of the chunk in the
        # clock profile) -- and four of the six specializations run measurably
        # faster with eight more registers there, taken from prep.  The two that
        # do not are the H64 fixed split build, whose duplicated prefix/emitting
        # prep bodies need every one of their 120 registers (+17.9%), and H96
        # mixed (+1.3%).  Lowering cg1 instead of prep works equally well on the
        # two H96 shapes that compile the narrow state epilogue but collapses
        # the varlen ones, whose `state_epilogue_wide` holds fr[64] and spills
        # below 96 registers (+45%/+37%), so cg1 stays at 96 everywhere.
        _rp, _rc0, _rc1 = {
            (96, "fixed"): (104, 128, 96),
            (96, "uniform"): (104, 128, 96),
            (64, "varlen"): (104, 128, 96),
            (64, "uniform"): (104, 128, 96),
        }.get((H, _cls), (_rp, _rc0, _rc1))
        _ra = 48
        if _cls == "uniform":
            # v168's H64-uniform optimum moved two cg1 tiers to the auxiliary
            # loader/storer/MMA group.  Apply the same point to H96 uniform,
            # which already derives its state-chain phases.
            _rc1, _ra = 80, 64
        assert 2 * _rp + _rc0 + _rc1 + _ra == 480
        prep = sp.role("prep", warps=list(range(8)), regs=_rp)
        cg0 = sp.role("cg0", warps=[8, 9, 10, 11], regs=_rc0)
        cg1 = sp.role("cg1", warps=[12, 13, 14, 15], regs=_rc1)
        auxg = sp.warpgroup("aux", warps=[16, 17, 18, 19], regs=_ra)
        mma0 = sp.role("mma0", warps=[16], group=auxg)  # idle (every MMA is issued by mma1 or cg0)
        loader = sp.role("loader", warps=[17], group=auxg)
        storer = sp.role("storer", warps=[18], group=auxg)
        mma1 = sp.role("mma1", warps=[19], group=auxg)  # state-chain GEMMs G5,G4,G6,G7

        smem = K.smem_pool()
        s_tmem_addr = smem.alloc((1,), K.i32, align=4)
        # ---- pipelines / barriers ---------------------------------------
        p_stage = K.Pipeline(smem, 2, full="tma", empty="tcgen05", init_empty=2)  # loader -> prep ; G1_0 (mma0) + V1 (after G5, mma1) -> loader
        p_v = K.Pipeline(smem, 1, full="tma", empty="tcgen05")  # loader -> mma(G3) ; G3 -> loader
        p_prep = K.Pipeline(smem, 2, full="mbar", empty="tcgen05", init_full=256)  # prep(8 warps) -> mma ; G7 -> prep (KG)
        p_j1 = K.Pipeline(smem, 2, full="mbar", empty="tcgen05", init_full=256)  # prep(8 warps) -> mma ; G1_1 -> prep
        p_beta = K.Pipeline(smem, 2, full="mbar", empty="mbar", init_full=32, init_empty=128)  # loader -> cg0
        p_g1 = K.Pipeline(smem, 2, full="tcgen05", empty="mbar", init_empty=128)  # G1_0 (D0, two TMEM buffers) -> cg0 ; cg0 (D0 read) -> mma0
        m_g1b = K.TCGen05Bar(smem, 1)  # G1_1 (D1) done -> cg0
        m_g1b.init(1)
        p_g1b = K.MBarrier(smem, 1)  # D1 consumed by cg0 (Akk and Aqk halves) -> mma1 may overwrite it with the next G1_1
        p_g1b.init(128)
        p_t = K.Pipeline(smem, 1, full="mbar", empty="tcgen05", init_full=128)  # cg0 -> mma ; G4'' (mma1, after G3) -> cg0
        p_aqk = K.Pipeline(smem, 1, full="mbar", empty="tcgen05", init_full=64)  # cg0 warps 2/3 -> mma1 (Aqk rows in the stage Y region; the region is recycled by prep after G7)
        p_u = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=128)  # G4 -> cg1 ; cg1 -> mma(G3)
        p_o = K.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=128)  # G6 -> cg1 ; cg1 -> mma(G5)
        p_osm = K.Pipeline(smem, 1, full="mbar", empty="mbar", init_full=128, init_empty=1)  # cg1 -> storer -> cg1
        m_v1 = K.TCGen05Bar(smem, 1)  # V1 done (mma1) -> cg1
        m_v1.init(1)
        m_v1b = K.MBarrier(smem, 1)  # V1 bf16 copy in TM_UB ready (cg1) -> mma1 (G4'')
        m_v1b.init(128)
        m_sacc = K.TCGen05Bar(smem, 1)  # G7 done -> cg1
        m_sacc.init(1)
        m_ub = K.MBarrier(smem, 1)  # U^T bf16 ready (cg1 -> mma)
        m_ub.init(128)
        m_s = K.MBarrier(smem, 1)  # S^T bf16 ready (cg1 -> mma)
        m_s.init(128)
        m_g3 = K.TCGen05Bar(smem, 1)  # G3 done (cg0) -> mma1 may accumulate G4'' into U^T
        m_g3.init(1)
        p_wid = K.Pipeline(smem, 4, full="mbar", empty="mbar", init_full=1, init_empty=640)

        # ---- shared memory plan ------------------------------------------
        s_stage = smem.alloc((2, STAGE_ROWS, D_HEAD), K.bf16, swizzle=K.SW128B)  # 96 KB
        s_j1 = smem.alloc((2, CHUNK, D_HEAD), K.bf16, swizzle=K.SW128B)  # 2 x 16 KB [X1(32); Q1(32)]
        s_T = smem.alloc((1, CHUNK, CHUNK), K.bf16, swizzle=K.SW128B)[0]  # 8 KB  TB = T*beta_j
        s_v = smem.alloc((1, CHUNK, D_HEAD), K.bf16, swizzle=K.SW128B)[0]  # 16 KB
        s_kg = smem.alloc((2, CHUNK, D_HEAD), K.bf16, swizzle=K.SW128B)  # 32 KB
        s_A = smem.alloc((CHUNK, CHUNK), K.f16, swizzle=K.SW128B)  # 8 KB inverse workspace
        s_o = smem.alloc((CHUNK, D_HEAD), K.bf16, swizzle=K.SW128B)  # 16 KB O [t][v]
        s_beta = smem.alloc((2, CHUNK), K.f32, align=16)
        # 4-deep ring: chunk n's vectors live in slot n&3; prep(n) writes it after the stage n-1 TMA, which
        # the loader issued only after G5(n-3), i.e. after cg1's state epilogue n-4 consumed chunk n-3's
        # vectors and (in program order) epilogue n-5 consumed chunk n-4's, the slot's previous tenant.
        # The midpoint-scaled state copy is consumed as bf16, so its factors are stored in that
        # destination precision.  End-of-chunk decay still uses fp32 factors and fp32 state.
        s_gate_r = smem.alloc((4, D_HEAD), K.bf16, align=16)  # [chunk&3]=bf16(2^R0)
        s_gate_e = smem.alloc((4, D_HEAD), K.f32, align=16)  # [chunk&3]=2^G2_63
        s_part = smem.alloc((2, 8, D_HEAD), K.f32, align=16)  # gate-scan partial totals, [chunk&1][group]
        s_norm = smem.alloc((2, 2, CHUNK), K.f32, align=16)
        # Dynamic work assignment. Packed-varlen sequences differ in length by
        # ~10x, so the static `work = cta_id + k * num_ctas` stride leaves one
        # CTA with up to 1.8x the average chunk count. The loader (the role that
        # runs furthest ahead) claims the next (sequence, head) with one global
        # atomic and publishes it through a 4-deep ring, so every role in the CTA
        # walks the same item sequence without a CTA-wide drain per item.
        # [stage] = the (sequence, head) work index the loader claimed for this
        # CTA; one i32 per ring slot, read by every other role via take_work().
        s_wid = smem.alloc((4,), K.i32, align=16)

        # The inverse never writes TB's strictly-upper 32x32 block; zero s_T once
        # so that block stays the required zero operand in every chunk.
        with K.If(K.thread_id() < 512), K.Then():
            zpad = K.alloc_local([4], "uint32")
            for i in range(4):
                K.assign(zpad[i], K.uint32(0))
            K.ptx["st.shared.v4.b32"](
                s_T.ptr_to(K.thread_id() >> 3, (K.thread_id() & 7) * 8), zpad[0], zpad[1], zpad[2], zpad[3]
            )
        if split:
            # The recompute prefix never TMAs q, so the stage's q rows are only
            # needed by the emitting run, whose TMA overwrites them before use.
            # Prefix G1_0 is M64 and prefix q norms skip their global tile, so
            # only J1's Q1 rows (still read by M64 G1_1) require initialization.
            with K.If(K.thread_id() < 512), K.Then():
                zq = K.alloc_local([4], "uint32")
                for i in range(4):
                    K.assign(zq[i], K.uint32(0))
                for stg in range(2):
                    # J1's Q1 rows are G1_1's dead Aqk operand rows.
                    K.ptx["st.shared.v4.b32"](
                        s_j1[stg].ptr_to(
                            CHUNK // 2 + (K.thread_id() >> 4), (K.thread_id() & 15) * 8
                        ),
                        zq[0], zq[1], zq[2], zq[3],
                    )
        K.ptx[FENCE_ASYNC]()
        with K.If(K.thread_id() == 0), K.Then():
            K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()
        # TMEM allocation by warp 8 (cg1 warp 0); the address is read by all after cta_sync.
        with K.If(K.warp_id() == 12), K.Then():
            K.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                K.address_of(s_tmem_addr[0]), K.uint32(TMEM_COLS)
            )
        K.cuda.cta_sync()

        # ---- trace-time layout facts ------------------------------------
        def row_off16(tile, rows):
            """16B-unit offset of row `rows` (column 0) inside one stage of `tile`."""
            t = tile.tile if isinstance(tile, K.KTileView) else tile
            c0 = t._coord(0, 0, 0) if t.stages is not None else t._coord(0, 0)
            c1 = t._coord(0, rows, 0) if t.stages is not None else t._coord(rows, 0)
            delta = t._phys(c1) - t._phys(c0)
            assert delta * 2 % 16 == 0
            return delta * 2 // 16

        OFF_Q = row_off16(s_stage, 64)
        OFF_Y0 = row_off16(s_stage, 128)
        OFF_Y1 = row_off16(s_stage, 160)
        assert (OFF_Q, OFF_Y0, OFF_Y1) == (64 * 8, 128 * 8, 160 * 8)

        # ---- shared helpers ------------------------------------------------
        class _NullCtx:
            """Placeholder so a trace-time guarded block can remain unguarded."""

            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

        def elected():
            return K.cuda.elect_sync() != K.uint32(0)

        def rng(name):
            if not iket:
                return None
            tok = K.alloc_local([1], "uint32")
            K.assign(tok[0], K.cuda.iket.range_start(name))
            return tok

        def rng_end(tok):
            if tok is not None:
                K.cuda.iket.range_end(tok[0])

        def elect_local():
            e = K.local_scalar("uint32")
            K.assign(e, K.cuda.elect_sync())
            return e

        def tmem_preamble():
            tm = K.alloc_local([1], "int32")
            K.ptx.ld.volatile.shared.s32(tm[0], K.address_of(s_tmem_addr[0]))
            return tm

        def pack_bf16x2(dst, lo, hi):
            # cvt.rn.bf16x2.f32 d, a, b puts b in the LOW half.
            K.ptx.cvt.rn.bf16x2.f32(dst, hi, lo)

        def pack_f16x2(dst, lo, hi):
            K.ptx.cvt.rn.f16x2.f32(dst, hi, lo)

        def bf16_lo(w):
            return K.reinterpret("float32", w << K.uint32(16))

        def bf16_hi(w):
            return K.reinterpret("float32", w & K.uint32(0xFFFF0000))

        def lds128(view, row, col, words, base):
            K.ptx["ld.shared.v4.b32"](
                words[base], words[base + 1], words[base + 2], words[base + 3], view.ptr_to(row, col)
            )

        def sts128(view, row, col, words, base):
            K.ptx["st.shared.v4.b32"](
                view.ptr_to(row, col), words[base], words[base + 1], words[base + 2], words[base + 3]
            )

        def f2(a, b):
            return K.cuda.make_float2(a, b)

        # Under the split there are 2*H pieces on ``num_ctas`` CTAs. The
        # ``num_work - num_ctas`` doubled CTAs must each get two prefix-free
        # piece-0 items, so the item order is
        #   [0, E)          piece 0, head = w
        #   [E, E+H)        piece 1, head = w - E
        #   [E+H, 2H)       piece 0, head = w - H
        # which puts item w and item w+num_ctas (w < E) on the same CTA and
        # makes both of them piece 0.
        split_e = 2 * H - num_ctas if split else 0

        def work_coords(work):
            """(sequence, head) of work item ``work``.

            Items are ordered longest sequence first. Combined with the greedy
            claim that is what bounds the makespan: a CTA that picks up the 48-chunk
            sequence last stretches the whole launch, so the long items have to be
            handed out while every CTA is still idle.
            """
            if split:
                _p1 = (work >= K.int32(split_e)) & (work < K.int32(split_e + H))
                return (
                    K.local_scalar("int32", init=K.Select(_p1, K.int32(1), K.int32(0))),
                    K.local_scalar(
                        "int32",
                        init=K.Select(
                            _p1,
                            work - K.int32(split_e),
                            K.Select(work < K.int32(split_e), work, work - K.int32(H)),
                        ),
                    ),
                )
            wu = K.Cast("uint32", work)
            rank = K.Cast("int32", wu // K.uint32(H))
            h_idx = work - rank * H
            # The eight-sequence workload is uniformly 1024 tokens, so the LPT
            # rank is exactly the sequence index.  Avoid reloading all eight
            # bounds and recomputing an N^2 rank in every role/work item.
            if whole_boxes:
                # Equal-length sequences make the LPT rank the identity.
                return rank, h_idx
            ln = []
            for t in range(nseq):
                bd = K.alloc_local([2], "int64")
                K.ptx.ld.global_.s64(bd[0], cu.ptr_to([K.int32(t)]))
                K.ptx.ld.global_.s64(bd[1], cu.ptr_to([K.int32(t + 1)]))
                ln.append(K.local_scalar("int32", init=K.Cast("int32", bd[1] - bd[0])))
            sel = K.local_scalar("int32", init=K.int32(0))
            for sq in range(nseq):
                rk = K.local_scalar("int32", init=K.int32(0))
                for t in range(nseq):
                    if t == sq:
                        continue
                    longer = ln[t] >= ln[sq] if t < sq else ln[t] > ln[sq]
                    K.assign(rk, rk + K.Cast("int32", longer))
                with K.If(rk == rank), K.Then():
                    K.assign(sel, K.int32(sq))
            return sel, h_idx

        def take_work(state, out):
            """Read this CTA's next work item from the ring (all roles but the loader)."""
            p_wid.full.wait(state.stage, state.phase)
            K.ptx.fence.acq_rel.cta()  # the ring read must not be hoisted above the wait
            K.ptx.ld.shared.s32(out, K.address_of(s_wid[state.stage]))

        def drop_work(state):
            p_wid.empty.arrive(state.stage)
            state.advance()

        def work_first(state, out):
            """Seed a role's work cursor (static stride, or the first ring entry)."""
            if dyn:
                take_work(state, out)
            else:
                K.assign(out, K.cta_id())

        def work_next(state, out):
            if dyn:
                drop_work(state)
                take_work(state, out)
            else:
                K.assign(out, out + num_ctas)

        def seq_bounds(b_idx):
            """(first token, token count, chunk count) of packed sequence ``b_idx``."""
            if uniform_pack:
                return b_idx * K.int32(seg), K.int32(seg), K.int32(seg // CHUNK)
            bd = K.alloc_local([2], "int64")
            K.ptx.ld.global_.s64(bd[0], cu.ptr_to([b_idx]))
            K.ptx.ld.global_.s64(bd[1], cu.ptr_to([b_idx + 1]))
            lo = K.local_scalar("int32", init=K.Cast("int32", bd[0]))
            ln = K.local_scalar("int32", init=K.Cast("int32", bd[1] - bd[0]))
            nch = K.local_scalar("int32", init=K.ceildiv(ln, K.int32(CHUNK)))
            return lo, ln, nch

        # Mutable per-CTA copy of the O tensor map: its token dimension is
        # narrowed to the current sequence's end so a partial tail chunk's TMA
        # store drops the rows that belong to the next packed sequence.
        d_o = desc_ws.ptr_to([K.Cast("int64", K.cta_id()) * K.int64(128)])

        def copy_desc(dsc, m):
            tmp = K.alloc_local([4], "uint64")
            s = K.reinterpret("uint64", K.address_of(m))
            d = K.reinterpret("uint64", dsc)
            for half in range(2):
                off = K.uint64(half * 32)
                K.ptx.ld.global_.v4.b64(
                    tmp[0], tmp[1], tmp[2], tmp[3], K.reinterpret("handle", s + off)
                )
                K.ptx.st.global_.v4.b64(
                    K.reinterpret("handle", d + off), tmp[0], tmp[1], tmp[2], tmp[3]
                )

        def desc_set_rows(dsc, rows):
            K.ptx["tensormap_replace.tile.global_dim.global.b1024.b32"](
                dsc, 1, K.Cast("uint32", rows)
            )

        # =====================================================================
        # PREP: gates, norms, gated bf16 operand tiles.       warps 0-3
        # =====================================================================
        with prep:
            st_stage = K.PipelineState(2, phase=0)  # consumer of p_stage.full
            st_prep = K.PipelineState(2, phase=1)  # producer of p_prep
            st_j1 = K.PipelineState(2, phase=1)  # producer of p_j1
            lane = K.lane_id()
            r_grp = K.local_scalar("int32", init=K.warp_id_in_role())  # token group 0..7 (one warp each)
            col0 = lane * 4  # channels 4*lane .. 4*lane+3
            hi_block = K.local_scalar("int32", init=K.Cast("int32", (K.warp_id_in_role() >= 4)))
            work = K.local_scalar("int32")
            st_wc = K.PipelineState(4, phase=0)  # consumer of p_wid

            # Byte offsets of this thread's 8-byte cell in token row 8r+i of a SW128B bf16 tile with
            # `rows` rows: atom column (lane/16) * rows*128 + row*128 + ((chunk ^ i) << 4) + (lane&1)*8,
            # chunk = (lane%16)/2.  Precomputed once; every tile access is then one add.
            def row_offsets(rows):
                offs = K.alloc_local([8], "int32")
                for i in range(8):
                    K.assign(
                        offs[i],
                        (lane >> 4) * (rows * 128)
                        + (r_grp * 8 + i) * 128
                        + ((((lane & 15) >> 1) ^ i) << 4)
                        + (lane & 1) * 8,
                    )
                return offs

            xo_stage = row_offsets(STAGE_ROWS)
            xo_adj = (lane >> 4) * ((STAGE_ROWS - CHUNK) * 128)  # xo_64[i] == xo_stage[i] - xo_adj (64-row tiles)
            # row norms: lanes 0..15 own this warp's eight k rows, lanes 16..31 its eight q rows, two lanes per
            # row, each lane one 64-column swizzle atom.  The atom's eight 16-byte chunks are summed in the
            # order j ^ (row & 7) ^ (atom << 2) (the sum does not care), which keeps the eight lanes of a
            # quarter-warp (four rows x two atoms) on eight distinct chunk positions: conflict-free LDS.128.
            norm_row = (lane >> 4) * CHUNK + r_grp * 8 + ((lane & 15) >> 1)
            norm_tok = r_grp * 8 + ((lane & 15) >> 1)
            norm_off = K.local_scalar("int32", init=(lane & 1) * (STAGE_ROWS * 128) + norm_row * 128)
            norm_x = K.local_scalar("int32", init=((norm_row & 7) ^ ((lane & 1) << 2)) << 4)
            norm_is_q = lane >= 16
            norm_wr = (lane & 1) == 0

            def lds64(base, off, w0, w1):
                K.ptx["ld.shared.v2.b32"](w0, w1, K.ptx.addr(base, off))

            def sts64(base, off, w0, w1):
                K.ptx["st.shared.v2.b32"](K.ptx.addr(base, off), w0, w1)

            def mul2(dst_pair, a0, a1, b0, b1):
                K.ptx.mul.rn.f32x2(dst_pair, f2(a0, a1), f2(b0, b1))

            gc = K.local_scalar("int32", init=K.int32(0))  # global chunk counter (ring indices)
            work_first(st_wc, work)
            with K.While(work < num_work):
                b_idx, h_idx = work_coords(work)
                if split:
                    lo_t = K.int32(0)
                    seqlen = K.int32(0)  # unused: a split shape is nseq==1
                    npre = K.local_scalar("int32", init=K.Select(b_idx != 0, K.int32(split), K.int32(0)))
                    pre_start = npre
                    NCH = K.local_scalar(
                        "int32", init=K.Select(b_idx != 0, K.int32(ntot_chunks - split), K.int32(split))
                    )
                    n_limit = K.local_scalar(
                        "int32", init=K.Select(b_idx != 0, K.int32(ntot_chunks), K.int32(split))
                    )
                else:
                    lo_t, seqlen, NCH = seq_bounds(b_idx)
                    npre = None
                    pre_start = None
                    n_limit = None
                # A_log is scalar per head. One lane computes exp(A_log) and
                # broadcasts it within each prep warp instead of all 32 lanes
                # issuing the identical global load and MUFU operation.
                ea_l2 = K.local_scalar("float32", init=K.float32(0.0))
                with K.If(lane == 0), K.Then():
                    ea = K.local_scalar("float32")
                    K.ptx.ld.global_.f32(ea, a_log.ptr_to([h_idx]))
                    K.ptx.ex2.approx.ftz.f32(ea_l2, ea * K.float32(LOG2E))
                ea_bits = K.local_scalar("uint32")
                K.ptx.shfl_sync.idx.b32(
                    ea_bits,
                    K.reinterpret("uint32", ea_l2),
                    K.uint32(0),
                    K.uint32(0x1F),
                    K.uint32(0xFFFFFFFF),
                )
                hea = K.local_scalar(
                    "float32",
                    init=K.reinterpret("float32", ea_bits) * K.float32(0.5),
                )  # exp(A_log)/2 for tanh sigmoid
                bias = K.alloc_local([4], "float32")
                bw = K.alloc_local([4], "uint32")
                K.ptx[LDG_V4](bw[0], bw[1], bw[2], bw[3], dt_bias.ptr_to([h_idx * D_HEAD + col0]))
                biash = K.alloc_local([4], "float32")  # bias * exp(A_log)/2: the tanh argument is one fma
                for j in range(4):
                    K.assign(bias[j], K.reinterpret("float32", bw[j]))
                    K.assign(biash[j], bias[j] * hea)
                # gate logits of this thread's 8 tokens x 4 channels, fetched from global one chunk
                # ahead (they only feed pass 1, which therefore no longer waits for the TMA stage)
                gw = K.alloc_local([16], "uint32")
                g_base = K.local_scalar(
                    "int64",
                    init=(K.Cast("int64", lo_t) * K.int64(H) + K.Cast("int64", h_idx)) * K.int64(D_HEAD)
                    + K.Cast("int64", col0),
                )

                def issue_g_load_token(nn, i):
                    # Mixed tails clamp padded rows to the sequence's last
                    # token; fixed/uniform chunks are all exactly full.
                    t = r_grp * 8 + i
                    if whole_boxes:
                        tc = nn * CHUNK + t
                    else:
                        tc = K.min(nn * CHUNK + t, seqlen - 1)
                    K.ptx["ld.global.L1::no_allocate.v2.b32"](
                        gw[2 * i], gw[2 * i + 1],
                        g.ptr_to([g_base + K.Cast("int64", tc) * K.int64(HK)]),
                    )

                def issue_g_loads(nn):
                    for i in range(8):
                        issue_g_load_token(nn, i)

                # gate sigmoid + chunk-local cumsum of one token into G[4i..4i+3] (registers only)
                G = K.alloc_local([32], "float32")

                def pass1_token(i, nn):
                    # Mixed tails already force padded khat/qhat rows to zero,
                    # and no final state is returned.  Their gate factors can
                    # therefore follow the clamped last-token logits: they can
                    # only alter the unobserved state after the final real row.
                    # Keeping the full straight-line gate pass also covers the
                    # deliberately discarded pass after every work item.
                    hfac = K.float32(NEG5LOG2E * 0.5)
                    if PASS1_FORM:
                        for pp in range(2):
                            w = gw[2 * i + pp]
                            arg = K.local_scalar("uint64")
                            K.ptx.fma.rn.f32x2(
                                arg,
                                f2(bf16_lo(w), bf16_hi(w)),
                                f2(hea, hea),
                                f2(biash[2 * pp], biash[2 * pp + 1]),
                            )
                            th0 = K.local_scalar("float32")
                            th1 = K.local_scalar("float32")
                            K.ptx.tanh.approx.f32(th0, K.cuda.float2_x(arg))
                            K.ptx.tanh.approx.f32(th1, K.cuda.float2_y(arg))
                            gdp = K.local_scalar("uint64")
                            K.ptx.fma.rn.f32x2(gdp, f2(th0, th1), f2(hfac, hfac), f2(hfac, hfac))
                            if PASS1_FORM == 2 or i == 0:
                                lo_v = K.cuda.float2_x(gdp)
                                hi_v = K.cuda.float2_y(gdp)
                                if i == 0:
                                    K.assign(G[2 * pp], lo_v)
                                    K.assign(G[2 * pp + 1], hi_v)
                                else:
                                    K.assign(G[i * 4 + 2 * pp], G[(i - 1) * 4 + 2 * pp] + lo_v)
                                    K.assign(G[i * 4 + 2 * pp + 1], G[(i - 1) * 4 + 2 * pp + 1] + hi_v)
                                continue
                            acc = K.local_scalar("uint64")
                            K.ptx.add.rn.f32x2(
                                acc,
                                f2(G[(i - 1) * 4 + 2 * pp], G[(i - 1) * 4 + 2 * pp + 1]),
                                gdp,
                            )
                            K.assign(G[i * 4 + 2 * pp], K.cuda.float2_x(acc))
                            K.assign(G[i * 4 + 2 * pp + 1], K.cuda.float2_y(acc))
                        return
                    for j in range(4):
                        wj = gw[2 * i] if j < 2 else gw[2 * i + 1]
                        gv = bf16_lo(wj) if j % 2 == 0 else bf16_hi(wj)
                        th = K.local_scalar("float32")
                        K.ptx.tanh.approx.f32(th, gv * hea + biash[j])
                        gd = K.local_scalar("float32")
                        K.ptx.fma.rn.f32(gd, th, hfac, hfac)
                        if i == 0:
                            K.assign(G[j], gd)
                        else:
                            K.assign(G[i * 4 + j], G[(i - 1) * 4 + j] + gd)

                # chunk 0's pass 1 runs here; every later chunk's pass 1 is fused into the previous chunk's pass 2
                issue_g_loads(K.int32(0))
                for i in range(8):
                    pass1_token(i, K.int32(0))
                issue_g_loads(K.int32(1))

                for _so, _cnt in phase_list(npre, NCH):
                    with K.serial(_cnt) as _nl:
                        n = _nl if (not split or _so) else _nl + pre_start
                        nlimit = n_limit if split else NCH
                        sv = s_stage[st_stage.stage]
                        base_k = sv.ptr_to(0, 0)  # rows 0..63 (k -> X0)
                        base_q = sv.ptr_to(64, 0)  # rows 64..127 (q -> Q0)
                        base_g = sv.ptr_to(128, 0)  # rows 128..191 (g -> Y)
                        base_j1 = s_j1[st_j1.stage].ptr_to(0, 0)
                        base_kg = s_kg[st_stage.stage].ptr_to(0, 0)
                        w0 = K.local_scalar("uint32")
                        w1 = K.local_scalar("uint32")
                        # (pass 1 of this chunk already ran, fused into the previous chunk's pass 2: G = cumsum)
                        _t = rng("P.scan_a")
                        # block totals -> smem partials [r][k]
                        K.ptx["st.shared.v4.f32"](K.address_of(s_part[gc & 1, r_grp, col0]), G[28], G[29], G[30], G[31])
                        rng_end(_t)
                        _t = rng("P.scan_bar1")
                        K.ptx.bar.sync(K.uint32(BAR_PREP), K.uint32(BAR_PREP_N))
                        rng_end(_t)
                        _t = rng("P.scan_b")
                        # exclusive prefix over token groups; R0=P(2), R1=P(6), e63=P(8)
                        P = K.alloc_local([4], "float32")
                        R0 = K.alloc_local([4], "float32")
                        R1 = K.alloc_local([4], "float32")
                        E63 = K.alloc_local([4], "float32")
                        for j in range(4):
                            K.assign(P[j], K.float32(0.0))
                            K.assign(E63[j], K.float32(0.0))
                        pw = K.alloc_local([4], "float32")
                        for rr in range(8):
                            K.ptx["ld.shared.v4.f32"](pw[0], pw[1], pw[2], pw[3], K.address_of(s_part[gc & 1, rr, col0]))
                            if rr == 2:
                                for j in range(4):
                                    K.assign(R0[j], E63[j])
                            if rr == 6:
                                for j in range(4):
                                    K.assign(R1[j], E63[j])
                            with K.If(rr == r_grp), K.Then():
                                for j in range(4):
                                    K.assign(P[j], E63[j])
                            for j in range(4):
                                K.assign(E63[j], E63[j] + pw[j])
                        rng_end(_t)
                        # (no second barrier: the partials are double-buffered, and the barrier of the next
                        #  chunk separates this chunk's reads from the write two chunks later)
                        _t = rng("P.scan_c")
                        # gate vectors for cg1's state epilogue (warp 0 covers all 128 channels)
                        with K.If(r_grp == 0), K.Then():
                            gv8 = K.alloc_local([8], "float32")
                            for j in range(4):
                                K.ptx.ex2.approx.ftz.f32(gv8[j], R0[j])
                                K.ptx.ex2.approx.ftz.f32(gv8[4 + j], E63[j])
                            gvr = K.alloc_local([2], "uint32")
                            pack_bf16x2(gvr[0], gv8[0], gv8[1])
                            pack_bf16x2(gvr[1], gv8[2], gv8[3])
                            K.ptx["st.shared.v2.b32"](K.address_of(s_gate_r[gc & 3, col0]), gvr[0], gvr[1])
                            K.ptx["st.shared.v4.f32"](
                                K.address_of(s_gate_e[gc & 3, col0]), gv8[4], gv8[5], gv8[6], gv8[7]
                            )
                        # per-thread block factors
                        Rsel = K.alloc_local([4], "float32")
                        fpr = K.alloc_local([4], "float32")  # 2^(G2_63 - R)   (KG factor)
                        f10 = K.alloc_local([4], "float32")  # 2^(R1-R0)
                        for j in range(4):
                            K.assign(Rsel[j], K.Select(hi_block != 0, R1[j], R0[j]))
                            K.ptx.ex2.approx.ftz.f32(fpr[j], E63[j] - Rsel[j])
                        for j in range(4):  # lo threads: f10 = 1 (their "own-reference" tile is the stage tile)
                            K.ptx.ex2.approx.ftz.f32(
                                f10[j], K.Select(hi_block != 0, R1[j] - R0[j], K.float32(0.0))
                            )
                        fprb = K.alloc_local([2], "uint32")
                        f10b = K.alloc_local([2], "uint32")
                        for p in range(2):
                            pack_bf16x2(fprb[p], fpr[2 * p], fpr[2 * p + 1])
                            pack_bf16x2(f10b[p], f10[2 * p], f10[2 * p + 1])
                        # G2_t - R (log2 exponent of this thread's 32 (t,k) cells) as one add per cell with the
                        # per-channel P - R precomputed: P, Rsel, R0/R1/E63 die here
                        PmR = K.alloc_local([4], "float32")
                        for j in range(4):
                            K.assign(PmR[j], P[j] - Rsel[j])
                        for i in range(8):
                            for j in range(4):
                                K.assign(G[i * 4 + j], G[i * 4 + j] + PmR[j])
                        rng_end(_t)
                        _t = rng("P.wait_stage")
                        p_stage.full.wait(st_stage.stage, st_stage.phase)
                        rng_end(_t)
                        _t = rng("P.norm")
                        # per-token 1/sqrt(sum x^2 + eps) of this warp's 8 k and 8 q rows (half a row per lane)
                        nacc = K.alloc_local([2], "uint64")
                        for pp in range(2):
                            K.assign(nacc[pp], f2(K.float32(0.0), K.float32(0.0)))
                        nw = K.alloc_local([4], "uint32")
                        # Square-accumulate in bf16x2 (sixteen terms per accumulator half,
                        # then one f32 combine): one instruction per two channels instead
                        # of unpack-unpack-fma.
                        nsq = K.alloc_local([2], "uint32")
                        for pp in range(2):
                            K.assign(nsq[pp], K.uint32(0))
                        # The state-only prefix does not consume qhat.  Its q
                        # lanes retain zero accumulators, but still participate
                        # in the full-warp shuffle and publish initialized dead
                        # normalization slots.
                        with (K.If(~norm_is_q) if _so else _NullCtx()), (
                            K.Then() if _so else _NullCtx()
                        ):
                            for j in range(8):
                                K.ptx["ld.shared.v4.b32"](
                                    nw[0], nw[1], nw[2], nw[3], K.ptx.addr(base_k, norm_off + (K.int32(16 * j) ^ norm_x))
                                )
                                for pp in range(4):
                                    K.ptx.fma.rn.bf16x2(nsq[pp & 1], nw[pp], nw[pp], nsq[pp & 1])
                        for pp in range(2):
                            K.ptx.fma.rn.f32x2(
                                nacc[pp],
                                f2(bf16_lo(nsq[pp]), bf16_hi(nsq[pp])),
                                f2(K.float32(1.0), K.float32(1.0)),
                                nacc[pp],
                            )
                        nhalf = K.local_scalar(
                            "float32",
                            init=(K.cuda.float2_x(nacc[0]) + K.cuda.float2_y(nacc[0]))
                            + (K.cuda.float2_x(nacc[1]) + K.cuda.float2_y(nacc[1])),
                        )
                        nother = K.local_scalar("uint32")
                        K.ptx.shfl_sync.bfly.b32(
                            nother, K.reinterpret("uint32", nhalf), K.uint32(1), K.uint32(0x1F), K.uint32(0xFFFFFFFF)
                        )
                        nr = K.local_scalar("float32")
                        K.ptx.rsqrt.approx.ftz.f32(nr, (nhalf + K.reinterpret("float32", nother)) + K.float32(1e-6))
                        with K.If(norm_is_q), K.Then():
                            K.assign(nr, nr * scale)
                        if varlen_pack:
                            with K.If(n * CHUNK + norm_tok >= seqlen), K.Then():
                                K.assign(nr, K.float32(0.0))  # tail padding: zero khat/qhat rows
                        with K.If(norm_wr), K.Then():
                            K.ptx.st.shared.f32(
                                K.address_of(s_norm[st_stage.stage, lane >> 4, r_grp * 8 + ((lane & 15) >> 1)]), nr
                            )
                        K.ptx["bar.warp.sync"](K.uint32(0xFFFFFFFF))
                        rq = K.alloc_local([8], "float32")
                        rk = K.alloc_local([8], "float32")
                        K.ptx["ld.shared.v4.f32"](rk[0], rk[1], rk[2], rk[3], K.address_of(s_norm[st_stage.stage, 0, r_grp * 8]))
                        K.ptx["ld.shared.v4.f32"](rk[4], rk[5], rk[6], rk[7], K.address_of(s_norm[st_stage.stage, 0, r_grp * 8 + 4]))
                        K.ptx["ld.shared.v4.f32"](rq[0], rq[1], rq[2], rq[3], K.address_of(s_norm[st_stage.stage, 1, r_grp * 8]))
                        K.ptx["ld.shared.v4.f32"](rq[4], rq[5], rq[6], rq[7], K.address_of(s_norm[st_stage.stage, 1, r_grp * 8 + 4]))
                        rng_end(_t)
                        _t = rng("P.wait_bufs")
                        # buffers this chunk writes: KG[stage] (after G7 two chunks ago), J1 (after G1_1 last chunk)
                        p_prep.empty.wait(st_prep.stage, st_prep.phase)
                        p_j1.empty.wait(st_j1.stage, st_j1.phase)
                        K.ptx[FENCE_ASYNC]()  # async-proxy (MMA) reads of those tiles precede our generic writes
                        rng_end(_t)
                        _t = rng("P.pass2")
                        # ---------------- pass 2: gated operand tiles ---------------
                        is_hi = hi_block != 0
                        ev = K.alloc_local([4], "float32")
                        eiv = K.alloc_local([4], "float32")
                        pk = K.alloc_local([2], "uint64")  # kh pairs
                        pq = K.alloc_local([2], "uint64")  # qh pairs
                        px = K.alloc_local([2], "uint64")
                        py = K.alloc_local([2], "uint64")
                        ow = K.alloc_local([4], "uint32")
                        for i in range(8):
                            t = r_grp * 8 + i
                            # eiv = 2^-G via a reciprocal: same MUFU count as a second
                            # ex2 but without the four FADD negations per token.
                            for j in range(4):
                                K.ptx.ex2.approx.ftz.f32(ev[j], G[i * 4 + j])
                            for j in range(4):
                                K.ptx.rcp.approx.ftz.f32(eiv[j], ev[j])
                            if not _so:
                                lds64(base_q, xo_stage[i], w0, w1)
                                mul2(pq[0], bf16_lo(w0), bf16_hi(w0), rq[i], rq[i])
                                mul2(pq[1], bf16_lo(w1), bf16_hi(w1), rq[i], rq[i])
                            lds64(base_k, xo_stage[i], w0, w1)
                            mul2(pk[0], bf16_lo(w0), bf16_hi(w0), rk[i], rk[i])
                            mul2(pk[1], bf16_lo(w1), bf16_hi(w1), rk[i], rk[i])
                            # Y = kh * einv -> g region row 128+t ; KG = Y * fpr -> s_kg[stage]
                            for p in range(2):
                                K.ptx.mul.rn.f32x2(py[p], pk[p], f2(eiv[2 * p], eiv[2 * p + 1]))
                                pack_bf16x2(ow[p], K.cuda.float2_x(py[p]), K.cuda.float2_y(py[p]))
                            sts64(base_g, xo_stage[i], ow[0], ow[1])
                            for p in range(2):
                                K.ptx.mul.rn.bf16x2(ow[p], ow[p], fprb[p])
                            sts64(base_kg, xo_stage[i] - xo_adj, ow[0], ow[1])
                            # X = kh * e ; Q = qh * e
                            for p in range(2):
                                K.ptx.mul.rn.f32x2(px[p], pk[p], f2(ev[2 * p], ev[2 * p + 1]))
                                if not _so:
                                    K.ptx.mul.rn.f32x2(pq[p], pq[p], f2(ev[2 * p], ev[2 * p + 1]))
                            # one straight-line path for both sub-blocks: hi threads store their own-reference
                            # X1/Q1 to J1 (single predicated stores), then everybody applies f10 (= 1 for lo
                            # threads) and stores the R0-referenced X0/Q0 to the stage
                            for p in range(2):
                                pack_bf16x2(ow[p], K.cuda.float2_x(px[p]), K.cuda.float2_y(px[p]))
                            with K.If(is_hi), K.Then():
                                sts64(base_j1, xo_stage[i] - xo_adj - 32 * 128, ow[0], ow[1])
                            if not _so:
                                for p in range(2):
                                    pack_bf16x2(ow[2 + p], K.cuda.float2_x(pq[p]), K.cuda.float2_y(pq[p]))
                                with K.If(is_hi), K.Then():
                                    sts64(base_j1, xo_stage[i] - xo_adj, ow[2], ow[3])
                            for p in range(2):
                                K.ptx.mul.rn.bf16x2(ow[p], ow[p], f10b[p])
                            sts64(base_k, xo_stage[i], ow[0], ow[1])
                            if not _so:
                                for p in range(2):
                                    K.ptx.mul.rn.bf16x2(ow[2 + p], ow[2 + p], f10b[p])
                                sts64(base_q, xo_stage[i], ow[2], ow[3])
                            # ---- pass 1 of chunk n+1 for this token: its gate sigmoids/cumsum go into the G slots
                            # freed by this token's exponents; then the token's gate logits of chunk n+2 are fetched
                            # into the freed gw words (for the last chunk this computes unused values) ----
                            pass1_token(i, n + 1)
                            with K.If(n + 2 < nlimit), K.Then():
                                issue_g_load_token(n + 2, i)
                        K.ptx[FENCE_ASYNC]()
                        rng_end(_t)
                        p_prep.full.arrive(st_prep.stage)
                        p_j1.full.arrive(st_j1.stage)
                        st_prep.advance()
                        st_j1.advance()
                        st_stage.advance()
                        K.assign(gc, gc + K.int32(1))
                work_next(st_wc, work)

        # =====================================================================
        # CG0: Akk epilogue, hierarchical inverse (-> TB bf16), G3 issue.      warps 8-11
        # =====================================================================
        with cg0:
            st_g1 = K.PipelineState(2, phase=0)  # consumer p_g1.full ; producer p_g1.empty (same index)
            st_g1b = K.PipelineState(1, phase=0)  # consumer of m_g1b (D1 done)
            st_t = K.PipelineState(1, phase=1)  # producer p_t
            st_beta = K.PipelineState(2, phase=0)  # consumer p_beta
            st_tc = K.PipelineState(1, phase=0)  # consumer of p_t.full (this warpgroup's own T, all 128 arrivals)
            st_v = K.PipelineState(1, phase=0)  # consumer of p_v.full (G3)
            st_ue = K.PipelineState(1, phase=1)  # waits p_u.empty before G3 overwrites U^T
            st_aqk0 = K.PipelineState(1, phase=1)  # producer of p_aqk (warps 2/3)
            dT0, offT0 = s_T.encode(major="k", mma_k=16)
            dV0, offV0 = s_v.encode(major="mn", mma_k=16)

            def mma_ss0(dcol, a_desc, a_off, b_desc, b_off, n_k, idesc, accumulate):
                a_base = K.local_scalar("uint64", init=a_desc)
                b_base = K.local_scalar("uint64", init=b_desc)
                with K.If(elected()), K.Then():
                    for kp in range(n_k):
                        K.ptx[MMA_SS](
                            K.Cast("uint32", tmem[0] + dcol),
                            a_base + K.uint64(a_off(kp)),
                            b_base + K.uint64(b_off(kp)),
                            K.uint32(idesc),
                            K.uint32(0),
                            K.uint32(0),
                            K.uint32(0),
                            K.uint32(0),
                            K.ptx.pred(K.uint32(1) if (accumulate or kp != 0) else K.uint32(0)),
                        )
            tmem = tmem_preamble()
            tid1 = K.tid_in_role()
            lane = K.lane_id()
            lw = K.warp_id_in_role()
            rowbits = (tid1 << 16) & 0x600000

            def tmem_at(col, rb=None):
                return K.Cast("uint32", tmem[0] + col + (rowbits if rb is None else rb))

            def ld32(regs, col):
                K.ptx[TC_LD32](*(regs[i] for i in range(32)), tmem_at(col))

            def bar_inv():
                K.ptx.bar.sync(K.uint32(BAR_INV), K.uint32(BAR_INV_N))

            # ---- GDN-style hierarchical inverse of (I + A) on the f16 tile s_A ----
            def neg_pack(dst_word, a, b):
                neg = K.local_scalar("uint64")
                K.ptx.sub.rn.f32x2(neg, f2(K.float32(0.0), K.float32(0.0)), f2(a, b))
                pack_f16x2(dst_word, K.cuda.float2_x(neg), K.cuda.float2_y(neg))

            def invert_diag_8x8(av, block8):
                r = block8 + (lane & 7)
                wds = K.alloc_local([4], "uint32")
                K.ptx["ld.shared.v4.b32"](wds[0], wds[1], wds[2], wds[3], av.ptr_to(r, block8))
                row = [K.local_scalar("float32") for _ in range(8)]
                for p in range(4):
                    K.idioms.cast_f16x2_to_f32x2(row, p, wds[p])
                for i in range(8):
                    with K.If((lane & 7) == i), K.Then():
                        K.assign(row[i], K.float32(1.0))
                rs = K.local_scalar("float32")
                pv = K.alloc_local([2], "uint32") if PAIR_INV else K.local_scalar("uint32")
                upd = K.local_scalar("uint64") if PAIR_INV else None
                for src in range(7):
                    K.ptx.neg.f32(rs, row[src])
                    if PAIR_INV:
                        # Each lane-local FP32 element keeps the incumbent
                        # shuffle/FMA dependency and rounding order; x2 only
                        # issues two independent updates together.
                        for p in range(src // 2):
                            for e in range(2):
                                K.ptx.shfl_sync.idx.b32(
                                    pv[e],
                                    K.reinterpret("uint32", row[2 * p + e]),
                                    K.uint32(src),
                                    K.uint32(0x181F),
                                    K.uint32(0xFFFFFFFF),
                                )
                            with K.If((lane & 7) > src), K.Then():
                                K.ptx.fma.rn.f32x2(
                                    upd,
                                    f2(rs, rs),
                                    f2(
                                        K.reinterpret("float32", pv[0]),
                                        K.reinterpret("float32", pv[1]),
                                    ),
                                    f2(row[2 * p], row[2 * p + 1]),
                                )
                                K.assign(row[2 * p], K.cuda.float2_x(upd))
                                K.assign(row[2 * p + 1], K.cuda.float2_y(upd))
                        if src & 1:
                            i = src - 1
                            K.ptx.shfl_sync.idx.b32(
                                pv[0],
                                K.reinterpret("uint32", row[i]),
                                K.uint32(src),
                                K.uint32(0x181F),
                                K.uint32(0xFFFFFFFF),
                            )
                            with K.If((lane & 7) > src), K.Then():
                                K.assign(row[i], row[i] + rs * K.reinterpret("float32", pv[0]))
                    else:
                        for i in range(7):
                            if i < src:
                                K.ptx.shfl_sync.idx.b32(
                                    pv,
                                    K.reinterpret("uint32", row[i]),
                                    K.uint32(src),
                                    K.uint32(0x181F),
                                    K.uint32(0xFFFFFFFF),
                                )
                                with K.If((lane & 7) > src), K.Then():
                                    K.assign(row[i], row[i] + rs * K.reinterpret("float32", pv))
                    with K.If((lane & 7) > src), K.Then():
                        K.assign(row[src], rs)
                for p in range(4):
                    pack_f16x2(wds[p], row[2 * p], row[2 * p + 1])
                K.ptx["st.shared.v4.b32"](av.ptr_to(r, block8), wds[0], wds[1], wds[2], wds[3])

            def ldm_x4(insn, dst, av, base_row, base_col):
                lm = lane >> 3
                row = base_row + (lane & 7) + (lm & 1) * 8
                col = base_col + (lm >> 1) * 8
                K.ptx[insn](dst[0], dst[1], dst[2], dst[3], av.ptr_to(row, col))

            def stm_x4(src, av, base_row, base_col):
                lm = lane >> 3
                row = base_row + (lane & 7) + (lm & 1) * 8
                col = base_col + (lm >> 1) * 8
                K.ptx[STM_X4](av.ptr_to(row, col), src[0], src[1], src[2], src[3])

            def mma_k8_zero(acc, a, b):
                K.ptx[MMA_K8](
                    acc[0], acc[1], acc[2], acc[3], a[0], a[1], b[0],
                    K.float32(0.0), K.float32(0.0), K.float32(0.0), K.float32(0.0),
                )  # fmt: skip

            def mma_k16(acc, a, b, acc_off, b_off, accumulate):
                c = [acc[acc_off + i] for i in range(4)] if accumulate else [K.float32(0.0)] * 4
                K.ptx[MMA_K16](
                    *(acc[acc_off + i] for i in range(4)), a[0], a[1], a[2], a[3], b[b_off], b[b_off + 1], *c
                )

            def inverse_8_to_16(av, b16):
                a = K.alloc_local([2], "uint32")
                b = K.alloc_local([1], "uint32")
                acc = K.alloc_local([4], "float32")
                dm = K.local_scalar("uint32")
                cm = K.local_scalar("uint32")
                K.ptx[LDM_X1](dm, av.ptr_to(b16 + 8 + (lane & 7), b16 + 8))
                K.ptx[LDM_X1T](cm, av.ptr_to(b16 + 8 + (lane & 7), b16))
                K.assign(a[0], dm)
                K.assign(a[1], dm)
                K.assign(b[0], cm)
                mma_k8_zero(acc, a, b)
                neg_pack(a[0], acc[0], acc[1])
                neg_pack(a[1], acc[2], acc[3])
                K.ptx[LDM_X1T](b[0], av.ptr_to(b16 + (lane & 7), b16))
                mma_k8_zero(acc, a, b)
                pack_f16x2(dm, acc[0], acc[1])
                K.ptx[STM_X1](av.ptr_to(b16 + 8 + (lane & 7), b16), dm)

            def inverse_16_to_32(av, b32):
                a = K.alloc_local([4], "uint32")
                b = K.alloc_local([4], "uint32")
                acc = K.alloc_local([8], "float32")
                out = K.alloc_local([4], "uint32")
                ldm_x4(LDM_X4, a, av, b32 + 16, b32 + 16)
                ldm_x4(LDM_X4T, b, av, b32 + 16, b32)
                mma_k16(acc, a, b, 0, 0, False)
                mma_k16(acc, a, b, 4, 2, False)
                for p in range(4):
                    neg_pack(a[p], acc[2 * p], acc[2 * p + 1])
                ldm_x4(LDM_X4T, b, av, b32, b32)
                mma_k16(acc, a, b, 0, 0, False)
                mma_k16(acc, a, b, 4, 2, False)
                for p in range(4):
                    pack_f16x2(out[p], acc[2 * p], acc[2 * p + 1])
                stm_x4(out, av, b32 + 16, b32)

            def inverse_32_to_64_tb(av, tb, half_warp, stg):
                """T21 = -T22 A21 T11 (f16 mma.sync), scaled by beta_j and written as bf16 TB rows
                32+16*half_warp.. of `tb`; s_A keeps A21 (never overwritten -> no intra-stage barrier)."""
                rb = 32 + half_warp * 16
                a0 = K.alloc_local([4], "uint32")
                a1 = K.alloc_local([4], "uint32")
                b00 = K.alloc_local([4], "uint32")
                b01 = K.alloc_local([4], "uint32")
                b10 = K.alloc_local([4], "uint32")
                b11 = K.alloc_local([4], "uint32")
                acc = K.alloc_local([16], "float32")
                pa = K.alloc_local([8], "uint32")
                o0 = K.alloc_local([4], "uint32")
                o1 = K.alloc_local([4], "uint32")
                ldm_x4(LDM_X4, a0, av, rb, 32)
                ldm_x4(LDM_X4, a1, av, rb, 48)
                ldm_x4(LDM_X4T, b00, av, 32, 0)
                ldm_x4(LDM_X4T, b01, av, 32, 16)
                ldm_x4(LDM_X4T, b10, av, 48, 0)
                ldm_x4(LDM_X4T, b11, av, 48, 16)
                mma_k16(acc, a0, b00, 0, 0, False)
                mma_k16(acc, a0, b00, 4, 2, False)
                mma_k16(acc, a0, b01, 8, 0, False)
                mma_k16(acc, a0, b01, 12, 2, False)
                mma_k16(acc, a1, b10, 0, 0, True)
                mma_k16(acc, a1, b10, 4, 2, True)
                mma_k16(acc, a1, b11, 8, 0, True)
                mma_k16(acc, a1, b11, 12, 2, True)
                for p in range(8):
                    neg_pack(pa[p], acc[2 * p], acc[2 * p + 1])
                ldm_x4(LDM_X4T, b00, av, 0, 0)
                ldm_x4(LDM_X4T, b01, av, 0, 16)
                ldm_x4(LDM_X4T, b10, av, 16, 0)
                ldm_x4(LDM_X4T, b11, av, 16, 16)
                for i in range(4):
                    K.assign(a0[i], pa[i])
                    K.assign(a1[i], pa[4 + i])
                mma_k16(acc, a0, b00, 0, 0, False)
                mma_k16(acc, a0, b00, 4, 2, False)
                mma_k16(acc, a0, b01, 8, 0, False)
                mma_k16(acc, a0, b01, 12, 2, False)
                mma_k16(acc, a1, b10, 0, 0, True)
                mma_k16(acc, a1, b10, 4, 2, True)
                mma_k16(acc, a1, b11, 8, 0, True)
                mma_k16(acc, a1, b11, 12, 2, True)
                # accumulator pair (acc[2pp], acc[2pp+1]) sits at column 8*(pp>>1) + 2*(lane&3) + {0,1}
                bq = K.alloc_local([8], "float32")
                for qq in range(4):
                    K.ptx["ld.shared.v2.f32"](bq[2 * qq], bq[2 * qq + 1], K.address_of(s_beta[stg, 8 * qq + 2 * (lane & 3)]))
                scaled = K.local_scalar("uint64")
                for p in range(4):
                    q0 = 2 * (p >> 1)
                    mul2(scaled, acc[2 * p], acc[2 * p + 1], bq[q0], bq[q0 + 1])
                    pack_bf16x2(o0[p], K.cuda.float2_x(scaled), K.cuda.float2_y(scaled))
                    mul2(scaled, acc[8 + 2 * p], acc[8 + 2 * p + 1], bq[4 + q0], bq[4 + q0 + 1])
                    pack_bf16x2(o1[p], K.cuda.float2_x(scaled), K.cuda.float2_y(scaled))
                stm_x4(o0, tb, rb, 0)
                stm_x4(o1, tb, rb, 16)

            def diag_blocks_to_tb(stg):
                """warps 2/3: T11 / T22 (f16, s_A) * beta_j -> bf16 TB blocks of s_T (one row per lane)."""
                blk = lw - 2
                trow = blk * 32 + lane
                tc0 = blk * 32
                for m in range(4):
                    lds128(s_A, trow, tc0 + 8 * m, wds, 4 * m)
                for p in range(16):
                    K.idioms.cast_f16x2_to_f32x2(acc32, p, wds[p])
                bcol = K.alloc_local([4], "float32")
                scaled = K.local_scalar("uint64")
                for m in range(8):
                    K.ptx["ld.shared.v4.f32"](
                        bcol[0], bcol[1], bcol[2], bcol[3], K.address_of(s_beta[stg, tc0 + 4 * m])
                    )
                    for p in range(2):
                        mul2(
                            scaled,
                            acc32[4 * m + 2 * p],
                            acc32[4 * m + 2 * p + 1],
                            bcol[2 * p],
                            bcol[2 * p + 1],
                        )
                        pack_bf16x2(wds[2 * m + p], K.cuda.float2_x(scaled), K.cuda.float2_y(scaled))
                for m in range(4):
                    sts128(s_T, trow, tc0 + 8 * m, wds, 4 * m)

            # zero the never-written upper 32x32 block of s_A once
            zw = K.alloc_local([4], "uint32")
            for i in range(4):
                K.assign(zw[i], K.uint32(0))
            with K.If(lw < 2), K.Then():  # 64 threads: rows 0..31, cols 32..63
                zr = lw * 16 + (lane >> 1)
                zc = 32 + (lane & 1) * 16
                sts128(s_A, zr, zc, zw, 0)
                sts128(s_A, zr, zc + 8, zw, 0)
            K.ptx[FENCE_ASYNC]()
            bar_inv()

            acc32 = K.alloc_local([32], "float32")
            wds = K.alloc_local([32], "uint32")  # [0..15] Akk/TB/W words, [16..31] Aqk D1 words (warps 2/3)
            work = K.local_scalar("int32")
            st_wc = K.PipelineState(4, phase=0)
            gc = K.local_scalar("int32", init=K.int32(0))
            work_first(st_wc, work)
            with K.While(work < num_work):
                b_idx, h_idx = work_coords(work)
                if split:
                    npre = K.local_scalar("int32", init=K.Select(b_idx != 0, K.int32(split), K.int32(0)))
                    pre_start = npre
                    NCH = K.local_scalar(
                        "int32", init=K.Select(b_idx != 0, K.int32(ntot_chunks - split), K.int32(split))
                    )
                    n_limit = K.local_scalar(
                        "int32", init=K.Select(b_idx != 0, K.int32(ntot_chunks), K.int32(split))
                    )
                else:
                    _lo0, _ln0, NCH = seq_bounds(b_idx)
                    npre = None
                    pre_start = None
                    n_limit = None
                for _so, _cnt in phase_list(npre, NCH):
                    with K.serial(_cnt) as _nl:
                        n = _nl if (not split or _so) else _nl + pre_start
                        nlimit = n_limit if split else NCH
                        stg = gc & 1
                        _t = rng("C0.wait_g1")
                        p_beta.full.wait(st_beta.stage, st_beta.phase)
                        p_g1.full.wait(st_g1.stage, st_g1.phase)
                        bar_inv()  # every warp finished reading s_A / s_T sources of the previous chunk
                        rng_end(_t)
                        _t = rng("C0.akk")
                        # ---- Akk -> s_A (f16, strictly-lower * beta_t) on warps 0/1; Aqk (causal, bf16)
                        # -> the dead Y rows 128..191 of this chunk's stage on warps 2/3 -------------
                        # Emitting D0 is M128: warp 0 owns Akk rows 0..31, warp 1 rows 32..63,
                        # and warps 2/3 own the corresponding Aqk rows.  The split prefix uses
                        # M64 because Aqk is discarded; its four warp fragments each expose
                        # sixteen Akk rows through lanes 0..15.
                        # D1 is M64 layout F: lanes 0..15 of warps 0/1 (2/3) own rows 32..47/48..63.
                        acc_b = K.alloc_local([32], "float32")
                        ld32(acc32, TM_D0 + stg * 64)
                        K.ptx[WAIT_LD]()
                        p_g1.empty.arrive(st_g1.stage)  # D0 buffer consumed: mma0 may run G1_0 of chunk n+2 into it
                        st_g1.advance()

                        def akk_rows(acc, row_base, col_base, diag_shift):
                            """Scale one 32-column Akk row block; diag_shift=None means a full block."""
                            t = row_base + lane
                            bt = K.local_scalar("float32")
                            K.ptx.ld.shared.f32(bt, K.address_of(s_beta[stg, t]))
                            scaled = K.local_scalar("uint64")
                            for p in range(16):
                                mul2(scaled, acc[2 * p], acc[2 * p + 1], bt, bt)
                                lo = K.cuda.float2_x(scaled)
                                hi = K.cuda.float2_y(scaled)
                                if diag_shift is not None:
                                    lo = K.Select(2 * p < lane + diag_shift, lo, K.float32(0.0))
                                    hi = K.Select(2 * p + 1 < lane + diag_shift, hi, K.float32(0.0))
                                pack_f16x2(wds[p], lo, hi)
                                if reorder_h96_mixed and p % 4 == 3:
                                    # Shorten the first atom's producer-to-store
                                    # distance without changing any memory access.
                                    sts128(s_A, t, col_base + 8 * (p // 4), wds, p - 3)
                            if not reorder_h96_mixed:
                                for m in range(4):
                                    sts128(s_A, t, col_base + 8 * m, wds, 4 * m)

                        if _so:
                            with K.If(lane < 16), K.Then():
                                akk_rows(acc32, lw * 16, 0, lw * 16)
                        else:
                            with K.If(lw == 0), K.Then():
                                akk_rows(acc32, 0, 0, 0)
                            with K.If(lw == 1), K.Then():
                                akk_rows(acc32, 32, 0, None)
                        if not _so:
                            with K.If(lw >= 2), K.Then():
                                with K.If(lw == 2), K.Then():
                                    for j in range(32):
                                        K.assign(acc32[j], K.Select(j <= lane, acc32[j], K.float32(0.0)))
                                for p in range(16):
                                    pack_bf16x2(wds[p], acc32[2 * p], acc32[2 * p + 1])
                        with K.If(lw < 2), K.Then():
                            m_g1b.wait(0, st_g1b.phase)
                            ld32(acc_b, TM_D1)
                            K.ptx[WAIT_LD]()
                            with K.If(lane < 16), K.Then():
                                with K.If(lw == 0):
                                    with K.Then():
                                        akk_rows(acc_b, 32, 32, 0)
                                    with K.Else():
                                        akk_rows(acc_b, 48, 32, 16)
                            p_g1b.arrive(0)  # Akk half of D1 consumed
                        bar_inv()
                        rng_end(_t)
                        _t = rng("C0.inv")
                        # warps 2/3 finish the Aqk epilogue in the windows where only warps 0/1 work:
                        # the D1 rows during the 8x8 diagonal inverse, the stores during the 16->32 merge
                        with K.If(lw < 2):
                            with K.Then():
                                invert_diag_8x8(s_A, ((lw * 32 + lane) >> 3) * 8)
                            with K.Else():
                                m_g1b.wait(0, st_g1b.phase)
                                if not _so:
                                    ld32(acc_b, TM_D1)
                                    K.ptx[WAIT_LD]()
                                    with K.If(lane < 16), K.Then():
                                        dshift = K.Select(lw == 2, K.int32(0), K.int32(16))
                                        for j in range(32):
                                            K.assign(
                                                acc_b[j],
                                                K.Select(j <= lane + dshift, acc_b[j], K.float32(0.0)),
                                            )
                                        for p in range(16):
                                            pack_bf16x2(wds[16 + p], acc_b[2 * p], acc_b[2 * p + 1])
                                p_g1b.arrive(0)  # Aqk half of D1 consumed
                        st_g1b.advance()
                        bar_inv()
                        inverse_8_to_16(s_A, lw * 16)
                        bar_inv()
                        with K.If(lw < 2):
                            with K.Then():
                                inverse_16_to_32(s_A, lw * 32)
                            with K.Else():
                                if not (_so and elide_even_prefix_sidepipes):
                                    # G1_1 (async proxy) finished reading the Y rows: publish Aqk there.
                                    K.ptx[FENCE_ASYNC]()
                                    if not _so:
                                        svy = s_stage[stg]
                                        arow = K.Select(lw == 2, lane, 32 + lane)
                                        for m in range(4):
                                            sts128(svy, 128 + arow, 8 * m, wds, 4 * m)
                                        with K.If(lane < 16), K.Then():
                                            arow2 = K.Select(lw == 2, 32 + lane, 48 + lane)
                                            for m in range(4):
                                                sts128(svy, 128 + arow2, 32 + 8 * m, wds, 16 + 4 * m)
                                        # the never-computed upper-right 32x32 block must be zero for G6
                                        zi = (lw - 2) * 32 + lane
                                        zr = zi >> 1
                                        zc = 32 + (zi & 1) * 16
                                        sts128(svy, 128 + zr, zc, zw, 0)
                                        sts128(svy, 128 + zr, zc + 8, zw, 0)
                                    K.ptx[FENCE_ASYNC]()
                                    p_aqk.full.arrive(st_aqk0.stage)
                        if not (_so and elide_even_prefix_sidepipes):
                            st_aqk0.advance()
                        bar_inv()
                        rng_end(_t)
                        _t = rng("C0.t64")
                        # ---- last merge writes TB = T * beta_j (bf16) straight into s_T: warps 0/1 compute
                        # T21 with mma.sync, warps 2/3 convert the diagonal blocks T11 / T22 meanwhile ----
                        p_t.empty.wait(st_t.stage, st_t.phase)  # G3 of the previous chunk finished reading s_T
                        K.ptx[FENCE_ASYNC]()
                        with K.If(lw < 2):
                            with K.Then():
                                inverse_32_to_64_tb(s_A, s_T, lw, stg)
                            with K.Else():
                                diag_blocks_to_tb(stg)
                        K.ptx[FENCE_ASYNC]()
                        p_t.full.arrive(st_t.stage)
                        st_t.advance()
                        p_beta.empty.arrive(st_beta.stage)  # last beta read of this chunk
                        st_beta.advance()
                        rng_end(_t)
                        # ---- G3: U^T = V^T TB^T, issued here by warp 0 (no handoff); TB is released by G4'' ----
                        _t = rng("C0.g3")
                        with K.If(lw == 0), K.Then():
                            p_t.full.wait(st_tc.stage, st_tc.phase)  # every cg0 thread's TB stores are published
                            st_tc.advance()
                            p_v.full.wait(st_v.stage, st_v.phase)
                            p_u.empty.wait(st_ue.stage, st_ue.phase)
                            st_ue.advance()
                            mma_ss0(TM_U, dV0.value, offV0, dT0.value, offT0, 4, ID_G23, False)
                            pr = elect_local()
                            p_v.empty.arrive(st_v.stage, pred=pr)
                            st_v.advance()
                            m_g3.arrive(0, pred=pr)
                        rng_end(_t)
                        K.assign(gc, gc + K.int32(1))
                work_next(st_wc, work)

        # =====================================================================
        # CG1: U / O / state epilogues.                                 warps 12-15
        # =====================================================================
        with cg1:
            # Re-test v76's state-chain phase derivation for H64 uniform after
            # the grid, cg0 budget and inverse issue schedule all changed.
            # Mixed keeps mutable cursors because its wide epilogue crosses a
            # severe spill cliff when gc parity is rematerialized.
            mutable_state_cursor = varlen_pack
            # Compose v162's output/publication derivation with v168's lower
            # cg1 tier; the changed register cliff may make the pairing useful.
            mutable_output_cursor = varlen_pack or (uniform_pack and H == 96)
            if mutable_state_cursor:
                st_u = K.PipelineState(1, phase=0)
            if mutable_output_cursor:
                st_o = K.PipelineState(1, phase=0)
            if mutable_state_cursor:
                st_sacc = K.PipelineState(1, phase=0)
            if mutable_output_cursor:
                st_osm = K.PipelineState(1, phase=1)
            if nseq != 1:
                st_pn = K.PipelineState(2, phase=0)
            if mutable_state_cursor:
                st_v1 = K.PipelineState(1, phase=0)
            tmem = tmem_preamble()
            tid2 = K.tid_in_role()
            lane = K.lane_id()
            lw = K.warp_id_in_role()
            rowbits = (tid2 << 16) & 0x600000

            def tmem_at2(col, extra_rows=0):
                return K.Cast("uint32", tmem[0] + col + rowbits + (extra_rows << 16))

            def ld32b(regs, col):
                K.ptx[TC_LD32](*(regs[i] for i in range(32)), tmem_at2(col))

            def st32b(col, regs):
                K.ptx[TC_ST32](tmem_at2(col), *(regs[i] for i in range(32)))

            fr = K.alloc_local([64 if varlen_pack else 32], "float32")
            wds = K.alloc_local([32], "uint32")
            fac = K.alloc_local([8], "float32")
            bfac = K.alloc_local([4], "uint32")

            def load_pack64(col):
                """TMEM f32[64] -> packed bf16x2[32], selected by shape."""
                if varlen_pack:
                    ld32b(fr, col)
                    K.ptx[TC_LD32](*(fr[32 + i] for i in range(32)), tmem_at2(col + 32))
                    K.ptx[WAIT_LD]()
                    for p in range(32):
                        pack_bf16x2(wds[p], fr[2 * p], fr[2 * p + 1])
                else:
                    for sub in range(2):
                        ld32b(fr, col + 32 * sub)
                        K.ptx[WAIT_LD]()
                        for p in range(16):
                            pack_bf16x2(wds[16 * sub + p], fr[2 * p], fr[2 * p + 1])

            def state_epilogue_narrow(nstg):
                """S~' = bf16(S * 2^R0(nstg)) -> TM_ST ; S *= 2^G2_63(nstg) in place."""
                state_pair = K.local_scalar("uint64")
                # Process each 32-column sub-round completely before loading the
                # next, keeping only 32 fp32 TMEM values live in this role.
                with K.serial(2) as half:
                    hc = half * 64
                    hs = half * 32
                    for sub in range(2):
                        qc = hc + 32 * sub
                        ld32b(fr, TM_S + qc)
                        K.ptx[WAIT_LD]()
                        # One packed load supplies eight bf16 midpoint factors; two fp32 loads
                        # supply the eight resident-state decay factors.
                        for m in range(4):
                            K.ptx["ld.shared.v4.b32"](
                                bfac[0], bfac[1], bfac[2], bfac[3], K.address_of(s_gate_r[nstg, qc + 8 * m])
                            )
                            K.ptx["ld.shared.v4.f32"](
                                fac[0], fac[1], fac[2], fac[3], K.address_of(s_gate_e[nstg, qc + 8 * m])
                            )
                            K.ptx["ld.shared.v4.f32"](
                                fac[4], fac[5], fac[6], fac[7], K.address_of(s_gate_e[nstg, qc + 8 * m + 4])
                            )
                            for p in range(4):
                                pack_bf16x2(
                                    wds[4 * m + p],
                                    fr[8 * m + 2 * p],
                                    fr[8 * m + 2 * p + 1],
                                )
                                K.ptx.mul.rn.bf16x2(
                                    wds[4 * m + p],
                                    wds[4 * m + p],
                                    bfac[p],
                                )
                                mul2(
                                    state_pair,
                                    fr[8 * m + 2 * p],
                                    fr[8 * m + 2 * p + 1],
                                    fac[2 * p],
                                    fac[2 * p + 1],
                                )
                                K.assign(fr[8 * m + 2 * p], K.cuda.float2_x(state_pair))
                                K.assign(fr[8 * m + 2 * p + 1], K.cuda.float2_y(state_pair))
                        K.ptx[TC_ST16](tmem_at2(TM_ST + hs + 16 * sub), *(wds[i] for i in range(16)))
                        st32b(TM_S + qc, fr)
                K.ptx[WAIT_ST]()

            def state_epilogue_wide(nstg):
                """Mixed-varlen path: overlap both 32-column TMEM loads."""
                state_pair = K.local_scalar("uint64")
                with K.serial(2) as half:
                    hc = half * 64
                    hs = half * 32
                    ld32b(fr, TM_S + hc)
                    K.ptx[TC_LD32](*(fr[32 + i] for i in range(32)), tmem_at2(TM_S + hc + 32))
                    K.ptx[WAIT_LD]()
                    for sub in range(2):
                        qc = hc + 32 * sub
                        fb = 32 * sub
                        for m in range(4):
                            K.ptx["ld.shared.v4.b32"](
                                bfac[0], bfac[1], bfac[2], bfac[3], K.address_of(s_gate_r[nstg, qc + 8 * m])
                            )
                            K.ptx["ld.shared.v4.f32"](
                                fac[0], fac[1], fac[2], fac[3], K.address_of(s_gate_e[nstg, qc + 8 * m])
                            )
                            K.ptx["ld.shared.v4.f32"](
                                fac[4], fac[5], fac[6], fac[7], K.address_of(s_gate_e[nstg, qc + 8 * m + 4])
                            )
                            for p in range(4):
                                pack_bf16x2(
                                    wds[4 * m + p],
                                    fr[fb + 8 * m + 2 * p],
                                    fr[fb + 8 * m + 2 * p + 1],
                                )
                                K.ptx.mul.rn.bf16x2(wds[4 * m + p], wds[4 * m + p], bfac[p])
                                mul2(
                                    state_pair,
                                    fr[fb + 8 * m + 2 * p],
                                    fr[fb + 8 * m + 2 * p + 1],
                                    fac[2 * p],
                                    fac[2 * p + 1],
                                )
                                K.assign(fr[fb + 8 * m + 2 * p], K.cuda.float2_x(state_pair))
                                K.assign(fr[fb + 8 * m + 2 * p + 1], K.cuda.float2_y(state_pair))
                        K.ptx[TC_ST16](tmem_at2(TM_ST + hs + 16 * sub), *(wds[i] for i in range(16)))
                    st32b(TM_S + hc, fr)
                    K.ptx[TC_ST32](tmem_at2(TM_S + hc + 32), *(fr[32 + i] for i in range(32)))
                K.ptx[WAIT_ST]()

            state_epilogue = state_epilogue_wide if varlen_pack else state_epilogue_narrow

            work = K.local_scalar("int32")
            st_wc = K.PipelineState(4, phase=0)
            gc = K.local_scalar("int32", init=K.int32(0))
            work_first(st_wc, work)
            with K.While(work < num_work):
                b_idx, h_idx = work_coords(work)
                if split:
                    npre = K.local_scalar("int32", init=K.Select(b_idx != 0, K.int32(split), K.int32(0)))
                    pre_start = npre
                    NCH = K.local_scalar(
                        "int32", init=K.Select(b_idx != 0, K.int32(ntot_chunks - split), K.int32(split))
                    )
                    n_limit = K.local_scalar(
                        "int32", init=K.Select(b_idx != 0, K.int32(ntot_chunks), K.int32(split))
                    )
                else:
                    _lo2, _ln2, NCH = seq_bounds(b_idx)
                    npre = None
                    pre_start = None
                    n_limit = None
                # ---- S^T <- initial_state[seq, head]; rows are v, columns k, which
                # is exactly FLA's state_v_first layout, so each lane reads one
                # contiguous 512B row.
                _sseq = K.int32(0) if split else b_idx
                sbase = K.local_scalar(
                    "int64",
                    init=(K.Cast("int64", _sseq) * K.int64(H) + K.Cast("int64", h_idx))
                    * K.int64(D_HEAD * D_HEAD)
                    + K.Cast("int64", tid2) * K.int64(D_HEAD),
                )
                if varlen_pack:
                    sbuf = (
                        [wds[i] for i in range(32)],
                        [fr[i] for i in range(32)],
                        [fr[32 + i] for i in range(32)],
                    )
                else:
                    # Keep two 32-column global-load rounds in flight, reusing
                    # each after its register-to-TMEM store issues.
                    sbuf = (
                        [wds[i] for i in range(32)],
                        [fr[i] for i in range(32)],
                    )

                def issue_state_round(c, buf):
                    # Each lane walks a contiguous 512 B row, so the eight 16 B
                    # loads that share a 128 B line must be allowed to hit L1;
                    # LDG_V4's L1::no_allocate would fetch every line 8 times.
                    for vec in range(8):
                        K.ptx["ld.global.nc.v4.b32"](
                            buf[vec * 4], buf[vec * 4 + 1], buf[vec * 4 + 2], buf[vec * 4 + 3],
                            istate.ptr_to([sbase + K.int64(c * 32 + vec * 4)]),
                        )

                if varlen_pack:
                    for c in range(3):
                        issue_state_round(c, sbuf[c])
                    for c in range(4):
                        K.ptx[TC_ST32](tmem_at2(TM_S + 32 * c), *(sbuf[c % 3][i] for i in range(32)))
                        if c == 0:
                            issue_state_round(3, sbuf[0])
                else:
                    for c in range(2):
                        issue_state_round(c, sbuf[c])
                    for c in range(4):
                        K.ptx[TC_ST32](tmem_at2(TM_S + 32 * c), *(sbuf[c & 1][i] for i in range(32)))
                        if c < 2:
                            issue_state_round(c + 2, sbuf[c & 1])
                K.ptx[WAIT_ST]()
                # chunk 0's gate vectors, then the same decay/copy epilogue the loop uses
                if nseq != 1:
                    p_prep.full.wait(st_pn.stage, st_pn.phase)
                    st_pn.advance()
                else:
                    # Every cg1 barrier advances exactly once per global chunk.
                    # Derive its cursor from gc instead of carrying seven mutable
                    # stage/phase scalars through this register-heavy loop.
                    p_prep.full.wait(gc & 1, (gc >> 1) & 1)
                state_epilogue(gc & 3)
                m_s.arrive(0)
                for _so, _cnt in phase_list(npre, NCH):
                    with K.serial(_cnt) as _nl:
                        n = _nl if (not split or _so) else _nl + pre_start
                        nlimit = n_limit if split else NCH
                        # H96 fixed launches one CTA per work item, so there is
                        # no next item and this hint is provably unreachable.
                        if not (H == 96 and nseq == 1):
                            # Pull the next work item's state into L2 a chunk ahead, so the
                            # seed above starts from L2 rather than DRAM.
                            with K.If((n == nlimit - state_pf_lead) & (work + num_ctas < num_work)), K.Then():
                                with K.If((lw == 0) & (K.cuda.elect_sync() != K.uint32(0))), K.Then():
                                    wn = K.local_scalar("int32", init=work + num_ctas)
                                    bn, hn = work_coords(wn)
                                    K.ptx[BULK_PREFETCH](
                                        istate.ptr_to([
                                            (K.Cast("int64", bn) * K.int64(H) + K.Cast("int64", hn))
                                            * K.int64(D_HEAD * D_HEAD)
                                        ]),
                                        K.uint32(D_HEAD * D_HEAD * 4),
                                    )
                        _t = rng("C1.wait_v1")
                        # ---- V1 = S~'^T X0^T f32 -> bf16 into the UB columns (A operand of G4''); UB(n-1) was
                        # consumed by G7(n-1), whose completion this warpgroup awaited before its state pass ----
                        if mutable_state_cursor:
                            m_v1.wait(0, st_v1.phase)
                            st_v1.advance()
                        else:
                            m_v1.wait(0, gc & 1)
                        rng_end(_t)
                        _t = rng("C1.v1")
                        load_pack64(v1_col)
                        st32b(TM_UB, wds)
                        K.ptx[WAIT_ST]()
                        m_v1b.arrive(0)
                        rng_end(_t)
                        _t = rng("C1.wait_u")
                        # ---- U^T f32 -> bf16 (A operand for G6/G7) -------------------
                        if mutable_state_cursor:
                            p_u.full.wait(st_u.stage, st_u.phase)
                        else:
                            p_u.full.wait(0, gc & 1)
                        rng_end(_t)
                        _t = rng("C1.u")
                        load_pack64(TM_U)
                        st32b(TM_UB, wds)
                        K.ptx[WAIT_ST]()
                        if mutable_state_cursor:
                            p_u.empty.arrive(st_u.stage)
                            st_u.advance()
                        else:
                            p_u.empty.arrive(0)
                        m_ub.arrive(0)
                        rng_end(_t)
                        _t = rng("C1.wait_o")
                        # ---- O^T f32 -> bf16 -> s_o[t][v] via stmatrix.trans ------------
                        if not (_so and elide_even_prefix_sidepipes):
                            if mutable_output_cursor:
                                p_o.full.wait(st_o.stage, st_o.phase)
                                p_osm.empty.wait(st_osm.stage, st_osm.phase)
                            else:
                                p_o.full.wait(0, gc & 1)
                                p_osm.empty.wait(0, (gc & 1) ^ 1)
                        rng_end(_t)
                        _t = rng("C1.o")
                        if not _so:
                            for hh in range(2):
                                K.ptx[TC_LD256](*(fr[i] for i in range(32)), tmem_at2(o_col, 16 * hh))
                                K.ptx[WAIT_LD]()
                                # regs[4i+2q+e] = O^T[v = 32w + 16hh + 8q + lane/4][t = 8i + 2(lane%4) + e]
                                for i in range(8):
                                    for qq in range(2):
                                        pack_bf16x2(
                                            wds[16 * hh + 2 * i + qq],
                                            fr[4 * i + 2 * qq],
                                            fr[4 * i + 2 * qq + 1],
                                        )
                            # 8 stmatrix.x4.trans calls: ci covers hh=ci//4, i in {2(ci%4), 2(ci%4)+1}, q in {0,1}
                            mm = lane >> 3
                            jj = lane & 7
                            for ci in range(8):
                                hh = ci // 4
                                ib = 2 * (ci % 4)
                                # matrix m = (i - ib)*2 + q  ->  i = ib + m//2, q = m%2
                                trow = (ib + (mm >> 1)) * 8 + jj
                                tcol = lw * 32 + 16 * hh + 8 * (mm & 1)
                                K.ptx[STM_X4T](
                                    s_o.ptr_to(trow, tcol),
                                    wds[16 * hh + 2 * ib + 0],
                                    wds[16 * hh + 2 * ib + 1],
                                    wds[16 * hh + 2 * ib + 2],
                                    wds[16 * hh + 2 * ib + 3],
                                )
                        if not (_so and elide_even_prefix_sidepipes):
                            K.ptx[FENCE_ASYNC]()
                            if mutable_output_cursor:
                                p_o.empty.arrive(st_o.stage)
                                st_o.advance()
                                p_osm.full.arrive(st_osm.stage)
                                st_osm.advance()
                            else:
                                p_o.empty.arrive(0)
                                p_osm.full.arrive(0)
                        rng_end(_t)
                        _t = rng("C1.wait_s")
                        # ---- S~'(n+1) = bf16(S_n * 2^R0(n+1)) ; S *= 2^(G2_63 of chunk n+1) — one TMEM pass --
                        # (decay must precede the next chunk's G7 accumulation; the bf16 operand copy is the
                        #  undecayed end-of-chunk state scaled per column k by the next chunk's 2^R0, which
                        #  makes G5's Q0 and G4's W' operands reference-free)
                        if mutable_state_cursor:
                            m_sacc.wait(0, st_sacc.phase)
                            st_sacc.advance()
                        else:
                            m_sacc.wait(0, gc & 1)
                        with K.If(n < nlimit - 1), K.Then():
                            if nseq != 1:
                                p_prep.full.wait(st_pn.stage, st_pn.phase)
                            else:
                                # chunk n+1's gate vectors
                                p_prep.full.wait((gc + 1) & 1, ((gc + 1) >> 1) & 1)
                        rng_end(_t)
                        _t = rng("C1.s")
                        with K.If(n < nlimit - 1), K.Then():
                            state_epilogue((gc + 1) & 3)
                            m_s.arrive(0)
                            if nseq != 1:
                                st_pn.advance()
                        K.assign(gc, gc + K.int32(1))
                        rng_end(_t)
                work_next(st_wc, work)
            # tmem dealloc by the allocating warp once every cg1 warp is done
            K.ptx.bar.sync(K.uint32(BAR_CG1), K.uint32(BAR_CG1_N))
            with K.If(lw == 0), K.Then():
                K.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
                K.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                    K.Cast("uint32", tmem[0]), K.uint32(TMEM_COLS)
                )

        with auxg:
            # =====================================================================
            # MMA issuer 0: chunk-local GEMMs G1 (Akk/Aqk), G2 (W'^T), G3 (U_pre^T).  warp 12
            # =====================================================================
            with mma0:
                # G1_0 of every chunk, issued as soon as prep publishes the chunk (double-buffered D0)
                st_prep = K.PipelineState(2, phase=0)
                st_g1 = K.PipelineState(2, phase=1)  # producer p_g1
                st_stg = K.PipelineState(2, phase=0)  # ledger for p_stage.empty commits (G1_0 reads X0/Q0 rows)
                tmem = tmem_preamble()

                def mma_ss(dcol, a_desc, a_off, b_desc, b_off, n_k, idesc, accumulate):
                    a_base = K.local_scalar("uint64", init=a_desc)
                    b_base = K.local_scalar("uint64", init=b_desc)
                    with K.If(elected()), K.Then():
                        for kp in range(n_k):
                            K.ptx[MMA_SS](
                                K.Cast("uint32", tmem[0] + dcol),
                                a_base + K.uint64(a_off(kp)),
                                b_base + K.uint64(b_off(kp)),
                                K.uint32(idesc),
                                K.uint32(0),
                                K.uint32(0),
                                K.uint32(0),
                                K.uint32(0),
                                K.ptx.pred(K.uint32(1) if (accumulate or kp != 0) else K.uint32(0)),
                            )

                def shifted(off_fn, extra):
                    return lambda kp: off_fn(kp) + extra

                dA0, offA = s_stage[0].encode(major="k", mma_k=16)
                work = K.local_scalar("int32")
                st_wc = K.PipelineState(4, phase=0)
                gc = K.local_scalar("int32", init=K.int32(0))
                work_first(st_wc, work)
                with K.While(work < num_work):
                    b_idx, h_idx = work_coords(work)
                    if split:
                        npre = K.local_scalar("int32", init=K.Select(b_idx != 0, K.int32(split), K.int32(0)))
                        pre_start = npre
                        NCH = K.local_scalar(
                            "int32", init=K.Select(b_idx != 0, K.int32(ntot_chunks - split), K.int32(split))
                        )
                        n_limit = K.local_scalar(
                            "int32", init=K.Select(b_idx != 0, K.int32(ntot_chunks), K.int32(split))
                        )
                    else:
                        _lom, _lnm, NCH = seq_bounds(b_idx)
                        npre = None
                        pre_start = None
                        n_limit = None
                    for _so, _cnt in phase_list(npre, NCH):
                        with K.serial(_cnt) as _nl:
                            n = _nl if (not split or _so) else _nl + pre_start
                            nlimit = n_limit if split else NCH
                            stg = gc & 1
                            dA = dA0.value + K.Cast("uint64", stg) * K.uint64(STAGE_UNITS)
                            _t = rng("M0.wait_g1")
                            p_g1.empty.wait(st_g1.stage, st_g1.phase)
                            p_prep.full.wait(st_prep.stage, st_prep.phase)
                            st_prep.advance()
                            rng_end(_t)
                            _t = rng("M0.g1")
                            mma_ss(
                                TM_D0 + stg * 64,
                                dA,
                                offA,
                                dA,
                                shifted(offA, OFF_Y0),
                                8,
                                ID_G1B if _so else ID_G1,
                                False,
                            )
                            pr = elect_local()
                            p_g1.full.arrive(st_g1.stage, pred=pr)  # D0 done (G1_0)
                            st_g1.advance()
                            p_stage.empty.arrive(st_stg.stage, pred=pr)  # G1_0 finished reading the stage's X0/Q0 rows
                            st_stg.advance()
                            rng_end(_t)
                            K.assign(gc, gc + K.int32(1))
                    work_next(st_wc, work)

            # =====================================================================
            # MMA issuer 1: G5 (O^T=S~Q~), V1 (S~X0), G4'' (U^T-=V1b TB^T), G1 of the next chunk, G6, G7.   warp 19
            # =====================================================================
            with mma1:
                st_stage = K.PipelineState(2, phase=0)
                st_prep = K.PipelineState(2, phase=0)  # ledger for p_prep.empty commits
                st_j1 = K.PipelineState(2, phase=0)
                st_g1b_e = K.PipelineState(1, phase=1)  # producer-side wait on p_g1b (D1 consumed)
                st_v1b = K.PipelineState(1, phase=0)  # consumer of m_v1b (V1 bf16 copy ready)
                st_t1 = K.PipelineState(1, phase=0)  # ledger for p_t.empty commits (G4'' is TB's last reader)
                st_aqk = K.PipelineState(1, phase=0)
                st_u = K.PipelineState(1, phase=1)  # producer p_u.full
                st_o = K.PipelineState(1, phase=1)  # producer p_o
                st_ub = K.PipelineState(1, phase=0)
                st_s = K.PipelineState(1, phase=0)
                st_g3 = K.PipelineState(1, phase=0)
                tmem = tmem_preamble()

                def mma_ts(dcol, a_col, b_desc, b_off, n_k, idesc, accumulate):
                    b_base = K.local_scalar("uint64", init=b_desc)
                    with K.If(elected()), K.Then():
                        for kp in range(n_k):
                            K.ptx[MMA_SS](
                                K.Cast("uint32", tmem[0] + dcol),
                                K.Cast("uint32", tmem[0] + a_col + kp * 8),
                                b_base + K.uint64(b_off(kp)),
                                K.uint32(idesc),
                                K.uint32(0),
                                K.uint32(0),
                                K.uint32(0),
                                K.uint32(0),
                                K.ptx.pred(K.uint32(1) if (accumulate or kp != 0) else K.uint32(0)),
                            )

                def mma_ss(dcol, a_desc, a_off, b_desc, b_off, n_k, idesc, accumulate):
                    """SS-form chain issued by one elected lane; descriptor phases are plain 64-bit adds
                    (the 16B-unit offsets never carry out of the 14-bit start-address field)."""
                    a_base = K.local_scalar("uint64", init=a_desc)
                    b_base = K.local_scalar("uint64", init=b_desc)
                    with K.If(elected()), K.Then():
                        for kp in range(n_k):
                            K.ptx[MMA_SS](
                                K.Cast("uint32", tmem[0] + dcol),
                                a_base + K.uint64(a_off(kp)),
                                b_base + K.uint64(b_off(kp)),
                                K.uint32(idesc),
                                K.uint32(0),
                                K.uint32(0),
                                K.uint32(0),
                                K.uint32(0),
                                K.ptx.pred(K.uint32(1) if (accumulate or kp != 0) else K.uint32(0)),
                            )

                def shifted(off_fn, extra):
                    return lambda kp: off_fn(kp) + extra

                dT1, offT1 = s_T.encode(major="k", mma_k=16)
                dJ, offJ = s_j1[0].encode(major="k", mma_k=16)
                dA0, offA = s_stage[0].encode(major="k", mma_k=16)
                dKG0, offKG = s_kg[0].encode(major="mn", mma_k=16)

                def issue_g1(par):
                    """G1_1 of the chunk in stage ``par``: [Akk;Aqk] block D1 (M64, J1 rows x Y1)."""
                    dA1 = dA0.value + K.Cast("uint64", par) * K.uint64(STAGE_UNITS)
                    _t = rng("M1.wait_g1")
                    p_g1b.wait(0, st_g1b_e.phase)  # cg0 consumed the previous D1
                    st_g1b_e.advance()
                    p_j1.full.wait(st_j1.stage, st_j1.phase)
                    rng_end(_t)
                    _t = rng("M1.g1")
                    dJs = dJ.value + K.Cast("uint64", st_j1.stage) * K.uint64(J1_UNITS)
                    mma_ss(TM_D1, dJs, offJ, dA1, shifted(offA, OFF_Y1), 8, ID_G1B, False)
                    pr = elect_local()
                    m_g1b.arrive(0, pred=pr)  # D1 done (G1_1)
                    p_j1.empty.arrive(st_j1.stage, pred=pr)
                    st_j1.advance()
                    rng_end(_t)

                work = K.local_scalar("int32")
                st_wc = K.PipelineState(4, phase=0)
                gc = K.local_scalar("int32", init=K.int32(0))
                work_first(st_wc, work)
                with K.While(work < num_work):
                    b_idx, h_idx = work_coords(work)
                    if split:
                        npre = K.local_scalar("int32", init=K.Select(b_idx != 0, K.int32(split), K.int32(0)))
                        pre_start = npre
                        NCH = K.local_scalar(
                            "int32", init=K.Select(b_idx != 0, K.int32(ntot_chunks - split), K.int32(split))
                        )
                        n_limit = K.local_scalar(
                            "int32", init=K.Select(b_idx != 0, K.int32(ntot_chunks), K.int32(split))
                        )
                    else:
                        _lo1, _ln1, NCH = seq_bounds(b_idx)
                        npre = None
                        pre_start = None
                        n_limit = None
                    issue_g1(gc & 1)
                    for _so, _cnt in phase_list(npre, NCH):
                        with K.serial(_cnt) as _nl:
                            n = _nl if (not split or _so) else _nl + pre_start
                            nlimit = n_limit if split else NCH
                            stg = gc & 1
                            dA = dA0.value + K.Cast("uint64", stg) * K.uint64(STAGE_UNITS)
                            dKG = dKG0.value + K.Cast("uint64", stg) * K.uint64(KG_UNITS)
                            # ---- G5: O^T = S~'^T Q0^T, then V1 = S~'^T X0^T (V1 releases the stage) -------
                            _t = rng("M1.wait_g5")
                            m_s.wait(0, st_s.phase)
                            st_s.advance()
                            p_prep.full.wait(st_prep.stage, st_prep.phase)  # Q0/X0 rows of this chunk published
                            if not (_so and elide_even_prefix_sidepipes):
                                p_o.empty.wait(st_o.stage, st_o.phase)
                            rng_end(_t)
                            _t = rng("M1.g5")
                            if fuse_g5_v1 and not _so:
                                # X0 and Q0 are adjacent 64-row regions in the
                                # K-major B descriptor.  One N128 chain emits
                                # V1 in the low half and O in the high half.
                                mma_ts(v1_col, TM_ST, dA, offA, 8, ID_G5F, False)
                            else:
                                if not _so:
                                    mma_ts(o_col, TM_ST, dA, shifted(offA, OFF_Q), 8, ID_G56, False)
                                # V1's accumulator was converted by cg1 before it published S~ of this chunk.
                                mma_ts(v1_col, TM_ST, dA, offA, 8, ID_G56, False)
                            pr = elect_local()
                            m_v1.arrive(0, pred=pr)  # V1 done -> cg1 converts it
                            p_stage.empty.arrive(st_stage.stage, pred=pr)  # stage fully consumed (G5 + V1)
                            st_stage.advance()
                            rng_end(_t)
                            # ---- G4'': U^T -= V1b TB^T (after G3 landed in U^T and cg1 published V1b) --------
                            _t = rng("M1.wait_g4")
                            m_v1b.wait(0, st_v1b.phase)
                            st_v1b.advance()
                            m_g3.wait(0, st_g3.phase)
                            st_g3.advance()
                            rng_end(_t)
                            _t = rng("M1.g4")
                            mma_ts(TM_U, TM_UB, dT1.value, offT1, 4, ID_G4B, True)
                            pr = elect_local()
                            p_u.full.arrive(st_u.stage, pred=pr)
                            st_u.advance()
                            p_t.empty.arrive(st_t1.stage, pred=pr)  # TB's last reader
                            st_t1.advance()
                            rng_end(_t)
                            # Aqk(n) was published by cg0 long ago; waiting for it before G1(n+1) keeps
                            # cg0's Aqk(n+1) arrival from ever running a phase ahead of this wait.
                            if not (_so and elide_even_prefix_sidepipes):
                                p_aqk.full.wait(st_aqk.stage, st_aqk.phase)
                            # ---- G1 of the next chunk sits between G4 and G6 in tensor-pipe order ----
                            with K.If(n + 1 < nlimit), K.Then():
                                issue_g1((gc + 1) & 1)
                            # ---- G6: O^T += U^T Aqk^T ; G7: S^T += U^T KG ----------------------
                            _t = rng("M1.wait_g6")
                            m_ub.wait(0, st_ub.phase)
                            st_ub.advance()
                            rng_end(_t)
                            _t = rng("M1.g6g7")
                            if reorder_h96_mixed:
                                # G7 feeds the next chunk's state dependency;
                                # output consumption of G6 is not on that chain.
                                mma_ts(TM_S, TM_UB, dKG, offKG, 4, ID_G7, True)
                            if not _so:
                                mma_ts(o_col, TM_UB, dA, shifted(offA, OFF_Y0), 4, ID_G56, True)
                            pr = elect_local()
                            if not (_so and elide_even_prefix_sidepipes):
                                p_o.full.arrive(st_o.stage, pred=pr)
                                st_o.advance()
                                st_aqk.advance()
                            if not reorder_h96_mixed:
                                mma_ts(TM_S, TM_UB, dKG, offKG, 4, ID_G7, True)
                                pr = elect_local()
                            m_sacc.arrive(0, pred=pr)
                            p_prep.empty.arrive(st_prep.stage, pred=pr)
                            st_prep.advance()
                            rng_end(_t)
                            K.assign(gc, gc + K.int32(1))
                    work_next(st_wc, work)

            # =====================================================================
            # TMA loader: q/k/g stages only (runs two chunks ahead).            warp 17
            # =====================================================================
            with loader:
                st_stage = K.PipelineState(2, phase=1)
                st_beta = K.PipelineState(2, phase=1)
                lane = K.lane_id()
                with K.If(elected()), K.Then():
                    for m in (q_map, k_map, g_map):
                        K.ptx.prefetch.tensormap(K.address_of(m))
                work = K.local_scalar("int32")
                st_wp = K.PipelineState(4, phase=1)  # producer of p_wid
                st_wc = K.PipelineState(4, phase=0)  # its own consumer cursor

                def claim_work():
                    """One lane takes the next item; the CTA reads it from the ring."""
                    p_wid.empty.wait(st_wp.stage, st_wp.phase)
                    pr = elect_local()
                    with K.If(pr != K.uint32(0)), K.Then():
                        wv = K.local_scalar("uint32")
                        # acq_rel at device scope: the claim is what orders one CTA's
                        # work against every other CTA's, so a relaxed atomic leaves
                        # the counter's read-from edges unordered across CTAs.
                        K.ptx.atom.acq_rel.gpu.global_.add.u32(
                            wv, wctr.ptr_to([K.int64(0)]), K.uint32(1)
                        )
                        K.ptx.st.shared.s32(K.address_of(s_wid[st_wp.stage]), K.Cast("int32", wv))
                    # The claim is a long-latency global atomic; without an explicit
                    # release the shared store of its result can be scheduled after
                    # the arrive, and a consumer then reads the slot's previous item.
                    K.ptx.fence.acq_rel.cta()
                    p_wid.full.arrive(st_wp.stage, pred=pr)
                    st_wp.advance()

                if dyn:
                    claim_work()
                work_first(st_wc, work)
                with K.While(work < num_work):
                    b_idx, h_idx = work_coords(work)
                    if split:
                        lo_t = K.int32(0)
                        seqlen = K.int32(0)  # unused: a split shape is nseq==1
                        npre = K.local_scalar("int32", init=K.Select(b_idx != 0, K.int32(split), K.int32(0)))
                        pre_start = npre
                        NCH = K.local_scalar(
                            "int32", init=K.Select(b_idx != 0, K.int32(ntot_chunks - split), K.int32(split))
                        )
                        n_limit = K.local_scalar(
                            "int32", init=K.Select(b_idx != 0, K.int32(ntot_chunks), K.int32(split))
                        )
                    else:
                        lo_t, seqlen, NCH = seq_bounds(b_idx)
                        npre = None
                        pre_start = None
                        n_limit = None
                    for _so, _cnt in phase_list(npre, NCH):
                        with K.serial(_cnt) as _nl:
                            n = _nl if (not split or _so) else _nl + pre_start
                            nlimit = n_limit if split else NCH
                            tok0 = lo_t + n * CHUNK
                            sv = s_stage[st_stage.stage]
                            _t = rng("L.wait_stage")
                            p_stage.empty.wait(st_stage.stage, st_stage.phase)
                            rng_end(_t)
                            _t = rng("L.issue")
                            with K.If(elected()), K.Then():
                                p_stage.full.arrive(
                                    st_stage.stage, tx_count=(KQ_BYTES // 2) if _so else KQ_BYTES
                                )
                                mb = K.cuda.cvta_generic_to_shared(p_stage.full.ptr_to([st_stage.stage]))
                                for tmap, row0 in (((k_map, 0),) if _so else ((k_map, 0), (q_map, 64))):
                                    for d0 in (0, 64):
                                        K.ptx[TMA_LD](
                                            sv.ptr_to(row0, d0),
                                            K.address_of(tmap),
                                            K.int32(d0),
                                            K.Cast("int32", tok0),
                                            K.Cast("int32", h_idx),
                                            mb,
                                        )
                                # q/k prefetch is always worthwhile. Gate prefetch is
                                # retained only for shapes where the controlled A/B
                                # showed a benefit; H and nseq are compile-time shape
                                # constants, not runtime input values.
                                with K.If(n + 1 < nlimit), K.Then():
                                    _pf = (
                                        (k_map, q_map, g_map)
                                        if (uniform_pack or (nseq == 1 and H == 96))
                                        else (k_map, q_map)
                                    )
                                    for tmap in (tuple(t for t in _pf if t is not q_map) if _so else _pf):
                                        for d0 in (0, 64):
                                            K.ptx[TMA_PREFETCH](
                                                K.address_of(tmap),
                                                K.int32(d0),
                                                K.Cast("int32", tok0 + CHUNK),
                                                K.Cast("int32", h_idx),
                                            )
                            st_stage.advance()
                            rng_end(_t)
                            # beta: sigmoid(bf16 logits) for the 64 tokens of this chunk
                            p_beta.empty.wait(st_beta.stage, st_beta.phase)
                            for i in range(2):
                                t = lane + 32 * i
                                if whole_boxes:
                                    tc = n * CHUNK + t
                                else:
                                    tc = K.min(n * CHUNK + t, seqlen - 1)  # tail padding reads the last token
                                bu = K.local_scalar("uint16")
                                K.ptx.ld.global_.nc.u16(
                                    bu,
                                    beta.ptr_to([(K.Cast("int64", lo_t) + K.Cast("int64", tc)) * K.int64(H) + K.Cast("int64", h_idx)]),
                                )
                                bfv = K.reinterpret("float32", K.Cast("uint32", bu) << K.uint32(16))
                                sg = K.idioms.sigmoid_tanh_approx_f32(bfv)
                                K.ptx.st.shared.f32(K.address_of(s_beta[st_beta.stage, t]), sg)
                            p_beta.full.arrive(st_beta.stage)
                            st_beta.advance()
                    # Claim late: taking the next item only once this one's loads are
                    # issued is what makes the schedule adapt to sequence length. A
                    # claim at the head of the item is grabbed by every CTA at once
                    # and degenerates back to the static round robin.
                    if dyn:
                        claim_work()
                    work_next(st_wc, work)
                # Reset the claim counter for the next launch. This must happen
                # only once every CTA has stopped claiming: a per-CTA compensating
                # subtract lets a CTA that finishes early rewind the counter, and a
                # still-running CTA then re-claims an item another CTA already ran.
                if dyn:
                    K.ptx.fence.acq_rel.gpu()
                    with K.If(elected()), K.Then():
                        dv = K.local_scalar("uint32")
                        K.ptx.atom.acq_rel.gpu.global_.add.u32(
                            dv, wctr.ptr_to([K.int64(1)]), K.uint32(1)
                        )
                        with K.If(dv == K.uint32(num_ctas - 1)), K.Then():
                            # Clear both slots with atomic AND-0 rather than a plain
                            # store: a plain store to a location every other CTA has
                            # been atomically incrementing is an unordered write/write
                            # pair even though it can only run after the last increment.
                            K.ptx["red.relaxed.gpu.global.and.b32"](
                                wctr.ptr_to([K.int64(0)]), K.uint32(0)
                            )
                            K.ptx["red.relaxed.gpu.global.and.b32"](
                                wctr.ptr_to([K.int64(1)]), K.uint32(0)
                            )

            # =====================================================================
            # v loads (one chunk ahead of the O store) + TMA store of O.        warp 18
            # =====================================================================
            with storer:
                st_osm = K.PipelineState(1, phase=0)
                st_v = K.PipelineState(1, phase=1)
                if whole_boxes:
                    store_desc = K.address_of(o_map)
                    with K.If(elected()), K.Then():
                        K.ptx.prefetch.tensormap(K.address_of(v_map))
                else:
                    store_desc = d_o
                    with K.If(elected()), K.Then():
                        copy_desc(d_o, o_map)
                        K.ptx.prefetch.tensormap(K.address_of(v_map))
                    K.cuda.warp_sync()
                    K.ptx.fence.acq_rel.cta()
                def issue_v(tok0, h_idx):
                    _t = rng("L.wait_v")
                    p_v.empty.wait(st_v.stage, st_v.phase)  # single v stage: free once G3 of the previous chunk completed
                    rng_end(_t)
                    with K.If(elected()), K.Then():
                        p_v.full.arrive(st_v.stage, tx_count=V_BYTES)
                        mbv = K.cuda.cvta_generic_to_shared(p_v.full.ptr_to([st_v.stage]))
                        for d0 in (0, 64):
                            K.ptx[TMA_LD](
                                s_v.ptr_to(0, d0),
                                K.address_of(v_map),
                                K.int32(d0),
                                K.Cast("int32", tok0),
                                K.Cast("int32", h_idx),
                                mbv,
                            )
                        for d0 in (0, 64):  # the chunk after this one into L2
                            K.ptx[TMA_PREFETCH](
                                K.address_of(v_map),
                                K.int32(d0),
                                K.Cast("int32", tok0 + CHUNK),
                                K.Cast("int32", h_idx),
                            )
                    st_v.advance()

                work = K.local_scalar("int32")
                st_wc = K.PipelineState(4, phase=0)
                work_first(st_wc, work)
                with K.While(work < num_work):
                    b_idx, h_idx = work_coords(work)
                    if split:
                        lo_t = K.int32(0)
                        seqlen = K.int32(0)  # unused: a split shape is nseq==1
                        npre = K.local_scalar("int32", init=K.Select(b_idx != 0, K.int32(split), K.int32(0)))
                        pre_start = npre
                        NCH = K.local_scalar(
                            "int32", init=K.Select(b_idx != 0, K.int32(ntot_chunks - split), K.int32(split))
                        )
                        n_limit = K.local_scalar(
                            "int32", init=K.Select(b_idx != 0, K.int32(ntot_chunks), K.int32(split))
                        )
                    else:
                        lo_t, seqlen, NCH = seq_bounds(b_idx)
                        npre = None
                        pre_start = None
                        n_limit = None
                    # bound the O map at this sequence's end before its first store
                    if not whole_boxes:
                        with K.If(elected()), K.Then():
                            K.ptx[BULK_WAIT](0)
                            desc_set_rows(d_o, lo_t + seqlen)
                        K.cuda.warp_sync()
                        K.ptx[TMAP_REL]()
                        with K.If(elected()), K.Then():
                            K.ptx[TMAP_ACQ](d_o)
                    issue_v(lo_t, h_idx)  # v(0)
                    for _so, _cnt in phase_list(npre, NCH):
                        with K.serial(_cnt) as _nl:
                            n = _nl if (not split or _so) else _nl + pre_start
                            nlimit = n_limit if split else NCH
                            tok0 = lo_t + n * CHUNK
                            with K.If(n < nlimit - 1), K.Then():
                                issue_v(tok0 + CHUNK, h_idx)  # v(n+1) ahead of the O(n) store
                            # O(n) store
                            if not (_so and elide_even_prefix_sidepipes):
                                _t = rng("S.wait")
                                p_osm.full.wait(st_osm.stage, st_osm.phase)
                                rng_end(_t)
                                _t = rng("S.store")
                                with K.If(elected()), K.Then():
                                    if not _so:
                                        K.ptx[FENCE_ASYNC]()
                                        for d0 in (0, 64):
                                            K.ptx[TMA_ST](
                                                store_desc,
                                                K.int32(d0),
                                                K.Cast("int32", tok0),
                                                K.Cast("int32", h_idx),
                                                s_o.ptr_to(0, d0),
                                            )
                                        K.ptx[BULK_COMMIT]()
                                        K.ptx[BULK_WAIT_READ](0)
                                    p_osm.empty.arrive(st_osm.stage)
                                st_osm.advance()
                                rng_end(_t)
                    work_next(st_wc, work)
                with K.If(elected()), K.Then():
                    K.ptx[BULK_WAIT](0)

    return kda_fwd


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------


class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_map(tensor, T_total, H):
    desc = _AlignedTensorMap()
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode(
        desc.ptr,
        "bfloat16",
        3,
        ctypes.c_void_p(int(tensor.data_ptr())),
        D_HEAD, T_total, H,  # dims (innermost first)
        2 * D_HEAD * H, 2 * D_HEAD,  # strides in bytes (dims 1, 2)
        64, 64, 1,  # box
        1, 1, 1,  # element strides
        0,  # interleave none
        3,  # swizzle 128B
        2,  # L2 promotion 128B
        0,  # oob fill none
    )  # fmt: skip
    return desc


# Fraction of a single sequence given to the first (prefix-free) piece.  With a
# state-only recompute costing r of an emitting chunk, the balanced cut is
# ntot/(2 - r); r measured 0.825 on B200, so the balanced value at ntot=128 is
# 128/(2 - 0.825) = 109.  The measured curve is 4-periodic in the cut because
# ptxas emits materially different code per trip-count immediate (14905 SASS
# instructions at 102/106/110 versus 15769 at 107 and 14457 at 112), so the cut
# was tuned rather than computed.  86/100 selects cut 110 at ntot=128, and the
# late retune of the cheaper M64 prefix closed the 106-110 bracket around it:
# cuts 109, 108, 107 and 106 each regressed the H64-fixed A/B (by 5.014%,
# 0.444%, 3.211% and 0.622%) against retained cut 110.
_SPLIT_NUM1 = 86
_SPLIT_DEN = 100

_MIXED_SEQ_LENS = (1300, 547, 2048, 963, 271, 3063)
_UNIFORM_SEQ_LENS = (1024,) * 8


@dataclass(frozen=True, slots=True)
class KDAForwardConfig:
    label: str = "b1_t8192_h96_fixed_k128_v128_bf16"
    batch_size: int = 1
    seq_lens: tuple[int, ...] = (8192,)
    num_qk_heads: int = 96
    num_v_heads: int = 96
    key_head_dim: int = D_HEAD
    value_head_dim: int = D_HEAD
    seed: int = 2766560907
    scale: float = 1.0 / math.sqrt(D_HEAD)
    lower_bound: float = -5.0

    def validate(self) -> None:
        if self.batch_size != 1:
            raise ValueError(f"packed KDA forward expects batch_size=1, got {self.batch_size}")
        if self.num_qk_heads != self.num_v_heads:
            raise ValueError(
                f"KDA forward expects num_qk_heads == num_v_heads, got "
                f"{self.num_qk_heads} and {self.num_v_heads}"
            )
        if self.num_qk_heads not in (64, 96):
            raise ValueError(f"unsupported agent-evolved KDA head count {self.num_qk_heads}")
        if (self.key_head_dim, self.value_head_dim) != (D_HEAD, D_HEAD):
            raise ValueError(
                f"unsupported agent-evolved KDA head dims "
                f"{(self.key_head_dim, self.value_head_dim)}"
            )
        if not self.seq_lens or any(int(t) <= 0 for t in self.seq_lens):
            raise ValueError(f"seq_lens must be non-empty and positive, got {self.seq_lens}")
        if self.total_tokens != 8192:
            raise ValueError(
                f"unsupported agent-evolved KDA token count {self.total_tokens} (expected 8192)"
            )
        if not math.isclose(self.scale, 1.0 / math.sqrt(D_HEAD), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"scale must be 1/sqrt({D_HEAD}), got {self.scale}")
        if self.lower_bound != -5.0:
            raise ValueError(f"lower_bound must be -5.0, got {self.lower_bound}")

    @property
    def num_heads(self) -> int:
        return self.num_qk_heads

    @property
    def num_seqs(self) -> int:
        return len(self.seq_lens)

    @property
    def total_tokens(self) -> int:
        return sum(int(t) for t in self.seq_lens)

    @property
    def packed(self) -> bool:
        """A single sequence is the ``cu_seqlens=None`` fixed case, not a pack."""
        return self.num_seqs > 1

    @property
    def use_initial_state(self) -> bool:
        return True

    @property
    def store_final_state(self) -> bool:
        return False


CONFIGS = [
    {
        "label": "b1_t8192_h96_fixed_k128_v128_bf16",
        "seq_lens": (8192,),
        "num_qk_heads": 96,
        "num_v_heads": 96,
        "seed": 2766560907,
    },
    {
        "label": "b1_t8192_h96_mixed_varlen_k128_v128_bf16",
        "seq_lens": _MIXED_SEQ_LENS,
        "num_qk_heads": 96,
        "num_v_heads": 96,
        "seed": 3560182230,
    },
    {
        "label": "b1_t8192_h96_uniform_varlen_k128_v128_bf16",
        "seq_lens": _UNIFORM_SEQ_LENS,
        "num_qk_heads": 96,
        "num_v_heads": 96,
        "seed": 651573149,
    },
    {
        "label": "b1_t8192_h64_fixed_k128_v128_bf16",
        "seq_lens": (8192,),
        "num_qk_heads": 64,
        "num_v_heads": 64,
        "seed": 958207609,
    },
    {
        "label": "b1_t8192_h64_mixed_varlen_k128_v128_bf16",
        "seq_lens": _MIXED_SEQ_LENS,
        "num_qk_heads": 64,
        "num_v_heads": 64,
        "seed": 2069778875,
    },
    {
        "label": "b1_t8192_h64_uniform_varlen_k128_v128_bf16",
        "seq_lens": _UNIFORM_SEQ_LENS,
        "num_qk_heads": 64,
        "num_v_heads": 64,
        "seed": 3106687487,
    },
]

KERNEL_META = {
    "name": "agent_evolved_kda_forward_b1_t8192_varlen",
    "category": "agent_evolved",
    "compute_capability": 10,
    "provenance": {
        "generator": "hmz",
        "run": "kda_forward-20260904-233859",
        "selected_version": "v238",
    },
}


def _cfg(**kwargs: Any) -> KDAForwardConfig:
    names = {field.name for field in fields(KDAForwardConfig)}
    values = {name: value for name, value in kwargs.items() if name in names}
    if "seq_len" in kwargs and "seq_lens" not in values:
        # agent_evolved_kda_forward_b1_t8192_h96 stores a scalar `seq_len` and
        # derives `seq_lens` from it as `(seq_len,) * batch_size`. Accept that
        # spelling so its call sites move to this kernel unchanged; a packing
        # with unequal lengths can only be given as `seq_lens`.
        values["seq_lens"] = (int(kwargs["seq_len"]),) * int(values.get("batch_size", 1))
    if "seq_lens" in values:
        values["seq_lens"] = tuple(int(t) for t in values["seq_lens"])
    cfg = KDAForwardConfig(**values)
    cfg.validate()
    return cfg


@dataclass(frozen=True, slots=True)
class _Dispatch:
    """The compile-time specialization and grid one config resolves to."""

    num_ctas: int
    num_work: int
    dyn: bool
    split: int
    ntot_chunks: int
    unif: bool
    seg: int


@lru_cache(maxsize=None)
def _dispatch(cfg: KDAForwardConfig) -> _Dispatch:
    """Resolve one config to a kernel specialization.

    Traced kernel and launch arguments must agree on ``num_work``, so the
    decision is taken once per config and cached; ``hardware_num_sms`` reports
    the compile profile without touching CUDA.
    """
    from tirx_kernels.runner import hardware_num_sms

    num_sms = hardware_num_sms()
    heads = cfg.num_heads
    lens = tuple(int(t) for t in cfg.seq_lens)
    num_seqs = len(lens)

    # Uniformity of the packing is a property of the sequence lengths, not of
    # the sequence count: a non-uniform pack of any N falls through to the
    # general varlen kernel.
    unif, seg = False, 0
    if num_seqs > 1 and len(set(lens)) == 1 and lens[0] % CHUNK == 0:
        unif, seg = True, lens[0]

    num_work = num_seqs * heads
    num_ctas = min(num_work, num_sms)
    if unif:
        # Equal-length work items make the launch a whole number of waves over
        # the work list, and that count does not change once the grid reaches
        # ceil(num_work / waves).  Every CTA above it only shortens the final,
        # partly idle wave while adding memory-system contention to all the
        # full ones.  Measured on B200 at a pinned 1965 MHz, so this is
        # contention and not clock throttling: the per-chunk cost of a fully
        # occupied launch rises from 2.98 us at 112 CTAs to 3.30 us at 148, and
        # wave-aligning both uniform grids to 128 is worth 3.3% on each.
        # Unequal items have no wave structure -- there every extra CTA really
        # does shorten the greedy makespan -- so they keep the full grid.
        waves = -(-num_work // num_ctas)
        num_ctas = -(-num_work // waves)

    split = 0
    ntot = 0
    if num_seqs == 1 and lens[0] % CHUNK == 0 and 2 * heads <= num_sms:
        # A single sequence leaves num_sms - H SMs idle. Cut it in two and let
        # the second CTA re-derive the state over the first piece's chunks
        # without emitting anything.  Only worth it when every piece gets its
        # own CTA: at H=96 the 44 doubled CTAs carry two pieces and the shape
        # measures 40% slower, so that case keeps the single-piece dispatch.
        ntot = lens[0] // CHUNK
        num_work = 2 * heads
        num_ctas = num_work
        split = max(1, min(ntot - 1, ntot * _SPLIT_NUM1 // _SPLIT_DEN))

    # One packed sequence cannot be load-imbalanced, so it skips the work-claim ring.
    return _Dispatch(
        num_ctas=num_ctas,
        num_work=num_work,
        dyn=num_seqs > 1,
        split=split,
        ntot_chunks=ntot,
        unif=unif,
        seg=seg,
    )


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    plan = _dispatch(cfg)
    return make_kernel(
        cfg.num_heads,
        plan.num_ctas,
        cfg.num_seqs,
        plan.dyn,
        split=plan.split,
        ntot_chunks=plan.ntot_chunks,
        unif=plan.unif,
        seg=plan.seg,
    ).func


def _gate_parameters(
    cfg: KDAForwardConfig, *, device: torch.device, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    a_log = torch.log(
        torch.empty(cfg.num_heads, dtype=torch.float32, device=device).uniform_(
            1.0, 16.0, generator=generator
        )
    )
    dt = torch.exp(
        torch.rand(
            cfg.num_heads * cfg.key_head_dim,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    ).clamp_(min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    return a_log, dt_bias


def _randn_bf16(
    shape: tuple[int, ...], *, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    return (
        torch.randn(shape, dtype=torch.float32, device=device, generator=generator) * 0.5
    ).to(torch.bfloat16)


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = torch.device(kwargs.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved KDA forward")

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    a_log, dt_bias = _gate_parameters(cfg, device=device, generator=generator)
    vector_shape = (
        cfg.batch_size,
        cfg.total_tokens,
        cfg.num_heads,
        cfg.key_head_dim,
    )
    q = _randn_bf16(vector_shape, device=device, generator=generator)
    k = _randn_bf16(vector_shape, device=device, generator=generator)
    v = _randn_bf16(vector_shape, device=device, generator=generator)
    g = _randn_bf16(vector_shape, device=device, generator=generator)
    beta = _randn_bf16(
        (cfg.batch_size, cfg.total_tokens, cfg.num_heads),
        device=device,
        generator=generator,
    )
    initial_state = (
        torch.randn(
            (cfg.num_seqs, cfg.num_heads, cfg.value_head_dim, cfg.key_head_dim),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.25
    )
    offsets = [0]
    for seq_len in cfg.seq_lens:
        offsets.append(offsets[-1] + int(seq_len))
    # The kernel always reads bounds from a cu_seqlens buffer; the fixed case
    # supplies the shape-derived [0, T] pair that the workload leaves implicit.
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device=device)

    out = torch.empty_like(v)
    tensors = {"q": q, "k": k, "v": v, "g": g, "out": out}
    tensor_maps = {
        name: _encode_map(tensor, cfg.total_tokens, cfg.num_heads)
        for name, tensor in tensors.items()
    }
    plan = _dispatch(cfg)
    return {
        "config": cfg,
        **tensors,
        "beta": beta,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "scale": cfg.scale,
        "tensor_maps": tensor_maps,
        # Per-CTA output tensormap scratch, and the work-claim counter.  The
        # kernel restores the counter to zero before it exits, so one buffer
        # serves every invocation.
        "desc_ws": torch.empty(plan.num_ctas * 128, dtype=torch.int8, device=device),
        "work_counter": torch.zeros(2, dtype=torch.uint32, device=device),
    }


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    cfg: KDAForwardConfig = case["config"]
    maps = case["tensor_maps"]
    return (
        case["q"].view(-1),
        case["k"].view(-1),
        case["v"].view(-1),
        case["g"].view(-1),
        case["beta"].view(-1),
        case["A_log"].contiguous().float(),
        case["dt_bias"].contiguous().float(),
        case["out"].view(-1),
        case["initial_state"].contiguous().view(-1),
        case["cu_seqlens"],
        case["desc_ws"],
        case["work_counter"],
        maps["q"].ptr,
        maps["k"].ptr,
        maps["v"].ptr,
        maps["g"].ptr,
        maps["out"].ptr,
        case["scale"],
        _dispatch(cfg).num_work,
    )


def _assert_sm100() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved KDA forward")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        raise SkipTest(f"agent-evolved KDA forward requires compute capability 10.x, got {capability}")


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


@lru_cache(maxsize=1)
def _load_fla_chunk_kda():
    from fla.ops.kda import chunk_kda

    return chunk_kda


def _run_fla_reference(case: dict[str, Any]) -> torch.Tensor:
    cfg: KDAForwardConfig = case["config"]
    with _native_fla_backend(), torch.inference_mode():
        output, _ = _load_fla_chunk_kda()(
            q=case["q"],
            k=case["k"],
            v=case["v"],
            g=case["g"],
            beta=case["beta"],
            scale=case["scale"],
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=True,
            safe_gate=True,
            lower_bound=cfg.lower_bound,
            A_log=case["A_log"],
            dt_bias=case["dt_bias"],
            initial_state=case["initial_state"].clone(),
            cu_seqlens=case["cu_seqlens"] if cfg.packed else None,
        )
    return output


def check_correctness(outputs: dict[str, Any], **kwargs: Any) -> None:
    _cfg(**kwargs)
    first = outputs["first"]
    actual = outputs["actual"]
    reference = outputs["reference"]
    for name, tensor in (("first", first), ("actual", actual), ("reference", reference)):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} output contains non-finite values")
    if not torch.equal(first, actual):
        max_abs = float((first.float() - actual.float()).abs().max())
        raise AssertionError(f"identical launches are not exactly repeatable; max abs diff={max_abs}")
    torch.testing.assert_close(actual, reference, atol=5e-2, rtol=5e-2)
    diff_rms = torch.sqrt(torch.mean((actual.float() - reference.float()).square()))
    reference_rms = torch.sqrt(torch.mean(reference.float().square()))
    rms_ratio = float(diff_rms / (reference_rms + 1e-8))
    if rms_ratio >= 1e-2:
        raise AssertionError(f"normalized RMS error ratio {rms_ratio:.6e} must be below 1e-2")


def run_test(**kwargs: Any) -> None:
    _assert_sm100()
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
    args = _tirx_args(case)

    case["out"].fill_(float("nan"))
    executable(*args)
    torch.cuda.synchronize()
    first = case["out"].clone()

    case["out"].fill_(42.0)
    executable(*args)
    torch.cuda.synchronize()
    actual = case["out"].clone()

    reference = _run_fla_reference(case)
    torch.cuda.synchronize()
    check_correctness(
        {"first": first, "actual": actual, "reference": reference},
        **kwargs,
    )


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
    _assert_sm100()
    config = dict(prepared["config"])
    config.update(kwargs)
    rounds = config.pop("rounds", 5)
    cooldown_s = config.pop("cooldown_s", 1.0)
    case = prepare_data(**config)
    executable = prepared["executable"]
    args = _tirx_args(case)

    executable(*args)
    torch.cuda.synchronize()

    def _flashkda_builder():
        from tirx_kernels.flashinfer.utils._flashkda_bench import prepare_flashkda_raw_reference

        reference_case = dict(case)
        cfg: KDAForwardConfig = case["config"]
        # The workload passes dt_bias flattened; the peer wants fp32 [H, K].
        reference_case["dt_bias"] = case["dt_bias"].view(cfg.num_heads, cfg.key_head_dim)
        return prepare_flashkda_raw_reference(reference_case).launch

    from tirx_kernels.runner import bench

    return bench(
        {"tirx": lambda: executable(*args)},
        references={"flash_kda": _flashkda_builder},
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
