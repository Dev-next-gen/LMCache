# A/B the real execute_cb_retrieve_plan_flat with PDL on/off (LMCACHE_CB_PDL).
# llama-8B-like plan: 32 layers, 256-token slots, 2048-elem fused K/V rows,
# 4 waves x 16 chunks, rope + scatter on every chunk.
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/weishu/LMCache")
from lmcache import device_ops as cuda_ops  # noqa: E402
import lmcache.lmcache_native as lmcache_native  # noqa: E402

DEV = torch.device("cuda")
NL, SPC, HID, ROT, HEADS = 32, 256, 2048, 128, 8
DT = torch.bfloat16
CHUNKS, WAVES = 16, 4
MAX_POS = 65536

def build():
    torch.manual_seed(0)
    cos_sin = torch.randn(MAX_POS, ROT, dtype=DT, device=DEV)
    # staging slots: 2 halves x CHUNKS, each [1, NL, SPC, HID]
    slots = [
        torch.zeros(1, NL, SPC, HID, dtype=DT, device=DEV)
        for _ in range(2 * CHUNKS)
    ]
    host = [
        torch.randn(1, NL, SPC, HID, dtype=DT).pin_memory()
        for _ in range(2 * CHUNKS)
    ]
    # paged KV pool: fused-packed rows (same format the unit test uses),
    # per layer [page_buffer_size, HID]; sized to cover every slot index.
    block_size = 256
    n_pos = WAVES * CHUNKS * SPC
    page_buffer_size = n_pos + block_size
    pool = torch.zeros(NL, page_buffer_size, HID, dtype=DT, device=DEV)
    paged_ptrs = torch.tensor(
        [pool[layer].data_ptr() for layer in range(NL)], dtype=torch.int64, device=DEV
    )
    slot_map = torch.arange(n_pos, dtype=torch.int64, device=DEV)
    spec = cuda_ops.CBGroupSpec(
        paged_kv_ptrs=paged_ptrs.data_ptr(),
        temp_buffer_ptrs=[s.data_ptr() for s in slots],
        num_layers=NL,
        slot_tokens=SPC,
        hidden_elems=HID,
        element_size=DT.itemsize,
        engine_kv_format=lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
        page_buffer_size=page_buffer_size,
        block_size=block_size,
        head_size=128,
        slot_mapping_base=slot_map.data_ptr(),
        slot_mapping_capacity=n_pos,
        cos_sin_cache=cos_sin.data_ptr(),
        rot_dim=ROT,
        rope_num_kv_heads=HEADS,
        rope_head_stride=2 * 128,
        key_scalar_type=15,
        is_neox=True,
        rope_base_offset=0,
    )
    chunk_bytes = NL * SPC * HID * DT.itemsize
    staging, ropes, scatters, offs = [], [], [], []
    for w in range(WAVES):
        half = w % 2
        for c in range(CHUNKS):
            slot = half * CHUNKS + c
            staging.append((slots[slot].data_ptr(), host[slot].data_ptr(), chunk_bytes, 0))
            ropes.append((0, slot, 1000 + c * SPC, 5000 + c * SPC))
            scatters.append((0, slot, (w * CHUNKS + c) * SPC, SPC))
        offs.append((len(staging), len(ropes), len(scatters)))
    return spec, (
        np.asarray(staging, dtype=np.int64),
        np.asarray(ropes, dtype=np.int64),
        np.asarray(scatters, dtype=np.int64),
        np.asarray(offs, dtype=np.int64),
    ), (slots, host, pool, paged_ptrs, slot_map, cos_sin)

def bench(spec, tables, iters=50):
    st, rp, sc, of = tables
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(5):
        cuda_ops.execute_cb_retrieve_plan_flat(DEV, 1 << 26, [spec], st, rp, sc, of)
    torch.cuda.synchronize()
    ev0.record()
    for _ in range(iters):
        cuda_ops.execute_cb_retrieve_plan_flat(DEV, 1 << 26, [spec], st, rp, sc, of)
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) * 1000 / iters  # us

if __name__ == "__main__":
    spec, tables, keep = build()
    results = {}
    for mode, env in (("pdl_on", "1"), ("pdl_off", "0")):
        os.environ["LMCACHE_CB_PDL"] = env
        for rep in range(3):
            us = bench(spec, tables)
            results.setdefault(mode, []).append(us)
    for mode, vals in results.items():
        print(f"{mode}: " + "  ".join(f"{v:8.1f}us" for v in vals))
    on = min(results["pdl_on"]); off = min(results["pdl_off"])
    print(f"best: pdl_on={on:.1f}us  pdl_off={off:.1f}us  delta={100*(off-on)/off:+.2f}%")
