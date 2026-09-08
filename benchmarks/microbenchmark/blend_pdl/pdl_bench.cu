// Standalone A/B benchmark: the blend retrieve's per-wave rope->scatter
// kernel chain, launched classically vs. with Programmatic Dependent Launch
// (PDL). Replicates execute_cb_retrieve_plan's grid shapes and memory
// traffic; no LMCache dependency, compiles in seconds.
//
//   nvcc -O3 -arch=sm_90 -o pdl_bench pdl_bench.cu && ./pdl_bench
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(x)                                                              \
  do {                                                                     \
    cudaError_t e = (x);                                                   \
    if (e != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e),   \
              __FILE__, __LINE__);                                         \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

#include <cuda_bf16.h>
using bf16 = __nv_bfloat16;

// ---------------- rope-like kernel ----------------
// grid(num_layers*slot_tokens, n_chunks), block(min(heads*rot/2,512))
// rewrites the rot window of each token row in the staging slot, in place.
struct RopeChunk { bf16* key; long old_st, new_st; };
struct RopePack { RopeChunk chunks[16]; };

__global__ void rope_kernel(RopePack pack, long slots, int rot_dim,
                            long key_stride, int num_kv_heads, int head_size,
                            long head_stride, bool pdl_trigger) {
  const int token_idx = blockIdx.x;
  const RopeChunk& c = pack.chunks[blockIdx.y];
  bf16* row = c.key + (long)token_idx * key_stride;
  const int embed_dim = rot_dim / 2;
  const long delta = c.new_st - c.old_st;  // stand-in for cos/sin math
  for (int i = threadIdx.x; i < num_kv_heads * embed_dim; i += blockDim.x) {
    const int h = i / embed_dim, d = i % embed_dim;
    bf16* p = row + (long)h * head_stride;
    float x = __bfloat162float(p[d]), y = __bfloat162float(p[d + embed_dim]);
    float s = __sinf((float)delta * 1e-4f * d), co = __cosf((float)delta * 1e-4f * d);
    p[d] = __float2bfloat16(x * co - y * s);
    p[d + embed_dim] = __float2bfloat16(y * co + x * s);
  }
#if __CUDA_ARCH__ >= 900
  if (pdl_trigger) cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ---------------- scatter-like kernel ----------------
// grid(max_tok, num_layers, 2*n_chunks), block(min(xwords,128))
// reads staging (rope-dependent), writes paged KV via slot_mapping.
struct XferChunk { const uint4* key_value; const long* slot_mapping; int n_tok; };
struct XferPack { XferChunk chunks[16]; };

__global__ void scatter_kernel(XferPack pack, uint4** paged, int k_or_v_size,
                               int xwords_per_token, int num_tokens,
                               int num_layers, int block_size, bool pdl_sync) {
  const int token_id = blockIdx.x;
  const int layer_id = blockIdx.y;
  const int chunk_id = blockIdx.z / k_or_v_size;
  const int k_or_v = blockIdx.z % k_or_v_size;
  const XferChunk& c = pack.chunks[chunk_id];
  if (token_id >= c.n_tok) return;
  // ---- preamble: rope-INDEPENDENT loads (slot mapping, pointer table) ----
  const long slot_idx = c.slot_mapping[token_id];   // written pre-rope
  uint4* paged_ptr = paged[layer_id];
  if (slot_idx < 0) return;
  const long src_base = ((long)k_or_v * num_layers + layer_id) * num_tokens +
                        token_id;
  const long dst_base = ((long)k_or_v * gridDim.x * num_layers +
                         (long)layer_id * gridDim.x + slot_idx);
#if __CUDA_ARCH__ >= 900
  // ---- only now do we need the rope's output ----
  if (pdl_sync) cudaGridDependencySynchronize();
#endif
  const uint4* src = c.key_value + src_base * xwords_per_token;
  uint4* dst = paged_ptr + (dst_base % ((long)gridDim.x * num_layers)) * xwords_per_token;
  for (int i = threadIdx.x; i < xwords_per_token; i += blockDim.x)
    dst[i] = src[i];
}

// ---------------- driver ----------------
struct Cfg { const char* name; int layers, heads, head_size, rot_dim,
             slot_tokens, chunks_per_wave, waves, groups; };

double run_chain(const Cfg& cfg, int mode, int iters) {
  const int L = cfg.layers, T = cfg.slot_tokens, C = cfg.chunks_per_wave;
  const long elems_per_tok = (long)cfg.heads * cfg.head_size;      // per K or V
  const int xwords = (int)(elems_per_tok * sizeof(bf16) / sizeof(uint4));
  const long slot_elems = 2L * L * T * elems_per_tok;

  // staging slots (2 halves x C), paged KV, slot mappings — per group
  std::vector<bf16*> slot_bufs; std::vector<uint4**> paged_tbls;
  std::vector<long*> slot_maps;
  for (int g = 0; g < cfg.groups; ++g) {
    bf16* s; CK(cudaMalloc(&s, sizeof(bf16) * slot_elems * 2 * C));
    slot_bufs.push_back(s);
    uint4* pool; CK(cudaMalloc(&pool, sizeof(uint4) * (long)xwords * T * L * 2 * 4));
    std::vector<uint4*> h_tbl(L);
    for (int l = 0; l < L; ++l) h_tbl[l] = pool + (long)l * xwords * T * 2;
    uint4** d_tbl; CK(cudaMalloc(&d_tbl, sizeof(uint4*) * L));
    CK(cudaMemcpy(d_tbl, h_tbl.data(), sizeof(uint4*) * L, cudaMemcpyHostToDevice));
    paged_tbls.push_back(d_tbl);
    std::vector<long> h_map(T * C);
    for (int i = 0; i < T * C; ++i) h_map[i] = (i * 7) % T;
    long* d_map; CK(cudaMalloc(&d_map, sizeof(long) * T * C));
    CK(cudaMemcpy(d_map, h_map.data(), sizeof(long) * T * C, cudaMemcpyHostToDevice));
    slot_maps.push_back(d_map);
  }

  cudaStream_t stream; CK(cudaStreamCreate(&stream));
  const long key_stride = 2 * elems_per_tok;  // fused-packed K/V rows
  dim3 rope_block(std::min<long>((long)cfg.heads * cfg.rot_dim / 2, 512));
  dim3 sc_block(std::min(xwords, 128));

  auto launch_wave = [&](int wave, int m) {
    const int half = wave % 2;
    for (int g = 0; g < cfg.groups; ++g) {
      RopePack rp{}; XferPack xp{};
      for (int c = 0; c < C; ++c) {
        bf16* slot = slot_bufs[g] + ((long)half * C + c) * slot_elems;
        rp.chunks[c] = {slot, 1000L + c * T, 5000L + c * T};
        xp.chunks[c] = {reinterpret_cast<const uint4*>(slot), slot_maps[g] + (long)c * T, T};
      }
      dim3 rope_grid((unsigned)(L * T), C);
      dim3 sc_grid((unsigned)T, L, 2 * C);
      // rope: classic launch either way (its predecessor edge is a false dep)
      rope_kernel<<<rope_grid, rope_block, 0, stream>>>(
          rp, T, cfg.rot_dim, key_stride, cfg.heads, cfg.head_size,
          2L * cfg.head_size, m == 1);
      if (m >= 1) {
        cudaLaunchConfig_t lc{};
        lc.gridDim = sc_grid; lc.blockDim = sc_block; lc.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        lc.attrs = attr; lc.numAttrs = 1;
        CK(cudaLaunchKernelEx(&lc, scatter_kernel, xp, paged_tbls[g], 2,
                              xwords, T, L, 256, true));
      } else {
        scatter_kernel<<<sc_grid, sc_block, 0, stream>>>(
            xp, paged_tbls[g], 2, xwords, T, L, 256, false);
      }
    }
  };

  // warmup
  for (int w = 0; w < cfg.waves; ++w) launch_wave(w, mode);
  CK(cudaStreamSynchronize(stream));

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0, stream));
  for (int it = 0; it < iters; ++it)
    for (int w = 0; w < cfg.waves; ++w) launch_wave(w, mode);
  CK(cudaEventRecord(e1, stream));
  CK(cudaEventSynchronize(e1));
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1));

  for (auto p : slot_bufs) cudaFree(p);
  for (auto p : slot_maps) cudaFree(p);
  CK(cudaStreamDestroy(stream));
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  return ms * 1000.0 / iters;  // us per request (all waves)
}

int main() {
  Cfg cfgs[] = {
      // llama-8B-like: 32L, 8 KV heads x128, chunk 256, 64 chunks -> 4 waves
      {"llama8B 16k (1 group, 4 waves)", 32, 8, 128, 128, 256, 16, 4, 1},
      // small request: 1 wave of 4 chunks (launch overhead dominant)
      {"llama8B 4k  (1 group, 1 wave/4ch)", 32, 8, 128, 128, 256, 4, 1, 1},
      // nemotron-like: 4 groups (rope on all here = upper bound), fewer layers
      {"hybrid 3 groups, 2 waves", 8, 2, 256, 64, 512, 8, 2, 3},
      // tiny: 2-chunk single wave, small layers — launch overhead regime
      {"tiny 2 chunks (launch-bound)", 8, 2, 64, 64, 64, 2, 1, 1},
  };
  int iters = 200;
  printf("%-40s %11s %12s %14s\n", "config", "classic(us)", "pdl+trig(us)", "pdl-notrig(us)");
  for (auto& c : cfgs) {
    for (int rep = 0; rep < 3; ++rep) {
      double base = run_chain(c, 0, iters);
      double pdl1 = run_chain(c, 1, iters);
      double pdl2 = run_chain(c, 2, iters);
      printf("%-40s %11.1f %6.1f (%+.1f%%) %6.1f (%+.1f%%)\n", c.name, base,
             pdl1, 100.0 * (pdl1 - base) / base,
             pdl2, 100.0 * (pdl2 - base) / base);
    }
  }
  return 0;
}
