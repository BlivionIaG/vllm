#pragma once

#include <torch/all.h>

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount);

torch::Tensor wvSplitK_int4_g(const at::Tensor& in_a, const at::Tensor& in_b,
                              const at::Tensor& in_scale,
                              const std::optional<at::Tensor>& in_zero_points,
                              const std::optional<at::Tensor>& in_bias,
                              const int64_t CuCount, const int64_t group_size);

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount);

void wvSplitKQ(const at::Tensor& in_a, const at::Tensor& in_b,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount);

torch::Tensor gptq_gemm_rdna2(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              torch::Tensor b_g_idx, bool use_v2_format);

torch::Tensor gptq_gemm_rdna2_prefill(torch::Tensor a, torch::Tensor b_q_weight,
                                      torch::Tensor b_qzeros,
                                      torch::Tensor b_scales,
                                      torch::Tensor b_g_idx,
                                      bool use_v2_format);

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              torch::Tensor b_g_idx, bool use_v2_format);

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales,
                                   torch::Tensor b_g_idx, bool use_v2_format);

void moe_gptq_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type);

// FA-RDNA2: Flash-Attention v2 hand-port for AMD RDNA2 (gfx1030).
// Dispatches a fast path inside RocmAttentionImpl.forward() for
// decode (split-K) and prefill (paged varlen). Gated by
// VLLM_USE_RDNA2_FA=1 and on_gfx10x().
//
// Definitions live at global namespace in fa_rdna2.cu. The device
// kernels (fa_decode_paged_splitk_kernel_*, fa_prefill_paged_varlen_kernel_*)
// live inside vllm::fa_rdna2:: because they share storage with the
// RDNA2 GEMM paths; the host launchers above are at global scope
// because they are called from torch registration which expects
// unqualified symbol names.
torch::Tensor fa_rdna2_decode_paged(torch::Tensor Q,
                                   torch::Tensor key_cache,
                                   torch::Tensor value_cache,
                                   torch::Tensor block_table,
                                   torch::Tensor seq_lens,
                                   int64_t block_size, int64_t kv_splits,
                                   int64_t sliding_window);

torch::Tensor fa_rdna2_prefill_paged_varlen(torch::Tensor Q,
                                           torch::Tensor key_cache,
                                           torch::Tensor value_cache,
                                           torch::Tensor block_table,
                                           torch::Tensor cu_query_lens,
                                           torch::Tensor seq_lens,
                                           int64_t block_size,
                                           int64_t causal,
                                           int64_t sliding_window);

torch::Tensor fa_rdna2_prefill_paged_varlen_short(
    torch::Tensor Q, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_table, torch::Tensor cu_query_lens,
    torch::Tensor seq_lens, int64_t block_size, int64_t causal,
    int64_t sliding_window);

torch::Tensor fa_rdna2_prefill_paged_varlen_splitk(
    torch::Tensor Q, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_table, torch::Tensor cu_query_lens,
    torch::Tensor seq_lens, int64_t block_size, int64_t causal,
    int64_t kv_splits, int64_t sliding_window);

void moe_gptq_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);

void moe_w8a16_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                           torch::Tensor b_q_weight, torch::Tensor b_scales,
                           torch::Tensor b_qzeros, torch::Tensor topk_weights,
                           torch::Tensor sorted_token_ids,
                           torch::Tensor expert_ids,
                           torch::Tensor num_tokens_post_padded,
                           int64_t top_k, int64_t block_size_m,
                           bool mul_topk_weight, int64_t output_topk);

void moe_mxfp4_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                           torch::Tensor b_q_weight, torch::Tensor b_scales,
                           torch::Tensor topk_weights,
                           torch::Tensor sorted_token_ids,
                           torch::Tensor expert_ids,
                           torch::Tensor num_tokens_post_padded,
                           int64_t top_k, int64_t block_size_m,
                           bool mul_topk_weight, int64_t output_topk);

void moe_w8a16_fp8_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                               torch::Tensor b_q_weight,
                               torch::Tensor b_scales,
                               torch::Tensor b_qzeros,
                               torch::Tensor topk_weights,
                               torch::Tensor sorted_token_ids,
                               torch::Tensor expert_ids,
                               torch::Tensor num_tokens_post_padded,
                               int64_t top_k, int64_t block_size_m,
                               bool mul_topk_weight, int64_t output_topk);

// W8A16-FP8 dense linear kernel for AMD RDNA2 (gfx1030).
// Per-tile FP8 (E4M3) -> fp16 dequant via 256-entry LUT, then v_dot2_f32_f16.
void gemm_w8a16_fp8_dense(torch::Tensor a, torch::Tensor b_q_weight,
                          torch::Tensor b_scales, torch::Tensor c,
                          int64_t group_size);

// W8A8-FP8 dense linear kernel for AMD RDNA2 (gfx1030). DeepSeek V4 Flash
// attention and shared experts: FP8 weights + FP8 activations, per-tile
// FP8 (E4M3) -> fp16 dequant via inline bit-trick (no LUT, no constant
// memory), then v_dot2_f32_f16. Per-row OR per-block-K activation scale,
// per-group weight scale. Atomic-add epilogue into a pre-zeroed fp16 output.
void gemm_w8a8_fp8_dense(torch::Tensor a_q, torch::Tensor a_scale,
                          torch::Tensor b_q_weight, torch::Tensor b_scales,
                          torch::Tensor c, int64_t group_size,
                          int64_t a_scale_K_groups);

// W4A4 MXFP4 dense linear kernel for AMD RDNA2 (gfx1030).
// E2M1 nibble -> fp16 via 16-entry constant LUT, UE8M0 scale per 32-elem
// group, then v_dot2_f32_f16. Used for non-MoE MXFP4 layers (attention,
// shared experts). Atomic-add epilogue into a pre-zeroed fp16 output.
void mxfp4_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                      torch::Tensor b_q_weight, torch::Tensor b_scales,
                      int64_t size_m, int64_t size_n, int64_t size_k);

// Paged MQA logits for DeepSeek V4 Lightning Indexer on AMD RDNA2
// (gfx1030). AITER is CDNA-only and crashes on gfx1030; this kernel
// replaces `rocm_aiter_sparse_attn_indexer`'s paged MQA logits stage
// with a fused FP8 dequant + dot-product + ReLU + per-head weighted
// sum kernel. Output is logits [B*next_n, max_model_len] fp32 with -inf
// in padded slots. Top-K selection is done by the standard upstream
// `top_k_per_row_decode` kernel (runs on gfx1030).
torch::Tensor paged_mqa_logits_decode_rdna2(
    torch::Tensor q_fp8, torch::Tensor kv_cache, torch::Tensor weights,
    torch::Tensor context_lens, torch::Tensor block_tables,
    int64_t max_model_len);

// Sparse MLA decode for DeepSeek V4 on AMD RDNA2 (gfx1030).
// Replaces the Triton `_sparse_attn_decode_ragged_kernel` path on
// gfx1030 (the AITER MLA path is CDNA-only and does not run on
// gfx1030). One CTA per query, 32 threads (wave32), 2 heads per thread;
// online softmax with full acc_nope/acc_rope state in registers.
// FP8 (E4M3 OCP) K_nope with E8M0 block scales, bf16 K_rope. Gated
// by VLLM_USE_RDNA2_MLA=1 and on_gfx10x().
void sparse_mla_decode_rdna2(
    torch::Tensor q,                  // [B, H, D] fp16 or bf16
    torch::Tensor main_cache,         // [num_blocks, block_size, 576] uint8
    torch::Tensor main_indices,       // [nnz] int32
    torch::Tensor main_indptr,        // [B+1] int32
    torch::Tensor extra_cache,        // [num_blocks, block_size, 576] uint8 (may be empty)
    torch::Tensor extra_indices,      // [nnz_extra] int32 (may be empty)
    torch::Tensor extra_indptr,       // [B+1] int32 (zeroed when no extra)
    int64_t main_block_size,
    int64_t main_num_rows,
    int64_t extra_block_size,
    int64_t extra_num_rows,
    double scale,
    torch::Tensor attn_sink,          // [H] fp32 or empty
    torch::Tensor out);               // [B, H, D] bf16

// Sparse MLA prefill for DeepSeek V4 on AMD RDNA2 (gfx1030).
// Replaces the Triton `_sparse_attn_prefill_ragged_kernel` path on
// gfx1030. Same online-softmax structure as sparse_mla_decode_rdna2,
// but the kv rows are plain fp16/bf16 (no fp8 slots, no E8M0 scales —
// the fp8_ds_mla cache encoding only applies post-encoder). One CTA per
// (query, head-group), 32 threads (wave32). Gated by
// VLLM_USE_RDNA2_MLA=1 and on_gfx10x().
void sparse_mla_prefill_rdna2(
    torch::Tensor q,                  // [T, H, D] fp16 or bf16
    torch::Tensor kv,                 // [skv, D] fp16/bf16 (contiguous rows)
    torch::Tensor indices,            // [nnz] int32
    torch::Tensor indptr,             // [T + 1] int32
    int64_t num_kv,
    double scale,
    torch::Tensor attn_sink,          // [H] fp32 or empty
    torch::Tensor out);               // [T, H, D] same dtype as q

// INT8 per-(token, head) KV-cache writer for AMD RDNA2 (gfx1030).
// Symmetric signed int8 quantize + write to the interleaved cache
// layout used by RDNA_ATTN backend: [2, num_blocks, H_kv, D+4, block_size]
// int8, with the last 4 int8 bytes per (block, head, slot) being the
// raw fp32 K/V scale. Used by fa_rdna2_decode_paged_int8 to populate
// the per-(token, head) scale tensor the kernel reads inside its
// cooperative load. Per the kv-int8.md wiki contract — fused i8 quant
// + scale computation in a single CTA per (token, head).
void reshape_and_cache_int8_rdna2(
    torch::Tensor key,         // [num_tokens, H_kv, D] fp16
    torch::Tensor value,       // [num_tokens, H_kv, D] fp16
    torch::Tensor kv_cache,    // [2, num_blocks, H_kv, D + 4, block_size] int8
    torch::Tensor slot_mapping // [num_tokens] int32 (-1 = skip)
);

// GatedDeltaNet (GDN) packed single-token decode for AMD RDNA2 (gfx1030).
// Hand port of fused_recurrent_gated_delta_rule_packed_decode_kernel
// (is_kda=False, scalar per-head sigmoid gating, qk-l2norm in kernel).
// Workgroup = one (token, value-head, V-tile); 256 threads hold the
// [32, 128] fp32 state tile in registers; K-reductions are warp-local
// __shfl_xor. head_k_dim must be 128; fp16 in/out, fp32 in-place state.
void gdn_decode_rdna2(
    torch::Tensor mixed_qkv,          // [B, 2*H*K + HV*V] fp16
    torch::Tensor a,                  // [B, HV] fp16
    torch::Tensor b,                  // [B, HV] fp16
    torch::Tensor A_log,              // [HV] fp32
    torch::Tensor dt_bias,            // [HV] fp32
    torch::Tensor out,                // [B, 1, HV, V] fp16
    torch::Tensor initial_state,      // [blocks, HV, V, K] fp32, in-place
    torch::Tensor ssm_state_indices,  // [B] int32
    double scale,
    bool use_qk_l2norm);
