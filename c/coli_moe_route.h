/* Shared MoE routing primitives: sigmoid + additive-bias top-k selection and
 * the post-selection renormalize/scale step.
 *
 * These two pieces are IDENTICAL between GLM-5.2 (colibri.c) and Laguna
 * (laguna_common.h): sigmoid the router logits, add e_score_correction_bias,
 * take top-k by the biased score, gather the UNBIASED scores at the selected
 * indices, renormalize them to sum 1, multiply by the routed scaling factor.
 * Both engines used to hand-roll it. The logic lives here so a fix or a
 * vectorization reaches every caller, and so an upstream sync has one small
 * file to diff instead of an inlined loop that has drifted in formatting.
 *
 * Deliberately NOT here: rope math (Laguna's apply step is half-split, every
 * other engine's is interleaved, so there is nothing shared to extract yet)
 * and the per-layer-type head-count macros (three one-liners, cheaper
 * duplicated than indirected). See docs/laguna.md's shared-migration table.
 */
#ifndef COLI_MOE_ROUTE_H
#define COLI_MOE_ROUTE_H

/* Top-k selection over `choice` (bias-augmented scores), weights gathered from
 * `weight` (the unbiased scores) at the selected indices.
 *
 * `fallback(best, kk, E, layer)` resolves a slot whose scan found no candidate,
 * which happens when the logits are non-finite: without it `best` stays -1 and
 * the caller indexes weight[-1]. Passed in rather than hardcoded so each engine
 * keeps its own diagnostic (colibri.c's router_best_or_fallback,
 * route_trace.h's rt_router_pick).
 *
 * O(K*E) with an O(K) dedup scan per slot, which is what both engines already
 * did: E is 256 and K is 8..10, and a heap loses to the flat scan at that size.
 */
static inline void coli_moe_pick_topk(const float *choice, const float *weight,
                                      int E, int K, int *idx, float *w,
                                      int (*fallback)(int, int, int, int), int layer) {
    for (int kk = 0; kk < K; kk++) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            int taken = 0;
            for (int j = 0; j < kk; j++) if (idx[j] == e) { taken = 1; break; }
            if (!taken && choice[e] > bv) { bv = choice[e]; best = e; }
        }
        best = fallback(best, kk, E, layer);
        idx[kk] = best; w[kk] = weight[best];
    }
}

/* Renormalize the selected weights to sum 1 (when the config asks for it), then
 * apply the routed scaling factor. Ke is the EFFECTIVE count: a top-p trim or a
 * causal ablation may have shortened the selection, and the renormalization has
 * to run over what is actually going to be computed. */
static inline void coli_moe_norm_scale(float *w, int Ke, int norm_topk, float routed_scale) {
    if (norm_topk) {
        float sm = 0;
        for (int kk = 0; kk < Ke; kk++) sm += w[kk];
        sm += 1e-20f;
        for (int kk = 0; kk < Ke; kk++) w[kk] /= sm;
    }
    for (int kk = 0; kk < Ke; kk++) w[kk] *= routed_scale;
}

#endif /* COLI_MOE_ROUTE_H */
