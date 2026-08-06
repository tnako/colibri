/* Laguna-S (poolside Laguna-S-2.1): 48 layers, D=3072, 256 experts, topk 10.
 *
 * Same engine as laguna_xs.c — see laguna_common.h for the forward pass and
 * docs/laguna.md for why one implementation covers both sizes. The differences
 * that matter (hidden size, layer count, experts per token, MoE width, sliding
 * layers' 72 attention heads, YaRN factor 128) are all read from config.json.
 */
#define LAGUNA_NAME "Laguna-S"
#define LAGUNA_REF_DEFAULT "ref_laguna_s.json"
#include "laguna_common.h"
