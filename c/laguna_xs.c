/* Laguna-XS (poolside Laguna-XS-2.1): 40 layers, D=2048, 256 experts, topk 8.
 *
 * The whole forward pass lives in laguna_common.h, shared with laguna_s.c —
 * XS and S are one architecture at two scales and every code path is identical
 * (see docs/ENGINEERING.md). This file exists so each size gets its own binary with
 * its own banner, matching how coli's engine_for() picks a binary per arch.
 * Nothing here is XS-specific except the name: geometry comes from the
 * checkpoint's config.json, never from a compile-time constant.
 */
#define LAGUNA_NAME "Laguna-XS"
#define LAGUNA_REF_DEFAULT "ref_laguna_xs.json"
#include "laguna_common.h"
