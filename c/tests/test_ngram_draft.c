/* ngram_draft (laguna_common.h, LAGUNA-FORK): speculative decoding's n-gram
 * draft source. Must never propose more than `maximum` tokens, must find the
 * LONGEST matching recent repeat (trigram before bigram), and must degrade to
 * "propose nothing" silently on inputs with no repeat rather than guess.
 *
 * Why this test exists: ngram_draft() feeds generate_stream()'s speculative
 * verify round directly. A wrong length or a wrong token here does not corrupt
 * anything by itself -- the batched step() call still runs the real model and
 * the accept-loop only takes tokens step() actually confirmed -- but a
 * over-long or mis-offset proposal would still waste a full verify round
 * (batch[] sized for spec_max+1) or, worse, read past draft[]/batch[]'s fixed
 * stack arrays in generate_stream. In-memory only, no model, no fixture. */
#define LAGUNA_NAME "Laguna-XS-test"
#define LAGUNA_REF_DEFAULT "unused.json"
#define main coli_laguna_main_unused
#include "../laguna_common.h"
#undef main

static int g_nfails = 0;

static void check(int cond, const char *what) {
    if (!cond) { printf("FAIL: %s\n", what); g_nfails++; }
}

int main(void) {
    int out[24];

    /* 1. trigram match wins over a shorter decoy bigram match earlier in the
     * same sequence. Match semantics: from the point right after the
     * earliest-found matching trigram, propose EVERYTHING through the end of
     * the sequence (capped at `maximum`) -- not just "the next few tokens".
     * seq is 1 2 3 [10 11 12] 13 14 15 [10 11 12]; the trailing (10,11,12) is
     * itself both a 3-gram tail AND the search target, so the scan (which
     * walks candidate start points from `count-gram-1` DOWNWARD) finds the
     * first (10,11,12) at index 3 as the earliest match for the tail's
     * (10,11,12), and proposes index 6 onward: 13,14,15,10,11,12. */
    {
        int seq[] = {1, 2, 3, 10, 11, 12, 13, 14, 15, 10, 11, 12};
        int n = ngram_draft(seq, 12, out, 8);
        check(n == 6, "trigram match proposes everything after the match point");
        check(n >= 1 && out[0] == 13, "first proposed token is 13");
        if (n == 6)
            check(out[1]==14 && out[2]==15 && out[3]==10 && out[4]==11 && out[5]==12,
                  "proposal is the contiguous tail from the match point");
    }

    /* 2. bigram fallback: no trigram repeats, but the last 2 tokens do. */
    {
        int seq[] = {5, 6, 7, 8, 7, 8};
        int n = ngram_draft(seq, 6, out, 8);
        check(n == 2, "bigram fallback proposes the tail run");
        check(n >= 1 && out[0] == 7, "bigram proposal starts with 7");
        if (n == 2) check(out[1] == 8, "bigram proposal is contiguous");
    }

    /* 3. cap: a long available run is still capped at `maximum`. */
    {
        int seq[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3};
        int n = ngram_draft(seq, 12, out, 2);
        check(n == 2, "proposal never exceeds the caller's cap");
    }

    /* 4. no repeat at all: silent, not a crash, not a guess. */
    {
        int seq[] = {100, 200, 300, 400};
        int n = ngram_draft(seq, 4, out, 8);
        check(n == 0, "no repeated n-gram proposes nothing");
    }

    /* 5. degenerate inputs: too short to hold even a bigram + its
     * continuation, NULL/zero args. */
    {
        int seq[] = {1, 2};
        check(ngram_draft(seq, 2, out, 8) == 0, "sequence too short proposes nothing");
        check(ngram_draft(NULL, 0, out, 8) == 0, "NULL sequence proposes nothing");
        int seq2[] = {1, 2, 3, 1, 2, 3};
        check(ngram_draft(seq2, 6, out, 0) == 0, "maximum=0 proposes nothing");
    }

    if (g_nfails) { printf("ngram_draft: %d FAILED\n", g_nfails); return 1; }
    printf("ngram_draft: trigram/bigram fallback, cap, absence ok\n");
    return 0;
}
