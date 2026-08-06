/* Cross-check c/oq.h's unpacker+dequant against vectors captured from
 * mlx.core.dequantize on real oQ bytes. Throwaway harness, not a repo test. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "/Users/anton.korshikov/GIT/colibri-laguna/c/json.h"
#include "/Users/anton.korshikov/GIT/colibri-laguna/c/oq.h"

static char *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(*n + 1); fread(b, 1, *n, f); b[*n] = 0; fclose(f); return b;
}

int main(void) {
    long n; char *buf = slurp("/tmp/oqchk/vec.json", &n);
    char *arena = NULL; jval *root = json_parse(buf, &arena);
    jval *cases = json_get(root, "cases");
    int fails = 0;
    for (int ci = 0; ci < cases->len; ci++) {
        jval *c = cases->kids[ci];
        const char *nm = json_get(c, "name")->str;
        int bits = (int)json_get(c, "bits")->num, gs = (int)json_get(c, "gs")->num;
        int rows = (int)json_get(c, "rows")->num, K = (int)json_get(c, "K")->num;
        int words = (int)json_get(c, "words")->num;
        jval *jc = json_get(c, "code"), *js = json_get(c, "scale"),
             *jb = json_get(c, "bias"), *jr = json_get(c, "ref");
        
        int ngroups = K / gs;
        uint32_t *code = malloc((size_t)rows * words * 4);
        float *scale = malloc((size_t)rows * ngroups * 4);
        float *bias  = malloc((size_t)rows * ngroups * 4);
        for (int i = 0; i < rows * words; i++) code[i] = (uint32_t)jc->kids[i]->num;
        for (int i = 0; i < rows * ngroups; i++) scale[i] = (float)js->kids[i]->num;
        for (int i = 0; i < rows * ngroups; i++) bias[i] = (float)jb->kids[i]->num;

        float *out = malloc((size_t)K * sizeof(float));
        double maxdiff = 0; int nbad = 0;
        for (int r = 0; r < rows; r++) {
            oq_dequant_row(code, scale, bias, r, K, bits, gs, out);
            for (int i = 0; i < K; i++) {
                double ref = jr->kids[(int64_t)r*K + i]->num;
                double d = fabs(out[i] - ref);
                if (d > maxdiff) maxdiff = d;
                if (d > 1e-6) nbad++;
            }
        }
        printf("%-52s bits=%d gs=%3d K=%6d  maxdiff=%.3e  bad=%d  %s\n",
               nm, bits, gs, K, maxdiff, nbad, nbad ? "FAIL" : "OK");
        if (nbad) fails++;

        /* also exercise the matvec path: y = x @ W^T with x = all ones, so
         * y[n] must equal the sum of row n's dequantized weights */
        float *x = malloc((size_t)K * sizeof(float));
        for (int i = 0; i < K; i++) x[i] = 1.0f;
        float *y = malloc((size_t)rows * sizeof(float));
        matmul_oq(y, x, code, scale, bias, 1, K, rows, bits, gs);
        double mvmax = 0;
        for (int r = 0; r < rows; r++) {
            double want = 0;
            for (int i = 0; i < K; i++) want += jr->kids[(int64_t)r*K + i]->num;
            double d = fabs(y[r] - want) / (fabs(want) + 1e-6);
            if (d > mvmax) mvmax = d;
        }
        printf("%-52s matvec rel-err=%.3e  %s\n", "", mvmax, mvmax < 1e-4 ? "OK" : "FAIL");
        if (!(mvmax < 1e-4)) fails++;
        free(out); free(x); free(y); free(code); free(scale); free(bias);
    }
    printf("\n%s\n", fails ? "FAILURES" : "ALL OK");
    return fails ? 1 : 0;
}
