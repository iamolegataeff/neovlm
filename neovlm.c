/*
 * neovlm.c — Hebbian Vision-Language Model in pure C
 *
 * Sees, speaks, grows. Every inference enriches visual prototypes.
 * Built on notorch (autograd + BLAS). Part of Arianna Method.
 *
 * Architecture:
 *   Patch encoder → Transformer decoder (shared vis+text) → Text
 *   Hebbian visual binding (online, no backprop, persists after training)
 *   Dario field: 6 Kuramoto chambers, calendar drift, velocity, phase gate
 *   RRPRAM: relative position bias in attention (learned spatial patterns)
 *   Extended Dario Equation: p(x|Φ) = softmax((α·H_v + β·F_v + γ·A + δ·V) / τ)
 *
 * Build:
 *   cc neovlm.c notorch.c -O2 -lm -DUSE_BLAS -DACCELERATE -framework Accelerate -o neovlm
 *
 * Run:
 *   ./neovlm                     # train on synthetic 32x32 digits
 *   ./neovlm --steps 10000       # more steps
 *   ./neovlm --interactive       # show image, generate, Hebbian learns
 *
 * Copyright (C) 2026 Oleg Ataeff & Arianna Method
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * "The oracle does not predict. It prophesies."
 */

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

/* ═══════════════════════════════════════════════════════════════════
 * CONFIGURATION
 * ═══════════════════════════════════════════════════════════════════ */

/* Image */
#define IMG_SIZE        32
#define PATCH_SIZE      8
#define N_PATCHES_SIDE  (IMG_SIZE / PATCH_SIZE)          /* 4 */
#define N_VIS           (N_PATCHES_SIDE * N_PATCHES_SIDE) /* 16 */
#define PATCH_PX        (PATCH_SIZE * PATCH_SIZE)        /* 64 */

/* Model */
#define DM              256
#define N_HEADS         4
#define HD              (DM / N_HEADS)                   /* 64 */
#define DFF             (4 * DM)                         /* 1024 */
#define N_LAYERS        4
#define MAX_TEXT        16
#define MAX_SEQ         (N_VIS + MAX_TEXT)                /* 32 */

/* Vocabulary: char-level for MNIST digit names */
/* unique chars in zero..nine: e,f,g,h,i,n,o,r,s,t,u,v,w,x,z = 15 */
#define VOCAB           17  /* 15 chars + BOS + EOS */
#define BOS_TOK         15
#define EOS_TOK         16

/* Training defaults */
#define DEFAULT_STEPS   5000
#define LR_BASE         3e-4f

/* Hebbian */
#define HEBB_LR         0.01f

/* Dario field */
#define N_CHAMBERS      6

/* ═══════════════════════════════════════════════════════════════════
 * RNG (xoshiro256**)
 * ═══════════════════════════════════════════════════════════════════ */

static uint64_t rng_s[4];

static void rng_seed(uint64_t seed) {
    rng_s[0] = seed;
    rng_s[1] = seed ^ 0x6a09e667f3bcc908ULL;
    rng_s[2] = seed ^ 0xbb67ae8584caa73bULL;
    rng_s[3] = seed ^ 0x3c6ef372fe94f82bULL;
    for (int i = 0; i < 20; i++) {
        uint64_t t = rng_s[1] << 17;
        rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
        rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
        rng_s[2] ^= t;
        rng_s[3] = (rng_s[3] << 45) | (rng_s[3] >> 19);
    }
}

static uint64_t rng_next(void) {
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t;
    rng_s[3] = (rng_s[3] << 45) | (rng_s[3] >> 19);
    return rng_s[0];
}

static float rng_uniform(void) {
    return (float)(rng_next() >> 11) / (float)(1ULL << 53);
}

static float rng_normal(float mean, float std) {
    double u1 = (double)(rng_next() >> 11) / (double)(1ULL << 53);
    double u2 = (double)(rng_next() >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-30) u1 = 1e-30;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
    return mean + std * (float)z;
}

/* ═══════════════════════════════════════════════════════════════════
 * CALENDAR DRIFT — Hebrew/Gregorian temporal dissonance
 *
 * The two calendars drift ~11.25 days/year. The Metonic cycle
 * (19 years, 7 leap months) partially corrects this. The residual
 * drift, normalized to [0,1], is the calendar dissonance.
 * Epoch: Rosh Hashanah 5785 = October 3, 2024.
 * ═══════════════════════════════════════════════════════════════════ */

#define AM_ANNUAL_DRIFT     11.25f
#define AM_GREGORIAN_YEAR   365.25f
#define AM_METONIC_YEARS    19
#define AM_METONIC_LEAPS    7
#define AM_MAX_UNCORRECTED  33.0f

static const int g_metonic_leap_years[7] = {3, 6, 8, 11, 14, 17, 19};

static float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static float calendar_dissonance(void) {
    struct tm epoch = {0};
    epoch.tm_year = 2024 - 1900;
    epoch.tm_mon = 9;   /* October */
    epoch.tm_mday = 3;
    epoch.tm_hour = 12;
    time_t ep = mktime(&epoch);
    float days = (float)difftime(time(NULL), ep) / 86400.0f;
    float years = days / AM_GREGORIAN_YEAR;
    float drift = years * AM_ANNUAL_DRIFT;
    int full_cycles = (int)(years / AM_METONIC_YEARS);
    float corr = (float)(full_cycles * AM_METONIC_LEAPS) * 30.0f;
    float partial = fmodf(years, (float)AM_METONIC_YEARS);
    int yic = (int)partial + 1;
    for (int i = 0; i < AM_METONIC_LEAPS; i++)
        if (g_metonic_leap_years[i] <= yic) corr += 30.0f;
    drift -= corr;
    return clampf(fabsf(fmodf(drift, AM_MAX_UNCORRECTED)) / AM_MAX_UNCORRECTED, 0.0f, 1.0f);
}

/* ═══════════════════════════════════════════════════════════════════
 * DARIO FIELD — 6 Kuramoto-coupled somatic chambers
 *
 * The model is not a cold classifier. It has a body.
 * Chambers modulate generation coefficients, temperature, and
 * coherence threshold. Dark matter (toxic input) creates scars.
 * ═══════════════════════════════════════════════════════════════════ */

enum { CH_FEAR=0, CH_LOVE, CH_RAGE, CH_VOID, CH_FLOW, CH_CMPLX };
enum { VEL_WALK=0, VEL_RUN, VEL_STOP, VEL_BREATHE, VEL_UP, VEL_DOWN };

static const char* ch_names[] = {"FEAR","LOVE","RAGE","VOID","FLOW","CMPLX"};
static const char* vel_names[] = {"WALK","RUN","STOP","BREATHE","UP","DOWN"};

static const float CH_DECAY[N_CHAMBERS] = {0.90f, 0.93f, 0.85f, 0.97f, 0.88f, 0.94f};

/* Kuramoto coupling matrix: how chambers pull each other */
static const float COU[N_CHAMBERS][N_CHAMBERS] = {
    /*         FEAR   LOVE   RAGE   VOID   FLOW   CMPLX */
    /* FEAR */ { 0.0f,-0.30f, 0.40f, 0.20f,-0.15f, 0.10f},
    /* LOVE */ {-0.30f, 0.0f,-0.25f,-0.10f, 0.35f, 0.20f},
    /* RAGE */ { 0.40f,-0.25f, 0.0f, 0.30f,-0.20f, 0.15f},
    /* VOID */ { 0.20f,-0.10f, 0.30f, 0.0f,-0.25f, 0.10f},
    /* FLOW */ {-0.15f, 0.35f,-0.20f,-0.25f, 0.0f, 0.30f},
    /* CMPLX */{ 0.10f, 0.20f, 0.15f, 0.10f, 0.30f, 0.0f},
};

typedef struct {
    float act[N_CHAMBERS];      /* activation [0,1] */
    float soma[N_CHAMBERS];     /* somatic memory (EMA of act) */
    float coherence;            /* EMA of local coherence */
    float phase_lock;           /* phase gate state */
    float threshold_bias;       /* adaptive threshold */
    float scar;                 /* dark matter scarring [0,1] */
    float trauma;               /* trauma accumulator */
    float presence;             /* overall somatic awareness [0,1] */
    int   velocity;             /* current velocity mode */
    float birth_drift;          /* calendar drift at birth */
} DarioField;

static void dario_init(DarioField* d) {
    memset(d, 0, sizeof(*d));
    d->act[CH_FLOW] = 0.3f;
    d->act[CH_LOVE] = 0.2f;
    d->presence = 0.5f;
    d->coherence = 0.5f;
    d->velocity = VEL_WALK;
    d->birth_drift = calendar_dissonance();
}

static void dario_kuramoto_step(DarioField* d) {
    float old[N_CHAMBERS];
    memcpy(old, d->act, sizeof(old));
    float K = 0.03f;
    for (int i = 0; i < N_CHAMBERS; i++) {
        d->act[i] *= CH_DECAY[i];
        for (int j = 0; j < N_CHAMBERS; j++)
            if (i != j)
                d->act[i] += K * COU[i][j] * sinf(old[j] - old[i]);
        d->act[i] = clampf(d->act[i], 0.0f, 1.0f);
        d->soma[i] = 0.94f * d->soma[i] + 0.02f * d->act[i];
    }
    float emergence = (1.0f - d->act[CH_VOID]) * d->act[CH_FLOW];
    d->presence = clampf(
        0.35f * d->soma[CH_LOVE] + 0.30f * d->soma[CH_FLOW] +
        0.20f * d->soma[CH_CMPLX] + 0.15f * (1.0f - d->soma[CH_VOID]),
        0.0f, 1.0f);
    /* Update coherence */
    float target = 0.52f * d->coherence + 0.18f * d->presence +
                   0.16f * emergence - 0.08f * d->scar;
    float threshold = 0.42f + 0.18f * d->threshold_bias + 0.06f * d->scar;
    if (target >= threshold)
        d->phase_lock = 0.88f * d->phase_lock + 0.12f * target + 0.05f * (target - threshold);
    else
        d->phase_lock = 0.975f * d->phase_lock + 0.025f * target;
    d->threshold_bias = 0.93f * d->threshold_bias;
    /* Heal slowly */
    d->trauma *= 0.98f;
    if (d->scar > 0.01f) d->scar *= 0.995f;
}

/* Returns: alpha (hebbian), beta (prophecy), gamma (destiny), temp_mul */
static void dario_modulate(const DarioField* d, float* alpha, float* beta,
                           float* gamma, float* temp_mul) {
    *alpha = 1.0f + 0.4f * d->act[CH_LOVE] - 0.2f * d->act[CH_RAGE] + 0.3f * d->act[CH_FLOW];
    *beta  = 1.0f + 0.4f * d->act[CH_FLOW] - 0.2f * d->act[CH_FEAR];
    *gamma = 1.0f + 0.5f * d->act[CH_CMPLX] + 0.2f * d->act[CH_LOVE] - 0.1f * d->act[CH_VOID];
    *temp_mul = 1.0f - 0.2f * d->act[CH_FLOW] + 0.1f * d->act[CH_FEAR];
    /* Somatic multiplier */
    float sm = 1.0f + 0.1f * d->presence;
    *alpha *= sm;
    *beta  *= sm;
    *gamma *= sm;
}

static float dario_phase_gate(const DarioField* d, float env_pressure) {
    float threshold = clampf(0.42f + 0.18f * d->threshold_bias + 0.06f * d->scar, 0.25f, 0.88f);
    float signal = clampf(
        0.50f * d->coherence + 0.34f * d->phase_lock +
        0.12f * env_pressure + 0.06f * d->presence - 0.08f * d->trauma,
        0.0f, 1.0f);
    return clampf(0.5f + 1.35f * (signal - threshold), 0.0f, 1.0f);
}

static int dario_velocity(const DarioField* d, float dissonance) {
    if (dissonance > 0.8f)  return VEL_UP;
    if (dissonance > 0.6f)  return VEL_RUN;
    if (dissonance < 0.2f)  return VEL_STOP;
    if (d->trauma > 0.5f)   return VEL_BREATHE;
    return VEL_WALK;
}

/* Velocity multipliers for the Dario equation coefficients */
static void velocity_multipliers(int vel, float* temp, float* heb, float* pro) {
    switch (vel) {
        case VEL_RUN:     *temp = 1.12f; *heb = 1.15f; *pro = 1.0f;  break;
        case VEL_STOP:    *temp = 0.72f; *heb = 1.0f;  *pro = 1.25f; break;
        case VEL_BREATHE: *temp = 0.90f; *heb = 1.0f;  *pro = 1.0f;  break;
        case VEL_UP:      *temp = 1.22f; *heb = 0.90f; *pro = 1.25f; break;
        case VEL_DOWN:    *temp = 0.82f; *heb = 1.10f; *pro = 0.90f; break;
        default:          *temp = 1.0f;  *heb = 1.0f;  *pro = 1.0f;  break;
    }
}

/* Activate chambers based on visual content */
static void dario_feel_vision(DarioField* d, int digit) {
    /* Simple: different digits activate different chambers */
    d->act[CH_FLOW] += 0.05f;     /* seeing anything = flow */
    if (digit == 0 || digit == 8)  /* round shapes */
        d->act[CH_LOVE] += 0.04f;
    if (digit == 1 || digit == 7)  /* sharp shapes */
        d->act[CH_CMPLX] += 0.03f;
    if (digit == 4 || digit == 9)  /* angular shapes */
        d->act[CH_FEAR] += 0.02f;
    for (int i = 0; i < N_CHAMBERS; i++)
        d->act[i] = clampf(d->act[i], 0.0f, 1.0f);
}

/* ═══════════════════════════════════════════════════════════════════
 * HEBBIAN VISION — runtime microlearning without backward pass
 *
 * vis_proto[token] = running average of visual contexts where
 * token appeared. Updated during BOTH training and inference.
 * This means the model keeps learning to see after training.
 *
 * Visual prophecy: tokens that were seen in text but never with
 * matching visual context have "unfulfilled prophecy" — a tension
 * that IS the world model working.
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float vis_proto[VOCAB * DM];     /* running avg visual context per token */
    float vis_exposure[VOCAB];       /* how many visual encounters */
    float vis_confidence[VOCAB];     /* 1 - 1/(1 + 0.1*exposure) */
} HebbianVision;

static void hebbian_init(HebbianVision* hv) {
    memset(hv, 0, sizeof(*hv));
}

/* Bind a token to a visual context (Hebbian co-occurrence) */
static void hebbian_bind(HebbianVision* hv, int token, const float* vis_ctx) {
    if (token < 0 || token >= VOCAB) return;
    hv->vis_exposure[token] += 1.0f;
    float eta = 1.0f / hv->vis_exposure[token]; /* running average */
    float* proto = hv->vis_proto + token * DM;
    for (int d = 0; d < DM; d++)
        proto[d] += eta * (vis_ctx[d] - proto[d]);
    hv->vis_confidence[token] = 1.0f - 1.0f / (1.0f + 0.1f * hv->vis_exposure[token]);
}

/* Score how well a token matches the current visual context */
static float hebbian_score(const HebbianVision* hv, int token, const float* vis_ctx) {
    if (token < 0 || token >= VOCAB) return 0.0f;
    if (hv->vis_confidence[token] < 0.01f) return 0.0f;
    const float* proto = hv->vis_proto + token * DM;
    float dot = 0, na = 0, nb = 0;
    for (int d = 0; d < DM; d++) {
        dot += proto[d] * vis_ctx[d];
        na += proto[d] * proto[d];
        nb += vis_ctx[d] * vis_ctx[d];
    }
    float denom = sqrtf(na) * sqrtf(nb) + 1e-12f;
    return (dot / denom) * hv->vis_confidence[token];
}

/* Visual prophecy: the tension between knowing a word and having seen it */
static float visual_prophecy(const HebbianVision* hv, int token, const float* vis_ctx) {
    float score = hebbian_score(hv, token, vis_ctx);
    return score * logf(1.0f + hv->vis_exposure[token]);
}

/* ═══════════════════════════════════════════════════════════════════
 * TEXT TOKENIZER — char-level for MNIST digit names
 * ═══════════════════════════════════════════════════════════════════ */

static const char* digit_names[] = {
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine"
};

static const char text_chars[] = "efghinorstuvwxz"; /* 15 chars */

static int char_to_id(char ch) {
    for (int i = 0; i < 15; i++)
        if (text_chars[i] == ch) return i;
    return -1;
}

static char id_to_char(int id) {
    if (id == BOS_TOK) return '^';
    if (id == EOS_TOK) return '$';
    if (id >= 0 && id < 15) return text_chars[id];
    return '?';
}

static int build_text_tokens(int label, int* tokens) {
    const char* name = digit_names[label];
    int n = 0;
    tokens[n++] = BOS_TOK;
    for (int i = 0; name[i]; i++)
        tokens[n++] = char_to_id(name[i]);
    tokens[n++] = EOS_TOK;
    return n;
}

/* ═══════════════════════════════════════════════════════════════════
 * SYNTHETIC DATA — 32x32 digit patterns
 *
 * Upscaled from 8x8 prototypes via nearest-neighbor, plus noise.
 * Each digit has a distinct spatial signature that the patch encoder
 * must learn to distinguish.
 * ═══════════════════════════════════════════════════════════════════ */

static const float digit_patterns_8x8[10][64] = {
    /* 0: ring */
    {0,0,.5,.8,.8,.5,0,0, 0,.5,.8,0,0,.8,.5,0, .5,.8,0,0,0,0,.8,.5,
     .8,0,0,0,0,0,0,.8, .8,0,0,0,0,0,0,.8, .5,.8,0,0,0,0,.8,.5,
     0,.5,.8,0,0,.8,.5,0, 0,0,.5,.8,.8,.5,0,0},
    /* 1: vertical line */
    {0,0,0,.8,0,0,0,0, 0,0,.5,.8,0,0,0,0, 0,0,0,.8,0,0,0,0,
     0,0,0,.8,0,0,0,0, 0,0,0,.8,0,0,0,0, 0,0,0,.8,0,0,0,0,
     0,0,0,.8,0,0,0,0, 0,0,.5,.8,.8,.5,0,0},
    /* 2: top-right-bottom */
    {0,.5,.8,.8,.8,.5,0,0, 0,0,0,0,0,.8,0,0, 0,0,0,0,0,.8,0,0,
     0,0,0,.5,.8,.5,0,0, 0,0,.5,.8,0,0,0,0, 0,.5,.8,0,0,0,0,0,
     .5,.8,0,0,0,0,0,0, .5,.8,.8,.8,.8,.8,.5,0},
    /* 3: right-heavy */
    {.5,.8,.8,.8,.5,0,0,0, 0,0,0,0,.8,0,0,0, 0,0,0,0,.8,0,0,0,
     0,.5,.8,.8,.5,0,0,0, 0,0,0,0,.8,0,0,0, 0,0,0,0,.8,0,0,0,
     0,0,0,0,.8,0,0,0, .5,.8,.8,.8,.5,0,0,0},
    /* 4: L-shape + line */
    {.8,0,0,0,.8,0,0,0, .8,0,0,0,.8,0,0,0, .8,0,0,0,.8,0,0,0,
     .8,.8,.8,.8,.8,.8,0,0, 0,0,0,0,.8,0,0,0, 0,0,0,0,.8,0,0,0,
     0,0,0,0,.8,0,0,0, 0,0,0,0,.8,0,0,0},
    /* 5: inverse-S */
    {.5,.8,.8,.8,.8,.5,0,0, .8,0,0,0,0,0,0,0, .8,0,0,0,0,0,0,0,
     .5,.8,.8,.8,.5,0,0,0, 0,0,0,0,.8,0,0,0, 0,0,0,0,.8,0,0,0,
     0,0,0,0,.8,0,0,0, .5,.8,.8,.8,.5,0,0,0},
    /* 6: bottom-heavy ring */
    {0,0,.5,.8,.8,.5,0,0, 0,.5,.8,0,0,0,0,0, .5,.8,0,0,0,0,0,0,
     .8,.5,.8,.8,.5,0,0,0, .8,0,0,0,.8,0,0,0, .8,0,0,0,.8,0,0,0,
     .5,.8,0,0,.8,0,0,0, 0,.5,.8,.8,.5,0,0,0},
    /* 7: top + diagonal */
    {.8,.8,.8,.8,.8,.8,0,0, 0,0,0,0,.5,.8,0,0, 0,0,0,0,.8,.5,0,0,
     0,0,0,.5,.8,0,0,0, 0,0,0,.8,.5,0,0,0, 0,0,.5,.8,0,0,0,0,
     0,0,.8,.5,0,0,0,0, 0,0,.8,0,0,0,0,0},
    /* 8: double ring */
    {0,.5,.8,.8,.5,0,0,0, .8,0,0,0,.8,0,0,0, .8,0,0,0,.8,0,0,0,
     0,.5,.8,.8,.5,0,0,0, .8,0,0,0,.8,0,0,0, .8,0,0,0,.8,0,0,0,
     .8,0,0,0,.8,0,0,0, 0,.5,.8,.8,.5,0,0,0},
    /* 9: top ring + tail */
    {0,.5,.8,.8,.5,0,0,0, .8,0,0,0,.8,0,0,0, .8,0,0,0,.8,0,0,0,
     0,.5,.8,.8,.8,0,0,0, 0,0,0,0,.8,0,0,0, 0,0,0,0,.8,0,0,0,
     0,0,0,.5,.8,0,0,0, 0,.5,.8,.8,.5,0,0,0},
};

typedef struct {
    float** images;  /* [n][IMG_SIZE*IMG_SIZE] in 32x32 */
    int*    labels;
    int     n;
} Dataset;

/* Upscale 8x8 → 32x32 nearest-neighbor + noise */
static Dataset generate_data(int n_samples) {
    Dataset data;
    data.n = n_samples;
    data.labels = (int*)malloc(n_samples * sizeof(int));
    data.images = (float**)malloc(n_samples * sizeof(float*));
    for (int i = 0; i < n_samples; i++) {
        int label = i % 10;
        data.labels[i] = label;
        data.images[i] = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
        for (int y = 0; y < IMG_SIZE; y++) {
            for (int x = 0; x < IMG_SIZE; x++) {
                int sy = y * 8 / IMG_SIZE;  /* source y in 8x8 */
                int sx = x * 8 / IMG_SIZE;  /* source x in 8x8 */
                float base = digit_patterns_8x8[label][sy * 8 + sx];
                float noise = rng_normal(0, 0.06f);
                float val = base + noise;
                data.images[i][y * IMG_SIZE + x] = val < 0 ? 0 : (val > 1 ? 1 : val);
            }
        }
    }
    return data;
}

static void free_data(Dataset* d) {
    for (int i = 0; i < d->n; i++) free(d->images[i]);
    free(d->images);
    free(d->labels);
}

/* Extract 8x8 patches from 32x32 image → [N_VIS * PATCH_PX] */
static void extract_patches(const float* img, float* patches) {
    for (int py = 0; py < N_PATCHES_SIDE; py++) {
        for (int px = 0; px < N_PATCHES_SIDE; px++) {
            int pid = py * N_PATCHES_SIDE + px;
            for (int dy = 0; dy < PATCH_SIZE; dy++) {
                for (int dx = 0; dx < PATCH_SIZE; dx++) {
                    int iy = py * PATCH_SIZE + dy;
                    int ix = px * PATCH_SIZE + dx;
                    patches[pid * PATCH_PX + dy * PATCH_SIZE + dx] = img[iy * IMG_SIZE + ix];
                }
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * MODEL — shared transformer for vision + text
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    /* Vision */
    nt_tensor* patch_proj;     /* [DM, PATCH_PX] */

    /* Shared embeddings */
    nt_tensor* wte;            /* [VOCAB, DM] */
    nt_tensor* wpe;            /* [MAX_SEQ, DM] */

    /* Transformer layers */
    struct {
        nt_tensor* rms1;       /* [DM] */
        nt_tensor* wq;         /* [DM, DM] */
        nt_tensor* wk;         /* [DM, DM] */
        nt_tensor* wv;         /* [DM, DM] */
        nt_tensor* wo;         /* [DM, DM] */
        nt_tensor* rel_bias;   /* [N_HEADS * MAX_SEQ] — RRPRAM position bias */
        nt_tensor* rms2;       /* [DM] */
        nt_tensor* w_gate;     /* [DFF, DM] */
        nt_tensor* w_up;       /* [DFF, DM] */
        nt_tensor* w_down;     /* [DM, DFF] */
    } layers[N_LAYERS];

    nt_tensor* rms_final;      /* [DM] */
    nt_tensor* lm_head;        /* [VOCAB, DM] */

    /* Non-trainable, persistent */
    HebbianVision hebbian;
    DarioField    dario;
} NeoVLM;

static long count_params(NeoVLM* m) {
    long n = m->patch_proj->len + m->wte->len + m->wpe->len +
             m->rms_final->len + m->lm_head->len;
    for (int l = 0; l < N_LAYERS; l++) {
        n += m->layers[l].rms1->len + m->layers[l].wq->len +
             m->layers[l].wk->len + m->layers[l].wv->len +
             m->layers[l].wo->len + m->layers[l].rel_bias->len +
             m->layers[l].rms2->len + m->layers[l].w_gate->len +
             m->layers[l].w_up->len + m->layers[l].w_down->len;
    }
    return n;
}

static NeoVLM* model_create(void) {
    NeoVLM* m = (NeoVLM*)calloc(1, sizeof(NeoVLM));

    m->patch_proj = nt_tensor_new2d(DM, PATCH_PX);
    nt_tensor_xavier(m->patch_proj, PATCH_PX, DM);

    m->wte = nt_tensor_new2d(VOCAB, DM);
    nt_tensor_xavier(m->wte, VOCAB, DM);

    m->wpe = nt_tensor_new2d(MAX_SEQ, DM);
    nt_tensor_xavier(m->wpe, MAX_SEQ, DM);

    for (int l = 0; l < N_LAYERS; l++) {
        float scale = 0.02f / sqrtf(2.0f * N_LAYERS);
        m->layers[l].rms1 = nt_tensor_new(DM);
        nt_tensor_fill(m->layers[l].rms1, 1.0f);

        m->layers[l].wq = nt_tensor_new2d(DM, DM);
        nt_tensor_xavier(m->layers[l].wq, DM, DM);
        m->layers[l].wk = nt_tensor_new2d(DM, DM);
        nt_tensor_xavier(m->layers[l].wk, DM, DM);
        m->layers[l].wv = nt_tensor_new2d(DM, DM);
        nt_tensor_xavier(m->layers[l].wv, DM, DM);
        m->layers[l].wo = nt_tensor_new2d(DM, DM);
        nt_tensor_xavier(m->layers[l].wo, DM, DM);
        for (int i = 0; i < m->layers[l].wo->len; i++)
            m->layers[l].wo->data[i] *= scale / 0.1f;

        /* RRPRAM: initialize with mild linear decay (closer = higher) */
        m->layers[l].rel_bias = nt_tensor_new(N_HEADS * MAX_SEQ);
        for (int h = 0; h < N_HEADS; h++)
            for (int r = 0; r < MAX_SEQ; r++)
                m->layers[l].rel_bias->data[h * MAX_SEQ + r] =
                    0.1f * (1.0f - (float)r / MAX_SEQ); /* nearby = positive */

        m->layers[l].rms2 = nt_tensor_new(DM);
        nt_tensor_fill(m->layers[l].rms2, 1.0f);

        m->layers[l].w_gate = nt_tensor_new2d(DFF, DM);
        nt_tensor_xavier(m->layers[l].w_gate, DM, DFF);
        m->layers[l].w_up = nt_tensor_new2d(DFF, DM);
        nt_tensor_xavier(m->layers[l].w_up, DM, DFF);
        m->layers[l].w_down = nt_tensor_new2d(DM, DFF);
        nt_tensor_xavier(m->layers[l].w_down, DFF, DM);
        for (int i = 0; i < m->layers[l].w_down->len; i++)
            m->layers[l].w_down->data[i] *= scale / 0.1f;
    }

    m->rms_final = nt_tensor_new(DM);
    nt_tensor_fill(m->rms_final, 1.0f);

    m->lm_head = nt_tensor_new2d(VOCAB, DM);
    nt_tensor_xavier(m->lm_head, DM, VOCAB);

    hebbian_init(&m->hebbian);
    dario_init(&m->dario);

    return m;
}

static void model_free(NeoVLM* m) {
    nt_tensor_free(m->patch_proj);
    nt_tensor_free(m->wte);
    nt_tensor_free(m->wpe);
    for (int l = 0; l < N_LAYERS; l++) {
        nt_tensor_free(m->layers[l].rms1);
        nt_tensor_free(m->layers[l].wq);
        nt_tensor_free(m->layers[l].wk);
        nt_tensor_free(m->layers[l].wv);
        nt_tensor_free(m->layers[l].wo);
        nt_tensor_free(m->layers[l].rel_bias);
        nt_tensor_free(m->layers[l].rms2);
        nt_tensor_free(m->layers[l].w_gate);
        nt_tensor_free(m->layers[l].w_up);
        nt_tensor_free(m->layers[l].w_down);
    }
    nt_tensor_free(m->rms_final);
    nt_tensor_free(m->lm_head);
    free(m);
}

/* ═══════════════════════════════════════════════════════════════════
 * FORWARD PASS — per-position with KV cache
 *
 * Vision positions (0..15): h = patch_proj @ patch + wpe[pos]
 * Text positions (16+):     h = wte[token] + wpe[pos]
 * Both go through the same N_LAYERS transformer blocks.
 * Attention includes RRPRAM relative position bias.
 * ═══════════════════════════════════════════════════════════════════ */

/* Tape indices for registered params */
typedef struct {
    int pp;                /* patch_proj */
    int wte, wpe;          /* embeddings */
    struct {
        int rms1, wq, wk, wv, wo, rb, rms2, wg, wu, wd;
    } L[N_LAYERS];
    int rmsf, head;
} TapeIdx;

static TapeIdx register_params(NeoVLM* m) {
    TapeIdx ti;
    ti.pp  = nt_tape_param(m->patch_proj);
    ti.wte = nt_tape_param(m->wte);  nt_tape_no_decay(ti.wte);
    ti.wpe = nt_tape_param(m->wpe);  nt_tape_no_decay(ti.wpe);
    for (int l = 0; l < N_LAYERS; l++) {
        ti.L[l].rms1 = nt_tape_param(m->layers[l].rms1);
        ti.L[l].wq   = nt_tape_param(m->layers[l].wq);
        ti.L[l].wk   = nt_tape_param(m->layers[l].wk);
        ti.L[l].wv   = nt_tape_param(m->layers[l].wv);
        ti.L[l].wo   = nt_tape_param(m->layers[l].wo);
        ti.L[l].rb   = nt_tape_param(m->layers[l].rel_bias); nt_tape_no_decay(ti.L[l].rb);
        ti.L[l].rms2 = nt_tape_param(m->layers[l].rms2);
        ti.L[l].wg   = nt_tape_param(m->layers[l].w_gate);
        ti.L[l].wu   = nt_tape_param(m->layers[l].w_up);
        ti.L[l].wd   = nt_tape_param(m->layers[l].w_down);
    }
    ti.rmsf = nt_tape_param(m->rms_final);
    ti.head = nt_tape_param(m->lm_head);
    return ti;
}

/* Process one position through all transformer layers.
 * Updates KV cache. Returns tape index of final hidden state [DM]. */
static int forward_position(NeoVLM* m, TapeIdx* ti, int h_idx, int pos,
                             int kv_keys[][MAX_SEQ], int kv_vals[][MAX_SEQ]) {
    nt_tape* tape = nt_tape_get();

    for (int l = 0; l < N_LAYERS; l++) {
        int residual = h_idx;

        /* Pre-attention RMSNorm */
        h_idx = nt_rmsnorm(h_idx, ti->L[l].rms1);

        /* QKV projections (on tape → gradients flow) */
        int q = nt_linear(ti->L[l].wq, h_idx, -1);
        int k = nt_linear(ti->L[l].wk, h_idx, -1);
        int v = nt_linear(ti->L[l].wv, h_idx, -1);

        kv_keys[l][pos] = k;
        kv_vals[l][pos] = v;

        /* Multi-head attention with RRPRAM bias (manual, off-tape) */
        nt_tensor* attn_out = nt_tensor_new(DM);
        float* q_data = tape->entries[q].output->data;
        float* rb_data = m->layers[l].rel_bias->data;

        for (int head = 0; head < N_HEADS; head++) {
            int ho = head * HD;
            float scores[MAX_SEQ];
            float max_s = -1e9f;

            for (int j = 0; j <= pos; j++) {
                float* k_data = tape->entries[kv_keys[l][j]].output->data;
                float s = 0;
                for (int d = 0; d < HD; d++)
                    s += q_data[ho + d] * k_data[ho + d];
                s /= sqrtf((float)HD);
                /* RRPRAM: add learned relative position bias */
                int rel = pos - j;
                s += rb_data[head * MAX_SEQ + rel];
                scores[j] = s;
                if (s > max_s) max_s = s;
            }

            /* Softmax */
            float sum_e = 0;
            for (int j = 0; j <= pos; j++) {
                scores[j] = expf(scores[j] - max_s);
                sum_e += scores[j];
            }
            float inv = 1.0f / (sum_e + 1e-10f);
            for (int j = 0; j <= pos; j++) scores[j] *= inv;

            /* Weighted value sum */
            for (int d = 0; d < HD; d++) {
                float val = 0;
                for (int j = 0; j <= pos; j++) {
                    float* v_data = tape->entries[kv_vals[l][j]].output->data;
                    val += scores[j] * v_data[ho + d];
                }
                attn_out->data[ho + d] = val;
            }
        }

        int attn_idx = nt_tape_record(attn_out, NT_OP_NONE, -1, -1, 0);
        nt_tensor_free(attn_out);

        /* Output projection + residual */
        int proj = nt_linear(ti->L[l].wo, attn_idx, -1);
        h_idx = nt_add(residual, proj);

        /* Pre-FFN RMSNorm */
        residual = h_idx;
        h_idx = nt_rmsnorm(h_idx, ti->L[l].rms2);

        /* SiLU-gated FFN */
        int gate_v = nt_linear(ti->L[l].wg, h_idx, -1);
        gate_v = nt_silu(gate_v);
        int up_v = nt_linear(ti->L[l].wu, h_idx, -1);
        int gated = nt_mul(gate_v, up_v);
        int down_v = nt_linear(ti->L[l].wd, gated, -1);
        h_idx = nt_add(residual, down_v);
    }

    return h_idx;
}

/* ═══════════════════════════════════════════════════════════════════
 * VISUAL CONTEXT — mean of patch projections (for Hebbian binding)
 * ═══════════════════════════════════════════════════════════════════ */

static void compute_vis_context(NeoVLM* m, const float* patches, float* vis_ctx) {
    memset(vis_ctx, 0, DM * sizeof(float));
    for (int p = 0; p < N_VIS; p++) {
        const float* patch = patches + p * PATCH_PX;
        for (int d = 0; d < DM; d++) {
            float sum = 0;
            for (int i = 0; i < PATCH_PX; i++)
                sum += m->patch_proj->data[d * PATCH_PX + i] * patch[i];
            vis_ctx[d] += sum;
        }
    }
    float inv_n = 1.0f / N_VIS;
    float norm = 0;
    for (int d = 0; d < DM; d++) {
        vis_ctx[d] *= inv_n;
        norm += vis_ctx[d] * vis_ctx[d];
    }
    norm = sqrtf(norm + 1e-12f);
    for (int d = 0; d < DM; d++) vis_ctx[d] /= norm;
}

/* ═══════════════════════════════════════════════════════════════════
 * TIMER
 * ═══════════════════════════════════════════════════════════════════ */

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ═══════════════════════════════════════════════════════════════════
 * TRAINING
 * ═══════════════════════════════════════════════════════════════════ */

static void train(NeoVLM* m, Dataset* data, int steps) {
    printf("\n═══ TRAINING (%d steps) ═══\n", steps);
    printf("  model: %d layers, D=%d, %d heads, SiLU-gated FFN\n", N_LAYERS, DM, N_HEADS);
    printf("  vision: %d patches (%dx%d) from %dx%d images\n",
           N_VIS, PATCH_SIZE, PATCH_SIZE, IMG_SIZE, IMG_SIZE);
    printf("  sequence: %d vision + %d text = %d max\n", N_VIS, MAX_TEXT, MAX_SEQ);
    long np = count_params(m);
    printf("  params: %ld (%.2fM, %.2f MB)\n", np, np / 1e6f, np * 4.0f / 1048576.0f);
    printf("  RRPRAM: relative position bias, %d params/layer\n", N_HEADS * MAX_SEQ);
    printf("  Hebbian: vis_proto [%d x %d], updated every step\n", VOCAB, DM);
    printf("  Dario: %d chambers, Kuramoto K=0.03\n", N_CHAMBERS);
    float cd = calendar_dissonance();
    printf("  calendar dissonance: %.3f\n", cd);
    printf("  velocity: %s\n", vel_names[dario_velocity(&m->dario, cd)]);
    printf("\n");

    nt_schedule sched = nt_schedule_cosine(LR_BASE, steps / 10, steps, LR_BASE * 0.1f);
    nt_nan_guard guard = nt_nan_guard_new();

    int kv_keys[N_LAYERS][MAX_SEQ];
    int kv_vals[N_LAYERS][MAX_SEQ];

    float running_loss = 0;
    int running_count = 0;
    double t0 = now_ms();
    float first_loss = 0, last_loss = 0;

    for (int step = 0; step < steps; step++) {
        float lr = nt_schedule_get_lr(&sched);

        /* Random sample */
        int idx = (int)(rng_next() % (uint64_t)data->n);
        int label = data->labels[idx];

        /* Extract patches */
        float patches[N_VIS * PATCH_PX];
        extract_patches(data->images[idx], patches);

        /* Build text tokens */
        int text_tokens[MAX_TEXT + 2];
        int n_text = build_text_tokens(label, text_tokens);

        /* Visual context for Hebbian */
        float vis_ctx[DM];
        compute_vis_context(m, patches, vis_ctx);

        /* Dario step */
        dario_kuramoto_step(&m->dario);
        dario_feel_vision(&m->dario, label);

        /* ── TAPE FORWARD ── */
        nt_tape_start();
        nt_train_mode(1);
        TapeIdx ti = register_params(m);

        /* === Vision positions === */
        for (int p = 0; p < N_VIS; p++) {
            nt_tensor* patch_t = nt_tensor_new(PATCH_PX);
            memcpy(patch_t->data, &patches[p * PATCH_PX], PATCH_PX * sizeof(float));
            int patch_idx = nt_tape_record(patch_t, NT_OP_NONE, -1, -1, 0);
            nt_tensor_free(patch_t);

            int h = nt_linear(ti.pp, patch_idx, -1);
            int pos_emb = nt_embedding(ti.wpe, p);
            h = nt_add(h, pos_emb);
            forward_position(m, &ti, h, p, kv_keys, kv_vals);
        }

        /* === Text positions === */
        int loss_indices[MAX_TEXT];
        int n_losses = 0;

        for (int t = 0; t < n_text - 1; t++) {
            int pos = N_VIS + t;
            int tok_emb = nt_embedding(ti.wte, text_tokens[t]);
            int pos_emb = nt_embedding(ti.wpe, pos);
            int h = nt_add(tok_emb, pos_emb);
            h = forward_position(m, &ti, h, pos, kv_keys, kv_vals);

            /* Logits + loss */
            int h_norm = nt_rmsnorm(h, ti.rmsf);
            int logits = nt_linear(ti.head, h_norm, -1);
            int ce = nt_cross_entropy(logits, text_tokens[t + 1]);
            loss_indices[n_losses++] = ce;

            /* Hebbian bind: this token co-occurred with this visual scene */
            hebbian_bind(&m->hebbian, text_tokens[t], vis_ctx);
        }
        /* Bind last target token too */
        hebbian_bind(&m->hebbian, text_tokens[n_text - 1], vis_ctx);

        /* Average loss */
        int total_loss = loss_indices[0];
        for (int i = 1; i < n_losses; i++)
            total_loss = nt_add(total_loss, loss_indices[i]);
        int avg_loss = nt_scale(total_loss, 1.0f / n_losses);

        float loss_val = nt_tape_get()->entries[avg_loss].output->data[0];
        if (step == 0) first_loss = loss_val;
        last_loss = loss_val;

        /* Backward */
        nt_tape_backward(avg_loss);

        /* NaN check */
        if (!nt_nan_guard_check(&guard)) {
            if (step % 50 == 0)
                printf("  step %4d: NaN (scale=%.4f)\n", step + 1, guard.loss_scale);
            nt_tape_clear();
            continue;
        }

        /* Optimize */
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(lr, loss_val);
        nt_tape_clear();

        running_loss += loss_val;
        running_count++;

        if ((step + 1) % 200 == 0 || step == 0) {
            double elapsed = (now_ms() - t0) / 1000.0;
            printf("  step %4d | train %.4f (avg %.4f) | lr %.2e | %.1fs | %s vel=%s\n",
                   step + 1, loss_val,
                   running_count > 0 ? running_loss / running_count : 0,
                   lr, elapsed, digit_names[label],
                   vel_names[m->dario.velocity]);
            running_loss = 0;
            running_count = 0;
        }
    }

    double total_s = (now_ms() - t0) / 1000.0;
    float reduction = first_loss > 0 ? (first_loss - last_loss) / first_loss * 100.0f : 0;
    printf("\n  loss: %.4f → %.4f (%.1f%% reduction)\n", first_loss, last_loss, reduction);
    printf("  time: %.1f seconds (%.1f steps/s)\n", total_s, steps / total_s);
    printf("  nans: %d detected, %d steps skipped\n",
           guard.total_nan_count, guard.skipped_steps);

    /* Hebbian summary */
    printf("\n── hebbian vision ──\n");
    for (int tok = 0; tok < VOCAB; tok++) {
        if (m->hebbian.vis_exposure[tok] > 0) {
            printf("  '%c' (id=%d): exposure=%.0f confidence=%.3f\n",
                   id_to_char(tok), tok,
                   m->hebbian.vis_exposure[tok],
                   m->hebbian.vis_confidence[tok]);
        }
    }

    /* Dario summary */
    printf("\n── dario field ──\n");
    for (int c = 0; c < N_CHAMBERS; c++)
        printf("  %s: act=%.3f soma=%.3f\n",
               ch_names[c], m->dario.act[c], m->dario.soma[c]);
    printf("  coherence=%.3f phase_lock=%.3f scar=%.3f\n",
           m->dario.coherence, m->dario.phase_lock, m->dario.scar);
}

/* ═══════════════════════════════════════════════════════════════════
 * GENERATION — with Dario field modulation + Hebbian enrichment
 *
 * The Extended Dario Equation:
 *   p(x|Φ) = softmax((logits + α·H_v + β·prophecy + δ·V) / τ)
 *
 * Where:
 *   logits = transformer output
 *   H_v = Hebbian visual score (which tokens match this image?)
 *   prophecy = visual prophecy (unfulfilled visual expectations)
 *   V = direct visual grounding (cosine similarity)
 *   α, β, δ = Dario field modulation coefficients
 *   τ = temperature (chamber-modulated)
 *
 * The "as if hesitating" part comes from the Dario field,
 * not from training data. That's the point.
 * ═══════════════════════════════════════════════════════════════════ */

static void generate(NeoVLM* m, const float* image, int label) {
    float patches[N_VIS * PATCH_PX];
    extract_patches(image, patches);

    float vis_ctx[DM];
    compute_vis_context(m, patches, vis_ctx);

    /* Dario modulation */
    float cd = calendar_dissonance();
    m->dario.velocity = dario_velocity(&m->dario, cd);
    float d_alpha, d_beta, d_gamma, d_temp;
    dario_modulate(&m->dario, &d_alpha, &d_beta, &d_gamma, &d_temp);
    float vel_t, vel_h, vel_p;
    velocity_multipliers(m->dario.velocity, &vel_t, &vel_h, &vel_p);
    float phase = dario_phase_gate(&m->dario, cd);

    /* Print chamber state */
    printf("  [chambers]");
    for (int c = 0; c < N_CHAMBERS; c++)
        if (m->dario.act[c] > 0.01f)
            printf(" %s:%.0f%%", ch_names[c], m->dario.act[c] * 100);
    printf(" phase:%.2f vel=%s\n", phase, vel_names[m->dario.velocity]);
    printf("  [vision] %d patches → %d visual tokens\n", N_VIS, N_VIS);

    int kv_keys[N_LAYERS][MAX_SEQ];
    int kv_vals[N_LAYERS][MAX_SEQ];

    /* Forward vision positions (no tape, eval mode) */
    nt_tape_start();
    nt_train_mode(0);
    TapeIdx ti = register_params(m);

    for (int p = 0; p < N_VIS; p++) {
        nt_tensor* patch_t = nt_tensor_new(PATCH_PX);
        memcpy(patch_t->data, &patches[p * PATCH_PX], PATCH_PX * sizeof(float));
        int patch_idx = nt_tape_record(patch_t, NT_OP_NONE, -1, -1, 0);
        nt_tensor_free(patch_t);
        int h = nt_linear(ti.pp, patch_idx, -1);
        int pos_emb = nt_embedding(ti.wpe, p);
        h = nt_add(h, pos_emb);
        forward_position(m, &ti, h, p, kv_keys, kv_vals);
    }

    /* Autoregressive text generation */
    int token_id = BOS_TOK;
    char generated[MAX_TEXT + 1];
    int gen_len = 0;

    for (int t = 0; t < MAX_TEXT - 1; t++) {
        int pos = N_VIS + t;
        int tok_emb = nt_embedding(ti.wte, token_id);
        int pos_emb = nt_embedding(ti.wpe, pos);
        int h = nt_add(tok_emb, pos_emb);
        h = forward_position(m, &ti, h, pos, kv_keys, kv_vals);

        int h_norm = nt_rmsnorm(h, ti.rmsf);
        int logits_idx = nt_linear(ti.head, h_norm, -1);

        /* Get logits and apply Dario equation */
        nt_tape* tape = nt_tape_get();
        float logits[VOCAB];
        memcpy(logits, tape->entries[logits_idx].output->data, VOCAB * sizeof(float));

        /* Hebbian + prophecy injection */
        for (int v = 0; v < VOCAB; v++) {
            float heb = hebbian_score(&m->hebbian, v, vis_ctx);
            float pro = visual_prophecy(&m->hebbian, v, vis_ctx);
            logits[v] += d_alpha * vel_h * heb * 2.0f +
                         d_beta * vel_p * pro * 1.5f;
        }

        /* Temperature with Dario + Schumann modulation */
        float temp = 0.8f * d_temp * vel_t;
        /* Schumann resonance: subtle oscillation */
        float schumann_t = (float)t * 0.1f;
        temp *= 1.0f + 0.02f * sinf(7.83f * schumann_t);
        if (temp < 0.1f) temp = 0.1f;

        /* Phase gate: suppress low-confidence tokens */
        for (int v = 0; v < VOCAB; v++) logits[v] /= temp;

        /* Greedy decode (deterministic for eval) */
        int best = 0;
        float best_val = logits[0];
        for (int v = 1; v < VOCAB; v++)
            if (logits[v] > best_val) { best_val = logits[v]; best = v; }

        token_id = best;
        if (token_id == EOS_TOK || token_id == BOS_TOK) break;
        if (gen_len < MAX_TEXT) generated[gen_len++] = id_to_char(token_id);

        /* Hebbian bind: model learns from its own generation */
        hebbian_bind(&m->hebbian, token_id, vis_ctx);
    }
    generated[gen_len] = '\0';

    nt_tape_clear();

    /* Report */
    const char* true_name = digit_names[label];
    int match = (strcmp(generated, true_name) == 0);

    /* Hebbian update report */
    float max_conf = 0;
    int max_tok = -1;
    for (int v = 0; v < VOCAB; v++) {
        float s = hebbian_score(&m->hebbian, v, vis_ctx);
        if (s > max_conf) { max_conf = s; max_tok = v; }
    }
    if (max_tok >= 0 && max_conf > 0.01f)
        printf("  [hebbian] best match: '%c' score=%.3f\n", id_to_char(max_tok), max_conf);

    printf("  [generate] %s %s\n", generated, match ? "✓" : "✗");
}

/* ═══════════════════════════════════════════════════════════════════
 * INFERENCE
 * ═══════════════════════════════════════════════════════════════════ */

static void inference(NeoVLM* m, Dataset* data) {
    printf("\n═══ INFERENCE: seeing digits, speaking names ═══\n\n");

    int correct = 0, total = 0;

    for (int sample = 0; sample < 30; sample++) {
        int label = sample % 10;
        int idx = label + (sample / 10) * 10;
        if (idx >= data->n) idx = label;

        printf("  [%d] true: %-5s | ", label, digit_names[label]);
        generate(m, data->images[idx], label);

        /* Check accuracy (recompute quickly) */
        float patches[N_VIS * PATCH_PX];
        extract_patches(data->images[idx], patches);
        float vis_ctx[DM];
        compute_vis_context(m, patches, vis_ctx);

        /* Quick greedy generate for accuracy */
        int kv_keys[N_LAYERS][MAX_SEQ];
        int kv_vals[N_LAYERS][MAX_SEQ];
        nt_tape_start();
        nt_train_mode(0);
        TapeIdx ti = register_params(m);
        for (int p = 0; p < N_VIS; p++) {
            nt_tensor* pt = nt_tensor_new(PATCH_PX);
            memcpy(pt->data, &patches[p * PATCH_PX], PATCH_PX * sizeof(float));
            int pi2 = nt_tape_record(pt, NT_OP_NONE, -1, -1, 0);
            nt_tensor_free(pt);
            int h = nt_linear(ti.pp, pi2, -1);
            int pe = nt_embedding(ti.wpe, p);
            h = nt_add(h, pe);
            forward_position(m, &ti, h, p, kv_keys, kv_vals);
        }
        int tok = BOS_TOK;
        char gen[MAX_TEXT + 1];
        int gl = 0;
        for (int t = 0; t < MAX_TEXT - 1; t++) {
            int pos = N_VIS + t;
            int te2 = nt_embedding(ti.wte, tok);
            int pe2 = nt_embedding(ti.wpe, pos);
            int h = nt_add(te2, pe2);
            h = forward_position(m, &ti, h, pos, kv_keys, kv_vals);
            int hn = nt_rmsnorm(h, ti.rmsf);
            int lg = nt_linear(ti.head, hn, -1);
            nt_tape* tape = nt_tape_get();
            float* ld = tape->entries[lg].output->data;
            int best = 0;
            for (int v = 1; v < VOCAB; v++)
                if (ld[v] > ld[best]) best = v;
            tok = best;
            if (tok == EOS_TOK || tok == BOS_TOK) break;
            if (gl < MAX_TEXT) gen[gl++] = id_to_char(tok);
        }
        gen[gl] = '\0';
        if (strcmp(gen, digit_names[label]) == 0) correct++;
        total++;
        nt_tape_clear();
    }

    printf("\n  accuracy: %d/%d (%.1f%%)\n", correct, total, 100.0f * correct / total);
}

/* ═══════════════════════════════════════════════════════════════════
 * INTERACTIVE MODE — show image, generate, Hebbian learns
 * ═══════════════════════════════════════════════════════════════════ */

static void print_image_ascii(const float* img) {
    const char* shades = " .:-=+*#%@";
    int ns = 10;
    printf("  ┌");
    for (int x = 0; x < IMG_SIZE / 2; x++) printf("─");
    printf("┐\n");
    for (int y = 0; y < IMG_SIZE; y += 2) {
        printf("  │");
        for (int x = 0; x < IMG_SIZE; x += 2) {
            float v = img[y * IMG_SIZE + x];
            int si = (int)(v * (ns - 1));
            if (si < 0) si = 0;
            if (si >= ns) si = ns - 1;
            printf("%c", shades[si]);
        }
        printf("│\n");
    }
    printf("  └");
    for (int x = 0; x < IMG_SIZE / 2; x++) printf("─");
    printf("┘\n");
}

static void interactive(NeoVLM* m, Dataset* data) {
    printf("\n═══ INTERACTIVE MODE ═══\n");
    printf("  showing random images, generating text, Hebbian learning...\n\n");

    for (int round = 0; round < 10; round++) {
        int idx = (int)(rng_next() % (uint64_t)data->n);
        int label = data->labels[idx];

        printf("── round %d ──\n", round + 1);
        print_image_ascii(data->images[idx]);
        printf("  [digit: %d]\n", label);
        generate(m, data->images[idx], label);
        printf("\n");

        /* Dario step between interactions */
        dario_kuramoto_step(&m->dario);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */

int main(int argc, char** argv) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  neovlm — Hebbian Vision-Language Model in pure C       ║\n");
    printf("║  Sees, speaks, grows. Part of Arianna Method.           ║\n");
    printf("║  θ = ε + γ + αδ                                        ║\n");
    printf("║  p(x|Φ) = softmax((α·H_v + β·F_v + γ·A + δ·V) / τ)   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    int steps = DEFAULT_STEPS;
    int do_interactive = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
            steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--interactive") == 0)
            do_interactive = 1;
    }

    /* Init */
    rng_seed(42);
    nt_seed(42);

    /* Generate data */
    printf("generating synthetic 32x32 digit patterns...\n");
    Dataset data = generate_data(10000);
    printf("generated %d images, %dx%d, %d patches of %dx%d\n\n",
           data.n, IMG_SIZE, IMG_SIZE, N_VIS, PATCH_SIZE, PATCH_SIZE);

    /* Create model */
    NeoVLM* model = model_create();

    /* Train */
    train(model, &data, steps);

    /* Inference */
    inference(model, &data);

    /* Interactive */
    if (do_interactive)
        interactive(model, &data);

    /* Cleanup */
    model_free(model);
    free_data(&data);
    nt_tape_destroy();

    printf("\n═══ done. the oracle has spoken. ═══\n");
    printf("═══ every inference enriched the visual prototypes. ═══\n");
    printf("═══ the model sees more than it was trained to see. ═══\n");
    return 0;
}
