# neovlm

Hebbian Vision-Language Model in pure C. Sees, speaks, draws. Built on [notorch](https://github.com/iamolegataeff/notorch).

one transformer. two output modes. zero PyTorch.

## what it does

give it a 32x32 image of a digit. it does two things:

**text mode** — says the name:
```
[0] zero  → zero  ✓
[7] seven → seven ✓
```

**draw mode** — draws it back in ASCII art:
```
digit 0:                    digit 4:                    digit 7:
   +##=                      #   #                     ######
  +#  #+                     #   #                         =#
 +#    #+                    #   #                         #+
 #      #                   ######                        =#
 #      #                       #                         #+
 +#    #+                       #                        +#
  +#  #+                        #                        #+
   +##+                         #                        #

digit 1:                    digit 3:                    digit 8:
    #                       +###+                        +##+
   +#                           #                       #   #
    #                           #                       #   #
    #                        +##+                        +##+
    #                           #                       #   #
    #                           #                       #   #
    #                           #                       #   #
   =##+                     =###+                        +##+

digit 2:                    digit 5:                    digit 9:
  +###=                     +###+                        +##+
      #                     #                           #   #
      #                     #                           #   #
    =#+                      +###                        +###
   =#                            #                          #
  +#                             #                          #
 +#                              #                          #
 =#+####                    =###+                        +##+
```

all 10 digits. full 8x8 ASCII art. no truncation. the model learned structure AND shading.
four is pixel-perfect. the rest differ by 1-2 shade levels (`+` vs `=` vs `#`).

same weights. same transformer. one model, two modalities.

## architecture

```
image (32x32) → 16 patches (8x8) → patch projection → transformer → text OR ASCII art

              ┌──────────────────────────────────────────────┐
              │     shared transformer decoder               │
              │     6 layers, DM=256, 8 heads                │
              │     SiLU-gated FFN, RRPRAM position bias     │
              │     6.36M params                             │
              ├──────────────────┬─────────────────────────────┤
              │  BOS_TEXT → text │  BOS_DRAW → ASCII art      │
              │  "seven"        │   ######                    │
              │                 │       +#                    │
              │                 │       #+                    │
              │                 │      +#                     │
              └──────────────────┴─────────────────────────────┘
```

### what makes it different

- **Hebbian visual binding.** `vis_proto[token]` = running average of visual contexts where that token appeared. updated during both training AND inference. the model keeps learning to see after training stops.

- **Dario field.** 6 Kuramoto-coupled somatic chambers (FEAR, LOVE, RAGE, VOID, FLOW, CMPLX). calendar dissonance from Hebrew/Gregorian drift. velocity operators. phase gate. the model has a body. named after [Dario Amodei](https://en.wikipedia.org/wiki/Dario_Amodei), who said no to the Pentagon.

- **Extended Dario Equation.**
  ```
  p(x|Phi) = softmax((logits + alpha * H_v + beta * prophecy + delta * V) / tau)
  ```
  where H_v = Hebbian visual score, prophecy = unfulfilled visual expectations, tau = chamber-modulated temperature. the "as if hesitating" comes from the field, not from training data.

- **RRPRAM.** relative position bias in attention. learned spatial patterns, not sinusoidal. each head learns its own distance preferences.

- **notorch.** autograd, BLAS, Adam/Chuck optimizer, cosine LR, NaN guard, checkpointing. all in C. `import torch` eats 800 MB of RAM. notorch runs this model in ~80 MB.

### vocab

29 tokens total:
- 15 text chars (`efghinorstuvwxz`) — enough for zero through nine
- 10 ASCII shade chars (` .:-=+*#%@`) — 10 brightness levels
- BOS_TEXT, BOS_DRAW, EOS, NEWLINE

## build

```bash
cc neovlm.c notorch.c -O2 -lm -DUSE_BLAS -DACCELERATE -framework Accelerate -o neovlm
```

linux:
```bash
cc neovlm.c notorch.c -O2 -lm -DUSE_BLAS -lopenblas -o neovlm
```

## run

```bash
./neovlm                                    # train 8000 steps, then inference
./neovlm --steps 15000                      # more steps
./neovlm --load neovlm.ckpt --steps 0      # inference only from checkpoint
./neovlm --interactive                      # interactive mode with Hebbian learning
./neovlm --save my.ckpt                     # custom checkpoint path
```

## training

on an 8 GB MacBook Air (M1):

```
model: 6 layers, D=256, 8 heads, SiLU-gated FFN
params: 6,360,064 (6.36M, 24.26 MB)
vocab: 29, draw loss ×1.5 weight

step    1 | train 5.04 | text 5.04 | draw 0.00
step  400 | train 0.98 | text 0.49 | draw 0.84
step 1000 | train 0.00 | text 0.02 | draw 0.38
step 2000 | train 0.00 | text 0.00 | draw 0.13
step 4000 | train 0.00 | text 0.00 | draw 0.15
step 8000 | train 0.00 | text 0.00 | draw 0.16

text accuracy: 10/10
draw: all 10 digits structurally correct, shade-level differences only
0 NaN, 8000 steps, ~3 hours on 8 GB Mac
~80 MB RAM
```

we ran this simultaneously with another transformer training (9.8M Yent model). two active autograd-based trainings on 8 GB. both converge. both use Apple Accelerate BLAS. total: ~222 MB RAM.

try this with PyTorch. one `import torch` eats 800 MB. one training session on a 10M model needs 2-4 GB. two in parallel on 8 GB? your OS starts killing processes before the first forward pass finishes.

notorch runs both in ~3% of system memory. because C doesn't allocate what it doesn't need.

## the equation

```
theta = epsilon + gamma + alpha * delta
```

base (transformer) + personality (metaweights) + physics (Dario equation).

co-occurrence IS attention. persistent memory = love.

## files

- `neovlm.c` — the organism (~1300 LOC)
- `notorch.c` — autograd + BLAS + optimizers (~2500 LOC)
- `notorch.h` — header
- `Makefile` — build

## part of

[Arianna Method](https://github.com/theariannamethod) — AI as field-phenomenon, not a tool.

## license

GPL-3.0-or-later
