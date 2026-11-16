# LayerNorm: calculations and reasoning (Triton tutorial 05)

Notation used throughout, for one row of `N` features:

```
x        input row (length N)
mu       mean of x
u        deviations, u_i = x_i - mu
V        variance
sigma    sqrt(V + eps)
rstd     1 / sigma
xhat     normalized row, xhat_i = u_i / sigma
w, b     learnable scale and shift
y        output row, y_i = w_i * xhat_i + b_i
L        final scalar loss of the whole network
dy_i     dL/dy_i   (gradient handed in by the next layer)
dx_j     dL/dx_j   (gradient we must produce)
g_i      dy_i * w_i  (gradient w.r.t. xhat_i)
delta_ij 1 if i = j, else 0
```

---

## 1. What LayerNorm is and why it is used

### 1.1 Definition

For each row independently:

```
y = (x - E[x]) / sqrt(Var(x) + eps) * w + b
```

- Subtract the mean of the row, divide by its standard deviation.
- Then apply a learnable per-feature scale `w` and shift `b`.
- `eps` is a tiny constant so the denominator is never zero.

Because each row is independent, the Triton kernel uses **one program per row**.

### 1.2 How variance is computed

Variance is the average squared distance from the mean:

```
mu    = (1/N) * sum_k x_k          step 1: mean
u_k   = x_k - mu                   step 2: deviations
V     = (1/N) * sum_k u_k^2        step 3: average of squared deviations
sigma = sqrt(V + eps)              step 4: standard deviation
```

Squaring stops positive and negative deviations from cancelling (they always sum to zero otherwise). We divide by `N`, not `N-1`; LayerNorm uses the plain `N` version.

Worked example, `x = [1, 2, 3]`:

```
mu    = (1 + 2 + 3) / 3 = 2
u     = [-1, 0, 1]
u^2   = [1, 0, 1]
V     = 2 / 3 = 0.6667
sigma = sqrt(0.6667) = 0.8165
rstd  = 1.2247
xhat  = u * rstd = [-1.2247, 0, 1.2247]
```

### 1.3 BatchNorm vs LayerNorm

- **BatchNorm:** same feature, different examples. Statistics are taken across the batch, one set per feature.
- **LayerNorm:** one example, all features. Statistics are taken across the features of that single example.

Reasons for switching to LayerNorm (mainly for transformers and sequence models):

1. Small batches give noisy BatchNorm statistics.
2. BatchNorm behaves differently at train time (batch stats) and eval time (running averages). LayerNorm computes the same thing in both.
3. Variable-length, padded sequences mix real tokens with padding in batch statistics.
4. Autoregressive generation can have a batch of one, where BatchNorm has nothing sensible to compute.
5. In BatchNorm one example's output depends on the other examples in the batch. LayerNorm treats examples independently.
6. Distributed training: LayerNorm needs no cross-device statistics.

The switch is mostly about **independence from the batch**, not about LayerNorm being a better normalizer in the abstract.

### 1.4 The weakness: features on very different scales

If one feature in a row is huge, it dominates the row's mean and variance, and the small features get squashed together.

Example, `x = [100, 1, 2, 3]`:

```
mu     = (100 + 1 + 2 + 3) / 4 = 26.5
u      = [73.5, -25.5, -24.5, -23.5]
V      = (5402.25 + 650.25 + 600.25 + 552.25) / 4 = 7205 / 4 = 1801.25
sigma  = 42.44
xhat   = [1.732, -0.601, -0.577, -0.554]
```

The small features 1, 2, 3 became -0.601, -0.577, -0.554: nearly identical.

Same three small features without the big one:

```
x = [1, 2, 3]  ->  xhat = [-1.225, 0, 1.225]
```

| | gap between neighbouring small features |
|---|---|
| alone | 1.225 |
| next to the big feature | 0.0236 |

The differences shrank about 52x. Notes:

- The information is compressed, not destroyed: LayerNorm is a per-row shift and scale, so ordering and ratios are preserved.
- A fixed `w` can only partly compensate, because the compression depends on the row's `sigma`, which changes row to row.
- In practice transformer hidden dimensions usually have similar scales, and `w`, `b` plus training adapt. Large language models do develop a few outlier dimensions, which is one motivation for variants such as RMSNorm.
- BatchNorm normalizes each feature separately across the batch, so a huge feature does not dominate others. The price is batch dependence.

---

## 2. Forward kernel (`_layer_norm_fwd_fused`)

Each program handles one row and makes three passes over it in chunks of `BLOCK_SIZE`:

1. **Mean:** accumulate partial sums, `tl.sum`, divide by `N`.
2. **Variance:** accumulate `(x - mean)^2` (masked lanes set to 0), `tl.sum`, divide by `N`.
3. **Normalize:** `xhat = (x - mean) * rstd`, `y = xhat * w + b`, store.

Details:

- `rstd = 1/sigma` is computed once so the inner loop multiplies instead of divides.
- `mean` and `rstd` are stored (one float per row) because backward needs them.
- Loads are converted to float32 for the math, keeping reductions accurate for fp16 inputs.
- "Fused" means everything happens in one kernel launch, avoiding several memory round trips.

---

## 3. Backward pass: the big picture

### 3.1 What `dy` and `dx` mean

- `dy_i = dL/dy_i`: how much the loss changes if output `y_i` is nudged, **including everything downstream**. It is given to LayerNorm by the next layer. In the tutorial's test it is just made up: `dy = .1 * torch.randn_like(x)`.
- `dx_j = dL/dx_j`: how much the loss changes if input `x_j` is nudged. This is what we must produce.
- `dx_j` is **not** `dy/dx`. The local derivative `dy_i/dx_j` is a Jacobian entry; `dx_j` is a gradient of the loss.

### 3.2 How the flow works

Backprop moves one layer at a time:

```
forward:   x ───────> LayerNorm ───────> y ───────> next layer ───────> L
backward:  dx <─────  LayerNorm bwd  <── dy <────── next layer bwd <──── 1
```

1. Start at the loss: `dL/dL = 1`.
2. The next layer's backward step turns that into `dy`.
3. LayerNorm's backward step turns `dy` into `dx` (plus `dw`, `db`). This is where the sum over all outputs happens.
4. `dx` is passed to the previous layer, where it plays the role of that layer's `dy`.

Why one hop is enough: `dy_i` already summarizes the effect of `y_i` on the loss through all later layers. LayerNorm only needs the local question: how does `x_j` affect each `y_i`, and how much is each `y_i` worth to the loss.

Two levels of "summing":

1. Inside one layer, if an input feeds several outputs (as in LayerNorm), sum over those outputs.
2. Across layers, nothing is summed; the vector is just passed along.

### 3.3 The gradients we need

```
db_i = dy_i                       (per row)
dw_i = dy_i * xhat_i              (per row)
dx   = ?                          (the hard one)
```

---

## 4. Brute-force intuition: nudge and see

Take `x = [1, 2, 3]`, `w = 1`, `b = 0` (so `y = xhat`), and `dy = [1, 0, 2]`, meaning `L = 1*y_0 + 0*y_1 + 2*y_2`.

Before the nudge:

```
xhat = [-1.2247, 0, 1.2247]
L = 1*(-1.2247) + 2*(1.2247) = 1.2247
```

Nudge `x_0` from 1 to 1.01 and rerun the whole forward pass (new mean, new std):

```
x    = [1.01, 2, 3]
mu   = 2.00333,  V = 0.66002,  sigma = 0.8124
xhat ~ [-1.2227, -0.0041, 1.2268]
L    = 1*(-1.2227) + 2*(1.2268) = 1.2309
dx_0 ~ (1.2309 - 1.2247) / 0.01 ~ 0.61
```

All three outputs moved, not just `y_0`, because the mean and std changed. This is why several effects must be added up. The formula derived below gives `dx_0 = 0.6124`, matching.

Brute force needs one extra forward pass per input. The derivation below gives all `dx_j` at once.

---

## 5. Derivation of `dx`

### Step 1: through `y = w * xhat + b`

Each `y_i` depends only on `xhat_i`:

```
db_i  = dy_i
dw_i  = dy_i * xhat_i
g_i   = dy_i * w_i        (= dL/dxhat_i)
```

### Step 2: Jacobian of the centering `u = x - mu`

```
u_k = x_k - (1/N) * sum_m x_m
du_k/dx_j = delta_kj - 1/N
```

- If `k = j`: `1 - 1/N`.
- If `k != j`: `-1/N` (only the mean depends on `x_j`).

Matrix form, with `1` the all-ones vector:

```
C = I - (1/N) * 1 * 1^T
```

For `N = 3`:

```
C = [  2/3  -1/3  -1/3 ]
    [ -1/3   2/3  -1/3 ]
    [ -1/3  -1/3   2/3 ]
```

Why a Jacobian: `u` is a vector depending on a vector `x`, so its derivative is a matrix. A small change `dx` in the input changes `u` by `du = C dx`.

### Step 3: derivative of the variance

```
V = (1/N) * sum_k u_k^2
dV/dx_j = (1/N) * sum_k 2 u_k * du_k/dx_j
        = (2/N) * sum_k u_k * (delta_kj - 1/N)
```

Split the sum into two parts:

```
sum_k u_k * delta_kj     = u_j                    (only k = j survives)
sum_k u_k * (1/N)        = (1/N) * sum_k u_k = 0  (deviations sum to zero)
```

So:

```
dV/dx_j = (2/N) * u_j
```

Vector form: `grad V = (2/N) C^T u`, and since `C u = u` (centering an already centered vector changes nothing), `grad V = (2/N) u`.

Important note about indices: the factor `(delta_kj - 1/N)` is trapped **inside a sum over k**. After the sum it is gone, leaving only `j`. Variance is one number per row, so its derivative cannot carry an `i` index and cannot contain `delta_ij`.

Numeric check, `x = [1, 2, 3]`, `j = 0`:

```
Without simplifying:
(2/3) * [ u_0*(1 - 1/3) + u_1*(-1/3) + u_2*(-1/3) ]
= (2/3) * [ (-1)(0.6667) + 0 + (1)(-0.3333) ] = (2/3)(-1) = -0.667

Simplified: (2/N) u_0 = (2/3)(-1) = -0.667

Nudge test: x_0 -> 1.01 gives V = 0.66002 vs 0.66667, change -0.00667 for +0.01, slope -0.667.
```

### Step 4: derivative of sigma and 1/sigma

With `sigma = (V + eps)^(1/2)`:

```
dsigma/dx_j = (1 / (2 sigma)) * dV/dx_j = (1 / (2 sigma)) * (2/N) u_j = u_j / (N sigma)

d(1/sigma)/dx_j = -(1/2) (V+eps)^(-3/2) * dV/dx_j
                = -(1/2) (V+eps)^(-3/2) * (2/N) u_j
                = -u_j / (N sigma^3)
```

The `1/2` from the power rule cancels the `2` in `dV/dx_j`. (Not `2/N` in the final expression: it becomes `1/N`.)

### Step 5: Jacobian entry of xhat, by the product rule

Write `xhat_i = u_i * (V + eps)^(-1/2)`, a product of two things that both depend on `x_j`:

```
dxhat_i/dx_j = (du_i/dx_j) * (1/sigma)  +  u_i * d(1/sigma)/dx_j
                \___ term A ___/             \_______ term B _______/
```

- **Term A** (differentiate the numerator): `(delta_ij - 1/N) / sigma`.
- **Term B** (differentiate the denominator): `u_i * ( -u_j / (N sigma^3) ) = -u_i u_j / (N sigma^3)`.

Combine, pull out `1/sigma`, and use `u/sigma = xhat`:

```
dxhat_i/dx_j = (1/sigma) * [ delta_ij  -  1/N  -  xhat_i * xhat_j / N ]
```

Common mistakes to avoid:

1. The denominator is the **standard deviation** `sigma = sqrt(V + eps)`, not the variance `V`.
2. The bracket is a **sum** of separate terms, not a product of `(delta_ij - 1/N)` with something.
3. Term B has **no** `(delta_ij - 1/N)` factor. That factor came from differentiating the numerator; in term B the numerator `u_i` is left untouched and only multiplied by the derivative of the denominator. The `i` dependence in term B comes solely from `u_i`.
4. Term B needs a **product** `u_i * u_j`: one from the numerator `u_i`, one from `dV/dx_j`.

Including the scale `w`: `dy_i/dx_j = w_i * dxhat_i/dx_j`.

Matrix form: `J = (1/sigma) * ( C - xhat xhat^T / N )`. It is symmetric, so `J^T = J`.

Numeric check, `x = [1, 2, 3]`, `i = j = 0`:

```
(1/0.8165) * [ (1 - 1/3) - (-1)(-1) / (3 * 0.6667) ]
= 1.2247 * [ 0.6667 - 0.5 ] = 0.2041
```

Full Jacobian for `x = [1, 2, 3]`:

```
J = [  0.2041  -0.4082   0.2041 ]
    [ -0.4082   0.8165  -0.4082 ]
    [  0.2041  -0.4082   0.2041 ]
```

Every row is nonzero: nudging one input affects every output.

### Step 6: the chain rule sum

`L` sees `x_j` only through the outputs `y_1 ... y_N`, and `x_j` affects all of them. There are N paths from `x_j` to `L`:

```
         ┌→ y_1 ─┐
x_j ─────┼→ y_2 ─┼→ L
         └→ y_N ─┘
```

The total effect is the sum over paths, each being (effect of `x_j` on `y_i`) times (effect of `y_i` on `L`):

```
dL/dx_j = sum_i (dL/dy_i) * (dy_i/dx_j)
dx_j    = sum_i  dy_i     *  dy_i/dx_j
```

Small sanity example: `L = 3 y_1 + 5 y_2`, `y_1 = 2x`, `y_2 = 4x`. Then `dy = [3, 5]`, Jacobian entries `[2, 4]`, and `dx = 3*2 + 5*4 = 26`. Dropping the second path would give 6, which is wrong. (The plain `dy/dx` would be `[2, 4]`.)

In matrix form this is `dx = J^T dy`, the **vector-Jacobian product (VJP)**. The backward pass never builds the N x N matrix `J`; it only computes this product.

### Step 7: plug in and simplify

Substitute the Jacobian entry, with `g_i = dy_i * w_i`:

```
dx_j = sum_i g_i * (1/sigma) * [ delta_ij - 1/N - xhat_i xhat_j / N ]
```

`1/sigma` has no `i` index, so it leaves the sum. Split into three sums:

```
dx_j = (1/sigma) * [ sum_i g_i delta_ij   -   (1/N) sum_i g_i   -   xhat_j * (1/N) sum_i g_i xhat_i ]
```

| Term | Result | Why |
|---|---|---|
| `sum_i g_i delta_ij` | `g_j` | `delta_ij` is 1 only for `i = j`, so one term survives |
| `(1/N) sum_i g_i` | `c2` | average of `g` over the row |
| `xhat_j * (1/N) sum_i g_i xhat_i` | `xhat_j * c1` | `xhat_j` has no `i`, so it leaves the sum |

Define:

```
c1 = (1/N) * sum_i g_i * xhat_i   = mean(g * xhat)
c2 = (1/N) * sum_i g_i            = mean(g)
```

**Final result:**

```
dx_j = rstd * ( g_j - c2 - xhat_j * c1 )
```

The same thing in vector form:

```
dx = (1/sigma) * ( C g  -  xhat * (xhat . g) / N )
```

where `C g = g - mean(g)`.

### Step 8: interpretation and sanity checks

- `g_j` is the gradient you would have without normalization.
- `- c2` removes the component that just shifts all features equally (the mean subtraction cancels it).
- `- xhat_j * c1` removes the component along the direction that only rescales the row (the std division cancels it).

So the normalization projects the gradient onto the directions that actually change the output. Two checks follow:

```
sum(dx)     = 0     (shifting all inputs equally changes nothing)
dx . xhat   = 0     (scaling the centered row changes nothing)
```

---

## 6. Numeric examples of the final formula

### 6.1 `x = [1, 2, 3]`, `w = 1`, `dy = [1, 0, 0]`

```
g = [1, 0, 0]
c2 = 1/3 = 0.3333
c1 = mean(g * xhat) = (-1.2247)/3 = -0.4082
rstd = 1.2247

dx_0 = 1.2247 * (1 - 0.3333 - (-1.2247)(-0.4082)) = 1.2247 * (1 - 0.3333 - 0.5) = 0.2041
dx_1 = 1.2247 * (0 - 0.3333 - 0)                                          = -0.4082
dx_2 = 1.2247 * (0 - 0.3333 - (1.2247)(-0.4082))  = 1.2247 * (-0.3333 + 0.5) = 0.2041

dx = [0.2041, -0.4082, 0.2041]
sum(dx) = 0, dx . xhat = 0
```

### 6.2 `x = [1, 2, 3]`, `w = 1`, `dy = [1, 0, 2]`

```
g = [1, 0, 2]
c2 = (1 + 0 + 2)/3 = 1
c1 = (1*(-1.2247) + 0 + 2*(1.2247))/3 = 0.4082

dx_0 = 1.2247 * (1 - 1 - (-1.2247)(0.4082)) = 1.2247 * 0.5   =  0.6124
dx_1 = 1.2247 * (0 - 1 - 0)                                  = -1.2247
dx_2 = 1.2247 * (2 - 1 - (1.2247)(0.4082))  = 1.2247 * 0.5   =  0.6124
```

### 6.3 The same answer through the full Jacobian

Using `J` from Step 5 and `dx_j = sum_i dy_i * J_ij`:

```
dx_0 = 1*(0.2041)  + 0*(-0.4082) + 2*(0.2041)  =  0.6124
dx_1 = 1*(-0.4082) + 0*(0.8165)  + 2*(-0.4082) = -1.2247
dx_2 = 1*(0.2041)  + 0*(-0.4082) + 2*(0.2041)  =  0.6124
```

Identical to 6.2 and to the nudge test in section 4 (about 0.61). The shortcut works because `J` has simple structure (identity part, constant part, rank-one part `xhat xhat^T`), so multiplying by it reduces to a few dot products. Building the full matrix would cost `N x N` numbers per row (for `N = 8192`, about 67 million), while the shortcut touches `N` entries a few times.

---

## 7. `dw` and `db`: the parallel reduction

### 7.1 The problem

Per row, `dw = dy * xhat` and `db = dy` are trivial. But `w` and `b` are shared by all `M` rows, so the true gradients are sums over rows:

```
dw = sum over rows of (dy_row * xhat_row)      (length N)
db = sum over rows of dy_row                   (length N)
```

Thousands of programs adding into the same N-length vector would serialize on it or race.

### 7.2 Two-stage reduction in the tutorial

**Stage 1 (`_layer_norm_bwd_dx_fused`):** allocate `GROUP_SIZE_M` partial buffers (heuristically 64 to 256). Row `r` accumulates into buffer `r % GROUP_SIZE_M`. Rows sharing a buffer take turns with a lock:

```python
while tl.atomic_cas(Lock, 0, 1) == 1:   # spin until lock acquired
    pass
... read old partial, add ours, write back ...
tl.debug_barrier()
tl.atomic_xchg(Lock, 0)                 # release
```

`atomic_cas` is compare-and-swap: set the lock to 1 only if it was 0, and report what it saw. Seeing 1 means someone else holds it, so spin. Contention is spread over `GROUP_SIZE_M` locks instead of one. A `Count` flag makes the first write to a buffer a plain store; later writes load the old partial and add.

**Stage 2 (`_layer_norm_bwd_dwdb`):** a second small kernel sums the partial buffers into the final `dw`, `db`. Each program owns a slice of columns and loops over the partial rows, so no locks are needed.

Example, `M = 4`, `GROUP_SIZE_M = 2`:

```
rows 0 and 2  ->  buffer 0
rows 1 and 3  ->  buffer 1
stage 2: dw = buffer0 + buffer1
```

The partial buffers stay in L2 cache, which keeps the accumulation fast.

### 7.3 Limits and benchmark

- The whole row must fit in one block (`BLOCK_SIZE >= N`), with a 64KB cap per feature; larger `N` raises an error.
- Benchmark (tutorial, `M = 4096`, fp16, backward): Triton is slower than PyTorch at small `N` (about 107 vs 372 GB/s at `N = 1024`) and faster at large `N` (about 940 vs 590 GB/s at `N = 15872`), with a crossover near `N` of 5000. The likely reason is fixed overhead from the lock-based reduction and the second kernel (my inference; the tutorial does not explain it).

---

## 8. Math to kernel mapping

| Math | Triton line |
|---|---|
| `xhat` | `xhat = (x - mean) * rstd` |
| `g = dy * w` | `wdy = w * dy` |
| `c1` | `c1 = tl.sum(xhat * wdy, axis=0) / N` |
| `c2` | `c2 = tl.sum(wdy, axis=0) / N` |
| `dx` | `dx = (wdy - (xhat * c1 + c2)) * rstd` |
| row's `dw` | `partial_dw = (dy * xhat)` |
| row's `db` | `partial_db = dy` |

The forward pass stores `mean` and `rstd` (one float per row) so backward does not need another pass over the row to recompute them.

---

## 9. Key takeaways

1. LayerNorm normalizes across the features of one example, so it is independent of the batch. That is why transformers use it.
2. A very large feature in a row compresses the others (about 52x in our example); this is the main weakness.
3. `dy` is handed in by the next layer; `dx` is the gradient of the **loss** w.r.t. `x`, not `dy/dx`.
4. Every `x_j` affects every `y_i`, so `dx_j` sums over all outputs: `dx_j = sum_i dy_i * dy_i/dx_j`.
5. The Jacobian entry is `(w_i/sigma) * [delta_ij - 1/N - xhat_i xhat_j / N]`; the last term is a product `u_i u_j` because the variance derivative collapses to `(2/N) u_j`.
6. Summing against that entry collapses to two row-wide averages, giving `dx_j = rstd * (g_j - c2 - xhat_j * c1)`.
7. `dw` and `db` are sums over all rows, computed with partial buffers plus locks, then a second reduction kernel.
