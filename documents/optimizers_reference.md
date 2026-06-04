# Optimizer Implementations — Algorithm Reference

A technical reference for the five from-scratch first-order optimizers used in Sections 4 and 7 of the notebook. Each section is written to be **plug-and-play**: any subsection here can be lifted into a Markdown cell in the notebook, or composed into the final report's "Methods" chapter. The math, the implementation, the design decisions, and the connection to the empirical results in §7 are all kept together so the reader can trace from theory to code to measurement in one pass.

> *Scope*: this document covers only the optimizer methods and learning-rate schedules. The loss functions (§5) and improvements ablations (§9) live in their own references.

---

## 0. Why implement these from scratch?

PyTorch ships production-grade implementations of all five optimizers in `torch.optim`, and the practical answer to "why not call them" is that calling them would have made this project an *application* of optimization algorithms rather than a *study* of them. The project assessment criterion explicitly rewards "the capacity to think in terms of optimization applied to AI and not merely to apply algorithms," and a from-scratch implementation does three things that opaque calls to `torch.optim.Adam` cannot:

1. **It exposes the design surface.** Each method's per-parameter state, its update rule, and its numerical-stability tricks (Adam's bias correction, AdamW's decoupled decay) appear directly in the code rather than being hidden inside a kernel. Choosing $\eta$, $\beta_1$, $\beta_2$, $\epsilon$, and $\mu$ becomes a deliberate act, and the consequences of those choices can be reasoned about line by line.
2. **It guarantees a fair comparison.** All five methods share an identical interface (`zero_grad()` / `step()`) and identical numerical conventions: same float precision, same in-place operations, same `@torch.no_grad()` boundary. Any difference observed in §7 is therefore attributable to the algorithm itself, not to engineering differences between `torch.optim` kernels.
3. **It exposes the substitution Adam's $v$ makes for second-order curvature** — visible in code as a single `addcmul_(g, g, ...)` line, but understood as a *diagonal Hessian surrogate*. This is the conceptual bridge from the second-order theory of Module 1 (Hessian eigenvalues, Sylvester tests) to the first-order practice used at this problem scale; see §1 below.

---

## 1. The first-order trade-off

Module 1 of the course classifies critical points of an objective $\mathcal{L}: \mathbb{R}^n \to \mathbb{R}$ by inspecting the Hessian $\nabla^2 \mathcal{L}$ at $\nabla \mathcal{L} = 0$: positive-definite ⇒ local minimum, negative-definite ⇒ maximum, indefinite ⇒ saddle. Newton's method, the canonical second-order optimizer, uses the inverse Hessian to step toward the critical point in a single, quadratic-convergence iteration.

For the NeRF MLP used here, $n \approx 58{,}000$ parameters; the dense Hessian would be a $58000 \times 58000$ matrix of $\sim 3.4 \times 10^9$ float32 entries, requiring $\sim 13$ GB of memory just to store, and many TFLOPs per iteration to invert. Forming it, let alone inverting it, is intractable. The first-order methods of this section are the field's response to that intractability: they discard the explicit Hessian and approximate the curvature information in two cheaper ways.

- **Momentum-class methods** (§§ 2.2, 2.3) approximate it *implicitly* by averaging consecutive gradients: the velocity vector $v$ accumulates a low-pass-filtered descent direction, which behaves as if the optimizer had taken a small step into the curvature and re-evaluated.
- **Adaptive methods** (§§ 2.4, 2.5) approximate it *diagonally*: Adam's second moment $\hat v$ is an exponentially-weighted average of squared gradients per parameter, used to normalise each parameter's step. This is precisely a diagonal proxy for $|\partial^2 \mathcal{L} / \partial \theta_i^2|^{1/2}$, recovered from gradient statistics alone.

The empirical comparison in §7 therefore quantifies what is gained and lost by each of these substitutions on a real non-convex deep-learning objective.

---

## 2. The five optimizers

### 2.1 Stochastic Gradient Descent (plain SGD)

**Update rule.**
$$
\theta_{t+1} = \theta_t - \eta\, g_t, \quad g_t = \nabla_\theta \mathcal{L}(\theta_t)
$$

**State.** None per parameter (`MySGD` allocates no buffers when `momentum=0`).

**Convergence properties.** For $\mathcal{L}$ that is $L$-smooth and $\mu$-strongly convex, with step size $\eta = 1/L$, SGD converges to the minimiser at rate $\mathcal{O}(\exp(-\mu/L \cdot t))$ — *linear* in iteration count, but with the constant degraded by the condition number $L/\mu$. Real neural-network objectives are not convex, but the local picture near a stationary point is dominated by the Hessian's largest and smallest eigenvalues, which behave like $L$ and $\mu$ — meaning ill-conditioning directly slows plain SGD.

**Properties this exposes.** SGD has a single global step size $\eta$ and no curvature information at all, which makes it the most exposed to ill-conditioning of the five: directions of high curvature force a small $\eta$, which then makes progress along low-curvature directions slow. The NeRF MLP, with its 10-frequency positional encoding multiplying input scales by orders of magnitude, is moderately ill-conditioned, so SGD is the most LR-sensitive method we test. This is what the §7.1 LR sweep visualises as a sharp single-LR sweet spot for SGD versus Adam's flat plateau.

**Implementation note (`MySGD`, cell 21).** The class implements plain SGD as the `momentum=0` branch of the same code path that handles momentum and Nesterov — selected at construction time. This eliminates a class of "wait, are these really being compared at parity?" bugs by collapsing the three methods to one tested execution path.

### 2.2 SGD with classical momentum

**Update rule.**
$$
v_{t+1} = \mu\, v_t + g_t, \quad
\theta_{t+1} = \theta_t - \eta\, v_{t+1}
$$
with $\mu \in [0, 1)$ — here $\mu = 0.9$, the de-facto standard.

**State.** A velocity buffer $v \in \mathbb{R}^n$ initialised to zero. One extra tensor of the same shape as $\theta$.

**What it does.** $v$ is an exponentially-weighted running sum of past gradients; the effective step is no longer along the instantaneous gradient but along a smoothed direction. The standard intuition is the ball rolling down a hill: classical SGD reacts only to the local slope; momentum builds up speed along consistent descent directions and damps oscillation across high-curvature directions where consecutive gradients alternate sign.

**Convergence properties.** On a strongly convex quadratic, Polyak (1964) shows momentum achieves the *accelerated* rate $\mathcal{O}(\exp(-\sqrt{\mu/L} \cdot t))$ — the square-root of the SGD rate, which is a strict improvement when $L/\mu$ is large (i.e., when the problem is ill-conditioned). The qualitative effect we expect on NeRF is faster convergence than plain SGD at the same $\eta$, with smoother loss curves.

**Implementation note.** The same `MySGD` class implements this as `momentum > 0, nesterov=False`. The velocity update is done in place via `v.mul_(self.mu).add_(g)`, which avoids allocating a new tensor every step — a tiny but real efficiency benefit for the inner training loop.

### 2.3 SGD with Nesterov accelerated gradient

**Update rule** (PyTorch-style form used here).
$$
v_{t+1} = \mu\, v_t + g_t, \quad
\theta_{t+1} = \theta_t - \eta\,(g_t + \mu\, v_{t+1})
$$

**State.** Same as classical momentum — one velocity buffer.

**What it does.** Nesterov's accelerated gradient (NAG; Nesterov 1983) modifies the step by adding a "look-ahead" term $\mu v_{t+1}$ to the gradient before applying it. Geometrically: instead of evaluating the gradient at $\theta_t$ and stepping by $v_{t+1}$, NAG evaluates as if the parameter had *already* been moved by the momentum component, and applies the correction. This is equivalent to measuring the gradient at the "where momentum is taking you" point rather than where you are.

**Convergence properties.** For convex $L$-smooth $\mathcal{L}$, NAG achieves the optimal first-order rate $\mathcal{O}(L/t^2)$ — provably the best any first-order method can achieve on this function class (Nesterov 1983). For *strongly* convex problems, NAG matches momentum's accelerated rate $\mathcal{O}(\exp(-\sqrt{\mu/L} \cdot t))$ but with a tighter constant. Empirically on neural networks the gain over classical momentum is small but consistent.

**Implementation note.** Implemented as the `nesterov=True` branch of `MySGD`. The one-line difference from classical momentum — `p.add_(g + self.mu * v if self.nesterov else v, alpha=-self.lr)` — keeps the comparison hermetic.

### 2.4 Adam

**Update rule.** For each parameter $\theta_i$:
$$
\begin{aligned}
m_{t+1} &= \beta_1\, m_t + (1 - \beta_1)\, g_t \\
v_{t+1} &= \beta_2\, v_t + (1 - \beta_2)\, g_t^2 \\
\hat m_{t+1} &= m_{t+1} / (1 - \beta_1^{t+1}) \\
\hat v_{t+1} &= v_{t+1} / (1 - \beta_2^{t+1}) \\
\theta_{t+1} &= \theta_t - \eta\, \hat m_{t+1} / (\sqrt{\hat v_{t+1}} + \epsilon)
\end{aligned}
$$
with defaults $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$.

**State.** Two buffers per parameter: first moment $m$ (a smoothed gradient, like momentum), and second moment $v$ (a smoothed squared gradient). Plus an iteration counter $t$ for the bias correction.

**What it does — two effects.**

1. **Per-parameter adaptive step.** The update divides $\hat m$ by $\sqrt{\hat v}$, giving each parameter its own effective step size: large where gradients are small and consistent (low $\hat v$), small where gradients are large or noisy (high $\hat v$). This makes Adam robust to vastly different parameter scales, which is exactly the regime the NeRF MLP produces — positional-encoding outputs span several orders of magnitude before the MLP weights, but the loss landscape is roughly self-similar across them, so Adam's per-coordinate normalisation pays off heavily.

2. **Bias correction.** Both $m_0$ and $v_0$ are initialised to zero, which biases the *moving averages* toward zero in the early iterations — they have not yet "warmed up." The terms $1 - \beta_1^{t+1}$ and $1 - \beta_2^{t+1}$ undo this bias exactly, so the very first steps are not undamped. Without bias correction, Adam takes vanishingly small first steps and "learns slowly" for the first ~$1/(1-\beta_2) = 1000$ iterations. This is a numerical-stability trick, not a method choice — but it is invisible in `torch.optim` and worth highlighting here.

**Convergence properties.** Adam was originally proven to converge by Kingma & Ba (2015), but Reddi et al. (2018) later showed the proof had a flaw and constructed counter-examples on which Adam diverges. Empirically Adam *does* converge on neural-network objectives and is the default choice for most deep-learning practitioners. The lack of a strong theoretical convergence guarantee on non-convex objectives is a known caveat; we accept it because the empirical record on this problem class is overwhelming. AMSGrad (Reddi et al. 2018) is a fix that we considered but did not implement, as it is rarely needed in practice.

**Connection to the Hessian framing of §1.** Adam's $v$ is an exponentially-weighted average of $g^2$; if the gradient is $g = \nabla \mathcal{L}$, then $\mathbb{E}[g^2]$ is the *diagonal of the Fisher information matrix*, which for a quadratic objective coincides with the diagonal of the Hessian. Adam's per-parameter step $\eta / \sqrt{\hat v}$ is therefore a *diagonal Newton step*, with $1/\sqrt{\hat v}$ playing the role of the diagonal inverse-Hessian. This is the precise sense in which Adam approximates the curvature information that Newton's method would use explicitly.

**Implementation note (`MyAdam`, cell 25).** The update uses in-place ops: `m.mul_(self.b1).add_(g, alpha=1-self.b1)` for $m$, `v.mul_(self.b2).addcmul_(g, g, value=1-self.b2)` for $v$, and `p.addcdiv_(m/bc1, v.div(bc2).sqrt().add_(self.eps), value=-self.lr)` for the parameter update. The bias correction divisors `bc1, bc2` are scalars computed once per step. The class does not implement AMSGrad or amsgrad-style enhancements.

### 2.5 AdamW

**Update rule.** Identical to Adam, with one additional term applied to the parameter directly:
$$
\theta_{t+1} = \theta_t - \eta\, \hat m_{t+1} / (\sqrt{\hat v_{t+1}} + \epsilon) - \eta\, \lambda\, \theta_t
$$
with $\lambda = 10^{-2}$ as the weight-decay coefficient.

**State.** Same as Adam plus the single scalar $\lambda$.

**What it does, and why it differs from "L2 regularisation."** Plain Adam with L2 regularisation would add $\lambda \theta$ to the gradient *before* the Adam normalisation step, so the weight-decay term would itself be divided by $\sqrt{\hat v}$ and become parameter-specific in magnitude. AdamW (Loshchilov & Hutter 2017) decouples the decay: it is applied directly to $\theta$, separately from the adaptive-gradient step, so the regularisation strength is uniform across parameters and independent of the gradient statistics. This is the behaviour usually intended by "weight decay."

**When does AdamW differ from Adam meaningfully?** When the loss already penalises parameter magnitudes through some other mechanism (architectural — e.g., batch normalisation — or through the loss term), AdamW and Adam at the same $\eta$ tend to behave similarly. The NeRF MLP has no batch norm and a small parameter count, and the chosen $\lambda = 10^{-2}$ is light, so we expect AdamW to track Adam closely in our §7 comparison. The comparison is included to *test* this prediction empirically and to honour the project's promise to characterise five methods.

**Implementation note (`MyAdamW`, cell 27).** Identical to `MyAdam` plus a single trailing line: `p.add_(p, alpha=-self.lr * self.wd)`. The decoupling is therefore visible as a one-line patch rather than an architectural fork.

---

## 3. Learning-rate schedules

The five optimizers above take a fixed base learning rate $\eta$. A *schedule* modulates $\eta$ over training, on top of any optimizer. Two schedules are implemented for use in §7 and §9.

### 3.1 Cosine annealing with linear warmup

$$
\eta_t =
\begin{cases}
  \eta_0 \cdot t / W & \text{if } t < W \quad \text{(warmup phase)}\\
  \eta_0 \cdot \tfrac{1}{2}\bigl(1 + \cos\bigl(\pi \, (t - W) / (T - W)\bigr)\bigr) & \text{otherwise (anneal phase)}
\end{cases}
$$
with $T$ = total iterations, $W$ = warmup steps.

**Why each phase.** The *warmup* ramps the rate from zero over the first $W$ steps so that the very first update — taken while Adam's moment estimates are still cold (before bias correction has fully kicked in) and while the random initialisation is far from any reasonable basin — does not destroy the model. The *anneal* eases $\eta$ smoothly to zero so the final iterations *settle* into the minimum rather than bouncing around it; this is what removes the late-training oscillation seen with fixed $\eta$.

**Implementation note (`cosine_warmup_lr`, cell 29).** Pure-function: `cosine_warmup_lr(step, base_lr, warmup_steps, total_steps) -> float`. The training loop calls it once per iteration and writes the result to `opt.lr` *before* `opt.step()`, so the schedule is applied uniformly regardless of which optimizer is in use.

### 3.2 SGDR — Stochastic Gradient Descent with Warm Restarts

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min})\bigl(1 + \cos(\pi \cdot \text{in-cycle progress})\bigr)
$$
with cycle $i$ of length $T_0 \cdot T_{\text{mult}}^i$.

**What it does.** SGDR (Loshchilov & Hutter 2017) replaces single-cycle cosine annealing with *cyclic* annealing: the rate decays toward $\eta_{\min}$, then jumps back to $\eta_0$ at the end of each cycle and decays again. Each cycle is longer than the last by a factor of $T_{\text{mult}}$ (here 2.0), so early restarts explore aggressively and later cycles refine.

**Why it might help a non-convex objective.** The hypothesis is that the warm restart "shakes" the parameters out of sharp local minima that fixed-schedule training can get stuck in. Empirically this is sometimes a strong improvement (the original paper showed it on CIFAR), sometimes a no-op (when training has already found a wide minimum). §9 tests this on our NeRF objective as Improvement A.

**Implementation note (`sgdr_lr`, cell 29).** Pure function. The training loop selects between `cosine_warmup_lr` and `sgdr_lr` based on `cfg.use_sgdr` and `cfg.use_schedule` flags.

---

## 4. Summary comparison

| Method | State per param | Update rule (sketch) | Best-case rate (convex) | Sensitivity to $\eta$ |
|---|---|---|---|---|
| **SGD** | none | $\theta \leftarrow \theta - \eta g$ | $\mathcal{O}(e^{-\mu t / L})$ | high |
| **Momentum** | velocity $v$ | $v \leftarrow \mu v + g; \theta \leftarrow \theta - \eta v$ | $\mathcal{O}(e^{-\sqrt{\mu / L} \cdot t})$ | medium |
| **Nesterov** | velocity $v$ | $v \leftarrow \mu v + g; \theta \leftarrow \theta - \eta(g + \mu v)$ | $\mathcal{O}(L / t^2)$ — optimal first-order | medium |
| **Adam** | $m, v$ + iter $t$ | $\theta \leftarrow \theta - \eta \hat m / (\sqrt{\hat v} + \epsilon)$ | none proven for non-convex; empirical | low |
| **AdamW** | $m, v$ + iter $t$ | Adam + decoupled $-\eta\lambda\theta$ | as Adam | low |

The §7.1 LR sweep makes the rightmost column visible directly: SGD has a sharp single-LR sweet spot, momentum and Nesterov a slightly broader one, Adam and AdamW a flat plateau across two orders of magnitude.

---

## 5. Connecting the implementation to the §7 results

The empirical results in §7 are meant to *test the predictions* this reference section sets up. The two that matter most for the optimization-thinking narrative:

1. **"SGD is the most LR-sensitive."** §7.1 confirms this: SGD's best-val-PSNR vs. LR curve is a sharp peak at a single LR; Adam's is flat across two orders of magnitude. This is a direct consequence of §2.1 (no per-parameter scaling) and §2.4 (diagonal curvature surrogate).
2. **"At per-method best LRs, the methods are much closer than at shared LR."** §7.2 confirms this dramatically: the SGD-vs-Adam test-PSNR gap shrinks from **12.77 dB** (shared LR) to **1.82 dB** (per-method LR). This proves the §7.0 finding — *the shared-LR comparison of the original poc was a learning-rate-mismatch artefact, not a property of the methods themselves*. This is the project's headline optimization-thinking moment: a fair comparison of optimizers requires per-method LR tuning, because what looks like a bad optimizer is often a mis-configured one.

§8.3 takes the same logic one step further by *automating* the per-method LR tuning with Bayesian optimization (TPE + median pruner), making the comparison fair not only in the limit of perfect manual tuning but in the practical regime where compute for per-method sweeps is limited.

---

## References

- Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, 4(5).
- Nesterov, Y. (1983). A method of solving a convex programming problem with convergence rate $O(1/k^2)$. *Soviet Mathematics Doklady*, 27.
- Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *ICLR*.
- Loshchilov, I. & Hutter, F. (2017). Decoupled weight decay regularization. *arXiv:1711.05101* / *ICLR 2019*.
- Reddi, S. J., Kale, S., & Kumar, S. (2018). On the convergence of Adam and beyond. *ICLR*.
