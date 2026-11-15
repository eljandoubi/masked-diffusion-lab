# Derivations

Notation. A sequence is $x \in [V]^L$ with $[V] = \{0, \dots, V-1\}$ and $\mathsf{m}$ is the mask symbol. The forward process keeps token $i$ with probability $\alpha_t = 1 - t$ and replaces it by $\mathsf{m}$ otherwise, independently across positions, for $t \in [0, 1]$. For a set $U$ of known positions, $c_i(y \mid x_U) = p(x_i = y \mid x_U)$ is the exact conditional of a masked position $i \notin U$. A denoiser returns conditionals $q_i(y \mid x_t)$ for the masked positions of a partial sequence $x_t$.

## 1. The loss

Take a reverse step from $t$ to $s < t$ and write $r = (\alpha_s - \alpha_t)/(1 - \alpha_t) = (t - s)/t$. Given $x$, the true step reveals each masked position with probability $r$ (to the true token $x^i$) and copies known positions. The model step is the same, except that a revealed token is drawn from $q_i(\cdot \mid x_t)$. At a masked position the two kernels are $(1-r)\,\delta_{\mathsf{m}} + r\,\delta_{x^i}$ and $(1-r)\,\delta_{\mathsf{m}} + r\,q_i$, so

$$\mathrm{KL}(\text{true step} \,\|\, \text{model step}) = r \sum_{i:\, x_t^i = \mathsf{m}} -\log q_i(x^i \mid x_t).$$

Summing these terms over a time grid, with $x_t$ drawn from the forward process, and refining the grid gives the negative ELBO. The prior term vanishes because both chains start from the all-masked sequence. With $-\alpha_t'/(1-\alpha_t) = 1/t$ for the linear schedule,

$$-\mathrm{ELBO}(x) = \int_0^1 \frac{1}{t}\, \mathbb{E}_{x_t \mid x}\Big[\sum_{i:\, x_t^i = \mathsf{m}} -\log q_i(x^i \mid x_t)\Big]\, dt. \tag{1}$$

A pattern $M$ of masked positions has probability $t^{|M|}(1-t)^{L-|M|}$ and $\int_0^1 t^{|M|-1}(1-t)^{L-|M|}\, dt = (|M|-1)!\,(L-|M|)!/L!$, so the integral is a finite sum:

$$-\mathrm{ELBO}(x) = \sum_{M \neq \emptyset} \frac{(|M|-1)!\,(L-|M|)!}{L!} \sum_{i \in M} -\log q_i(x^i \mid x_{M^c}). \tag{2}$$

The weight of a pair $(i, U)$ with $U = M^c$ is $u!\,(L-1-u)!/L!$ for $u = |U|$. This is the probability that a uniformly random ordering of the positions puts exactly the set $U$ before $i$. So (2) is the chain-rule negative log-likelihood averaged over all orderings. With the exact conditionals every ordering gives $-\log p(x)$, hence $-\mathrm{ELBO}(x) = -\log p(x)$ for every $x$, not only on average.

Checked in `tests/test_elbo.py`: the closed form against the pattern sum integrated over $t$ with Gauss-Legendre nodes, against the average over all orderings (for conditionals that do not come from any joint law), and the tightness for exact conditionals on every sequence of the Markov and Potts laws.

## 2. The minimiser

Take $x \sim p$ and group the terms of (2) by the pair $(i, U)$:

$$\mathbb{E}_{x \sim p}\big[-\mathrm{ELBO}(x)\big] = \sum_i \sum_{U \not\ni i} w_{|U|}\; \mathbb{E}_{x_U}\Big[\mathrm{CE}\big(c_i(\cdot \mid x_U),\, q_i(\cdot \mid x_U)\big)\Big], \qquad w_u = \frac{u!\,(L-1-u)!}{L!} > 0.$$

Each cross-entropy is $H(c_i) + \mathrm{KL}(c_i \,\|\, q_i)$, and it can be minimised separately for every pair and every value $x_U$. By Gibbs' inequality the minimiser is $q_i(\cdot \mid x_U) = c_i(\cdot \mid x_U)$. The minimum is $H(p)$, the chain rule averaged over orderings, and in general

$$\mathbb{E}_{x \sim p}\big[-\mathrm{ELBO}(x)\big] = H(p) + \underbrace{\sum_i \sum_{U \not\ni i} w_{|U|}\; \mathbb{E}_{x_U}\, \mathrm{KL}\big(c_i(\cdot \mid x_U) \,\|\, q_i(\cdot \mid x_U)\big)}_{\text{ELBO gap}}. \tag{3}$$

Any positive weighting of the pairs has the same minimiser, so the clipped weight used in training (Part 3) does not move the optimum, only the trade-off when the model is too small.

**Training on the exact expectation.** Group the terms of (2) by the partial sequence $s$ that a mask pattern produces, with masked set $M(s)$ and known part $x_U$. For any law $\pi$ over sequences, for example the empirical law of $n$ training sequences,

$$\mathbb{E}_{x \sim \pi}\big[-\mathrm{ELBO}(x)\big] = -\sum_{s} w_{|M(s)|} \sum_{i \in M(s)} \sum_{y} \pi(x_U,\, x_i = y)\, \log q_i(y \mid s).$$

This is a weighted cross-entropy on the $(V+1)^L - V^L$ partial sequences that have at least one mask. With the weight $1/\max(t, \text{floor})$ in place of $1/t$, only the numbers $w_m$ change (`pattern_weights(L, floor)`). Training can therefore minimise the expectation over $t$ and the mask pattern exactly instead of sampling them. Normalising the weights over $y$ gives the minimiser: the exact conditionals when $\pi = p$ and the plain conditional frequencies of the training sequences when $\pi$ is empirical (undefined where a context never occurs). `elbo.loss_targets` builds the weights, and `tests/test_elbo.py` checks that they reproduce the expected loss and that they normalise to the exact conditionals.

The error measure of Part 3 is the conditional KL: draw $t$ uniformly, draw the mask pattern from the forward process, and average the KL of (3) over the masked positions of the pattern (zero if none is masked). Since $\int_0^1 t^m (1-t)^{L-m}\, dt \,/\, m = (m-1)!\,(L-m)!/(L+1)!$, it equals the ELBO gap divided by $L + 1$. Both are computed exactly by summing over all $(V+1)^L$ partial sequences (`Reference.conditional_kl`), and `tests/test_reference.py` checks them against the expected loss minus the entropy and against an explicit sum over $t$ and mask patterns.

## 3. The K-step sampler

Take the uniform grid $\alpha_{t_k} = k/K$. At step $k = 0, \dots, K-1$ each still-masked position is revealed with probability

$$r_k = \frac{\alpha_{t_{k+1}} - \alpha_{t_k}}{1 - \alpha_{t_k}} = \frac{1}{K - k}.$$

A position survives steps $0, \dots, k-1$ with probability $\prod_{j<k}(1 - r_j) = (K-k)/K$, so its reveal step $\tau_i$ satisfies $P(\tau_i = k) = \frac{K-k}{K} \cdot \frac{1}{K-k} = 1/K$: the reveal steps are independent and uniform on $\{0, \dots, K-1\}$. Positions revealed in the same step draw their tokens independently, each from its exact conditional given the partial sequence at the start of the step. With $U(i) = \{j : \tau_j < \tau_i\}$ the output law is

$$q_K(x) = \mathbb{E}_\tau \prod_{i=1}^{L} c_i\big(x_i \mid x_{U(i)}\big). \tag{4}$$

`sampler_law` computes this by dynamic programming over the $(V+1)^L$ partial sequences. The tests compare it with the explicit sum over $\tau$ and with $10^6$ simulated sequences.

**No ties.** If all $\tau_i$ differ, the product in (4) is the chain rule in the order given by $\tau$, so it equals $p(x)$.

**One position per step.** Revealing one uniformly chosen masked position per step (the schedule `random_order_schedule([1] * L, L)`) averages the chain rule over uniformly random orders, so $q = p$ exactly. The program returns a KL below $10^{-14}$. Groups of equal size in random order (the schedule `random_order_schedule([L // K] * K, L)` for $K$ dividing $L$) are the same construction with groups instead of single positions, so ties inside a group remain.

**Large K.** Let $\epsilon_K$ be the probability that two positions share a step, $\epsilon_K = 1 - \frac{K!}{(K-L)!\,K^L} \le \frac{L(L-1)}{2K}$. Then $q_K = (1 - \epsilon_K)\, p + \epsilon_K\, r_K$, where $r_K$ is the output law given a tie. Hence $q_K \to p$. When $p > 0$, $q_K \ge (1-\epsilon_K)\,p$ and $\mathrm{KL}(p \,\|\, q) \le \log(1 + \chi^2(p \,\|\, q))$ give

$$\mathrm{KL}(p \,\|\, q_K) \le \log\Big(1 + \frac{\epsilon_K^2}{1 - \epsilon_K}\, \chi^2(r_K \,\|\, p)\Big) = O(K^{-2}).$$

The error is quadratic, not linear, in $1/K$: $q_K - p$ is $O(1/K)$ at every point and KL is quadratic in that difference. To first order only one tied pair $\{i, j\}$ matters. It occurs with probability $1/K$, the other steps differ, and the pair is revealed behind a subset $U$ of the other $L - 2$ positions whose size is uniform on $\{0, \dots, L-2\}$ and which is uniform given its size. That pair multiplies the law by $p(x_i \mid x_U)\,p(x_j \mid x_U) / p(x_i, x_j \mid x_U)$. So

$$\mathrm{KL}(p \,\|\, q_K) = \frac{c}{K^2}\,\big(1 + O(1/K)\big), \qquad c = \frac12 \sum_x p(x)\, g(x)^2, \qquad g(x) = \sum_{i<j} \mathbb{E}_U\Big[\frac{p(x_i \mid x_U)\, p(x_j \mid x_U)}{p(x_i, x_j \mid x_U)} - 1\Big]. \tag{5}$$

The fraction inside is $e^{-\mathrm{pmi}}$ for the pointwise mutual information of $x_i$ and $x_j$ given $x_U$, so $c = 0$ for a product law and $c > 0$ as soon as some pair stays dependent given some subset. For $L = 2$ the mixture is exact, $q_K = (1 - 1/K)\,p + (1/K)\,p_1 \otimes p_2$. `maskdiff.asymptotics.leading_coefficient` evaluates $c$, and `tests/test_asymptotics.py` compares $K^2\,\mathrm{KL}$ with it up to $K = 2048$.

**One step.** For $K = 1$ every position is revealed at step 0 from the empty context, so $q_1 = \prod_i p_i$ and $\mathrm{KL}(p \,\|\, q_1) = \sum_i H(x_i) - H(x)$, the total correlation. It is zero only for a product law. For a product law every conditional is a marginal, ties change nothing and $q_K = p$ for every $K$.

**Link to the loss.** Use the $K$-step sampler as the model in the construction of section 1. The path version of (1) is the Riemann sum $\frac1K \sum_k \frac{1}{t_k}\,\mathbb{E}[\dots]$ with $t_k = (K-k)/K$. It is an upper bound on $-\log q_K(x)$, with equality at $K = 1$, and it tends to (2) at first order in $1/K$ (`tests/test_discrete_elbo.py`). Averaged over $x \sim p$ the bound reads $\mathrm{KL}(p \,\|\, q_K) \le \mathbb{E}_p[-\mathrm{ELBO}_K] - H(p)$, and the right-hand side tends to the ELBO gap of (3). So for large $K$ the gap bounds the error of a sampler that runs on learned conditionals from above, up to a term that vanishes with $K$. It is not a lower bound, see Part 4.

## 4. The exact conditionals do not minimise the K-step error

Part 2 shows that the exact conditionals minimise the training loss. The error of the sampler is another function of the conditionals and has another minimiser. Take $L = 2$, $V = 2$ and let $p_\rho$ be the Markov chain with persistence $\rho$. Its marginals are uniform and $p_\rho = \rho\, d + (1-\rho)\, u$, where $d$ puts mass $1/2$ on each of the two sequences with equal tokens and $u$ is uniform on the four sequences.

Give the sampler of Part 3 the exact conditionals of $p_{\rho'}$. The two reveal steps differ with probability $1 - 1/K$, and then (4) is the chain rule, so the output is $p_{\rho'}$. With probability $1/K$ they coincide and the output is the product of the marginals, which is $u$. Hence

$$q_K = \Big(1 - \frac{1}{K}\Big)\, p_{\rho'} + \frac{1}{K}\, u = p_{\rho'(1 - 1/K)}.$$

The exact conditionals ($\rho' = \rho$) give persistence $\rho(1 - 1/K)$ and

$$\mathrm{KL}(p_\rho \,\|\, q_K) = \frac{1+\rho}{2} \log \frac{1+\rho}{1+\rho(1-1/K)} + \frac{1-\rho}{2} \log \frac{1-\rho}{1-\rho(1-1/K)}.$$

The conditionals of $p_{\rho'}$ with $\rho' = \rho/(1 - 1/K)$, which needs $\rho < 1 - 1/K$, give $q_K = p_\rho$, so the KL is zero, although this table has a positive conditional KL against the data. A table with more dependence than the data offsets the independent draws of positions that share a step. For $\rho = 0.4$ and $K = 16$ the two errors are $3.664 \cdot 10^{-4}$ and 0, with $\rho' = 0.4267$.

`tests/test_compensation.py` checks the closed form and the exact output law against the program for five pairs $(\rho, K)$. For the six-position law of the README the same direction helps numerically, see `scripts/compensation.py`.
