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
