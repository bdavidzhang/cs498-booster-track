★ **Q1:** Williams' Theorem 1 says (r − b_ij) ∂ ln g_i / ∂ w_ij is an *unbiased*
estimate of ∂E{r | W}/∂w_ij. In your own words, what does unbiased mean here, and
why must the baseline b_ij not depend on the output y_i? (S&B's baseline b(s) in
eq. 13.10 likewise may depend on the state, but not on the action.)

An unbiased estimator $\hat{\theta}$ of a random variable $\theta$ satisfies 
$$
    E[\hat{\theta}] = \theta.
$$

In this case, if $\alpha$ is a constant learning rate, the theorem suggests that 
$$
    E[\Delta W \mid W] = \alpha \nabla_W E[r \mid W],
$$
where $\Delta W = \alpha(b_{ij} - r) \frac{\partial g_i}{\partial w_{ij}}$ 
is the increment in reward. It follows that for each element $w_{ij}$, the equality also holds after swapping order of operation between expectation and differentiation, so 
$$
   (b_{ij} - r) \frac{\partial g_i}{\partial w_{ij}}  = \frac{\partial E[r \mid W] }{\partial w_{ij}}.
$$
In other words, this justifies that the aggregate work $\Delta W$ is an identially good approximation 
for the reward accumulation, providng a formula for calculation.  



★ **Q2:** Williams' episodic update (eq. 11) multiplies *every* step's
eligibility by the *whole* episode's reinforcement. S&B's REINFORCE (eq. 13.8) and
HW2's `compute_returns` weight step t by the return *from t onward* instead. Why
is that still an unbiased estimate? (Williams' "causal" remark at the end of §5
is the idea.) Answer after the notebook, and use the variance you measured there.




★ **Q3:** `GaussianActor` learns `log_std`, not σ. Connect that choice to
Williams' footnote 2 of §6 and to S&B's eq. 13.20. What is ∂ ln π / ∂ (ln σ)?



★ **Q4:** Section 7 of the notebook shows two different errors from trying to
backpropagate through log-probabilities produced during collection. Explain
both. Then: `update()` never uses the stored `old_log_probs` — so why does the
storage keep them?


★ **Q5:** How many environment steps per second did this run get (from the
Slurm log)? Describe the reward curve's shape in two sentences.