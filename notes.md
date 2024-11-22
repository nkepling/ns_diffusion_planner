$\newcommand{\x}{\mathbf{x}} \newcommand{\p}{\mathbf{\theta}} imports$  
### score function

We have some data:
$$
\{ \mathbf {x}_0, \mathbf {x}_1, \dotsm, \mathbf {x}_N \} = D
$$
We assume this data was drawn from some underlying data distribution.
$$
\x_i \in D \mid \x_i \sim p(\x) \text{ i.i.d } i = 1, 2, \dotsm, N 
$$
Learning a model to approximate $p(\x)$, denoted as $p_\p(\x)$ with learnable parameters $\p$ is difficult because we must learn the normalizing constant:
$$
p_\p(\x) = \frac {e^{f_\p(\x)}}{Z_\p} 
$$
 If we are learning the p.d.f (or c.d.f. in discrete spaces), we must ensure that our model is a valid probability, hence the normalizing constant. There are methods to learn such a model, but we can avoid it all together by learning the score of the model instead. The score of our unknown data distribution is: 
$$
\nabla_x \log p(\x)
$$
The above is the true score function. It represents some vector field. 
$$
\begin{matrix}
\rightarrow & \rightarrow & \cdot \\
\rightarrow & \nearrow &  \uparrow\\
\nearrow & \uparrow &  \uparrow\\
\end{matrix}
$$
We want to model the score of the true data using a model:
$$
\nabla_\x \log p_\p(\x) = s_\p(\x)
$$
Note that this formulation allows us to sidestep the need of computing the normalizing constant
$$
s_\p(\x) = \nabla_\x f_\p(\x) \cancel{-\nabla_\x Z_\p}
$$
In other words, the score is simply the gradient of the energy. To learn $\p$ , thereby finding the an approximation of the true data's score function, we can use Fisher's divergence to compare vector fields:
$$
\left(
\begin{vmatrix}
\rightarrow & \rightarrow & \cdot \\
\rightarrow & \nearrow &  \uparrow\\
\nearrow & \uparrow &  \uparrow\\
\end{vmatrix}

- 

\begin{vmatrix}
\uparrow & \uparrow & \uparrow \\
\uparrow & \uparrow &  \uparrow\\
\uparrow & \uparrow &  \uparrow\\
\end{vmatrix}_\p
\right) 
^2
$$
$$
\mathbb{E}_{p(\x)} \left[\|  \nabla_\x \log p(\x) - s_\p(\x) \|^2_2 \right]
$$
Since we don't know the true score of the underlying data distribution, we can rearrange, via integration by parts, to write the Fisher Divergence only interns of our model score
$$
\mathbb{E}_{p(\x)} \left[  Tr(\nabla_\x s_\p(\x)) - \frac 1 2 \|s_\p(\x) \|^2_2 \right] + const.
$$
Boom! the true score is gone. However, we need to compute the hessian on the right hand side! We do this by using propagation  on each output of the U-Net.  Lets avoid this by using sliced score matching:

$$
\mathbb{E}_{p(\x)} \left[  \mathbf{v}^T\nabla_\x s_\p(\x) \mathbf{v} - \frac 1 2 (\mathbf{v}^T s_\p(\x))^2 \right] + const.
$$

$$
 = \mathbb{E}_{p(\x)} \left[  \mathbf{v}^T\nabla_\x \mathbf{v}^T s_\p(\x)  - \frac 1 2 (\mathbf{v}^T s_\p(\x))^2 \right] + const.
$$
While we still have a hessian on the right hand side. The hessian vector product can be computed with one pass, by back-propagating the vector $\mathbf v$ through the network. $\mathbf v$ is sampled randomly from some distribution $p(\mathbf v)$. Just use a multivariate random normal. 

However, our learned score function will not be good the subspace with low support. We can learn a set of probability distributions, each one is expanded by some variance to expand its support. We can make this continuous along time $t$. We need to define our model to take into account the time. We can use SDEs. Since we have a continuous time dimension $t$, we need to take an expectation over all $t$ 
$$
 = \mathbb E_t \left[ \lambda(t) \space \mathbb{E}_{p(\x)} \left[  \mathbf{v}^T\nabla_\x \mathbf{v}^T s_\p(\x)  - \frac 1 2 (\mathbf{v}^T s_\p(\x;t))^2 \right]\right].
$$
This is continuous time sliced score matching. $\lambda: \mathbb R \rightarrow \mathbb R_{>0}$ is a weighting function, weighting each time specific score matched objective. To approximate the SDE, we end up discretizing t anyway, making it similar to Noise Conditional Diffusion Models. $\lambda$ is designed by hand. We set T to range from $[0, 1]$. We implement a variance preserving (VP) SDE:
$$
d\x = -\frac 1 2 \beta(t) \x \space dt + \sqrt{\beta(t)} \space dw 
$$
We must carefully formulate $\beta(t)$ so that the following evaluates to 1:
$$
\int_0^T \beta(t) \space dt \approx 1
$$
We define a linear noise schedule for simplicity:
$$
\beta(t) = \beta_0 + \alpha t
$$