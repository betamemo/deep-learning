Calculate the forward and backward paths. Put the forward values on top and gradient values on the bottom vertices. 

In this assignment, we consider $x$ as the object features we cannot change (constants for our graph) and $b_1$, $b_2$, $c_1$, $c_2$ as variables. That means we do not calculate the gradient for $x_1$ and $x_2$. We calculate gradients only for $b_1$, $b_2$, $c_1$, $c_2$ and the intermediate $z_1$, $z_2$, $z_3$, $z_4$, $z_5$, $z_6$, $z_7$, $z_8$, $z_9$.

> [!NOTE]
> - $.$ is the multiplication
> - $+$ is summation
> - $𝜎$ is sigmoid function
> - $\tanh$ is hyperbolic tangent function: $\frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}$

Checking the video https://youtu.be/i94OvYb6noo?t=323 or this article https://cs231n.github.io/optimization-2.
