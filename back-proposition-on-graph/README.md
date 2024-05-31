Calculate the forward and backward paths. Put the forward values on top and gradient values on the bottom vertices. 

In this assignment, we consider $x$ as the object features we cannot change (constants for our graph) and b1, b2, c1, c2 as variables. That means we do not calculate the gradient for $x_1$ and $x_2$. We calculate gradients only for b1, b2, c1, c2 and the intermediate z1, z2, z3, z4, z5, z6, z7, z8, z9.

> [!NOTE]
> - $.$ is the multiplication
> - $+$ is summation
> - $𝜎$ is sigmoid function
> - $\tanh$ is hyperbolic tangent function: (exp(x) - exp(-x)) / (exp(x) + exp(-x))

Checking the video https://youtu.be/i94OvYb6noo?t=323 or this article https://cs231n.github.io/optimization-2.
