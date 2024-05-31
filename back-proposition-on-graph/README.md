Calculate the forward and backward paths, the same as we did in class. Put the forward values on top and gradient values on the bottom vertices. 

Unlike the task that we did in class, where both x and w were variables; in this assignment, we think of x as the object features that we cannot change (constants for our graph), and b1, b2, c1, c2 as variables. That means we do not calculate the gradient for x1 and x2. We calculate gradients only for b1, b2, c1, c2 and the intermediate z1, z2, z3, z4, z5, z6, z7, z8, z9.

In the graph:
. – is the multiplication
+ – summation
𝜎 – sigmoid function
tanh - hyperbolic tangent function: (exp(x) - exp(-x)) / (exp(x) + exp(-x))

Checking the video https://youtu.be/i94OvYb6noo?t=323 or this article might help: https://cs231n.github.io/optimization-2