The goal of numerics is to learn the POVM and estimates $Theta = {hat(M)_mu, bold(xi)_mu}$ that minimise the cost function:
$
L(Theta) = & sum_mu integral d bold(theta) P(bold(theta)) "Tr"[hat(M)_mu e^(- i hat(H)(bold(theta)) t) hat(rho) e^(+ i hat(H)(bold(theta)) t)] [bold(xi)_mu - bold(theta)]^2\

"subject to"\
hat(M)_mu succ.eq & 0\
sum_(mu) hat(M)_mu = & II.
$<loss>

To minimise the cost function, we will use the typical Newton-Raphson Method, treating $Theta$ as a one-dimensional vector parameterising all $hat(M)_mu$ and $bold(xi)_mu$.
Expanding the loss function simplifies the problem:
$
L(Theta) & = "Tr"[ sum_mu hat(M)_mu (bold(xi)_mu^2 hat(F)[1] - 2 bold(xi)_mu dot hat(F)[bold(theta)] + hat(F)[bold(theta)^2])],
$
where the matrix-valued function $hat(F)[...]$ is defined via integration
$
hat(F)[g(bold(theta))] = integral d bold(theta) P(bold(theta)) e^(- i hat(H)(bold(theta)) t) hat(rho) e^(+ i hat(H)(bold(theta)) t) g(bold(theta)).
$


The three matrices $hat(F)[1], hat(F)[bold(theta)], hat(F)[bold(theta)^2]$ are evaluated to arbitrary accuracy via Monte Carlo integration. This is straightforward from sampling the prior:
$
hat(F)[g(bold(theta))]&  = EE_(bold(theta)~P(bold(theta)))[e^(- i hat(H)(bold(theta)) t) hat(rho) e^(+ i hat(H)(bold(theta)) t) g(bold(theta))] \
& approx 1/N sum_(i=1)^N [e^(- i hat(H)(bold(theta)_i) t) hat(rho) e^(+ i hat(H)(bold(theta)_i) t) g(bold(theta)_i)], bold(theta)~P(bold(theta)).
$

The time-evolved density matrices
$
e^(- i hat(H)(bold(theta)_i) t) hat(rho) e^(+ i hat(H)(bold(theta)_i) t)
$
can be calculated using a variety of techniques. The one used in this paper makes use of the singular value decomposition of $hat(rho)$:
$
hat(rho) = hat(U) hat(S) hat(U)^dagger.
$
Therefore, 
$
e^(- i hat(H)(bold(theta)_i) t) hat(rho) e^(+ i hat(H)(bold(theta)_i) t) = [e^(- i hat(H)(bold(theta)_i) t) hat(U) sqrt(hat(S))][e^(- i hat(H)(bold(theta)_i) t) hat(U) sqrt(hat(S))]^dagger.
$
In this form, the density matrix is the square of one matrix.

The loss function from @loss is constrained: POVM must satisfy the two conditions.
Positive semidefiniteness is simple to enforce since every positive semidefinite matrix admits a unique Cholesky decomposition:
$
hat(M)_mu = hat(L)_mu hat(L)_mu^dagger,
$<cholesky>
where $hat(L)_mu$ is a lower triangular complex matrix. 
That is, we re-parameterise the problem with these lower triangular matrices.
The remaining constraint is non-trivial to enforce. To enforce the constraint, we relax the loss function with a term that penalises deviations from the constraint:
$
L(Theta) -> L(Theta) + lambda || II - sum_mu M_mu ||^2_2.
$
The upshot is that we must increase $lambda$ during the optimisation to enforce the constraint.

We may perform Newton-Raphson minimisation since the problem is amenable to auto-differentiation. Updating the parameters allow us to find a minima: 
$
Theta_a' = Theta_a - sum_(b)[H^(-1)]_(a b) D_b,
$
where
$
D_a = & (diff L)/(diff Theta_a)\
H_(a b) = & (diff ^2 L)/(diff Theta_a diff Theta_b).
$