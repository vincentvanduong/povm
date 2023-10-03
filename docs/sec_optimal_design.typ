== Baye's theorem

We will study how to construct a quantum measurement that allows parameters to be estimated with minimum variance. 
The approach is Bayesian, allowing us to account for prior knowledge of the parameters and the uncertainty in our observations.
(This approach draws strong parallels to a partially observable Markov decision process (POMPD) #cite("d677280f-74cf-4f52-bc01-2a8f358f4f92", "10.1287/opre.26.2.282"). The POMPD framework could construct the optimal sequence of measurements that sharpens the posterior probability distribution.)

There are two sources of randomness in our problem: quantum and classical.
Measuring a quantum state $rho$ leads to distinct outcomes with varying probabilities; and
on the other hand, the parameter's prior distribution contributes to the posterior probability distribution.

Consider a quantum state $rho$ with POVM $M_mu$ acting on the Hilbert space. (Each $mu$ labels a different measurement outcome.)
In a typical system, the parameters are encoded in its state: $rho = rho(phi.alt)$ and have a prior probability distribution $phi.alt$.
Following Baye's theorem, we update our estimates after measuring the system that minimises the variance (or increases the information gained).
For example, after observing outcome $mu$, we can arbitrarily estimate the parameter to be $xi_mu$. The square loss (or ) is
$
integral dif phi.alt p(phi.alt|mu) epsilon(phi.alt, xi_mu),
$
where $epsilon(phi.alt, xi_mu) = (phi.alt - xi_mu)^2$ is the square loss. 

The cost function $cal(C)$, following the work of #cite("PRXQuantum.4.020333"), is the total variance of the parameters $phi.alt$:
$
cal(C) = sum_mu p(mu) integral dif phi.alt p(mu|phi.alt) p(phi.alt) epsilon(phi.alt, xi_mu).
$
The variable $mu$ labels the possible outcomes from measuring the quantum system $rho$.
The prior probability density of the parameters is $p(phi.alt)$

== Simple spin

We study a single spin-1/2 particle in a random magnetic field. We show that fluctuations in the random magnetic field lead to a long-time behaviour that is non-trivial. The Hamiltonian of the system is
$
H = bold(B) dot bold(L),
$
where $bold(B)$ is the external magnetic field, and $bold(L)$ is the total angular momentum of the particle.

There are six kinds of measurements that we can make:
$
M_1 =& mat(1, 1; 1, 1)\
M_2 =& mat(1, -1; -1, 1)\
M_3 =& mat(1, i; -i, 1)\
M_4 =& mat(1, -i; i, 1)\
M_5 =& mat(1, 0; 0, 0)\
M_6 =& mat(0, 0; 0, 1)\
$
Turning the crank, we find that 
$
integral dif phi.alt p(mu|phi.alt) p(phi.alt) = mat(
    1; 1; 1; 1;
    1/3 - e^(-2 t^2 sigma^2)/3 (1 - 4 t^2 sigma^2);
    2/3 + e^(-2 t^2 sigma^2)/3 (1 - 4 t^2 sigma^2)
),
$
and
$
integral dif phi.alt p(mu|phi.alt) p(phi.alt) epsilon(phi.alt, 0) = 3 sigma^2 mat(
    1; 1; 1; 1;
    1/3 - e^(-2 t^2 sigma^2) / 9 (3 - 24 t^2 sigma^2 + 16 t^4 sigma^4);
    2/3 + e^(-2 t^2 sigma^2) /9 (3 - 24 t^2 sigma^2 + 16 t^4 sigma^4);
)
$
When measurements are projectors of the initial conditions, the probability distribution varies in time due to fluctuations. This is an artefact that reappears in more complicated dynamics. 