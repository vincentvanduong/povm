Now that we introduced POVMs, I will make contact with quantum channels.
It is helpful to get a notion of distances between two channels to obtain a sense of convergence.
This section will introduce the completely bounded trace norm on quantum channels.

To start, we focus on the distance measure between density matrices, and extend these ideas for quantum channels.
The trace distance $T$ measures the distance between two density matrices $rho, sigma$:
$
T(rho, sigma) = 1/2"Tr"[sqrt((rho - sigma)^dagger (rho - sigma))].
$
The trace distance is a quantum analogue of the total variation between two classical probability distributions.
The trace distance enjoys some valuable properties, so we would like to use it for measuring the distance between quantum channels instead of density matrices.

Let's be more general and consider a quantum channel $Phi$ induced by Kraus operators $L_mu$.
I write the quantum channel as
$
Phi(hat(X)) = sum_mu L_mu hat(X) L_mu^dagger.
$
My channel is a linear operator: $Phi: cal(H) times cal(H) -> cal(H) times cal(H)$. The domain and codomain of $Phi$ have bases, so I can express the quantum channel $Phi$ as a matrix. This matrix is known as the Choi matrix. It is straightforward to write the Choi matrix:
$
hat(Phi) = sum_(i,j) hat(E)_(i j) times.circle Phi(hat(E)_(i j)),
$
where the matrices $hat(E)_(i j)$ forms a basis for $cal(H) times cal(H)$. (For simplicity, it is best to choose the canonical basis of matrices $hat(E)_(i j)$ that has matrix elements $[hat(E)_(i j)]_(a b) = delta_(i a) delta_(j b)$.) Choi's theorem shows $hat(Phi)$ is completely positive! I.e., it has non-negative eigenvalues. Therefore, I can treat $hat(Phi)$ analogously to a density matrix.

How can I calculate the distance $d$ between two quantum channels $Phi, Psi$? Let's use the trace distance with the matrices induced from interpreting $Phi, Psi$ as density matrices:
$
d(Phi, Psi) = 1/2"Tr"[sqrt((hat(Phi) - hat(Psi))^dagger (hat(Phi) - hat(Psi)))].
$
Finally, I can calculate the distance between two POVMs $M_mu, N_nu$ by measuring their trace distance on their induced Choi matrix. That is,
$
d(M_mu, N_nu) = 1/2"Tr"[sqrt((hat(Phi) - hat(Psi))^dagger (hat(Phi) - hat(Psi)))],
$
where the Choi matrices $hat(Phi), hat(Psi)$ for each channel is induced from their linear map:
$
Phi(X) = & sum_mu M_mu X M_mu^dagger\
Psi(X) = & sum_nu N_nu X N_nu^dagger\
hat(Phi) = & sum_(i,j) hat(E)_(i j) times.circle Phi(hat(E)_(i j))\
hat(Psi) = & sum_(i,j) hat(E)_(i j) times.circle Psi(hat(E)_(i j)).
$
In this section, I have shown how to measure distances between sets of POVMs, $d(M_mu, N_nu)$, by measuring the induced norm on their Choi matrices.