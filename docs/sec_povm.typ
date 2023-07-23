We briefly review the positive operator valued measure (POVM) formalism. POVMs are operators that generalise classical events. POVMs are a set of operators $hat(M)_mu$ that satisfy two conditions:
$
sum_mu hat(M)_mu = &II \ 
hat(M)_mu succ.eq & 0.
$
If the system's density matrix is $hat(rho)$, then the probability of measuring event $mu$ is given by a trace:
$
p(mu) = "Tr"[hat(M)_mu hat(rho)].
$
The first condition (resolution of the identity) is necessary to satisfy the law of total probability. The second condition (semidefiniteness) ensures that all measurement probabilities are positive.

The simplest POVM is the orthogonal projectors. This reduces to undergraduate quantum mechanics. For example, $hat(M)_mu = |phi_mu angle.r angle.l phi_mu|$, where $|phi_mu angle.r$ is an orthonormal basis for the Hilbert space. Both POVM conditions are satisfied. Moreover, the measurement probability reduces to the probabilities of measuring a certain state:
$
p(mu) = & "Tr"[hat(M)_mu hat(rho)] = "Tr"[ |phi_mu angle.r angle.l phi_mu| hat(rho)]\
= & sum_nu angle.l phi_nu |phi_mu angle.r angle.l phi_mu| hat(rho) |phi_nu angle.r\
= & angle.l phi_mu |hat(rho) |phi_mu angle.r.
$

The critical benefit of a POVM is its ability to capture measurements that are not projectors, such as entangled measurements. Unfortunately, this means they are challenging to realise experimentally. Naimark's dilation theorem saves us from that: all POVM can be expressed as rank-1 projectors of a larger Hilbert space. All we need to do is combine the initial Hilbert space with an ancillary Hilbert space and perform rank-1 measurements on the combined space.