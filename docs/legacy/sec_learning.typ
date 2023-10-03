The optimal POVMs are typically non-projective.
How can we realise these measurements in the laboratory?
The answer requires ancilla qubits.
The Stinespring Dilatation theorem shows that with sufficiently many qubits, the non-projective measurements can become projective in the augmented Hilbert space.
In this section, we provide a technique to determine the number of ancilla qubits necessary and, critically, the unitary operation that realises the desired measurement.

Suppose I have
+ a (possibly non-projective) POVM $M_mu$ on the Hilbert space $cal(H)_1$ spanned by $N$ qubits
+ access to ancillary Hilbert space $cal(H)_2$ spanned by a single qubit.
Furthermore, suppose I am given a basis of necessarily projective measurements $Pi_mu$ on the augmented Hilbert space $cal(H)$ spanned by $N+1$ qubits. (The target $M_mu$ are size $2^(N) times 2^(N)$, and $Pi_mu = |phi_mu angle.r angle.l phi_mu|$ are size $2^(N+1) times 2^(N+1)$.) The goal is to find the unitary operator $U$ that best approximates the target POVM $M_mu$ after tracing out the ancilla Hilbert space:
$
M_mu approx "Tr"_(cal(H)_2)[U Pi_mu U^dagger]
$
What is the best $U$ that realises the approximation? Let us warm-up with a simple example.

== Warm-up with two qubits

#let bra(f) = $lr(angle.l #f|)$
#let ket(f) = $lr(|#f angle.r)$

- Let $cal(H_1)$ have basis $cal(B)_1 = {ket(+z), ket(-z)}$
- Let $cal(H_2)$ have basis $cal(B)_2 = {ket(+z), ket(-z)}$
- Create the augmented Hilbert space $cal(H) = cal(H)_1 times.circle cal(H)_2$ which has a canonical basis
$
cal(B) & = {ket(+z)_1ket(+z)_2, ket(+z)_1ket(-z)_2, ket(-z)_1ket(+z)_2, ket(-z)_1ket(-z)_2}\
& =  {ket(e_1), ket(e_2), ket(e_3), ket(e_4)}
$
- The canonical basis of projective measurements $Pi_mu = ket(e_mu) bra(e_mu)$.
- A unitary $U$ acting on the augmented space $cal(H)$ can be expressed as
$
U = & sum_(i, j)^(4, 4) U_(i j)ket(e_i)bra(e_j)\
= & sum_(a, a', b, b')^(2, 2, 2, 2) U_(a a' b b')ket(a) ket(a') bra(b) bra(b')
$

We can complete the calculation by writing
$
"Tr"_(cal(H)_2)[...] = & sum_(a') bra(a')_2 [...] ket(a')_2\
ket(e_mu) = & sum_(a b)A(mu)_(a b)ket(a)_1 ket(b)_2\
U = & sum_(mu nu)U_(mu nu) ket(e_mu) bra(e_nu)\
Pi_mu = & ket(e_mu) bra(e_mu)
$
Note that
$
U ket(e_alpha) = & sum_(mu nu) U_(mu nu) ket(e_mu) bra(e_nu) ket(e_alpha)\
= & sum_(mu) U_(mu alpha) ket(e_mu)
$
Writing down the transformed POVM:
$
U Pi_alpha U^dagger = & sum_(mu) U_(mu alpha) ket(e_mu) sum_(nu) U_(mu alpha)^* bra(e_nu)\
"Tr"_(cal(H)_2)[U Pi_alpha U^dagger] = & sum_(mu) U_(mu alpha) sum_(nu) U_(mu alpha)^* "Tr"_(cal(H)_2)[ket(e_mu) bra(e_nu)]
$
Note that
$
"Tr"_(cal(H)_2)[ket(e_mu) bra(e_nu)] = & "Tr"_(cal(H)_2)[sum_(a b)A(mu)_(a b)ket(a)_1 ket(b)_2 sum_(a' b')A(nu)_(a' b')^*bra(a')_1 bra(b')_2]\
= & sum_(a b a^' b^')A(mu)_(a b) A(nu)_(a^' b^')^* ket(a)_1 bra(a^')_1 "Tr"_(cal(H)_2)[ket(b)_2 bra(b^')_2]\
= & sum_(a a^' b)A(mu)_(a b) A(nu)_(a^' b)^* ket(a)_1 bra(a^')_1.
$
Finally,
$
"Tr"_(cal(H)_2)[U Pi_alpha U^dagger] = sum_(a a') sum_(mu nu b)[U_(mu alpha) U_(nu alpha)^* A^(mu)_(a b) scripts(A^nu_(a^' b))^*]ket(a)_1 bra(a^')_1.
$
Calculating the inner product between the target POVM $M_beta$ becomes
$
angle.l M_beta, "Tr"_(cal(H)_2)[U Pi_alpha U^dagger] angle.r = & sum_(a a') sum_(mu nu b)[M^(beta)_(a^' a) U_(mu alpha) U_(nu alpha)^* A^(mu)_(a b) scripts(A^nu_(a^' b))^*].
$