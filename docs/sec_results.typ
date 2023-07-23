In this section, we provide evidence that optimal sensing can only be performed within a certain time interval.
Inside the time interval, we can reduce the total uncertainty on the external magnetic field by measuring the system with a POVM.
I.e., any POVM measurement outside this window yields no added benefit.

== Set-up

The first step is to provide a Hamiltonian $hat(H)$ that describes our sensor. The Hamiltonian $hat(H)$ that we considered is the Ising model coupled to an unknown magnetic field $bold(B)$:
$
hat(H) = J sum_(i j) hat(sigma)_i^z hat(sigma)_j^z + bold(B) dot hat(bold(L)),
$
where $bold(B)$ is the unknown external field and $hat(bold(L))$ is the total angular momentum operator. In our analyses, the Ising coupling $J$ is known. (We studied whether large or small $J$ increase or reduce sensing abailities.)

The second step is to include our prior knowledge of the field that we want to sense. The only information available about the external field $bold(B)$ is its prior $P(bold(B))$, which we take to be a spherically symmetric Gaussian probability distribution:
$
P(bold(B)) = (2 pi sigma^2)^(-3/2) e^(-(1)/(2 sigma^2) bold(B)^2).
$
In the following analyses, we fix the width of the distribution: $sigma = 0.1$.

The third step is to intialise the state of the system $hat(rho)$. For simplicity, we initialise each spin in the $|-z angle.r$ state.
Though this corresponds to a pure state $hat(rho) = |psi angle.r angle.l psi |$, we can (and should) consider mixed states. 

The fourth step is to let the system evolve for a duration $t$:
$
hat(rho)(t) = e^(- i hat(H) t) hat(rho) e^(+ i hat(H) t).
$
In our analyses, the evolution time $t$ is fixed. (We will find that there is an interval where sensing can be made optimal.)

The fifth, and final step is to make a judicious choice of POVM ${hat(M)_mu}$ that provides the most information about the magnetic field $bold(B)$. Put another way, we want to minimise the total uncertainty on the external field $bold(B)$. This is the most challenging step because finding the optimal POVM is non-trivial. The technique used in this paper is outlined in the Numerics section. 

In the following sub-sections, we chart the regions where there is a quantum advantage.
These are regions where the cost (or loss) function can be reduced via a judicious choice of POVM and estimates.
The regions where there is a quantum advantage take place when the evolution time $t$ and Ising strength $J$ take particular values -- one should tune $t$ and $J$ to guarantee optimal performance. 


== $N = 2$
This is a two-body system. The system is initiated with all spins in $|-z angle.r$. The number of POVM is 
#figure(
    grid(
        columns: 2,
        gutter: 2mm,
        image("two-body2.png", width: 100%),
        image("two-body3.png", width: 100%),
    ),
    caption: "some caption"
)
== Three spins
In the
#figure(
    grid(
        columns: 2,
        gutter: 2mm,
        image("three-body2.png", width: 100%),
        image("three-body3.png", width: 100%),
    ),
    caption: "some caption"
)
== Four spins
In the
#figure(
    grid(
        columns: 2,
        gutter: 2mm,
        image("four-body2.png", width: 100%),
        image("four-body3.png", width: 100%),
    ),
    caption: "some caption"
)