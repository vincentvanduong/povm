#import "template.typ": *

// Take a look at the file `template.typ` in the file panel
// to customise this template and discover how it works.
#show: project.with(
  title: "Note on POVM and Control",
  authors: (
    (name: "Vincent Văn Dương", email: "vv2102@nyu.edu", affiliation: "New York University"),
  ),
)

// We generated the example code below so you can see how
// your document will look. Go ahead and replace it with
// your own content!
// typst watch main.typ

= Introduction

#include("sec_intro.typ")

= Quantum Fisher Information
  - Introduce the Classical Fisher Information Matrix
  - Introduce the Cramer-Rao Bound 
  - Introduce the Quantum Fisher Information (QFI)
  - Derive the Quantum Cramer-Rao bound
  - Explain why QFI is impractical: it shows a bound but fails to provide a parameter estimate and fails to provide the optimal measurement.

= Quantum Channels and the Positive Operator-Valued Measure (POVM)



== Positive operator-valued measure (POVM)

#include("sec_povm.typ")

== Completely Bounded Trace Norm

#include("sec_norm.typ")


= Quantum Parameter Estimation

#include("sec_quantumparameterestimation.typ")

== The Spin-1/2 system

== Uncertainty and optimisation

== Optimisation (Dual Problem)

== Two spin-1/2 system

== Spin-1/2 Tomography

= Quantum Control
  - Introduce the efficiency advantage of designing measurement devices that use the Markov Decision Process Theory
  - Find a system that is exactly solvable using the theory. Maybe this is two qubits in a Magnetic Field?

#include("reinforcement.typ")

= Numerical strategy

#include("sec_numerical.typ")

= Results

#include("sec_results.typ")

= Learning

#include("sec_learning.typ")

#pagebreak()
#bibliography("bibliography.bib")
