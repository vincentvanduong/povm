  - Introduce the problem of quantum parameter estimation
  - Provide practical examples, such as NMR spectroscopy or qubit readout using 
  - Introduce the Bayesian approach to parameter estimation
  - Introduce the optimisation procedure for finding the optimal parameters of a Hamiltonian
  - Verify that entanglement improves parameter estimation
  - Study a simple system numerically

How do we build sensors from quantum devices?
The idea is to couple the quantum device to the external field that we want to sense, and then perform inference.
In what ways is this different from classical physics?
The measurement outcome after coupling the system to the external field is probabilistic, due to the quantum mechanics.
This is commonly referred to as 'Hamiltonian Learning.'

Suppose that I have a Hamiltonian $H$ with unknown parameters $theta$.
How can I learn the parameters?
I run an experiment that depends on the Hamiltonian to perform inference on the unknown parameters $theta$.
Then, I measure the outcome of my experiment and update my knowledge about the parameters.
This strategy is well understood and studied under the Bayesian framework.

How do I learn the parameters optimally?
The optimal strategy consists of constructing an experiment that allows one to extract as much information about the parameters as possible.
I.e., One reduces the total uncertainty on the inferred parameters.
This is a minimisation problem!

