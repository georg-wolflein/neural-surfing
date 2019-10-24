# Neural Surfing

My senior honours project at the University of St Andrews. 

The progress is documented [here](research/progress/main.pdf).

## Goal
The goal is to design and implement an approach to training neural networks that takes inspiration from potential field techniques in robotics. To achieve this, a framework will be designed for a simplified scenario that lends itself for visualisation (this will likely have few weights and output dimensions). During development, this framework will be used to benchmark and compare different versions of the approach to find the optimal algorithm. The approach will be generalized to an arbitrary number of dimensions and its effectiveness evaluated in comparison to other optimisation algorithms, especially with regard to the local minimum problem and other costing issues in such algorithms. 

## Objectives

### Primary objectives
1.	Design a generic framework that can be used for various neural training algorithms with a clear set of inputs and outputs at each step. This framework should include benchmarking capabilities.
2.	For a simple case of this framework (when the dimensionality of the control space and output space are suitably low), implement a visualisation tool that shows the algorithm’s steps.
3.	Implement a particular training algorithm for the framework that uses potential field techniques.
4.	Evaluate the performance of this and other algorithms on tasks of differing complexity, especially with regard to the local minimum problem and similar issues.

### Secondary objectives
1.	Investigate how this approach can be generalized to any numerical optimisation problems.

### Tertiary objectives
1.	Investigate if and how reinforcement learning could be used for such an algorithm.
