# Neural Surfing

My senior honours project at the University of St Andrews. 

The progress is documented [here](research/progress/main.pdf).

## Goal
The goal is to design and implement an approach to training neural networks that takes inspiration from potential field techniques in robotics. To achieve this, a framework will be designed for a simplified scenario that lends itself for visualisation (this will likely have few weights and output dimensions). During development, this framework will be used to benchmark and compare different versions of the approach to find the optimal algorithm. The approach will be generalized to an arbitrary number of dimensions and its effectiveness evaluated in comparison to other optimisation algorithms, especially with regard to the local minimum problem and other costing issues in such algorithms. 

## Objectives

### Primary objectives
1.	Contrive a minimalist version of the stripe problem and show that it provides a strong basis for investigating and resolving the suboptimal local minimum problem for neural networks.
2.	Investigate goal-connecting paths for this problem and design a “neural surfer” that attempts to find such a goal-connecting path.
3.	Design a generic framework with a well-defined interface for implementing different gradient-based and derivative-free neural training algorithms and implement such algorithms.
4.	For this framework, implement a tool that facilitates the comparison of neural training algorithms on a given problem (dataset) by visualising arbitrary user-specified metrics (including weight and output trajectories) during training in real time.

### Secondary objectives
1.	Investigate how the neural surfer can be generalized to more complex problems.