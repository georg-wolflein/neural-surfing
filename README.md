# Neural Surfing

My senior honours project at the University of St Andrews. 

The report is available [here](report/report.pdf).

## Abstract
Gradient methods based on backpropagation are widely used in training multilayer feedforward neural networks. 
However, such algorithms often converge to suboptimal weight configurations known as local minima. 
This report presents a novel minimal example of the local minimum problem with only three training samples and demonstrates its suitability for investigating and resolving said problem by analysing its mathematical properties and conditions leading to the failure of conventional training regimes. 
A different perspective for training neural networks is introduced that concerns itself with neural spaces and is applied to study the local minimum example.
This gives rise to the concept of setting intermediate subgoals during training which is demonstrated to be a viable and effective means of overcoming the local minimum problem. 
The versatility of subgoal-based approaches is highlighted by showing their potential for training more generally. 
An example of a subgoal-based training regime using sampling and an adaptive clothoid for establishing a goal-connecting path is suggested as a proof of concept for further research. 
In addition, this project includes the design and implementation of a software framework for monitoring the performance of different neural training algorithms on a given problem simultaneously and in real time. 
This framework can be used to reproduce the findings of how classical algorithms fail to find the global minimum in the aforementioned example.

