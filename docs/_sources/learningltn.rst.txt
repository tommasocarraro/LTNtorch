Introduction to Learning in Logic Tensor Networks
=================================================
.. _notelearning:

In order to train a Logic Tensor Network, one has to define:

1. a First-Order Logic knowledge base containing some logical axioms;
2. some learnable predicates, functions, and/or logical constants appearing in the axioms;
3. some data.

Given these three components, the LTN workflow is the following:

1. **grounding phase**: data is used to ground (instantiate) the logical axioms included in the knowledge base;
2. **forward phase**: the truth values of the logical axioms are computed based on the given grounding (instantiation);
3. **aggregation phase**: the truth values of the axioms are aggregated to compute the overall satisfaction level of the knowledge base;
4. **loss function computation**: the gap between the overall satisfaction level and the truth (1) has to be minimized;
5. **backward phase**: the parameters of the learnable predicates, functions, and/or constants are changed in such a way to maximize the overall satisfaction level.

The training ends up with a solution which maximally satisfies all the logical axioms in the knowledge base. This
`tutorial <https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/main/tutorials/3-knowledgebase-and-learning.ipynb>`_
shows how to use the satisfaction of a First-Order Logic knowledge base as an objective to learn a Logic Tensor Network.

In this documentation, you will find how to create a First-Order Logic knowledge base containing learnable predicates (:class:`ltn.core.Predicate`),
functions (:class:`ltn.core.Function`), and/or constants (:class:`ltn.core.Constant`) using LTNtorch.