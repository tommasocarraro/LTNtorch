Stable Fuzzy Semantics
======================

In LTNtorch, connectives and quantifiers are :ref:`grounded <notegrounding>` using fuzzy semantics.
Despite fuzzy logic enables the incorporation of logic and learning, not all fuzzy semantics are equally
suited for gradient-descent optimization. Many fuzzy logic operators can lead to vanishing or exploding gradients.
Some operators are also *single-passing*, in that they propagate gradients to only one input at a time.

If you are
interested in an analysis of differentiable fuzzy semantics and gradient problems in LTN, see this
`tutorial <https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/main/tutorials/2b-operators-and-gradients.ipynb>`_.

In the following, there are some examples of gradient problems in LTN:

- the Gougen fuzzy conjunction (:class:`ltn.fuzzy_ops.AndProd`) has vanishing gradients on the edge case :math:`x = y = 0`;
- the Gougen fuzzy disjunction (:class:`ltn.fuzzy_ops.OrProbSum`) has vanishing gradients on the edge case :math:`x = y = 1`;
- the Gougen fuzzy implication (:class:`ltn.fuzzy_ops.ImpliesGoguen`) has vanishing gradients on the edge case :math:`x = 0, y = 1`;
- for other examples, refer to the appendix of the `LTN paper <https://arxiv.org/abs/2012.13635>`_.

To address these problems, LTNtorch provides additional stable versions for the fuzzy operators and aggregators with
gradient problems (unstable operators). In particular, the stable version can be accessed by setting the boolean
parameter `stable` of the constructor method of the operator to `True`. If the parameter `stable` does not appear in the
signature of the constructor, it means that the selected operator does not have gradient problems, so a stable version
for that specific operator is not required. See :class:`ltn.fuzzy_ops.AndProd` for an example of
unstable operator. Notice the parameter `stable` appears in the signature of the constructor.

The stable versions are obtained in LTNtorch by applying the following projection functions to the inputs of the operators:

- :math:`\pi_0:[0,1] \rightarrow ]0,1]: x \rightarrow(1-\epsilon) x+\epsilon`, to avoid having zeros in input to the operator;
- :math:`\pi_1:[0,1] \rightarrow [0,1[: x \rightarrow(1-\epsilon) x`, to avoid having ones in input to the operator.

In LTNtorch, :math:`\epsilon` is set to 0.0001.