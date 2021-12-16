Grounding in Logic Tensor Networks
==================================
.. _notegrounding:

To make learning possible, LTN uses a differentiable first-order logic language, called Real Logic, which enables
the incorporation of data and logic.

Real Logic defines the concept of `grounding` (different from the grounding of logic), which is a mapping from the
logical domain (i.e., constants, variables, and logical symbols) to tensors in the Real field or operations based on
tensors. These operations could be, for instance, mathematical functions or learnable neural networks. In other words,
a `grounding`, denoted as :math:`\mathcal{G}`, is a function which maps a logical symbol into a real tensor or an
operation on tensors.

In particular, the grounding is defined as follows. Let us assume that :math:`c` is a constant, :math:`x` is a logical
variable, :math:`P` is a predicate, and :math:`f` is a logical function:

- :math:`\mathcal{G}(c) = \mathbb{R}^{d_1 \times \dots \times d_n}`: a logical constant is grounded as a tensor (individual) of **any order** (e.g., :math:`\mathbb{R}^4$ or $\mathbb{R}^{5 \times 4 \times 4}`);
- :math:`\mathcal{G}(x) = \mathbb{R}^{m \times d}`: a logical variable is grounded as a **sequence** of :math:`m` tensors (individuals) of the same shape :math:`d`;
- :math:`\mathcal{G}(f \mid \theta) = x \mapsto MLP_{\theta}(x)`: a logical function is grounded as a (learnable) **mathematical function** which takes as input some tensors (individuals) and returns a tensor. In this definition, :math:`\theta` are the learnable parameters of the function, while :math:`MLP_{\theta}` is the neural network representing the function, parametrized by :math:`\theta`. Note that the given definition has one input :math:`x`, however, an LTN function can take multiple inputs;
- :math:`\mathcal{G}(P \mid \theta) = x \mapsto \sigma (MLP_{\theta}(x))`: a logical formula (atomic or not) is grounded as a mathematical function which takes as input some tensors (individuals) and returns a value in **[0., 1.]**. In this case, the logistic function :math:`\sigma` assures the output to be in the range [0., 1.], resulting in a value which can be interpreted as a fuzzy truth value. Note that the given definition has one input :math:`x`, however, an LTN predicate (or formula) can take multiple inputs.

The `grounding` defines also how the logical connectives (:math:`\land, \lor, \lnot, \implies, \leftrightarrow`) and quantifiers
(:math:`\forall, \exists`) are mapped in the Real domain. In particular, logical connectives are grounded using fuzzy logic semantics, while
quantifiers are grounded using fuzzy aggregators. Please, **carefully** read the `paper <https://arxiv.org/abs/2012.13635>`_ if you have some doubts on these notions.

Examples of possible groundings are showed in the figure below. In particular, :math:`friend(Mary, John)` is an
atomic formula (predicate), while :math:`\forall x (friend(John, x) \implies friend(Mary, x))` is a closed formula (all the variables are
quantified). The letter :math:`\mathcal{G}`, again, is the grounding, the function which maps the logical domain into the Real domain.

.. image:: ../../images/framework_grounding.png