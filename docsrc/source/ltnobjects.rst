LTN objects
===========
.. _noteltnobject:

In LTNtorch, non-logical symbols (constants and variables) and the output of logical symbols (predicates, functions,
formulas, connectives, and quantifiers) are wrapped inside :class:`ltn.core.LTNObject` instances.

An `LTNObject` represents a generic symbol (non-logical or logical) used by LTNtorch. Every `LTNObject` instance is defined by two important attributes:

1. `value`, which contains the grounding of the symbol (`LTNObject`). For example, if the grounding of variable :math:`x` is :math:`\mathcal{G}(x) = [1., 2., 3.]`, then the `value` attribute for variable :math:`x` will contain the vector :math:`[1., 2., 3.]`;
2. `free_vars`, which contains the list of the labels of the free variables contained in the `LTNObject` instance. For example, if we have the formula :math:`\forall x P(x, y)`, the `free_vars` attribute for this formula will be `['y']`. In fact, :math:`x` is quantified by :math:`\forall`, while :math:`y` is not quantified, namely it is a free variable.

For those unfamiliar with logic, a free variable is a variable which is not quantified by a universal (:math:`\forall`) or
existential (:math:`\exists`) quantifier.