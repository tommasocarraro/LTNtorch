LTNtorch's documentation
========================

Welcome to the LTNtorch's documentation!

`LTNtorch <https://github.com/bmxitalia/LTNtorch>`_ is a **fully tested** and **well documented** `PyTorch <https://pytorch.org/>`_ implementation of `Logic Tensor Networks (LTNs) <https://arxiv.org/abs/2012.13635>`_, a
Neural-Symbolic approach which allows learning neural networks using the satisfaction of a First-Order
Logic (FOL) knowledge base as an objective.

The documentation is organized as follows:

- **Notes**: contains some information that may be useful for those unfamiliar with the LTNtorch framework;
- **LTNtorch's modules**:

    - `ltn.core`, which contains the definition of constants, variables, predicates, functions, connectives, and quantifiers;
    - `ltn.fuzzy_ops`, which contains the definition of some of the most common fuzzy semantics (connective operators and aggregators).

.. toctree::
    :caption: Notes
    :maxdepth: 1

    grounding
    learningltn
    ltnobjects
    broadcasting
    quantification
    stableconf

.. toctree::
    :caption: Modules
    :maxdepth: 1

    core
    fuzzy_ops

Indices
==================

* :ref:`genindex`
* :ref:`modindex`
