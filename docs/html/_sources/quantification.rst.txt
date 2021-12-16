Quantification in Logic Tensor Networks
=======================================

.. _quantification:

Base quantification
-------------------

The logical quantifiers, namely :math:`\forall` and :math:`\exists`, are implemented in LTN using fuzzy aggregators. An
example of fuzzy aggregator is the mean. The :class:`ltn.fuzzy_ops` module contains the implementation
of the most common fuzzy aggregators.

In the note regarding the :ref:`LTN broadcasting <broadcasting>`, we have seen that the output of logical predicates, for
example :math:`P(x, y)`, is organized in a tensor where the dimensions are related with the free variables appearing in the
predicate. In the case of the atomic formula :math:`P(x, y)`, the result is a tensor :math:`\mathbf{out} \in [0., 1.]^{m \times n}`,
where :math:`m` is the number of individuals of variable :math:`x`, while :math:`n` is the number of individuals of variable :math:`y`.

Notice that:

1. the tensor :math:`\mathbf{out}` has two dimensions because the atomic formula :math:`P(x, y)` has two free variables, namely :math:`x` and :math:`y`;
2. variable :math:`x` has been placed on the first dimension, while variable :math:`y` in the second dimension, according to their order of appearance in the atomic formula;
3. the output of the predicate is wrapped inside an :ref:`LTN object <noteltnobject>` where the `value` attribute contains the tensor :math:`\mathbf{out}`, while the `free_vars` attribute contains the list `['x', 'y']`.

Now, we can extend this example with the formula: :math:`\forall x P(x, y)`. In order to compute the result of this formula,
LTNtorch must compute the result of the atomic formula :math:`P(x, y)` first, and then apply a fuzzy aggregator to this
result for performing the desired quantification on variable :math:`x`.

We know that the output of :math:`P(x, y)` is the tensor :math:`\mathbf{out} \in [0., 1.]^{m \times n}`, and that the
first dimension is related to variable :math:`x` while second dimension to variable :math:`y`. In order to perform the
quantification of :math:`P(x, y)` on variable :math:`x`, LTNtorch simply performs an aggregation of tensor
:math:`\mathbf{out}` on the first dimension. Let us assume that the selected fuzzy aggregator for :math:`\forall` is the mean.
LTNtorch simply computes `torch.mean(out, dim=0)`, where `out` is the tensor :math:`\mathbf{out}` and `dim=0` means that
we aggregate on the first dimension of `out`, namely the dimension related to variable :math:`x`.

After the application of the quantifier, a new :ref:`LTN object <noteltnobject>` is created. The attribute `value` will contain
the result of :math:`\forall x P(x, y)`, that will be the tensor :math:`\mathbf{out}' \in [0., 1.]^n`, while the attribute
`free_vars` will contain the list `['y']`.

Notice that:

1. the result has one less dimension within respect to the result of :math:`P(x, y)`. This is due to the fact that we have aggregated one of the two dimensions, namely the dimension of :math:`x`. The only dimension left is the dimension related with :math:`y`, and it is for this reason that :math:`\mathbf{out}` is now a vector in :math:`[0., 1.]^n`;
2. the attribute `free_vars` contains now only variable :math:`y`. This is due to the fact that variable :math:`x` is not free anymore since it has been quantified by :math:`\forall`.

Finally, notice that if the formula would have been :math:`\forall x \forall y P(x, y)`, the quantification would have been
`torch.mean(out, dim=(0,1))`, namely both dimensions of :math:`\mathbf{out}` would have been aggregated. Also, the output
would have been a scalar in :math:`[0., 1.]` and no more free variables would have been in the formula, since both variables
have been quantified.

Diagonal quantification
-----------------------
.. _diagonal:

Given 2 (or more) variables, there are scenarios where one wants to express statements about specific pairs (or tuples)
of values only, instead of all the possible combinations of values of the variables. This is allowed in LTN thanks to the concept
of diagonal quantification.

To make the concept clear, let us make a simple example. Suppose that we have the formula :math:`\forall x \forall y P(x, y)`.
Suppose also that variables :math:`x` and :math:`y` have the same number of individuals, and that this number is :math:`n`.

With the usual :ref:`LTN broadcasting <broadcasting>`, the predicate :math:`P(x, y)` is computed on all the possible
combinations of individuals of :math:`x` and :math:`y`. In other words, the result of :math:`P(x, y)` would be a tensor
:math:`\mathbf{out} \in [0., 1.]^{n \times n}`. Then, after the quantification on both :math:`x` and :math:`y`, we obtain
a scalar in :math:`[0., 1.]`.

With diagonal quantification (:math:`\forall Diag(x, y) P(x, y)`), instead, predicate :math:`P(x, y)` is computed only on the tuples :math:`(x_i, y_i)`, where
:math:`x_i` is the :math:`i_{th}` individual of :math:`x`, while :math:`y_i` is the :math:`i_{th}` individual of :math:`y`.
In other words, the output would be a tensor :math:`\mathbf{out} \in [0., 1.]^n`. Notice that the output has one single dimension
since the predicate has been computed on :math:`n` tuples only, namely the tuples created constructing a one-to-one correspondence
between the individuals of :math:`x` and the individuals :math:`y`. At the end, after the quantification, a scalar is obtained like in the
case with the previous case.

The advantages of diagonal quantification are mani-fold:

1. it is a way to disable the :ref:`LTN broadcasting <broadcasting>`. Even if it has been designed to work with quantified statements, it could serve as a way to temporarily disable the :ref:`LTN broadcasting <broadcasting>` when computing some formulas. In fact, sometimes it is not necessary to compute a predicate on all the possible combinations of individuals of the variables in input;
2. it is more efficient compared to the usual quantification since it allows to avoid to compute a predicate on all the possible combinations of individuals of the variables appearing in the predicate;
3. it is useful when dealing with variables which represent machine learning examples. In many tasks, the dataset comes with some labelled examples. One variable could contain the examples, while another variable could contain the labels of the examples. With diagonal quantification we are able to force LTNtorch to use these variables with a one-to-one correspondence. This allows to avoid to compute formulas on combination of individuals which do not make any sense, for example a data sample labelled with a wrong label.

Notice that diagonal quantification expects the variables to have the same number of individuals, since a one-to-one correspondence has to be created.

In order to use diagonal quantification in LTNtorch, it is possible to use :func:`ltn.core.diag`.

Guarded quantification
----------------------
.. _guarded:

In some cases, it could be useful to quantify formulas only on variables' individuals which satisfy a given condition.
This is allowed in LTN by using the guarded quantification.

To make the concept clear, let us make a simple example. Suppose that we have a binary predicate :math:`P(a, b)`. Then, we have two variables,
:math:`x` and :math:`y`, containing sequences of points in :math:`\mathbb{R}^2`. Specifically,
:math:`\mathcal{G}(x) \in \mathbb{R}^{m_1 \times 2}` and :math:`\mathcal{G}(y) \in \mathbb{R}^{m_2 \times 2}`. So, :math:`x` contains
:math:`m_1` points, while :math:`m_2` contains :math:`m_2` points.

We have already seen that LTN allows to compute the formula: :math:`\forall x \forall y P(x, y)`. Also,
we know that LTNtorch firstly computes :math:`P(x, y)` and then aggregates on the dimensions specified by the quantified variables.

Suppose now that we want to compute the same formula :math:`\forall x \forall y P(x, y)` but quantifying only on the pairs of
points whose distance is lower than a certain threshold. We represent this threshold with the constant :math:`th`. In Real Logic,
it is possible to formalize this statement as :math:`\forall x \forall y : (dist(x, y) < th) \text{ } P(x, y)`, where :math:`dist` is
a function which computes the Euclidean distance between two points in :math:`\mathbb{R}^2`.

In order to compute this formula, LTNtorch follows this procedure:

1. it computes the result of the atomic formula :math:`P(x, y)`, which is a tensor :math:`\mathbf{out} \in [0., 1.]^{m_1 \times m_2}`;
2. it masks the truth values in the tensor :math:`\mathbf{out}` which do not satisfy the given condition (:math:`dist(x, y) < th`);
3. it aggregates the tensor :math:`mathbf{out}` on both dimensions, since both variables have been quantified. In this aggregation, the truth values masked by the previous step are not considered. Since both variables have been quantified, the result is a scalar in :math:`[0, 1]`.

For applying guarded quantification in LTNtorch, see :class:`ltn.core.Quantifier`. In particular, see `mask_fn`
and `cond_vars` parameters.