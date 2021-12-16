LTN broadcasting
================

.. _broadcasting:

LTN predicate case
------------------

In LTNtorch, when a predicate (:class:`ltn.core.Predicate`), function (:class:`ltn.core.Function`), or connective (:class:`ltn.core.Connective`) is
called, the framework automatically performs the broadcasting of the inputs.

To make a simple example, assume that we have two variables, :math:`x` and :math:`y`, with the following :ref:`groundings <notegrounding>`:

- :math:`\mathcal{G}(x)=[[1.6, 1.8, 2.3], [9.3, 4.5, 3.4]] \in \mathbb{R}^{2 \times 3}`;
- :math:`\mathcal{G}(y)=[[1.2, 1.3, 2.7, 10.4], [4.3, 5.6, 9.5, 1.3], [5.4, 1.5, 9.5, 8.4]] \in \mathbb{R}^{3 \times 4}``.

Variable :math:`x` has two individuals with three features each, while variable :math:`y` has three individuals with four
features each.

Now, let us assume that we have a binary predicate :math:`P(a, b)`, :ref:`grounded <notegrounding>` as
:math:`\mathcal{G}P(a, b | \theta) = a, b \mapsto \sigma(MLP_{\theta}(a, b))`. :math:`P(a, b)` is a learnable
predicate which maps from :math:`\mathbb{R}^7` to :math:`[0., 1.]`. In the notation, :math:`MLP_{\theta}` is a neural network,
parametrized by :math:`\theta`, with 7 input neurons, some hidden layers, and 1 output neuron. At the last layer has been applied
a logistic function to assure the output to be in the range :math:`[0., 1.]`. By doing so, the output of :math:`P(a, b)`
can be interpreted as fuzzy truth value.

Now, suppose that we want to compute :math:`P(x, y)`. LTNtorch automatically broadcasts the two variables before
computing the predicate. After the broadcasting, we will have the following inputs for our predicate:

:math:`\begin{bmatrix} 1.6 & 1.8 & 2.3\\ 1.6 & 1.8 & 2.3\\ 1.6 & 1.8 & 2.3\\ 9.3 & 4.5 & 3.4\\ 9.3 & 4.5 & 3.4\\ 9.3 & 4.5 & 3.4 \end{bmatrix} \in \mathbb{R}^{6 \times 3}`
for :math:`x`, and :math:`\begin{bmatrix} 1.2 & 1.3 & 2.7 & 10.4\\ 4.3 & 5.6 & 9.5 & 1.3\\ 5.4 & 1.5 & 9.5 & 8.4\\ 1.2 & 1.3 & 2.7 & 10.4\\ 4.3 & 5.6 & 9.5 & 1.3\\ 5.4 & 1.5 & 9.5 & 8.4 \end{bmatrix} \in \mathbb{R}^{6 \times 4}` for :math:`y`.

Now, it is possible to observe that if we concatenate these two tensors on the first dimension (`torch.cat([x, y], dim=1)`), we obtain the following input for our predicate:

:math:`\begin{bmatrix} 1.6 & 1.8 & 2.3 & 1.2 & 1.3 & 2.7 & 10.4\\ 1.6 & 1.8 & 2.3 & 4.3 & 5.6 & 9.5 & 1.3\\ 1.6 & 1.8 & 2.3 & 5.4 & 1.5 & 9.5 & 8.4\\ 9.3 & 4.5 & 3.4 & 1.2 & 1.3 & 2.7 & 10.4\\ 9.3 & 4.5 & 3.4 & 4.3 & 5.6 & 9.5 & 1.3\\ 9.3 & 4.5 & 3.4 & 5.4 & 1.5 & 9.5 & 8.4 \end{bmatrix} \in \mathbb{R}^{6 \times 7}`.

This tensor contains all the possible combinations of the individuals of
the two variables, that are 6. After the computation of the predicate, LTNtorch organizes the output in a tensor :math:`\mathbf{out} \in [0., 1.]^{2 \times 3}`, where
the first dimension is related with variable :math:`x`, while the second dimension with variable :math:`y`.
In :math:`\mathbf{out}[0, 0]` there will be the result of the evaluation of :math:`P(x, y)` on the first individual of
:math:`x`, namely :math:`[1.6, 1.8, 2.3]`, and first individual of :math:`y`, namely :math:`[1.2, 1.3, 2.7, 10.4]`, in :math:`\mathbf{out}[0, 1]` there will be the result of the evaluation of :math:`P(x, y)` on the first individual of
:math:`x`, namely :math:`[1.6, 1.8, 2.3]`, and second individual of :math:`y`, namely :math:`[4.3, 5.6, 9.5, 1.3]`, and so forth.

To conclude this note, in LTNtorch, the output of predicates, functions, connectives, and quantifiers are
:ref:`LTNObject <noteltnobject>` instances. In the case of our example, the output of predicate :math:`P` is
an `LTNObject` with the following attributes:

- `value` :math:`\in [0., 1.]^{2 \times 3}`;
- `free_vars` = `['x', 'y']`.

Note that we have analyzed just an atomic formula (predicate) in this scenario. Since the variables appearing in the formula are not quantified, the
free variables in the output are both :math:`x` and :math:`y`. If instead of :math:`P(x, y)` we had to compute :math:`\forall x P(x, y)`,
the `free_vars` attribute would have been equal to `['y']`. Finally, if we had to compute :math:`\forall x \forall y P(x, y)`,
the `free_vars` attribute would have been an empty list.

LTN function case
-----------------

The same scenario explained above can be applied to an LTN function (:class:`ltn.core.Function`) instead of an LTN predicate (:class:`ltn.core.Predicate`). Suppose we have the same
variables, :math:`x` and :math:`y`, with the same :ref:`groundings <notegrounding>`, :math:`\mathcal{G}(x)` and :math:`\mathcal{G}(y)`.
Then, suppose we have a 2-ary (2 inputs) logical function :math:`f`, :ref:`grounded <notegrounding>` as
:math:`\mathcal{G}f(a, b | \theta) = a, b \mapsto MLP_{\theta}(a, b)`.

In this case, :math:`MLP_{\theta}` is a neural network, parametrized by :math:`\theta`, with 7 input neurons, some hidden layers, and
five output neurons. In other words, :math:`f` is a learnable function which maps from individuals in :math:`\mathbb{R}^7` to individuals in :math:`\mathbb{R}^5`.
Note that, in this case, we do not have applied a logistic function to the output. In fact, logical functions do not have
the constraint of having outputs in the range :math:`[0., 1.]`.

LTNtorch applies the same broadcasting that we have seen above to the inputs of function :math:`f`. The only difference is
on how the output is organized. In the case of an LTN function, the output is organized in a tensor where the first :math:`k`
dimensions are related with the variables given in input, while the remaining dimensions are related with the features of the individuals in output.

In our scenario, the output of :math:`f(x, y)` is a tensor :math:`\mathbf{out} \in \mathbb{R}^{2 \times 3 \times 5}`. The first
dimension is related with variable :math:`x`, the second dimension with variable :math:`y`, while the third dimension with the
features of the individuals in output. In :math:`\mathbf{out}[0, 0]` there will be the result of the evaluation of :math:`f(x, y)` on the first individual of
:math:`x`, namely :math:`[1.6, 1.8, 2.3]`, and first individual of :math:`y`, namely :math:`[1.2, 1.3, 2.7, 10.4]`, in :math:`\mathbf{out}[0, 1]` there will be the result of the evaluation of :math:`f(x, y)` on the first individual of
:math:`x`, namely :math:`[1.6, 1.8, 2.3]`, and second individual of :math:`y`, namely :math:`[4.3, 5.6, 9.5, 1.3]`, and so forth.

LTN connective case
-------------------

LTNtorch applies the LTN broadcasting also before computing a logical connective. To make the concept clear, let us make
a simple example.

Suppose that we have variables :math:`x`, :math:`y`, :math:`z`, and :math:`u`, with :ref:`groundings <notegrounding>`:

- :math:`mathcal{G}(x) \in \mathbb{R}^{2 \times 3}`, namely :math:`x` contains two individuals in :math:`\mathbb{R}^3`;
- :math:`mathcal{G}(y) \in \mathbb{R}^{4 \times 3 \times 5}`, namely :math:`y` contains four individuals in :math:`\mathbb{R}^{3 \times 5}`;
- :math:`mathcal{G}(z) \in \mathbb{R}^{3 \times 5}`, namely :math:`z` contains three individuals in :math:`\mathbb{R}^5`;
- :math:`mathcal{G}(u) \in \mathbb{R}^{6 \times 2}`, namely :math:`u` contains six individuals in :math:`\mathbb{R}^2`.

Then, suppose that we have two binary predicates, :math:`P(a, b)` and :math:`Q(a, b)`. :math:`P` maps from :math:`\mathbb{R}^18` to
:math:`[0., 1.]`, while :math:`Q` maps from :math:`\mathbb{R}^7` to :math:`[0., 1.]`.

Suppose now that we want to compute the formula: :math:`P(x, y) \land Q(z, u)`. In order to evaluate this formula, LTNtorch
follows the following procedure:

1. it computes the result of the atomic formula :math:`P(x, y)`, which is a tensor :math:`\mathbf{out_1} \in [0., 1.]^{2 \times 4}`. Note that before the computation of :math:`P(x, y)`, LTNtorch performed the LTN broadcasting of variables :math:`x` and :math:`y`;
2. it computes the result of the atomic formula :math:`Q(z, u)`, which is a tensor :math:`\mathbf{out_2} \in [0., 1.]^{3 \times 6}`. Note that before the computation of :math:`P(z, u)`, LTNtorch performed the LTN broadcasting of variables :math:`z` and :math:`u`;
3. it performs the LTN broadcasting of :math:`\mathbf{out_1}` and :math:`\mathbf{out_2}`;
4. it applies the fuzzy conjunction :math:`\mathbf{out_1} \land_{fuzzy} \mathbf{out_2}`. The result is a tensor :math:`\mathbf{out} \in [0., 1.]^{2 \times 4 \times 3 \times 6}`.

Notice that the output of a logical connective is always wrapped into an :ref:`LTN object <noteltnobject>`, like it happens for predicates and functions.
In this simple example, the `LTNObject` produced by the fuzzy conjunction has the following attributes:

- `value` = :math:`\mathbf{out}`;
- `free_vars = ['x', 'y', 'z', 'u']`.

Notice that `free_vars` contains the labels of all the variables appearing in :math:`P(x, y) \land Q(z, u)`. This is due
to the fact that all the variables are free in the formula since are not quantified by any logical quantifier. Notice also
that :math:`\mathbf{out}` has four dimensions, one for each variable appearing in the formula. These dimensions can be
indexed to retrieve the evaluation of :math:`P(x, y) \land Q(z, u)` on a specific combination of individuals of :math:`x,y,z,u`.
For example, :math:`\mathbf{out}[0, 0, 0, 0]` contains the evaluation of :math:`P(x, y) \land Q(z, u)` on the first individuals
of all the variables, while :math:`\mathbf{out}[0, 1, 0, 0]` contains the evaluation of :math:`P(x, y) \land Q(z, u)` on the first individuals
of :math:`x,z,u` and second individual of :math:`y`.