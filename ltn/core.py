"""
The `ltn.core` module contains the main functionalities of LTNtorch. In particular, it contains the definitions of
constants, variables, predicates, functions, connectives, and quantifiers.
"""

import copy
import torch
from torch import nn
import numpy as np
import ltn
import types


class LTNObject:
    r"""
    Class representing a generic :ref:`LTN object <noteltnobject>`.

    In LTNtorch, LTN objects are constants, variables, and outputs of predicates, formulas, functions, connectives,
    and quantifiers.

    Parameters
    ----------
    value : :class:`torch.Tensor`
        The :ref:`grounding <notegrounding>` (value) of the LTN object.
    var_labels : :obj:`list` of :obj:`str`
        The labels of the free variables contained in the LTN object.

    Raises
    ------
    :class:`TypeError`
        Raises when the types of the input parameters are incorrect.

    Attributes
    ----------
    value : :class:`torch.Tensor`
        See `value` parameter.
    free_vars : :obj:`list` of :obj:`str`
        See `var_labels` parameter.

    Notes
    -----
    - in LTNtorch, the :ref:`groundings <notegrounding>` of the LTN objects (symbols) are represented using PyTorch tensors, namely :class:`torch.Tensor` instances;
    - `LTNObject` is used by LTNtorch internally. The user should not create `LTNObject` instances by his/her own, unless strictly necessary.
    """

    def __init__(self, value, var_labels):
        # check inputs before creating the object
        if not isinstance(value, torch.Tensor):
            raise TypeError("LTNObject() : argument 'value' (position 1) must be a torch.Tensor, not "
                            + str(type(value)))
        if not (isinstance(var_labels, list) and (all(isinstance(x, str) for x in var_labels) if var_labels else True)):
            raise TypeError("LTNObject() : argument 'var_labels' (position 2) must be a list of strings, not "
                            + str(type(var_labels)))
        self.value = value
        self.free_vars = var_labels

    def __repr__(self):
        return "LTNObject(value=" + str(self.value) + ", free_vars=" + str(self.free_vars) + ")"

    def shape(self):
        """
        Returns the shape of the :ref:`grounding <notegrounding>` of the LTN object.

        Returns
        -------
        :class:`torch.Size`
            The shape of the :ref:`grounding <notegrounding>` of the LTN object.
        """
        return self.value.shape


class Constant(LTNObject):
    r"""
    Class representing an LTN constant.

    An LTN constant denotes an individual :ref:`grounded <notegrounding>` as a tensor in the Real field.
    The individual can be pre-defined (fixed data point) or learnable (embedding).

    Parameters
    ----------
    value : :class:`torch.Tensor`
        The :ref:`grounding <notegrounding>` of the LTN constant. It can be a tensor of any order.
    trainable : :obj:`bool`, default=False
        Flag indicating whether the LTN constant is trainable (embedding) or not.

    Notes
    -----
    - LTN constants are :ref:`LTN objects <noteltnobject>`. :class:`ltn.core.Constant` is a subclass of :class:`ltn.core.LTNObject`;
    - the attribute `free_vars` for LTN constants is an empty list, since a constant does not have variables by definition;
    - if parameter `trainable` is set to `True`, the LTN constant becomes trainable, namely an embedding;
    - if parameter `trainable` is set to `True`, then the `value` attribute of the LTN constant will be used as an initialization for the embedding of the constant.

    Examples
    --------
    Non-trainable constant

    >>> import ltn
    >>> import torch
    >>> c = ltn.Constant(torch.tensor([3.4, 5.4, 4.3]))
    >>> print(c)
    Constant(value=tensor([3.4000, 5.4000, 4.3000]), free_vars=[])
    >>> print(c.value)
    tensor([3.4000, 5.4000, 4.3000])
    >>> print(c.free_vars)
    []
    >>> print(c.shape())
    torch.Size([3])

    Trainable constant

    >>> t_c = ltn.Constant(torch.tensor([[3.4, 2.3, 5.6],
    ...                                  [6.7, 5.6, 4.3]]), trainable=True)
    >>> print(t_c)
    Constant(value=tensor([[3.4000, 2.3000, 5.6000],
            [6.7000, 5.6000, 4.3000]], requires_grad=True), free_vars=[])
    >>> print(t_c.value)
    tensor([[3.4000, 2.3000, 5.6000],
            [6.7000, 5.6000, 4.3000]], requires_grad=True)
    >>> print(t_c.free_vars)
    []
    >>> print(t_c.shape())
    torch.Size([2, 3])
    """
    def __init__(self, value, trainable=False):
        # create sub-object of type LTNObject
        super(Constant, self).__init__(value, [])
        self.value = self.value.to(ltn.device)
        if trainable:
            # we need to ensure that the tensor is float to set the required_grad to True, since PyTorch needs a float
            # tensor in this case
            self.value = self.value.float()
            self.value.requires_grad = trainable

    def __repr__(self):
        return "Constant(value=" + str(self.value) + ", free_vars=" + str(self.free_vars) + ")"


class Variable(LTNObject):
    r"""
    Class representing an LTN variable.

    An LTN variable denotes a sequence of individuals. It is :ref:`grounded <notegrounding>` as a sequence of
    tensors (:ref:`groundings <notegrounding>` of individuals) in the real field.

    Parameters
    ----------
    var_label : :obj:`str`
        Name of the variable.
    individuals : :class:`torch.Tensor`
        Sequence of individuals (tensors) that becomes the :ref:`grounding <notegrounding>` the LTN variable.
    add_batch_dim : :obj:`bool`, default=True
        Flag indicating whether a batch dimension (first dimension) has to be added to the
        `vale` of the variable or not. If `True`, a dimension will be added only if the
        `value` attribute of the LTN variable has one single dimension. In all the other cases, the first dimension
        will be considered as batch dimension, so no dimension will be added.

    Raises
    ------
    :class:`TypeError`
        Raises when the types of the input parameters are not correct.
    :class:`ValueError`
        Raises when the value of the `var_label` parameter is not correct.

    Notes
    -----
    - LTN variables are :ref:`LTN objects <noteltnobject>`. :class:`ltn.core.Variable` is a subclass of :class:`ltn.core.LTNObject`;
    - the first dimension of an LTN variable is associated with the number of individuals in the variable, while the other dimensions are associated with the features of the individuals;
    - setting `add_batch_dim` to `False` is useful, for instance, when an LTN variable is used to denote a sequence of indexes (for example indexes for retrieving values in tensors);
    - variable labels starting with '_diag' are reserved for diagonal quantification (:func:`ltn.core.diag`).

    Examples
    --------
    `add_batch_dim=True` has no effects on the variable since its `value` has more than one dimension, namely there is
    already a batch dimension.

    >>> import ltn
    >>> import torch
    >>> x = ltn.Variable('x', torch.tensor([[3.4, 4.5],
    ...                                     [6.7, 9.6]]), add_batch_dim=True)
    >>> print(x)
    Variable(value=tensor([[3.4000, 4.5000],
            [6.7000, 9.6000]]), free_vars=['x'])
    >>> print(x.value)
    tensor([[3.4000, 4.5000],
            [6.7000, 9.6000]])
    >>> print(x.free_vars)
    ['x']
    >>> print(x.shape())
    torch.Size([2, 2])

    `add_bath_dim=True` adds a batch dimension to the `value` of the variable since it has only one dimension.

    >>> y = ltn.Variable('y', torch.tensor([3.4, 4.5, 8.9]), add_batch_dim=True)
    >>> print(y)
    Variable(value=tensor([[3.4000],
            [4.5000],
            [8.9000]]), free_vars=['y'])
    >>> print(y.value)
    tensor([[3.4000],
            [4.5000],
            [8.9000]])
    >>> print(y.free_vars)
    ['y']
    >>> print(y.shape())
    torch.Size([3, 1])

    `add_batch_dim=False` tells to LTNtorch to not add a batch dimension to the `value` of the variable. This is useful
    when a variable contains a sequence of indexes.

    >>> z = ltn.Variable('z', torch.tensor([1, 2, 3]), add_batch_dim=False)
    >>> print(z)
    Variable(value=tensor([1, 2, 3]), free_vars=['z'])
    >>> print(z.value)
    tensor([1, 2, 3])
    >>> print(z.free_vars)
    ['z']
    >>> print(z.shape())
    torch.Size([3])
    """
    def __init__(self, var_label, individuals, add_batch_dim=True):
        # check inputs
        if not isinstance(var_label, str):
            raise TypeError("Variable() : argument 'var_label' (position 1) must be str, not " + str(type(var_label)))
        if var_label.startswith("diag_"):
            raise ValueError("Labels starting with 'diag_' are reserved for diagonal quantification.")
        if not isinstance(individuals, torch.Tensor):
            raise TypeError("Variable() : argument 'individuals' (position 2) must be a torch.Tensor, not "
                            + str(type(individuals)))

        super(Variable, self).__init__(individuals, [var_label])

        if isinstance(self.value, torch.DoubleTensor):
            # we ensure that the tensor will be a float tensor and not a double tensor to avoid type incompatibilities
            self.value = self.value.float()

        if len(self.value.shape) == 1 and add_batch_dim:
            # adds a dimension to transform the input in a sequence of individuals in the case in which it is not
            # already a sequence of individuals but just a tensor with only one dimension
            # the dimension added will be the batch dimension
            # Example: [3, 1, 2] is transformed into [[3], [1], [2]] if individuals has one dimension and add_batch_dim
            # is set to True
            self.value = self.value.view(self.value.shape[0], 1)

        self.value = self.value.to(ltn.device)
        self.latent_var = var_label

    def __repr__(self):
        return "Variable(value=" + str(self.value) + ", free_vars=" + str(self.free_vars) + ")"


def process_ltn_objects(objects):
    """
    This function prepares the list of LTN objects given in input for a predicate, function, or connective computation.
    In particular, it makes the shapes of the objects compatible, in such a way the logical operation that has to be
    computed after this pre-processing can be done by using element-wise operations.
    For example, if we have two variables in input that have different shapes or number of individuals, this function
    will change the shape of one variable to match the shape of the second one. This reshaping is done by adding new
    dimensions and repeating the existing ones along the new dimensions.

    After these operations have been computed, the objects with compatible shapes are returned as a list and are ready
    for the computation of the predicate, function, or connective. Along with the processed objects, the labels of the
    variables contained in these objects and the number of individuals of each variable are returned. This is needed
    to perform reshapes of the output after the computation of a predicate or function, since each axis of the output
    has to be related to one specific variable.

    Parameters
    ----------
    objects: :obj:`list`
        List of LTN objects of potentially different shapes for which we need to make the shape compatible.

    Returns
    ----------
    :obj:`list`
        The same list given in input but with new LTN objects which now have compatible shapes.
    :obj:`list`
        List of labels of all the variables which appear in the LTN objects given in input.
    :obj:`list`
        List of integers which contains the number of individuals of each variable contained in the previous list.

    Raises
    ----------
    :class:`TypeError`
        Raises when the type of the input parameter is incorrect.
    """
    # check inputs
    if not (isinstance(objects, list) and all(isinstance(x, LTNObject) for x in objects)):
        raise TypeError("The objects should be a list of LTNObject")
    # we perform a deep copy to avoid problems if the LTN objects given in input are used in other formulas
    # we want to give the user the possibility to use the same object in different formulas
    # if we do not perform a deep copy, the object itself will be changed by this function even outside of the function
    # due to a side effect
    # note that we copy only if the input object is a constant/variable with grad_fn or if the object has not
    # grad_fn attribute, namely it is a leaf tensor
    objects_ = [LTNObject(torch.clone(o.value), copy.deepcopy(o.free_vars))
                if (o.value.grad_fn is None or (isinstance(o, (Constant, Variable)) and o.value.grad_fn is not None))
                else o for o in objects]
    # this deep copy is necessary to avoid the function directly changes the free
    # variables and values contained in the given LTN objects. We do not want to directly change the input objects
    # Instead, we need to create new objects based on the input objects since it is possible we have to reuse the
    # input objects again in coming steps.
    vars_to_n = {}  # dict which maps each var to the number of its individuals
    for o in objects_:
        for (v_idx, v) in enumerate(o.free_vars):
            vars_to_n[v] = o.shape()[v_idx]
    vars = list(vars_to_n.keys())  # list of var labels
    n_individuals_per_var = list(vars_to_n.values())  # list of n individuals for each var
    proc_objs = []  # list of processed objects
    for o in objects_:
        vars_in_obj = o.free_vars
        vars_not_in_obj = list(set(vars).difference(vars_in_obj))
        for new_var in vars_not_in_obj:
            new_var_idx = len(vars_in_obj)
            o.value = torch.unsqueeze(o.value, dim=new_var_idx)
            # repeat existing dims along the new dim related to the new variable that has to be added to the object
            o.value = torch.repeat_interleave(o.value, repeats=vars_to_n[new_var], dim=new_var_idx)
            vars_in_obj.append(new_var)

        # permute the dimensions of the object in such a way the shapes of the processed objects is the same
        # the shape is computed based on the order in which the variables are found at the beginning of this function
        dims_permutation = [vars_in_obj.index(var) for var in vars] + list(range(len(vars_in_obj), len(o.shape())))
        o.value = o.value.permute(dims_permutation)

        # this flats the batch dimension of the processed LTN object if the flat is set to True
        flatten_shape = [-1] + list(o.shape()[len(vars_in_obj)::])
        o.value = torch.reshape(o.value, shape=tuple(flatten_shape))

        # change the free variables of the LTN object since it contains now new variables on it
        # note that at the end of this function, all the LTN objects given in input will be defined on the same
        # variables since they are now compatible to be processed by element-wise predicate, function, or connective
        # operators
        o.free_vars = vars
        proc_objs.append(o)

    return proc_objs, vars, n_individuals_per_var


class LambdaModel(nn.Module):
    """
    Simple `nn.Module` that implements a non-trainable model based on a lambda function or a function.

    Parameters
    ----------
    func: :class:`function`
        Lambda function or function that has to be applied in the forward of the model.

    Attributes
    ---------
    func: :class:`function`
        See func parameter.
    """
    def __init__(self, func):
        super(LambdaModel, self).__init__()
        self.func = func

    def forward(self, *x):
        return self.func(*x)


class Predicate(nn.Module):
    """
    Class representing an LTN predicate.

    An LTN predicate is :ref:`grounded <notegrounding>` as a mathematical function (either pre-defined or learnable)
    that maps from some n-ary domain of individuals to a real number in [0,1] (fuzzy), which can be interpreted as a
    truth value.

    In LTNtorch, the inputs of a predicate are automatically broadcasted before the computation of the predicate,
    if necessary. Moreover, the output is organized in a tensor where each dimension is related to
    one variable given in input. See :ref:`LTN broadcasting <broadcasting>` for more information.

    Parameters
    ----------
    model : :class:`torch.nn.Module`, default=None
        PyTorch model that becomes the :ref:`grounding <notegrounding>` of the LTN predicate.
    func : :obj:`function`, default=None
        Function that becomes the :ref:`grounding <notegrounding>` of the LTN predicate.

    Notes
    -----
    - the output of an LTN predicate is always an :ref:`LTN object <noteltnobject>` (:class:`ltn.core.LTNObject`);
    - LTNtorch allows to define a predicate using a trainable model **or** a python function, not both;
    - defining a predicate using a python function is suggested only for simple and non-learnable mathematical operations;
    - examples of LTN predicates could be similarity measures, classifiers, etc;
    - the output of an LTN predicate must be always in the range [0., 1.]. Outputs outside of this range are not allowed;
    - evaluating a predicate with one variable of :math:`n` individuals yields :math:`n` output values, where the :math:`i_{th}` output value corresponds to the predicate calculated with the :math:`i_{th}` individual;
    - evaluating a predicate with :math:`k` variables :math:`(x_1, \dots, x_k)` with respectively :math:`n_1, \dots, n_k` individuals each, yields a result with :math:`n_1 * \dots * n_k` values. The result is organized in a tensor where the first :math:`k` dimensions can be indexed to retrieve the outcome(s) that correspond to each variable;
    - the attribute `free_vars` of the `LTNobject` output by the predicate tells which dimension corresponds to which variable in the `value` of the `LTNObject`. See :ref:`LTN broadcasting <broadcasting>` for more information;
    - to disable the :ref:`LTN broadcasting <broadcasting>`, see :func:`ltn.core.diag()`.

    Attributes
    ----------
    model : :class:`torch.nn.Module` or :obj:`function`
        The :ref:`grounding <notegrounding>` of the LTN predicate.

    Raises
    ----------
    :class:`TypeError`
        Raises when the types of the input parameters are incorrect.

    :class:`ValueError`
        Raises when the values of the input parameters are incorrect.

    Examples
    --------
    Unary predicate defined using a :class:`torch.nn.Sequential`.

    >>> import ltn
    >>> import torch
    >>> predicate_model = torch.nn.Sequential(
    ...                         torch.nn.Linear(4, 2),
    ...                         torch.nn.ELU(),
    ...                         torch.nn.Linear(2, 1),
    ...                         torch.nn.Sigmoid()
    ...                   )
    >>> p = ltn.Predicate(model=predicate_model)
    >>> print(p)
    Predicate(model=Sequential(
      (0): Linear(in_features=4, out_features=2, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=2, out_features=1, bias=True)
      (3): Sigmoid()
    ))

    Unary predicate defined using a function. Note that `torch.sum` is performed on `dim=1`. This is because in LTNtorch
    the first dimension (`dim=0`) is related to the batch dimension, while other dimensions are related to the features
    of the individuals. Notice that the output of the print is `Predicate(model=LambdaModel())`. This indicates that the
    LTN predicate has been defined using a function, through the `func` parameter of the constructor.

    >>> p_f = ltn.Predicate(func=lambda x: torch.nn.Sigmoid()(torch.sum(x, dim=1)))
    >>> print(p_f)
    Predicate(model=LambdaModel())

    Binary predicate defined using a :class:`torch.nn.Module`. Note the call to `torch.cat` to merge
    the two inputs of the binary predicate.

    >>> class PredicateModel(torch.nn.Module):
    ...     def __init__(self):
    ...         super(PredicateModel, self).__init__()
    ...         elu = torch.nn.ELU()
    ...         sigmoid = torch.nn.Sigmoid()
    ...         self.dense1 = torch.nn.Linear(4, 5)
    ...         self.dense2 = torch.nn.Linear(5, 1)
    ...
    ...     def forward(self, x, y):
    ...         x = torch.cat([x, y], dim=1)
    ...         x = self.elu(self.dense1(x))
    ...         out = self.sigmoid(self.dense2(x))
    ...         return out
    ...
    >>> predicate_model = PredicateModel()
    >>> b_p = ltn.Predicate(model=predicate_model)
    >>> print(b_p)
    Predicate(model=PredicateModel(
      (dense1): Linear(in_features=4, out_features=5, bias=True)
      (dense2): Linear(in_features=5, out_features=1, bias=True)
    ))

    Binary predicate defined using a function. Note the call to `torch.cat` to merge the two inputs of the
    binary predicate.

    >>> b_p_f = ltn.Predicate(func=lambda x, y: torch.nn.Sigmoid()(
    ...                                             torch.sum(torch.cat([x, y], dim=1), dim=1)
    ...                                         ))
    >>> print(b_p_f)
    Predicate(model=LambdaModel())

    Evaluation of a unary predicate on a constant. Note that:

    - the predicate returns a :class:`ltn.core.LTNObject` instance;
    - since a constant has been given, the `LTNObject` in output does not have free variables;
    - the shape of the `LTNObject` in output is empty since the predicate has been evaluated on a constant, namely on one single individual;
    - the attribute `value` of the `LTNObject` in output contains the result of the evaluation of the predicate;
    - the `value` is in the range [0., 1.] since it has to be interpreted as a truth value. This is assured thanks to the *sigmoid function* in the definition of the predicate.

    >>> c = ltn.Constant(torch.tensor([0.5, 0.01, 0.34, 0.001]))
    >>> out = p_f(c)
    >>> print(type(out))
    <class 'ltn.core.LTNObject'>
    >>> print(out)
    LTNObject(value=tensor(0.7008), free_vars=[])
    >>> print(out.value)
    tensor(0.7008)
    >>> print(out.free_vars)
    []
    >>> print(out.shape())
    torch.Size([])

    Evaluation of a unary predicate on a variable. Note that:

    - since a variable has been given, the `LTNObject` in output has one free variable;
    - the shape of the `LTNObject` in output is 2 since the predicate has been evaluated on a variable with two individuls.

    >>> v = ltn.Variable('v', torch.tensor([[0.4, 0.3],
    ...                                     [0.32, 0.043]]))
    >>> out = p_f(v)
    >>> print(out)
    LTNObject(value=tensor([0.6682, 0.5898]), free_vars=['v'])
    >>> print(out.value)
    tensor([0.6682, 0.5898])
    >>> print(out.free_vars)
    ['v']
    >>> print(out.shape())
    torch.Size([2])

    Evaluation of a binary predicate on a variable and a constant. Note that:

    - like in the previous example, the `LTNObject` in output has just one free variable, since only one variable has been given to the predicate;
    - the shape of the `LTNObject` in output is 2 since the predicate has been evaluated on a variable with two individuals. The constant does not add dimensions to the output.

    >>> v = ltn.Variable('v', torch.tensor([[0.4, 0.3],
    ...                                     [0.32, 0.043]]))
    >>> c = ltn.Constant(torch.tensor([0.4, 0.04, 0.23, 0.43]))
    >>> out = b_p_f(v, c)
    >>> print(out)
    LTNObject(value=tensor([0.8581, 0.8120]), free_vars=['v'])
    >>> print(out.value)
    tensor([0.8581, 0.8120])
    >>> print(out.free_vars)
    ['v']
    >>> print(out.shape())
    torch.Size([2])

    Evaluation of a binary predicate on two variables. Note that:

    - since two variables have been given, the `LTNObject` in output has two free variables;
    - the shape of the `LTNObject` in output is `(2, 3)` since the predicate has been evaluated on a variable with two individuals and a variable with three individuals;
    - the first dimension is dedicated to variable `x`, which is also the first one appearing in `free_vars`, while the second dimension is dedicated to variable `y`, which is the second one appearing in `free_vars`;
    - it is possible to access the `value` attribute for getting the results of the predicate. For example, at position `(1, 2)` there is the evaluation of the predicate on the second individual of `x` and third individuals of `y`.

    >>> x = ltn.Variable('x', torch.tensor([[0.4, 0.3],
    ...                                     [0.32, 0.043]]))
    >>> y = ltn.Variable('y', torch.tensor([[0.4, 0.04, 0.23],
    ...                                     [0.2, 0.04, 0.32],
    ...                                     [0.06, 0.08, 0.3]]))
    >>> out = b_p_f(x, y)
    >>> print(out)
    LTNObject(value=tensor([[0.7974, 0.7790, 0.7577],
            [0.7375, 0.7157, 0.6906]]), free_vars=['x', 'y'])
    >>> print(out.value)
    tensor([[0.7974, 0.7790, 0.7577],
            [0.7375, 0.7157, 0.6906]])
    >>> print(out.free_vars)
    ['x', 'y']
    >>> print(out.shape())
    torch.Size([2, 3])
    >>> print(out.value[1, 2])
    tensor(0.6906)
    """
    def __init__(self, model=None, func=None):
        """
        Initializes the LTN predicate in two different ways:
            1. if `model` is not None, it initializes the predicate with the given PyTorch model;
            2. if `model` is None, it uses the `func` as a function to define
            the LTN predicate. Note that, in this case, the LTN predicate is not learnable. So, the lambda function has
            to be used only for simple predicates.
        """
        super(Predicate, self).__init__()
        if model is not None and func is not None:
            raise ValueError("Both model and func parameters have been specified. Expected only one of "
                             "the two parameters to be specified.")

        if model is None and func is None:
            raise ValueError("Both model and func parameters have not been specified. Expected one of the two "
                             "parameters to be specified.")

        if model is not None:
            if not isinstance(model, nn.Module):
                raise TypeError("Predicate() : argument 'model' (position 1) must be a torch.nn.Module, "
                                "not " + str(type(model)))
            self.model = model
        else:
            if not isinstance(func, types.LambdaType):
                raise TypeError("Predicate() : argument 'func' (position 2) must be a function, "
                                "not " + str(type(model)))
            self.model = LambdaModel(func)

    def __repr__(self):
        return "Predicate(model=" + str(self.model) + ")"

    def forward(self, *inputs, **kwargs):
        """
        It computes the output of the predicate given some :ref:`LTN objects <noteltnobject>` in input.

        Before computing the predicate, it performs the :ref:`LTN broadcasting <broadcasting>` of the inputs.

        Parameters
        ----------
        inputs : :obj:`tuple` of :class:`ltn.core.LTNObject`
            Tuple of :ref:`LTN objects <noteltnobject>` for which the predicate has to be computed.

        Returns
        ----------
        :class:`ltn.core.LTNObject`
            An :ref:`LTNObject <noteltnobject>` whose `value` attribute contains the truth values representing the result of the
            predicate, while `free_vars` attribute contains the labels of the free variables contained in the result.

        Raises
        ----------
        :class:`TypeError`
            Raises when the types of the inputs are incorrect.

        :class:`ValueError`
            Raises when the values of the output are not in the range [0., 1.].
        """
        inputs = list(inputs)
        if not all(isinstance(x, LTNObject) for x in inputs):
            raise TypeError("Expected parameter 'inputs' to be a tuple of LTNObject, but got " + str([type(i)
                                                                                                      for i in inputs]))

        proc_objs, output_vars, output_shape = process_ltn_objects(inputs)

        # the management of the input is left to the model or the lambda function
        output = self.model(*[o.value for o in proc_objs], **kwargs)

        # check if output of predicate contains only truth values, namely values in the range [0., 1.]
        if not torch.all(torch.where(torch.logical_and(output >= 0., output <= 1.), 1., 0.)):
            raise ValueError("Expected the output of a predicate to be in the range [0., 1.], but got some values "
                             "outside of this range. Check your predicate implementation!")

        output = torch.reshape(output, tuple(output_shape))
        # we assure the output is float in the case it is double to avoid type incompatibilities
        output = output.float()

        return LTNObject(output, output_vars)


class Function(nn.Module):
    r"""
    Class representing LTN functions.

    An LTN function is :ref:`grounded <notegrounding>` as a mathematical function (either pre-defined or learnable)
    that maps from some n-ary domain of individuals to a tensor (individual) in the Real field.

    In LTNtorch, the inputs of a function are automatically broadcasted before the computation of the function,
    if necessary. Moreover, the output is organized in a tensor where the first :math:`k` dimensions are related
    with the :math:`k` variables given in input, while the last dimensions are related with the features of the
    individual in output. See :ref:`LTN broadcasting <broadcasting>` for more information.

    Parameters
    ----------
    model : :class:`torch.nn.Module`, default=None
        PyTorch model that becomes the :ref:`grounding <notegrounding>` of the LTN function.
    func : :obj:`function`, default=None
        Function that becomes the :ref:`grounding <notegrounding>` of the LTN function.

    Attributes
    ----------
    model : :class:`torch.nn.Module` or :obj:`function`
        The :ref:`grounding <notegrounding>` of the LTN function.

    Raises
    ----------
    :class:`TypeError`
        Raises when the types of the input parameters are incorrect.

    :class:`ValueError`
        Raises when the values of the input parameters are incorrect.

    Notes
    -----
    - the output of an LTN function is always an :ref:`LTN object <noteltnobject>` (:class:`ltn.core.LTNObject`);
    - LTNtorch allows to define a function using a trainable model **or** a python function, not both;
    - defining an LTN function using a python function is suggested only for simple and non-learnable mathematical operations;
    - examples of LTN functions could be distance functions, regressors, etc;
    - differently from LTN predicates, the output of an LTN function has no constraints;
    - evaluating a function with one variable of :math:`n` individuals yields :math:`n` output values, where the :math:`i_{th}` output value corresponds to the function calculated with the :math:`i_{th}` individual;
    - evaluating a function with :math:`k` variables :math:`(x_1, \dots, x_k)` with respectively :math:`n_1, \dots, n_k` individuals each, yields a result with :math:`n_1 * \dots * n_k` values. The result is organized in a tensor where the first :math:`k` dimensions can be indexed to retrieve the outcome(s) that correspond to each variable;
    - the attribute `free_vars` of the `LTNobject` output by the function tells which dimension corresponds to which variable in the `value` of the `LTNObject`. See :ref:`LTN broadcasting <broadcasting>` for more information;
    - to disable the :ref:`LTN broadcasting <broadcasting>`, see :func:`ltn.core.diag()`.

    Examples
    --------
    Unary function defined using a :class:`torch.nn.Sequential`.

    >>> import ltn
    >>> import torch
    >>> function_model = torch.nn.Sequential(
    ...                         torch.nn.Linear(4, 3),
    ...                         torch.nn.ELU(),
    ...                         torch.nn.Linear(3, 2)
    ...                   )
    >>> f = ltn.Function(model=function_model)
    >>> print(f)
    Function(model=Sequential(
      (0): Linear(in_features=4, out_features=3, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=3, out_features=2, bias=True)
    ))

    Unary function defined using a function. Note that `torch.sum` is performed on `dim=1`. This is because in LTNtorch
    the first dimension (`dim=0`) is related to the batch dimension, while other dimensions are related to the features
    of the individuals. Notice that the output of the print is `Function(model=LambdaModel())`. This indicates that the
    LTN function has been defined using a function, through the `func` parameter of the constructor.

    >>> f_f = ltn.Function(func=lambda x: torch.repeat_interleave(
    ...                                              torch.sum(x, dim=1, keepdim=True), 2, dim=1)
    ...                                         )
    >>> print(f_f)
    Function(model=LambdaModel())

    Binary function defined using a :class:`torch.nn.Module`. Note the call to `torch.cat` to merge
    the two inputs of the binary function.

    >>> class FunctionModel(torch.nn.Module):
    ...     def __init__(self):
    ...         super(FunctionModel, self).__init__()
    ...         elu = torch.nn.ELU()
    ...         self.dense1 = torch.nn.Linear(4, 5)
    ...         dense2 = torch.nn.Linear(5, 2)
    ...
    ...     def forward(self, x, y):
    ...         x = torch.cat([x, y], dim=1)
    ...         x = self.elu(self.dense1(x))
    ...         out = self.dense2(x)
    ...         return out
    ...
    >>> function_model = FunctionModel()
    >>> b_f = ltn.Function(model=function_model)
    >>> print(b_f)
    Function(model=FunctionModel(
      (dense1): Linear(in_features=4, out_features=5, bias=True)
    ))

    Binary function defined using a function. Note the call to `torch.cat` to merge the two inputs of the
    binary function.

    >>> b_f_f = ltn.Function(func=lambda x, y:
    ...                                 torch.repeat_interleave(
    ...                                     torch.sum(torch.cat([x, y], dim=1), dim=1, keepdim=True), 2,
    ...                                     dim=1))
    >>> print(b_f_f)
    Function(model=LambdaModel())

    Evaluation of a unary function on a constant. Note that:

    - the function returns a :class:`ltn.core.LTNObject` instance;
    - since a constant has been given, the `LTNObject` in output does not have free variables;
    - the shape of the `LTNObject` in output is `(2)` since the function has been evaluated on a constant, namely on one single individual, and returns individuals in :math:`\mathbb{R}^2`;
    - the attribute `value` of the `LTNObject` in output contains the result of the evaluation of the function.

    >>> c = ltn.Constant(torch.tensor([0.5, 0.01, 0.34, 0.001]))
    >>> out = f_f(c)
    >>> print(type(out))
    <class 'ltn.core.LTNObject'>
    >>> print(out)
    LTNObject(value=tensor([0.8510, 0.8510]), free_vars=[])
    >>> print(out.value)
    tensor([0.8510, 0.8510])
    >>> print(out.free_vars)
    []
    >>> print(out.shape())
    torch.Size([2])

    Evaluation of a unary function on a variable. Note that:

    - since a variable has been given, the `LTNObject` in output has one free variable;
    - the shape of the `LTNObject` in output is `(2, 2)` since the function has been evaluated on a variable with two individuls and returns individuals in :math:`\mathbb{R}^2`.

    >>> v = ltn.Variable('v', torch.tensor([[0.4, 0.3],
    ...                                     [0.32, 0.043]]))
    >>> out = f_f(v)
    >>> print(out)
    LTNObject(value=tensor([[0.7000, 0.7000],
            [0.3630, 0.3630]]), free_vars=['v'])
    >>> print(out.value)
    tensor([[0.7000, 0.7000],
            [0.3630, 0.3630]])
    >>> print(out.free_vars)
    ['v']
    >>> print(out.shape())
    torch.Size([2, 2])

    Evaluation of a binary function on a variable and a constant. Note that:

    - like in the previous example, the `LTNObject` in output has just one free variable, since only one variable has been given to the predicate;
    - the shape of the `LTNObject` in output is `(2, 2)` since the function has been evaluated on a variable with two individuals and returns individuals in :math:`\mathbb{R}^2`. The constant does not add dimensions to the output.

    >>> v = ltn.Variable('v', torch.tensor([[0.4, 0.3],
    ...                                     [0.32, 0.043]]))
    >>> c = ltn.Constant(torch.tensor([0.4, 0.04, 0.23, 0.43]))
    >>> out = b_f_f(v, c)
    >>> print(out)
    LTNObject(value=tensor([[1.8000, 1.8000],
            [1.4630, 1.4630]]), free_vars=['v'])
    >>> print(out.value)
    tensor([[1.8000, 1.8000],
            [1.4630, 1.4630]])
    >>> print(out.free_vars)
    ['v']
    >>> print(out.shape())
    torch.Size([2, 2])

    Evaluation of a binary function on two variables. Note that:

    - since two variables have been given, the `LTNObject` in output has two free variables;
    - the shape of the `LTNObject` in output is `(2, 3, 2)` since the function has been evaluated on a variable with two individuals, a variable with three individuals, and returns individuals in :math:`\mathbb{R}^2`;
    - the first dimension is dedicated to variable `x`, which is also the first one appearing in `free_vars`, the second dimension is dedicated to variable `y`, which is the second one appearing in `free_vars`, while the last dimensions is dedicated to the features of the individuals in output;
    - it is possible to access the `value` attribute for getting the results of the function. For example, at position `(1, 2)` there is the evaluation of the function on the second individual of `x` and third individuals of `y`.

    >>> x = ltn.Variable('x', torch.tensor([[0.4, 0.3],
    ...                                     [0.32, 0.043]]))
    >>> y = ltn.Variable('y', torch.tensor([[0.4, 0.04, 0.23],
    ...                                     [0.2, 0.04, 0.32],
    ...                                     [0.06, 0.08, 0.3]]))
    >>> out = b_f_f(x, y)
    >>> print(out)
    LTNObject(value=tensor([[[1.3700, 1.3700],
             [1.2600, 1.2600],
             [1.1400, 1.1400]],
    <BLANKLINE>
            [[1.0330, 1.0330],
             [0.9230, 0.9230],
             [0.8030, 0.8030]]]), free_vars=['x', 'y'])
    >>> print(out.value)
    tensor([[[1.3700, 1.3700],
             [1.2600, 1.2600],
             [1.1400, 1.1400]],
    <BLANKLINE>
            [[1.0330, 1.0330],
             [0.9230, 0.9230],
             [0.8030, 0.8030]]])
    >>> print(out.free_vars)
    ['x', 'y']
    >>> print(out.shape())
    torch.Size([2, 3, 2])
    >>> print(out.value[1, 2])
    tensor([0.8030, 0.8030])
    """

    def __init__(self, model=None, func=None):
        """
        Initializes the LTN function in two different ways:
            1. if `model` is not None, it initializes the function with the given PyTorch model;
            2. if `model` is None, it uses the `func` as a lambda function or a function to represent
            the LTN function. Note that, in this case, the LTN function is not learnable. So, the lambda function has
            to be used only for simple functions.
        """
        super(Function, self).__init__()
        if model is not None and func is not None:
            raise ValueError("Both model and func parameters have been specified. Expected only one of "
                             "the two parameters to be specified.")

        if model is None and func is None:
            raise ValueError("Both model and func parameters have not been specified. Expected one of the two "
                             "parameters to be specified.")

        if model is not None:
            if not isinstance(model, nn.Module):
                raise TypeError("Function() : argument 'model' (position 1) must be a torch.nn.Module, "
                                "not " + str(type(model)))
            self.model = model
        else:
            if not isinstance(func, types.LambdaType):
                raise TypeError("Function() : argument 'func' (position 2) must be a function, "
                                "not " + str(type(model)))
            self.model = LambdaModel(func)

    def __repr__(self):
        return "Function(model=" + str(self.model) + ")"

    def forward(self, *inputs, **kwargs):
        """
        It computes the output of the function given some :ref:`LTN objects <noteltnobject>` in input.

        Before computing the function, it performs the :ref:`LTN broadcasting <broadcasting>` of the inputs.

        Parameters
        ----------
        inputs : :obj:`tuple` of :class:`ltn.core.LTNObject`
            Tuple of :ref:`LTN objects <noteltnobject>` for which the function has to be computed.

        Returns
        ----------
        :class:`ltn.core.LTNObject`
            An :ref:`LTNObject <noteltnobject>` whose `value` attribute contains the result of the
            function, while `free_vars` attribute contains the labels of the free variables contained in the result.

        Raises
        ----------
        :class:`TypeError`
            Raises when the types of the inputs are incorrect.
        """
        inputs = list(inputs)
        if not all(isinstance(x, LTNObject) for x in inputs):
            raise TypeError("Expected parameter 'inputs' to be a tuple of LTNObject, but got " + str([type(i)
                                                                                                      for i in inputs]))

        proc_objs, output_vars, output_shape = process_ltn_objects(inputs)

        # the management of the input is left to the model or the lambda function
        output = self.model(*[o.value for o in proc_objs], **kwargs)

        output = torch.reshape(output, tuple(output_shape + list(output.shape[1::])))

        output = output.float()

        return LTNObject(output, output_vars)


def diag(*vars):
    """
    Sets the given LTN variables for :ref:`diagonal quantification <diagonal>`.

    The diagonal quantification disables the :ref:`LTN broadcasting <broadcasting>` for the given variables.

    Parameters
    ----------
    vars : :obj:`tuple` of :class:`ltn.core.Variable`
        Tuple of LTN variables for which the diagonal quantification has to be set.

    Returns
    ---------
    :obj:`list` of :class:`ltn.core.Variable`
        List of the same LTN variables given in input, prepared for the use of :ref:`diagonal quantification <diagonal>`.

    Raises
    ----------
    :class:`TypeError`
        Raises when the types of the input parameters are incorrect.

    :class:`ValueError`
        Raises when the values of the input parameters are incorrect.

    Notes
    -----
    - diagonal quantification has been designed to work with quantified statements, however, it could be used also to reduce the combinations of individuals for which a predicate has to be computed, making the computation more efficient;
    - diagonal quantification is particularly useful when we need to compute a predicate, or function, on specific tuples of variables' individuals only;
    - diagonal quantification expects the given variables to have the same number of individuals.

    See Also
    --------
    :func:`ltn.core.undiag`
        It allows to disable the diagonal quantification for the given variables.

    Examples
    --------
    Behavior of a predicate without diagonal quantification. Note that:

    - if diagonal quantification is not used, LTNtorch applies the :ref:`LTN broadcasting <broadcasting>` to the variables before computing the predicate;
    - the shape of the `LTNObject` in output is `(2, 2)` since the predicate has been computed on two variables with two individuals each;
    - the `free_vars` attribute of the `LTNObject` in output contains two variables, namely the variables on which the predicate has been computed.

    >>> import ltn
    >>> import torch
    >>> p = ltn.Predicate(func=lambda a, b: torch.nn.Sigmoid()(
    ...                                         torch.sum(torch.cat([a, b], dim=1), dim=1)
    ...                                     ))
    >>> x = ltn.Variable('x', torch.tensor([[0.3, 0.56, 0.43], [0.3, 0.5, 0.04]]))
    >>> y = ltn.Variable('y', torch.tensor([[0.4, 0.004], [0.3, 0.32]]))
    >>> out = p(x, y)
    >>> print(out.value)
    tensor([[0.8447, 0.8710],
            [0.7763, 0.8115]])
    >>> print(out.free_vars)
    ['x', 'y']
    >>> print(out.shape())
    torch.Size([2, 2])

    Behavior of the same predicate with diagonal quantification. Note that:

    - diagonal quantification requires the two variables to have the same number of individuals;
    - diagonal quantification has disabled the :ref:`LTN broadcasting <broadcasting>`, namely the predicate is not computed on all the possible combinations of individuals of the two variables (that are 2x2). Instead, it is computed only on the given tuples of individuals (that are 2), namely on the first individual of `x` and first individual of `y`, and on the second individual of `x` and second individual of `y`;
    - the shape of the `LTNObject` in output is `(2)` since diagonal quantification has been set and the variables have two individuals;
    - the `free_vars` attribute of the `LTNObject` in output has just one variable, even if two variables have been given to the predicate. This is due to diagonal quantification;
    - when diagonal quantification is set, you will se a variable label starting with `diag_` in the `free_Vars` attribute.

    >>> x, y = ltn.diag(x, y)
    >>> out = p(x, y)
    >>> print(out.value)
    tensor([0.8447, 0.8115])
    >>> print(out.free_vars)
    ['diag_x_y']
    >>> print(out.shape())
    torch.Size([2])

    See the examples under :class:`ltn.core.Quantifier` to see how to use :func:`ltn.core.diag` with quantifiers.
    """
    vars = list(vars)
    # check if a list of LTN variables has been passed
    if not all(isinstance(x, Variable) for x in vars):
        raise TypeError("Expected parameter 'vars' to be a tuple of Variable, but got " + str([type(v) for v in vars]))
    # check if the user has given only one variable
    if not len(vars) > 1:
        raise ValueError("Expected parameter 'vars' to be a tuple of more than one Variable, but got just one Variable.")
    # check if variables have the same number of individuals, assuming the first dimension is the batch dimension
    n_individuals = [var.shape()[0] for var in vars]
    if not len(
        set(n_individuals)) == 1:
        raise ValueError("Expected the given LTN variables to have the same number of individuals, "
                         "but got the following numbers of individuals " + str([v.shape()[0] for v in vars]))
    diag_label = "diag_" + "_".join([var.latent_var for var in vars])
    for var in vars:
        var.free_vars = [diag_label]
    return vars


def undiag(*vars):
    """
    Resets the :ref:`LTN broadcasting <broadcasting>` for the given LTN variables.

    In other words, it removes the :ref:`diagonal quantification <diagonal>` setting from the given variables.

    Parameters
    ----------
    vars : :obj:`tuple` of :class:`ltn.core.Variable`
        Tuple of LTN variables for which the :ref:`diagonal quantification <diagonal>` setting has to be removed.

    Returns
    ----------
    :obj:`list`
        List of the same LTN variables given in input, with the :ref:`diagonal quantification <diagonal>` setting removed.

    Raises
    ----------
    :class:`TypeError`
        Raises when the types of the input parameters are incorrect.

    See Also
    --------
    :func:`ltn.core.diag`
        It allows to set the :ref:`diagonal quantification <diagonal>` for the given variables.

    Examples
    --------
    Behavior of predicate with diagonal quantification. Note that:

    - diagonal quantification requires the two variables to have the same number of individuals;
    - diagonal quantification has disabled the :ref:`LTN broadcasting <broadcasting>`, namely the predicate is not computed on all the possible combinations of individuals of the two variables (that are 2x2). Instead, it is computed only on the given tuples of individuals (that are 2), namely on the first individual of `x` and first individual of `y`, and on the second individual of `x` and second individual of `y`;
    - the shape of the `LTNObject` in output is `(2)` since diagonal quantification has been set and the variables have two individuals;
    - the `free_vars` attribute of the `LTNObject` in output has just one variable, even if two variables have been given to the predicate. This is due to diagonal quantification;
    - when diagonal quantification is set, you will se a variable label starting with `diag_` in the `free_Vars` attribute.

    >>> import ltn
    >>> import torch
    >>> p = ltn.Predicate(func=lambda a, b: torch.nn.Sigmoid()(
    ...                                         torch.sum(torch.cat([a, b], dim=1), dim=1)
    ...                                     ))
    >>> x = ltn.Variable('x', torch.tensor([[0.3, 0.56, 0.43], [0.3, 0.5, 0.04]]))
    >>> y = ltn.Variable('y', torch.tensor([[0.4, 0.004], [0.3, 0.32]]))
    >>> x, y = ltn.diag(x, y)
    >>> out = p(x, y)
    >>> print(out.value)
    tensor([0.8447, 0.8115])
    >>> print(out.free_vars)
    ['diag_x_y']
    >>> print(out.shape())
    torch.Size([2])

    :func:`ltn.core.undiag` can be used to restore the :ref:`LTN broadcasting <broadcasting>` for the two variables. In
    the following, it is shown the behavior of the same predicate without diagonal quantification. Note that:

    - since diagonal quantification has been disabled, LTNtorch applies the :ref:`LTN broadcasting <broadcasting>` to the variables before computing the predicate;
    - the shape of the `LTNObject` in output is `(2, 2)` since the predicate has been computed on two variables with two individuals each;
    - the `free_vars` attribute of the `LTNObject` in output contains two variables, namely the variables on which the predicate has been computed.

    >>> x, y = ltn.undiag(x, y)
    >>> out = p(x, y)
    >>> print(out.value)
    tensor([[0.8447, 0.8710],
            [0.7763, 0.8115]])
    >>> print(out.free_vars)
    ['x', 'y']
    >>> print(out.shape())
    torch.Size([2, 2])
    """
    vars = list(vars)
    # check if a list of LTN variables has been passed
    if not all(isinstance(x, Variable) for x in vars):
        raise TypeError("Expected parameter 'vars' to be a tuple of Variable, but got " + str([type(v) for v in vars]))

    for var in vars:
        var.free_vars = [var.latent_var]
    return vars


class Connective:
    """
    Class representing an LTN connective.

    An LTN connective is :ref:`grounded <notegrounding>` as a fuzzy connective operator.

    In LTNtorch, the inputs of a connective are automatically broadcasted before the computation of the connective,
    if necessary. Moreover, the output is organized in a tensor where each dimension is related to
    one variable appearing in the inputs. See :ref:`LTN broadcasting <broadcasting>` for more information.

    Parameters
    ----------
    connective_op : :class:`ltn.fuzzy_ops.ConnectiveOperator`
        The unary/binary fuzzy connective operator that becomes the :ref:`grounding <notegrounding>` of the LTN connective.

    Attributes
    -----------
    connective_op : :class:`ltn.fuzzy_ops.ConnectiveOperator`
        See `connective_op` parameter.

    Raises
    ----------
    :class:`TypeError`
        Raises when the type of the input parameter is incorrect.

    Notes
    -----
    - the LTN connective supports various fuzzy connective operators. They can be found in :ref:`ltn.fuzzy_ops <fuzzyop>`;
    - the LTN connective allows to use these fuzzy operators with LTN formulas. It takes care of combining sub-formulas which have different variables appearing in them (:ref:`LTN broadcasting <broadcasting>`).
    - an LTN connective can be applied only to :ref:`LTN objects <noteltnobject>` containing truth values, namely values in :math:`[0., 1.]`;
    - the output of an LTN connective is always an :ref:`LTN object <noteltnobject>` (:class:`ltn.core.LTNObject`).

    .. automethod:: __call__

    See Also
    --------
    :class:`ltn.fuzzy_ops`
        The `ltn.fuzzy_ops` module contains the definition of common fuzzy connective operators that can be used with LTN connectives.

    Examples
    --------
    Use of :math:`\\land` to create a formula which is the conjunction of two predicates. Note that:

    - a connective operator can be applied only to inputs which represent truth values. In this case with have two predicates;
    - LTNtorch provides various semantics for the conjunction, here we use the Goguen conjunction (:class:`ltn.fuzzy_ops.AndProd`);
    - LTNtorch applies the :ref:`LTN broadcasting <broadcasting>` to the variables before computing the predicates;
    - LTNtorch applies the :ref:`LTN brodcasting <broadcasting>` to the operands before applying the selected conjunction operator;
    - the result of a connective operator is a :class:`ltn.core.LTNObject` instance containing truth values in [0., 1.];
    - the attribute `value` of the `LTNObject` in output contains the result of the connective operator;
    - the shape of the `LTNObject` in output is `(2, 3, 4)`. The first dimension is associated with variable `x`, which has two individuals, the second dimension with variable `y`, which has three individuals, while the last dimension with variable `z`, which has four individuals;
    - it is possible to access to specific results by indexing the attribute `value`. For example, at index `(0, 1, 2)` there is the evaluation of the formula on the first individual of `x`, second individual of `y`, and third individual of `z`;
    - the attribute `free_vars` of the `LTNObject` in output contains the labels of the three variables appearing in the formula.

    >>> import ltn
    >>> import torch
    >>> p = ltn.Predicate(func=lambda a: torch.nn.Sigmoid()(
    ...                                     torch.sum(a, dim=1)
    ...                                  ))
    >>> q = ltn.Predicate(func=lambda a, b: torch.nn.Sigmoid()(
    ...                                         torch.sum(torch.cat([a, b], dim=1),
    ...                                     dim=1)))
    >>> x = ltn.Variable('x', torch.tensor([[0.3, 0.5],
    ...                                     [0.04, 0.43]]))
    >>> y = ltn.Variable('y', torch.tensor([[0.5, 0.23],
    ...                                     [4.3, 9.3],
    ...                                     [4.3, 0.32]]))
    >>> z = ltn.Variable('z', torch.tensor([[0.3, 0.4, 0.43],
    ...                                     [0.4, 4.3, 5.1],
    ...                                     [1.3, 4.3, 2.3],
    ...                                     [0.4, 0.2, 1.2]]))
    >>> And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    >>> print(And)
    Connective(connective_op=AndProd(stable=True))
    >>> out = And(p(x), q(y, z))
    >>> print(out)
    LTNObject(value=tensor([[[0.5971, 0.6900, 0.6899, 0.6391],
             [0.6900, 0.6900, 0.6900, 0.6900],
             [0.6878, 0.6900, 0.6900, 0.6889]],
    <BLANKLINE>
            [[0.5325, 0.6154, 0.6153, 0.5700],
             [0.6154, 0.6154, 0.6154, 0.6154],
             [0.6135, 0.6154, 0.6154, 0.6144]]]), free_vars=['x', 'y', 'z'])
    >>> print(out.value)
    tensor([[[0.5971, 0.6900, 0.6899, 0.6391],
             [0.6900, 0.6900, 0.6900, 0.6900],
             [0.6878, 0.6900, 0.6900, 0.6889]],
    <BLANKLINE>
            [[0.5325, 0.6154, 0.6153, 0.5700],
             [0.6154, 0.6154, 0.6154, 0.6154],
             [0.6135, 0.6154, 0.6154, 0.6144]]])
    >>> print(out.free_vars)
    ['x', 'y', 'z']
    >>> print(out.shape())
    torch.Size([2, 3, 4])
    """

    def __init__(self, connective_op):
        if not isinstance(connective_op, ltn.fuzzy_ops.ConnectiveOperator):
            raise TypeError("Connective() : argument 'connective_op' (position 1) must be a "
                            "ltn.fuzzy_ops.ConnectiveOperator, not " + str(type(connective_op)))
        self.connective_op = connective_op

    def __repr__(self):
        return "Connective(connective_op=" + str(self.connective_op) + ")"

    def __call__(self, *operands, **kwargs):
        """
        It applies the selected fuzzy connective operator (`connective_op` attribute) to the operands
        (:ref:`LTN objects <noteltnobject>`) given in input.

        Parameters
        -----------
        operands : :obj:`tuple` of :class:`ltn.core.LTNObject`
            Tuple of :ref:`LTN objects <noteltnobject>` representing the operands to which the fuzzy connective
            operator has to be applied.

        Returns
        ----------
        :class:`ltn.core.LTNObject`
            The `LTNObject` that is the result of the application of the fuzzy connective operator to the given
            :ref:`LTN objects <noteltnobject>`.

        Raises
        ----------
        :class:`TypeError`
            Raises when the types of the input parameters are incorrect.

        :class:`ValueError`
            Raises when the values of the input parameters are incorrect.
            Raises when the truth values of the operands given in input are not in the range [0., 1.].
        """
        operands = list(operands)

        if isinstance(self.connective_op, ltn.fuzzy_ops.UnaryConnectiveOperator) and len(operands) != 1:
            raise ValueError("Expected one operand for a unary connective, but got " + str(len(operands)))
        if isinstance(self.connective_op, ltn.fuzzy_ops.BinaryConnectiveOperator) and len(operands) != 2:
            raise ValueError("Expected two operands for a binary connective, but got " + str(len(operands)))

        if not all(isinstance(x, LTNObject) for x in operands):
            raise TypeError("Expected parameter 'operands' to be a tuple of LTNObject, but got " +
                            str([type(o) for o in operands]))

        # check if operands are in [0., 1.]
        ltn.fuzzy_ops.check_values(*[o.value for o in operands])

        proc_objs, vars, n_individuals_per_var = process_ltn_objects(operands)
        # the connective operator needs the values of the objects and not the objects themselves
        proc_objs = [o.value for o in proc_objs]
        output = self.connective_op(*proc_objs)
        # reshape the output according to the dimensions given by the processing function
        # we need to give this shape in order to have different axes associated to different variables, as usual
        output = torch.reshape(output, n_individuals_per_var)
        return LTNObject(output, vars)


class Quantifier:
    """
    Class representing an LTN quantifier.

    An LTN quantifier is :ref:`grounded <notegrounding>` as a fuzzy aggregation operator. See :ref:`quantification in LTN <quantification>`
    for more information about quantification.

    Parameters
    ----------
    agg_op : :class:`ltn.fuzzy_ops.AggregationOperator`
        The fuzzy aggregation operator that becomes the :ref:`grounding <notegrounding>` of the LTN quantifier.
    quantifier : :obj:`str`
        String indicating the quantification that has to be performed ('e' for ∃, or 'f' for ∀).

    Attributes
    -----------
    agg_op : :class:`ltn.fuzzy_ops.AggregationOperator`
        See `agg_op` parameter.
    quantifier : :obj:`str`
        See `quantifier` parameter.

    Raises
    ----------
    :class:`TypeError`
        Raises when the type of the `agg_op` parameter is incorrect.

    :class:`ValueError`
        Raises when the value of the `quantifier` parameter is incorrect.

    Notes
    -----
    - the LTN quantifier supports various fuzzy aggregation operators, which can be found in :class:`ltn.fuzzy_ops`;
    - the LTN quantifier allows to use these fuzzy aggregators with LTN formulas. It takes care of selecting the formula (`LTNObject`) dimensions to aggregate, given some LTN variables in arguments.
    - boolean conditions (by setting parameters `mask_fn` and `mask_vars`) can be used for :ref:`guarded quantification <guarded>`;
    - an LTN quantifier can be applied only to :ref:`LTN objects <noteltnobject>` containing truth values, namely values in :math:`[0., 1.]`;
    - the output of an LTN quantifier is always an :ref:`LTN object <noteltnobject>` (:class:`ltn.core.LTNObject`).

    .. automethod:: __call__

    See Also
    --------
    :class:`ltn.fuzzy_ops`
        The `ltn.fuzzy_ops` module contains the definition of common fuzzy aggregation operators that can be used with
        LTN quantifiers.

    Examples
    --------
    Behavior of a binary predicate evaluated on two variables. Note that:

    - the shape of the `LTNObject` in output is `(2, 3)` since the predicate has been computed on a variable with two individuals and a variable with three individuals;
    - the attribute `free_vars` of the `LTNObject` in output contains the labels of the two variables given in input to the predicate.

    >>> import ltn
    >>> import torch
    >>> p = ltn.Predicate(func=lambda a, b: torch.nn.Sigmoid()(
    ...                                         torch.sum(torch.cat([a, b], dim=1),
    ...                                     dim=1)))
    >>> x = ltn.Variable('x', torch.tensor([[0.3, 1.3],
    ...                                     [0.3, 0.3]]))
    >>> y = ltn.Variable('y', torch.tensor([[2.3, 0.3, 0.4],
    ...                                     [1.2, 3.4, 1.3],
    ...                                     [2.3, 1.4, 1.4]]))
    >>> out = p(x, y)
    >>> print(out)
    LTNObject(value=tensor([[0.9900, 0.9994, 0.9988],
            [0.9734, 0.9985, 0.9967]]), free_vars=['x', 'y'])
    >>> print(out.value)
    tensor([[0.9900, 0.9994, 0.9988],
            [0.9734, 0.9985, 0.9967]])
    >>> print(out.free_vars)
    ['x', 'y']
    >>> print(out.shape())
    torch.Size([2, 3])

    Universal quantification on one single variable of the same predicate. Note that:

    - `quantifier='f'` means that we are defining the fuzzy semantics for the universal quantifier;
    - the result of a quantification operator is always a :class:`ltn.core.LTNObject` instance;
    - LTNtorch supports various sematics for quantifiers, here we use :class:`ltn.fuzzy_ops.AggregPMeanError` for :math:`\\forall`;
    - the shape of the `LTNObject` in output is `(3)` since the quantification has been performed on variable `x`. Only the dimension associated with variable `y` has left since the quantification has been computed by LTNtorch as an aggregation on the dimension related with variable `x`;
    - the attribute `free_vars` of the `LTNObject` in output contains only the label of variable `y`. This is because variable `x` has been quantified, namely it is not a free variable anymore;
    - in LTNtorch, the quantification is performed by computing the value of the predicate first and then by aggregating on the selected dimensions, specified by the quantified variables.

    >>> Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier='f')
    >>> print(Forall)
    Quantifier(agg_op=AggregPMeanError(p=2, stable=True), quantifier='f')
    >>> out = Forall(x, p(x, y))
    >>> print(out)
    LTNObject(value=tensor([0.9798, 0.9988, 0.9974]), free_vars=['y'])
    >>> print(out.value)
    tensor([0.9798, 0.9988, 0.9974])
    >>> print(out.free_vars)
    ['y']
    >>> print(out.shape())
    torch.Size([3])

    Universal quantification on both variables of the same predicate. Note that:

    - the shape of the `LTNObject` in output is empty since the quantification has been performed on both variables. No dimension has left since the quantification has been computed by LTNtorch as an aggregation on both dimensions of the `value` of the predicate;
    - the attribute `free_vars` of the `LTNObject` in output contains no labels of variables. This is because both variables have been quantified, namely they are not free variables anymore.

    >>> out = Forall([x, y], p(x, y))
    >>> print(out)
    LTNObject(value=tensor(0.9882), free_vars=[])
    >>> print(out.value)
    tensor(0.9882)
    >>> print(out.free_vars)
    []
    >>> print(out.shape())
    torch.Size([])

    Universal quantification on one variable, and existential quantification on the other variable, of the same predicate.
    Note that:

    - the only way in LTNtorch to apply two different quantifiers to the same formula is a nested syntax;
    - `quantifier='e'` means that we are defining the fuzzy semantics for the existential quantifier;
    - LTNtorch supports various sematics for quantifiers, here we use :class:`ltn.fuzzy_ops.AggregPMean` for :math:`\\exists`.

    >>> Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(), quantifier='e')
    >>> print(Exists)
    Quantifier(agg_op=AggregPMean(p=2, stable=True), quantifier='e')
    >>> out = Forall(x, Exists(y, p(x, y)))
    >>> print(out)
    LTNObject(value=tensor(0.9920), free_vars=[])
    >>> print(out.value)
    tensor(0.9920)
    >>> print(out.free_vars)
    []
    >>> print(out.shape())
    torch.Size([])

    Guarded quantification. We perform a universal quantification on both variables of the same predicate, considering
    only the individuals of variable `x` whose sum of features is lower than a certain threshold. Note that:

    - guarded quantification requires the parameters `cond_vars` and `cond_fn` to be set;
    - `cond_vars` contains the variables on which the guarded condition is based on. In this case, we have decided to create a condition on `x`;
    - `cond_fn` contains the function which is the guarded condition. In this case, it verifies if the sum of features of the individuals of `x` is lower than 1. (our threshold);
    - the second individual of `x`, which is `[0.3, 0.3]`, satisfies the condition, namely it will not be considered when the aggregation has to be performed. In other words, all the results of the predicate computed using the second individual of `x` will not be considered in the aggregation;
    - notice the result changes compared to the previous example (:math:`\\forall x \\forall y P(x, y)`). This is due to the fact that some truth values of the result of the predicate are not considered in the aggregation due to guarded quantification. These values are at positions `(1, 0)`, `(1, 1)`, and `(1, 2)`, namely all the positions related with the second individual of `x` in the result of the predicate;
    - notice that the shape of the `LTNObject` in output and its attribute `free_vars` remain the same compared to the previous example. This is because the quantification is still on both variables, namely it is perfomed on both dimensions of the result of the predicate.

    >>> out = Forall([x, y], p(x, y),
    ...             cond_vars=[x],
    ...             cond_fn=lambda x: torch.less(torch.sum(x.value, dim=1), 1.))
    >>> print(out)
    LTNObject(value=tensor(0.9844, dtype=torch.float64), free_vars=[])
    >>> print(out.value)
    tensor(0.9844, dtype=torch.float64)
    >>> print(out.free_vars)
    []
    >>> print(out.shape())
    torch.Size([])

    Universal quantification of both variables of the same predicate using diagonal quantification
    (:func:`ltn.core.diag`). Note that:

    - the variables have the same number of individuals since it is a constraint for applying diagonal quantification;
    - since diagonal quantification has been set, the predicate will not be computed on all the possible combinations of individuals of the two variables (that are 4), namely the :ref:`LTN broadcasting <broadcasting>` is disabled;
    - the predicate is computed only on the given tuples of individuals in a one-to-one correspondence, namely on the first individual of `x` and `y`, and second individual of `x` and `y`;
    - the result changes compared to the case without diagonal quantification. This is due to the fact that we are aggregating a smaller number of truth values since the predicate has been computed only two times.

    >>> x = ltn.Variable('x', torch.tensor([[0.3, 1.3],
    ...                                     [0.3, 0.3]]))
    >>> y = ltn.Variable('y', torch.tensor([[2.3, 0.3],
    ...                                    [1.2, 3.4]]))
    >>> out = Forall(ltn.diag(x, y), p(x, y)) # with diagonal quantification
    >>> out_without_diag = Forall([x, y], p(x, y)) # without diagonal quantification
    >>> print(out_without_diag)
    LTNObject(value=tensor(0.9788), free_vars=[])
    >>> print(out_without_diag.value)
    tensor(0.9788)
    >>> print(out)
    LTNObject(value=tensor(0.9888), free_vars=[])
    >>> print(out.value)
    tensor(0.9888)
    >>> print(out.free_vars)
    []
    >>> print(out.shape())
    torch.Size([])
    """
    def __init__(self, agg_op, quantifier):
        if not isinstance(agg_op, ltn.fuzzy_ops.AggregationOperator):
            raise TypeError("Quantifier() : argument 'agg_op' (position 1) must be a "
                            "ltn.fuzzy_ops.AggregationOperator, not " + str(type(agg_op)))
        self.agg_op = agg_op
        if quantifier not in ["f", "e"]:
            raise ValueError("Expected parameter 'quantifier' to be the string 'e', "
                             "for existential quantifier, or the string 'f', "
                             "for universal quantifier, but got " + str(quantifier))
        self.quantifier = quantifier

    def __repr__(self):
        return "Quantifier(agg_op=" + str(self.agg_op) + ", quantifier='" + self.quantifier + "')"

    def __call__(self, vars, formula, cond_vars=None, cond_fn=None, **kwargs):
        """
        It applies the selected aggregation operator (`agg_op` attribute) to the formula given in input based on the
        selected variables.

        It allows also to perform a :ref:`guarded quantification <guarded>` by setting `cond_vars` and `cond_fn`
        parameters.

        Parameters
        -----------
        vars : :obj:`list` of :class:`ltn.core.Variable`
            List of LTN variables on which the quantification has to be performed.
        formula : :class:`ltn.core.LTNObject`
            Formula on which the quantification has to be performed.
        cond_vars : :obj:`list` of :class:`ltn.core.Variable`, default=None
            List of LTN variables that appear in the :ref:`guarded quantification <guarded>` condition.
        cond_fn : :class:`function`, default=None
            Function representing the :ref:`guarded quantification <guarded>` condition.

        Raises
        ----------
        :class:`TypeError`
            Raises when the types of the input parameters are incorrect.

        :class:`ValueError`
            Raises when the values of the input parameters are incorrect.
            Raises when the truth values of the formula given in input are not in the range [0., 1.].
        """
        # first of all, check if user has correctly set the condition vars and the condition function
        if cond_vars is not None and cond_fn is None:
            raise ValueError("Since 'cond_fn' parameter has been set, 'cond_vars' parameter must be set as well, "
                             "but got None.")
        if cond_vars is None and cond_fn is not None:
            raise ValueError("Since 'cond_vars' parameter has been set, 'cond_fn' parameter must be set as well, "
                             "but got None.")
        # check that vars is a list of LTN variables or a single LTN variable
        if not all(isinstance(x, Variable) for x in vars) if isinstance(vars, list) else not isinstance(vars, Variable):
            raise TypeError("Expected parameter 'vars' to be a list of Variable or a "
                            "Variable, but got " + str([type(v) for v in vars])
                            if isinstance(vars, list) else type(vars))

        # check that formula is an LTNObject
        if not isinstance(formula, LTNObject):
            raise TypeError("Expected parameter 'formula' to be an LTNObject, but got " + str(type(formula)))

        # check if formula is in [0., 1.]
        ltn.fuzzy_ops.check_values(formula.value)

        if isinstance(vars, Variable):
            vars = [vars]  # this serves in the case vars is just a variable and not a list of variables

        # aggregation_vars contains the labels of the variables on which the quantification has to be performed
        aggregation_vars = set([var.free_vars[0] for var in vars])

        # check if guarded quantification has to be performed
        if cond_fn is not None and cond_vars is not None:
            # check that cond_vars are LTN variables
            if not all(isinstance(x, Variable) for x in cond_vars) if isinstance(cond_vars, list) \
                    else not isinstance(cond_vars, Variable):
                raise TypeError("Expected parameter 'cond_vars' to be a list of Variable or a "
                                "Variable, but got " + str([type(v) for v in cond_vars])
                                if isinstance(cond_vars, list) else type(cond_vars))
            # check that cond_fn is a function
            if not isinstance(cond_fn, types.LambdaType):
                raise TypeError("Expected parameter 'cond_fn' to be a function, but got " + str(type(cond_fn)))

            if isinstance(cond_vars, Variable):
                cond_vars = [cond_vars]  # this serves in the case vars is just a variable and not a list of variables

            # create the mask for applying the guarded quantification
            formula, mask = self.compute_mask(formula, cond_vars, cond_fn, list(aggregation_vars))

            # we perform the desired quantification
            # we give the mask to the aggregator for performing the guarded quantification
            aggregation_dims = [formula.free_vars.index(var) for var in aggregation_vars]
            output = self.agg_op(formula.value, aggregation_dims, mask=mask.value, **kwargs)

            # For some values in the formula, the mask can result in aggregating with empty variables.
            #    e.g. forall X ( exists Y:condition(X,Y) ( p(X,Y) ) )
            #       For some values of X, there may be no Y satisfying the condition
            # The result of the aggregation operator in such case is often not defined (e.g. NaN).
            # We replace the result with 0.0 if the semantics of the aggregator is exists,
            # or 1.0 if the semantics of the aggregator is forall.
            rep_value = 1. if self.quantifier == "f" else 0.
            output = torch.where(
                torch.isnan(output),
                rep_value,
                output.double()
            )
        else:  # in this case, the guarded quantification has not to be performed
            # aggregation_dims are the dimensions on which the aggregation has to be performed
            # the aggregator aggregates only on the axes given by aggregations_dims
            aggregation_dims = [formula.free_vars.index(var) for var in aggregation_vars]
            # the aggregation operator needs the values of the formula and not the LTNObject containing the values
            output = self.agg_op(formula.value, dim=tuple(aggregation_dims), **kwargs)

        undiag(*vars)
        # update the free variables on the output object by removing variables that have been aggregated
        return LTNObject(output, [var for var in formula.free_vars if var not in aggregation_vars])

    @staticmethod
    def compute_mask(formula, cond_vars, cond_fn, aggregation_vars):
        """
        It computes the mask for performing the guarded quantification on the formula given in input.

        Parameters
        ----------
        formula: :class:`LTNObject`
            Formula that has to be quantified.
        cond_vars: :obj:`list`
            List of LTN variables that appear in the guarded quantification condition.
        cond_fn: :class:`function`
            Function which implements the guarded quantification condition.
        aggregation_vars: :obj:`list`
            List of labels of the variables on which the quantification has to be performed.

        Returns
        ----------
        (:class:`LTNObject`, :class:`LTNObject`)
            Tuple where the first element is the input formula transposed in such a way that the
            guarded variables are in the first dimensions, while the second element is the mask that has to be applied over
            the formula in order to perform the guarded quantification. The formula and the mask will have the same shape in
            order to apply the mask to the formula by element-wise operations.
        """
        # reshape the formula in such a way it now includes the dimensions related to the variables in the condition
        # this has to be done only if the formula does not include some variables in the condition yet
        cond_vars_not_in_formula = [var for var in cond_vars if var.free_vars[0] not in formula.free_vars]
        if cond_vars_not_in_formula:
            proc_objs, _, n_ind = process_ltn_objects([formula] + cond_vars_not_in_formula)
            formula = proc_objs[0]
            formula.value = formula.value.view(tuple(n_ind))

        # set the masked (guarded) vars on the first axes
        vars_in_cond_labels = [var.free_vars[0] for var in cond_vars]
        vars_in_cond_not_agg_labels = [var for var in vars_in_cond_labels if var not in aggregation_vars]
        vars_in_cond_agg_labels = [var for var in vars_in_cond_labels if var in aggregation_vars]
        vars_not_in_cond_labels = [var for var in formula.free_vars if var not in vars_in_cond_labels]
        formula_new_vars_order = vars_in_cond_not_agg_labels + vars_in_cond_agg_labels + vars_not_in_cond_labels

        # we need to construct a dict to remove duplicate diag entries inside the labels
        # these duplicates would interfere with the transposition operation
        formula = Quantifier.transpose_vars(formula, list(dict.fromkeys(formula_new_vars_order)))

        # compute the boolean mask using the variables in the condition
        # make the shapes of the variables in the condition compatible in order to apply the condition element-wisely
        cond_vars, vars_order_in_cond, n_individuals_per_var = process_ltn_objects(cond_vars)
        mask = cond_fn(*cond_vars)  # creates the mask
        # give the mask the correct shape after the condition has been computed
        mask = torch.reshape(mask, tuple(n_individuals_per_var))

        # transpose the mask dimensions according to the var order in formula_grounding
        # create LTN object for the mask with associated free variables
        mask = LTNObject(mask, vars_order_in_cond)

        cond_new_vars_order = vars_in_cond_not_agg_labels + vars_in_cond_agg_labels

        # we need to construct a dict to remove duplicate diag entries inside the labels
        # these duplicates would interfere with the transposition operation
        mask = Quantifier.transpose_vars(mask, list(dict.fromkeys(cond_new_vars_order)))

        # make formula and mask of the same shape in order to apply the mask element-wisely
        if formula.shape() != mask.shape():
            # I have to rearrange the size of the mask if it has a different size respect to the formula
            # this is needed to apply an element-wise torch.where in order to apply the mask
            (formula, mask), vars, n_individuals_per_var = process_ltn_objects([formula, mask])
            mask.value = mask.value.view(n_individuals_per_var)
            # this fix the side effect due to call to process_ltn_objects
            formula.value = formula.value.view(n_individuals_per_var)

        return formula, mask

    @staticmethod
    def transpose_vars(object, new_vars_order):
        """
        It transposes the input LTN object using the order of variables given in `new_vars_order` parameter.

        Parameters
        ----------
        object: :class:`LTNObject`
            LTN object that has to be transposed.
        new_vars_order: :obj:`list`
            List containing the order of variables (expressed by labels) to transpose the input object.

        Returns
        -----------
        :class:`LTNObject`
            The input LTN object transposed according to the order in `new_vars_order` parameter.
        """
        perm = [object.free_vars.index(var) for var in new_vars_order]
        object.value = torch.permute(object.value, perm)
        object.free_vars = new_vars_order
        return object
