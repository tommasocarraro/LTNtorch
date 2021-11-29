import copy
import torch
from torch import nn
import numpy as np
import ltn
import types


class LTNObject:
    """
    This class represents a generic LTN object (constants, variables, outputs of predicates and functions).
    
    In LTN, each logical object (constants, variables, outputs of predicates
    and functions) is represented by a torch.Tensor, which represents the value of the object, and a list of variables
    labels (list of strings).
    The variables labels represent the free variables that appear in the LTN object. A variable is free when it is not
    quantified through a logical quantifier (existential, universal). For example, the formula IsSon(x, y) contains
    two free variables, namely x and y, while ∀x IsSon(x, y) contains only one free variable, namely y, since x is
    universally quantified. So, in the case of IsSon(x, y), the `value` attribute will contain a tensor of truth values
    corresponding to the predicate IsSon computed on all the possible combinations of the individuals in x and y, while
    the attribute `free_vars` will contain the list of variables labels ['x', 'y'], since both variables are not
    quantified, namely are free.

    Parameters
    ----------
    value: :class:`torch.Tensor`
        The value of the LTN object. The value could be real in case of features or integer in case of indexes.
    var_labels: :obj:`list`
        The labels of the free variables contained in the LTN object. It must be a list of strings.

    Attributes
    ----------
    value: :class:`torch.Tensor`
        See value parameter.
    free_vars: :obj:`list`
        See var_labels parameter.

    Raises
    ------
    :class:`TypeError`
        Raises when the types of the input parameters are incorrect.
    """

    def __init__(self, value, var_labels):
        # check inputs before creating the object
        if not isinstance(value, torch.Tensor):
            raise TypeError("The value parameter must be a torch.Tensor")
        if not (isinstance(var_labels, list) and (all(isinstance(x, str) for x in var_labels) if var_labels else True)):
            raise TypeError("The var_labels parameter must be a list of strings")
        self.value = value
        self.free_vars = var_labels

    def shape(self):
        """
        Methods to be used as a short-cut to access the LTN object value's shape.

        Returns
        -------
        :class:`torch.Size`
            The shape of the value of the LTN object.
        """
        return self.value.shape


class Constant(LTNObject):
    """
    This class represents an LTN constant.

    An LTN constant denotes an individual (`LTNobject`) grounded as a tensor in the Real field.
    The individual can be pre-defined (fixed data point) or learnable (embedding).

    Parameters
    ----------
        value: :class:`torch.Tensor`
            The value of the LTN constant. The value can be a tensor of any order,
            depending of the domain on the constant. The value of the constant is the so-called `grounding` in the
            paper terminology.
        trainable: :obj:`bool` [optional]
            Flag indicating the LTN constant is trainable or not. If False, the PyTorch subgraph containing the constant
            will be excluded from the gradient computation. If True, the constant is initialized using the value
            parameter and then will change during learning. Defaults to False.
    Attributes
    ----------
        Attributes are inherited by the super class.
        Note that the attribute `free_vars` will be an empty list in this case, since a constant does not have
        variables.

    Raises
    ------
    :class:`TypeError`
        Raises when the types of the input parameters are incorrect.
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


class Variable(LTNObject):
    """
    This class represents an LTN variable.

    An LTN variable denotes a sequence of individuals. It is grounded as a sequence of tensors (groundings of
    individuals) in the real field.
    In particular, axis 0 is the batch dimension (it is associated with the number of individuals in the grounding of
    the variable).
    So, if `x` is an `ltn.Variable`, `x[0]` gives the first individual, `x[1]` gives the second individual,
    and so forth, i.e., the usual way.

    Parameters
    ----------
    var_label: :obj:`str`
        String containing the name of the variable, for example 'x'.
    individuals: :class:`torch.Tensor`
        Tensor containing the sequence of individuals (tensors) that becomes the grounding the LTN
        variable. The first dimension should be related to the number of individuals of the LTN variable.
    add_batch_dim: :obj:`bool` [optional]
        Boolean flag indicating whether the batch dimension has to be added to the variable or
        not. Since a variable represents a sequence of individuals, the batch dimension should be added if it is missed.
        Note that the batch dimension is added if and only if the input sequence has one single dimension. If, instead,
        the input sequence has more than one dimension, the first dimension is considered as batch dimension. The
        default value of this parameter is True. If this value is set to False and the input sequence has only one
        dimension, no batch dimension will be added. This could serve in rare cases, for example when the individuals
        are just indexes to be used in future operations.

    Attributes
    ----------
    The attributes are inherited from the super class.
    Note that in this case the `free_vars` attribute will contain only the label of the variable itself, since a
    variable has only one free variable, the variable itself.
    latent_var: :obj:'str'
        This attribute contains the variable label given in input and it is used on the 'undiag' operation
        of the framework to restore the `free_vars` attribute of desired LTN objects after a diagonal quantification
        has been performed.
    """
    def __init__(self, var_label, individuals, add_batch_dim=True):
        # check inputs
        if not isinstance(var_label, str):
            raise TypeError("The label of the LTN variable must be a string.")
        if var_label.startswith("diag_"):
            raise ValueError("Labels starting with diag_ are reserved for diagonal quantification.")
        if not isinstance(individuals, torch.Tensor):
            raise TypeError("The individuals of the variable must be contained in a torch.Tensor.")

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
    # note that we copy only if the input object is a constant with grad_fn or if the object has not grad_fn attribute,
    # namely it is a leaf tensor
    objects_ = [LTNObject(torch.clone(o.value), copy.deepcopy(o.free_vars))
                if (o.value.grad_fn is None or (isinstance(o, Constant) and o.value.grad_fn is not None)) else o
                for o in objects]
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
    Class for LTN predicates.

    An LTN predicate is a mathematical function (either pre-defined or learnable) that maps
    from some n-ary domain of individuals to a real number in [0,1] (fuzzy) that can be interpreted as a truth value.
    Examples of LTN predicates can be similarity measures, classifiers, etc.

    Predicates can be defined using any operations in PyTorch. They can be linear functions, Deep Neural Networks,
    and so forth. The only constraint for the predicates is that their output must be in the range [0., 1.]. If the
    output is outside of this range, the LTN framework will behave unpredictably.

    An LTN predicate implements a `nn.Module` instance that can "broadcast" LTN terms as follows:
    1. Evaluating a predicate with one variable of n individuals yields n output values,
    where the i-th output value corresponds to the predicate calculated with the i-th individual.
    2. Evaluating a predicate with k variables (x1,...,xk) with respectively n1,...,nk
    individuals each, yields a result with n1*...*nk values. The result is organized in a tensor
    where the first k dimensions can be indexed to retrieve the outcome(s) that correspond to each variable.
    The attribute `free_vars` tells which axis corresponds to which variable in the LTNObject
    output by the predicate (using the name of the variable).

    Parameters
    ----------
    model: :class:`torch.nn.Module` [optional]
        PyTorch model that becomes the grounding of the predicate.
    func: :class:`function` [optional]
        If a model is not given, it is possible to give a lambda function or a function.
        In this case, the function will be used to define a non-trainable model for the LTN predicate.

    Attributes
    ----------
    model: :class:`torch.nn.Module` or :class:`function`
        The definition (`grounding`) of the LTN predicate. The grounding of a predicate is a non-trainable
        model implemented using a lambda function or a function, or a learnable model. When some LTN objects are given
        in input to the predicate model, the model returns a real value in [0, 1] for each combination of the values
        of the objects given in input.
        Then, an LTNObject is created as a result of the computation. The attribute `free_vars` will contain the list
        of free variables contained in the LTNObject in output.

    Raises
    ----------
    :class:`TypeError`
        Raises when the types of the input parameters are incorrect.

    :class:`ValueError`
        Raises when the values of the input parameters are incorrect.
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
            raise ValueError("Both model and func are not None. Only one of the two ways to define the "
                             "predicate can be used.")

        if model is None and func is None:
            raise ValueError("Both model and func are None. At least one of the two ways to define the "
                             "predicate has to be defined.")

        if model is not None:
            if not isinstance(model, nn.Module):
                raise TypeError("The given model is not a PyTorch model. Only PyTorch models are allowed for"
                                " the model parameter.")
            self.model = model
        else:
            if not isinstance(func, types.LambdaType):
                raise TypeError("The given model is not a lambda function or a function. Only "
                                "lambda functions or functions are allowed for the func parameter.")
            self.model = LambdaModel(func)

    def forward(self, *inputs, **kwargs):
        """Encapsulates the "self.model.forward()" to handle the ltn-broadcasting (by using
        process_ltn_objects() method).

        Parameters
        ----------
            inputs: :obj:`list` or :obj:`tuple`
                List or tuple of LTN objects for which the predicate has to be computed.
        Returns
        ----------
            :class:`LTNObject`
                An `LTNObject` whose `value` parameter contains the truth values representing the result of the
                predicate, and `free_vars` parameter contains the labels of the free variables contained in the result.

        Raises
        ----------
        :class:`TypeError`
            Raises when the types of the inputs are incorrect.

        :class:`ValueError`
            Raises when the values of the output are not in the range [0., 1.].
        """
        inputs = list(inputs)
        if not all(isinstance(x, LTNObject) for x in inputs):
            raise TypeError("Only LTNObject objects are valid inputs for predicate.")

        proc_objs, output_vars, output_shape = process_ltn_objects(inputs)

        # the management of the input is left to the model or the lambda function
        output = self.model(*[o.value for o in proc_objs], **kwargs)

        # check if output of predicate contains only truth values, namely values in the range [0., 1.]
        if not torch.all(torch.where(torch.logical_and(output >= 0., output <= 1.), 1., 0.)):
            raise ValueError("The output of a predicate must be in the range [0., 1.]. Some of the values contained"
                             " in the output are not inside the range. Check your predicate implementation!")

        output = torch.reshape(output, tuple(output_shape))
        # we assure the output is float in the case it is double to avoid type incompatibilities
        output = output.float()

        return LTNObject(output, output_vars)


class Function(nn.Module):
    """
    Class for LTN functions.

    An ltn function is a mathematical function (pre-defined or learnable) that maps n individuals to one individual
    in the tensor domain.
    Examples of functions can be distance functions, regressors, etc.

    Functions can be defined using any operations in PyTorch.
    They can be linear functions, Deep Neural Networks, and so forth. Unlike the LTN predicates, for the LTN functions
    there are not constraints on the type of output produced.

    An LTN function implements a `torch.nn.Module` instance that can "broadcast" LTN terms as follows:
    1. Evaluating a term with one variable of n individuals yields n output values,
    where the i-th output value corresponds to the term calculated with the i-th individual.
    2. Evaluating a term with k variables (x1,...,xk) with respectively n1,...,nk
    individuals each, yields a result with n1*...*nk values. The result is organized in a tensor
    where the first k dimensions can be indexed to retrieve the outcome(s) that correspond to each variable.
    The attribute `free_vars` tells which axis corresponds to which variable in the LTNObject output by
    the function (using the name of the variable).

    Parameters
    ----------
    model: :class:`torch.nn.Module` [optional]
        PyTorch model that becomes the definition (grounding) of the function.
    func: :class:`function` [optional]
        If a model is not given, it is possible to give a lambda function or a function.
        In this case, the lambda function will be used to define a non-trainable model for the LTN function.

    Attributes
    ----------
    model: :class:`torch.nn.Module` or :class:`function`
        The definition (grounding) of the LTN function. The grounding of a function is a non-trainable model
        implemented using a lambda function or a function, or a learnable model. When some objects are given in input
        to the function model, the model returns a tensor in the real filed for each combination of the values of the
        objects.
        Then, an LTNObject is created as a result of the computation. The attribute `free_vars` will contain the list
        of free variables contained in the LTNObject in output.

    Raises
    ----------
    :class:`TypeError`
        Raises when the types of the input parameters are incorrect.

    :class:`ValueError`
        Raises when the values of the input parameters are incorrect.
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
            raise ValueError("Both model and lambda_func are not None. Only one of the two ways to define the "
                             "predicate can be used.")

        if model is None and func is None:
            raise ValueError("Both model and func are None. At least one of the two has to be defined.")

        if model is not None:
            if not isinstance(model, nn.Module):
                raise TypeError("The given model is not a PyTorch model. Only PyTorch models are allowed "
                                "for model parameter.")
            self.model = model
        else:
            if not isinstance(func, types.LambdaType):
                raise TypeError("The given model is not a lambda function or a function. "
                                "Only lambda functions or functions are allowed for func parameter.")
            self.model = LambdaModel(func)

    def forward(self, *inputs, **kwargs):
        """Encapsulates the "self.model.forward()" to handle the ltn-broadcasting (by using
        process_ltn_objects() method).

        Parameters
        ----------
        inputs: :obj:`list` or :obj:`tuple`
            List or tuple of LTN objects for which the function has to be computed.
        Returns
        ----------
        :class:`LTNObject`
            An `LTNObject` whose `value` parameter contains the result of the function computation, and
            `free_vars` parameter contains the labels of the free variables contained in the result.
        """
        inputs = list(inputs)
        if not all(isinstance(x, LTNObject) for x in inputs):
            raise TypeError("Only LTNObject objects are valid inputs for function.")

        proc_objs, output_vars, output_shape = process_ltn_objects(inputs)

        # the management of the input is left to the model or the lambda function
        output = self.model(*[o.value for o in proc_objs], **kwargs)

        output = torch.reshape(output, tuple(output_shape + list(output.shape[1::])))

        output = output.float()

        return LTNObject(output, output_vars)


def diag(*vars):
    """
    Sets the given LTN variables for diagonal quantification. The diagonal quantification disables the broadcasting
    for these variables (broadcasting is achieved via process_ltn_objects()). It means that when they are given to
    a predicate, or a function, their shape is not changed as it happens with broadcasting. This is
    useful when it is not needed to compute a predicate or a function on all the possible combinations of individuals
    of the variables, but on specific tuples of individuals only.

    Note that diagonal quantification should be used when the given variables share the same number of individuals, for
    example, when we have images and associated labels, which come in a one-to-one correspondence, like it happens for
    computer vision samples.

    Given 2 (or more) LTN variables, there are scenarios where one wants to express statements about
    specific pairs (or tuples) only, such that the i-th tuple contains the i-th instances of the variables.
    We allow this using `ltn.diag`.
    Note: diagonal quantification assumes that the variables have the same number of individuals.
    Given a predicate `P(x,y)` with two variables `x` and `y`,
    the usual broadcasting followed by an aggregation would compute (in Python pseudo-code):
        ```
        for i,x_i in enumerate(x):
            for j,y_j in enumerate(y):
                results[i,j]=P(x_i,y_i)
        aggregate(results)
        ```
    In contrast, diagonal quantification would compute:
        ```
        for i,(x_i, y_i) in enumerate(zip(x,y)):
            results[i].append(P(x_i,y_i))
        aggregate(results)
        ```
    LTN computes only the "zipped" results when diagonal quantification is performed.

    Parameters
    ----------
    vars: :obj:`list` or :obj:`tuple`
        List or tuple of LTN variables for which the diagonal quantification has to be set.

    Returns
    ---------
    :obj:`list`
        List of the same LTN variables given in input, where the attribute `free_vars` has been changed to allow
        the use of diagonal quantification.

    Raises
    ----------
    :class:`TypeError`
        Raises when the types of the input parameters are incorrect.

    :class:`ValueError`
        Raises when the values of the input parameters are incorrect.
    """
    vars = list(vars)
    # check if a list of LTN variables has been passed
    if not all(isinstance(x, Variable) for x in vars):
        raise TypeError("Diagonal quantification only accepts LTN variables.")
    # check if the user has given only one variable
    if not len(vars) > 1:
        raise ValueError("It is not possible to perform diagonal quantification on a single "
                         "variable. At least two variables have to be given.")
    # check if variables have the same number of individuals, assuming the first dimension is the batch dimension
    n_individuals = [var.shape()[0] for var in vars]
    if not len(
        set(n_individuals)) == 1:
        raise ValueError("The given variables have a different number of individuals between each other."
                                  " It is not possible to perform diagonal quantification between variables that"
                                  " have a different number of individuals.")
    diag_label = "diag_" + "_".join([var.latent_var for var in vars])
    for var in vars:
        var.free_vars = [diag_label]
    return vars


def undiag(*vars):
    """
    Resets the usual broadcasting strategy for the given LTN variables. In other words, it removes the diagonal
    quantification from the variables.

    In practice, `ltn.diag` is designed to be used with quantifiers.
    Every quantifier automatically calls `ltn.undiag` after the aggregation has been performed.
    By doing so, the variables can continue to keep their normal behavior outside of the formula and it is not needed
    that the user calls `ltn.undiag` by his own.

    It is recommended to use `ltn.diag` only in quantified formulas since it has been designed for that reason.
    An example of use of `ltn.diag` is the following:
        ```
        Forall(ltn.diag(x,l), C(x,l))
        ```

    Parameters
    ----------
    vars: :obj:`list` or :obj:`tuple`
        List or tuple of LTN variables for which the diagonal quantification setting has to be removed.
    Returns
    ----------
    :obj:`list`
        List of the same LTN variables given in input with the attribute `free_vars` changed in such a way that
        the diagonal quantification setting has been removed.

    Raises
    ----------
    :class:`TypeError`
        Raises when the types of the input parameters are incorrect.
    """
    vars = list(vars)
    # check if a list of LTN variables has been passed
    if not all(isinstance(x, Variable) for x in vars):
        raise TypeError("Undiagonal quantification only accepts LTN variables.")

    for var in vars:
        var.free_vars = [var.latent_var]
    return vars


class Connective:
    """
    Class to wrap unary/binary connective operators to use them within LTN formulas.

    LTN supports various logical connectives. They are grounded using fuzzy semantics.
    The implementation of some common fuzzy logic operators using PyTorch primitives is in `ltn.fuzzy_ops`.
    The wrapper `ltn.Connective` allows to use these fuzzy operators with LTN formulas.
    It takes care of combining sub-formulas that have different variables appearing in them
    (the sub-formulas may have different dimensions that need to be "broadcasted").

    Parameters
    ----------
    connective_op: :class:`ltn.fuzzy_ops.ConnectiveOperator`
        The original unary/binary fuzzy connective operator (without broadcasting).

    Attributes
    -----------
    connective_op: :class:`ltn.fuzzy_ops.ConnectiveOperator`
        The original unary/binary fuzzy connective operator (without broadcasting).

    Raises
    ----------
    :class:`TypeError`
        Raises when the type of the input parameter is incorrect.
    """

    def __init__(self, connective_op):
        if not isinstance(connective_op, ltn.fuzzy_ops.ConnectiveOperator):
            raise TypeError("A ConnectiveOperator object has to be given.")
        self.connective_op = connective_op

    def __call__(self, *operands, **kwargs):
        """
        It applies the selected fuzzy connective operator to the operands (LTN objects) given in input. To do so, it
        firstly broadcasts the input objects to make them compatible to apply the operator element-wise.

        Parameters
        -----------
        operands: :obj:'list'
            List of LTN objects representing the operands to which the fuzzy connective operator has to
            be applied. These operands are tensors filled with truth values on which the connective has to be applied.
            These truth values are the result of the evaluation of the operands in previous steps.

        Returns
        ----------
        :class:`LTNObject`
            The LTNObject that is the result of the application of the fuzzy connective operator to the input
            LTN objects.

        Raises
        ----------
        :class:`TypeError`
            Raises when the types of the input parameters are incorrect.

        :class:`ValueError`
            Raises when the values of the input parameters are incorrect.
        """
        operands = list(operands)

        if isinstance(self.connective_op, ltn.fuzzy_ops.UnaryConnectiveOperator) and len(operands) != 1:
            raise ValueError("You have given more than one operand or no operand to a unary connective.")
        if isinstance(self.connective_op, ltn.fuzzy_ops.BinaryConnectiveOperator) and len(operands) != 2:
            raise ValueError("You have given more than two operands or less than two operands to a binary connective.")

        if not all(isinstance(x, LTNObject) for x in operands):
            raise TypeError("A connective only accepts LTN operands.")

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
    Class to wrap quantification operators to use them within LTN formulas.

    LTN supports universal and existential quantification. They are grounded using fuzzy aggregation operators.
    The implementation of some common aggregators using PyTorch primitives is in `ltn.fuzzy_ops`.
    The wrapper allows to use the quantifiers with LTN formulas.
    It takes care of selecting the tensor dimensions to aggregate, given some variables in arguments.
    Additionally, boolean conditions (by setting parameters `mask_fn` and `mask_vars`) can be used for
    guarded quantification.

    Parameters
    ----------
    agg_op: :class:`ltn.fuzzy_ops.AggregationOperator`
        The fuzzy aggregation operator to perform the desired quantification.
    quantifier: :obj:`string`
        String indicating the quantification that has to be performed ('e' for exists, or 'f'
        for forall).

    Attributes
    -----------
    agg_op: :class:`ltn.fuzzy_ops.AggregationOperator`
        See agg_op parameter.
    quantifier: :obj:`string`
        See quantifier parameter.

    Raises
    ----------
    :class:`TypeError`
        Raises when the type of the agg_op parameter is incorrect.

    :class:`ValueError`
        Raises when the value of the quantifier parameter is incorrect.
    """
    def __init__(self, agg_op, quantifier):
        if not isinstance(agg_op, ltn.fuzzy_ops.AggregationOperator):
            raise TypeError("An AggregationOperator object has to be given.")
        self.agg_op = agg_op
        if quantifier not in ["f", "e"]:
            raise ValueError("The keywords for quantifiers are 'f' for \"forall\" and 'e' \"exists\".")
        self.quantifier = quantifier

    def __call__(self, vars, formula, cond_vars=None, cond_fn=None, **kwargs):
        """
        It applies the desired quantification at the formula given in input based on the selected variables. In
        particular, the formula is an LTNObject containing the truth values which are the result of the evaluation
        of the formula in previous steps. Specifically, the quantification is performed by aggregating the dimensions
        of the formula's tensor specified by the variables given in input.
        It is also possible to perform a guarded quantification, namely a quantification applied to truth values which
        satisfy a given condition. In that case, `cond_vars` and `cond_fn` have to be set properly.
        If 'mask_vars' and 'mask_fn' are left `None`, it means that no guarded quantification has to be performed.
        An example of guarded quantification is the following:
        ∀u1,u2,i : u1 ~= u2 (Sim(u1, u2) and Likes(u1, i) -> Likes(u2, i))
        This formula means: for all the pair of users such that they are different users, if they are similar, they
        tend to give the same ratings to the same items.
        Here, the condition is u1 ~= u2, and the condition variables are u1 and u2.

        Parameters
        -----------
        vars: :obj:`list`
            List of LTN variables on which the quantification has to be performed.
        formula: :class:`LTNObject`
            LTNObject which contains the truth values that have to be aggregated. These truth values are the
            result of a logical formula.
        cond_vars: :obj:`list` [optional]
            List of LTN variables that appear in the guarded quantification condition.
        cond_fn: :class:`function` [optional]
            Function which implements the guarded quantification condition. The condition is based on the
            variables contained in `cond_vars` parameter.

        Raises
        ----------
        :class:`TypeError`
            Raises when the types of the input parameters are incorrect.

        :class:`ValueError`
            Raises when the values of the input parameters are incorrect.
        """
        # first of all, check if user has correctly set the condition vars and the condition function
        if cond_vars is not None and cond_fn is None:
            raise ValueError("You have set a condition function, but you do not have specified condition variables.")
        if cond_vars is None and cond_fn is not None:
            raise ValueError("You have specified condition variables, but you do not have set a condition function.")
        # check that vars is a list of LTN variables or a single LTN variable
        if not all(isinstance(x, Variable) for x in vars) if isinstance(vars, list) else not isinstance(vars, Variable):
            raise TypeError("vars argument must be a list of LTN variables or one LTN variable.")

        # check that formula is an LTNObject
        if not isinstance(formula, LTNObject):
            raise TypeError("formula argument must be a LTNObject")

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
                raise TypeError("cond_vars argument must be a list of LTN variables or one LTN variable.")
            # check that cond_fn is a function
            if not isinstance(cond_fn, types.LambdaType):
                raise TypeError("cond_fn argument must be a function or lambda function.")

            if isinstance(cond_vars, Variable):
                cond_vars = [cond_vars]  # this serves in the case vars is just a variable and not a list of variables

            # create the mask for applying the guarded quantification
            formula, mask = self.compute_mask(formula, cond_vars, cond_fn, list(aggregation_vars))

            # we apply the mask to the truth values of the formula
            # the idea is to put NaN values where the mask is False (condition unsatisfied), while the rest of the
            # truth values are kept untouched
            masked_formula = torch.where(
                ~mask.value,
                np.nan,
                formula.value.double()  # necessary for type incompatibilities
            )

            # we perform the desired quantification after the mask has been applied
            aggregation_dims = [formula.free_vars.index(var) for var in aggregation_vars]
            output = self.agg_op(masked_formula, aggregation_dims, **kwargs)

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
                output
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
