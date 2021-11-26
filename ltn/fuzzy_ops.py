import torch
import numpy as np
from ltn import LTNObject
"""
This module of the LTN framework contains the PyTorch implementation of some common fuzzy logic operators. Refer to the
LTN paper for a detailed description of these operators (see the Appendix).
The operators support the traditional NumPy/PyTorch broadcasting.

In order to use these fuzzy operators with LTN formulas (broadcasting w.r.t. LTN variables appearing in a formula), 
it is necessary to wrap the operators with `ltn.Connective` or `ltn.Quantifier`. 
"""
# these are the projection functions to make the Product Real Logic stable. These functions help to change the input
# of particular fuzzy operators in such a way they do not lead to gradient problems (vanishing, exploding).
eps = 1e-4  # epsilon is set to small value in such a way to not change the input too much


def pi_0(x):
    """
    Function that has to be used when we need to assure that the truth value in input to a fuzzy operator is never equal
    to zero, in such a way to avoid gradient problems. It maps the interval [0, 1] in the interval ]0, 1], where the 0
    is excluded.

    Parameters
    -----------
    x: :class:`torch.Tensor`
        A truth value.

    Returns
    -----------
    :class:`torch.Tensor`
        The input truth value changed in such a way to prevent gradient problems (0 is changed with a small number
        near 0).
    """
    return (1 - eps) * x + eps


def pi_1(x):
    """
    Function that has to be used when we need to assure that the truth value in input to a fuzzy operator is never equal
    to one, in such a way to avoid gradient problems. It maps the interval [0, 1] in the interval [0, 1[, where the 1
    is excluded.

    Parameters
    -----------
    x: :class:`torch.Tensor`
        A truth value.

    Returns
    -----------
    :class:`torch.Tensor`
        The input truth value changed in such a way to prevent gradient problems (1 is changed with a small number
        near 1).
    """
    return (1 - eps) * x


# utility function to check the input of connectives and quantifiers
def check_values(*values):
    """
    This function checks the input values are in the range [0., 1.] and raises an exception if it is not the case.

    Parameters
    -----------
    values: :obj:`list` or :obj:`tuple`
        List or tuple of :class:`torch.Tensor` containing the truth values of the operands.

    Raises
    -----------
    :class:`ValueError`
        Raises when the values of the input parameters are incorrect.
    """
    values = list(values)
    for v in values:
        if not torch.all(torch.where(torch.logical_or(torch.logical_and(v >= 0., v <= 1.), torch.isnan(v)), 1., 0.)):
            raise ValueError("The input/inputs must contain truth values in [0., 1.]. Some of the values of "
                             "the input/inputs are out of this range.")


# here, it begins the implementation of fuzzy operators in PyTorch

class ConnectiveOperator:
    """
    Abstract class for connective operators.

    Every connective operator implemented in LTN must inherit from this class and implement the __call__ method.

    Raises
    -----------
    :class:`NotImplementedError`
        Raised when __call__ is not implemented in the sub-class.
    """
    def __call__(self, *args, **kwargs):
        """
        Call method for the connective operator.

        This method implements the behavior of the connective operator, which is usually an element-wise operation.
        """
        raise NotImplementedError()


class UnaryConnectiveOperator(ConnectiveOperator):
    """
    Abstract class for unary connective operators.

    Every unary connective operator implemented in LTN must inherit from this class and implement the __call__ method.

    Raises
    -----------
    :class:`NotImplementedError`
        Raised when __call__ is not implemented in the sub-class.
    """
    def __call__(self, *args, **kwargs):
        """
        Call method for the connective operator.

        This method implements the behavior of the connective operator, which is usually an element-wise operation.
        """
        raise NotImplementedError()


class BinaryConnectiveOperator(ConnectiveOperator):
    """
    Abstract class for binary connective operators.

    Every binary connective operator implemented in LTN must inherit from this class and implement the __call__ method.

    Raises
    -----------
    :class:`NotImplementedError`
        Raised when __call__ is not implemented in the sub-class.
    """
    def __call__(self, *args, **kwargs):
        """
        Call method for the connective operator.

        This method implements the behavior of the connective operator, which is usually an element-wise operation.
        """
        raise NotImplementedError()

# NEGATION


class NotStandard(UnaryConnectiveOperator):
    """
    Implementation of the standard fuzzy negation.
    """
    def __call__(self, x):
        """
        Method __call__ for the standard fuzzy negation operator.

        Parameters
        -----------
        x: :class:`torch.Tensor`
            The input truth value.

        Returns
        ----------
        :class:`torch.Tensor`
            The standard negation of the input.

        Raises
        ----------
        :class:`ValueError`
            Raises when the values of the operand are not in [0., 1.].
        """
        check_values(x)
        return 1. - x


class NotGodel(UnaryConnectiveOperator):
    """
    Implementation of the Godel fuzzy negation.
    """
    def __call__(self, x):
        """
        Method __call__ for the Godel fuzzy negation operator.

        Parameters:
        -----------
        x: :class:`torch.Tensor`
            The input truth value.

        Returns
        -----------
        :class:`torch.Tensor`
            The Godel negation of the input.

        Raises
        ----------
        :class:`ValueError`
            Raises when the values of the operand are not in [0., 1.].
        """
        check_values(x)
        return torch.eq(x, 0.)

# CONJUNCTION


class AndMin(BinaryConnectiveOperator):
    """
    Implementation of the Godel fuzzy conjunction (min operator).
    """
    def __call__(self, x, y):
        """
        Method __call__ for the Godel fuzzy conjunction operator.

        Parameters
        ----------
        x: :class:`torch.Tensor`
            The truth value of the first input;
        y: :class:`torch.Tensor`
            The truth value of the second input.

        Returns
        ----------
        :class:`torch.Tensor`
            The Godel conjunction of the two inputs.

        Raises
        ----------
        :class:`ValueError`
            Raises when the values of the operands are not in [0., 1.].
        """
        check_values(x, y)
        return torch.minimum(x, y)


class AndProd(BinaryConnectiveOperator):
    """
    Implementation of the Goguen fuzzy conjunction (product operator).
    """
    def __init__(self, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the Goguen fuzzy conjunction or not.

        Parameters
        -----------
        stable: :obj:`bool`
            A boolean flag indicating whether it has to be used the stable version of the operator or not.
        """
        self.stable = stable

    def __call__(self, x, y, stable=None):
        """
        Method __call__ for the Goguen fuzzy conjunction operator.

        Parameters
        ----------
        x: :class:`torch.Tensor`
            The truth value of the first input.
        y: :class:`torch.Tensor`
            The truth value of the second input.
        stable: :obj:`bool`
            A boolean flag indicating whether it has to be used the stable version of the operator or not.

        Returns
        -----------
        :class:`torch.Tensor`
            The Goguen conjunction of the two inputs.

        Raises
        ----------
        :class:`ValueError`
            Raises when the values of the operands are not in [0., 1.].
        """
        check_values(x, y)
        stable = self.stable if stable is None else stable
        if stable:
            x, y = pi_0(x), pi_0(y)
        return torch.mul(x, y)


class AndLuk(BinaryConnectiveOperator):
    """
    Implementation of the Lukasiewicz fuzzy conjunction.
    """
    def __call__(self, x, y):
        """
        Method __call__ for the Lukasiewicz fuzzy conjunction operator.

        Parameters
        -----------
        x: :class:`torch.Tensor`
            The truth value of the first input.
        y: :class:`torch.Tensor`
            The truth value of the second input.

        Returns
        -----------
        :class:`torch.Tensor`
            The Lukasiewicz fuzzy conjunction of the two inputs.

        Raises
        ----------
        :class:`ValueError`
            Raises when the values of the operands are not in [0., 1.].
        """
        check_values(x, y)
        zeros = torch.zeros_like(x)
        return torch.maximum(x + y - 1., zeros)

# DISJUNCTION


class OrMax(BinaryConnectiveOperator):
    """
    Implementation of the Godel fuzzy disjunction (max operator).
    """
    def __call__(self, x, y):
        """
        Method __call__ for the Godel fuzzy disjunction operator.

        Parameters
        -----------
        x: :class:`torch.Tensor`
            The truth value of the first input.
        y: :class:`torch.Tensor`
            The truth value of the second input.

        Returns
        -----------
        :class:`torch.Tensor`
            The Godel fuzzy disjunction of the two inputs.

        Raises
        ----------
        :class:`ValueError`
            Raises when the values of the operands are not in [0., 1.].
        """
        check_values(x, y)
        return torch.maximum(x, y)


class OrProbSum(BinaryConnectiveOperator):
    """
    Implementation of the Goguen fuzzy disjunction (probabilistic sum operator).
    """
    def __init__(self, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the Goguen fuzzy disjunction or not.

        Parameters
        -----------
        stable: :obj:`bool`
            A boolean flag indicating whether it has to be used the stable version of the operator or not.
        """
        self.stable = stable

    def __call__(self, x, y, stable=None):
        """
        Method __call__ for the Goguen fuzzy disjunction operator.

        Parameters
        ----------
        x: :class:`torch.Tensor`
            The truth value of the first input.
        y: :class:`torch.Tensor`
            The truth value of the second input.
        stable: :obj:`bool`
            A boolean flag indicating whether it has to be used the stable version of the operator or not.

        Returns
        -----------
        :class:`torch.Tensor`
            The Goguen fuzzy disjunction of the two inputs.

        Raises
        ----------
        :class:`ValueError`
            Raises when the values of the operands are not in [0., 1.].
        """
        check_values(x, y)
        stable = self.stable if stable is None else stable
        if stable:
            x, y = pi_1(x), pi_1(y)
        return x + y - torch.mul(x, y)


class OrLuk(BinaryConnectiveOperator):
    """
    Implementation of the Lukasiewicz fuzzy disjunction.
    """
    def __call__(self, x, y):
        """
        Method __call__ for the Lukasiewicz fuzzy disjunction operator.

        Parameters
        -----------
        x: :class:`torch.Tensor`
            The truth value of the first input.
        y: :class:`torch.Tensor`
            The truth value of the second input.

        Returns
        -----------
        :class:`torch.Tensor`
            The Lukasiewicz fuzzy disjunction of the two inputs.

        Raises
        ----------
        :class:`ValueError`
            Raises when the values of the operands are not in [0., 1.].
        """
        check_values(x, y)
        ones = torch.ones_like(x)
        return torch.minimum(x + y, ones)

# IMPLICATION (differences between strong and residuated implications can be found in the Appendix of the LTN paper)


class ImpliesKleeneDienes(BinaryConnectiveOperator):
    """
    Implementation of the Kleene Dienes fuzzy implication.
    """
    def __call__(self, x, y):
        """
        Method __call__ for the Kleene Dienes implication operator.

        Parameters
        ----------
        x: :class:`torch.Tensor`
            The truth value of the first input.
        y: :class:`torch.Tensor`
            The truth value of the second input.

        Returns
        -----------
        :class:`torch.Tensor`
            The Kleene Dienes implication of the two inputs.

        Raises
        ----------
        :class:`ValueError`
            Raises when the values of the operands are not in [0., 1.].
        """
        check_values(x, y)
        return torch.maximum(1. - x, y)


class ImpliesGodel(BinaryConnectiveOperator):
    """
    Implementation of the Godel fuzzy implication.
    """
    def __call__(self, x, y):
        """
        Method __call__ for the Godel implication operator.

        Parameters
        ----------
        x: :class:`torch.Tensor`
            The truth value of the first input.
        y: :class:`torch.Tensor`
            The truth value of the second input.

        Returns
        -----------
        :class:`torch.Tensor`
            The Godel implication of the two inputs.

        Raises
        ----------
        :class:`ValueError`
            Raises when the values of the operands are not in [0., 1.].
        """
        check_values(x, y)
        return torch.where(torch.le(x, y), torch.ones_like(x), y)


class ImpliesReichenbach(BinaryConnectiveOperator):
    """
    Implementation of the Reichenbach fuzzy implication.
    """
    def __init__(self, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the Reichenbach fuzzy implication or not.

        Parameters
        -----------
        stable: :obj:`bool`
            A boolean flag indicating whether it has to be used the stable version of the operator or not.
        """
        self.stable = stable

    def __call__(self, x, y, stable=None):
        """
        Method __call__ for the Reichenbach implication operator.

        Parameters
        ----------
        x: :class:`torch.Tensor`
            The truth value of the first input.
        y: :class:`torch.Tensor`
            The truth value of the second input.
        stable: :obj:`bool`
            A boolean flag indicating whether it has to be used the stable version of the operator or not.

        Returns
        -----------
        :class:`torch.Tensor`
            The Reichenbach implication of the two inputs.

        Raises
        ----------
        :class:`ValueError`
            Raises when the values of the operands are not in [0., 1.].
        """
        check_values(x, y)
        stable = self.stable if stable is None else stable
        if stable:
            x, y = pi_0(x), pi_1(y)
        return 1. - x + torch.mul(x, y)


class ImpliesGoguen(BinaryConnectiveOperator):
    """
    Implementation of the Goguen fuzzy implication.
    """
    def __init__(self, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the Goguen fuzzy implication or not.

        Parameters
        -----------
        stable: :obj:`bool`
            A boolean flag indicating whether it has to be used the stable version of the operator or not.
        """
        self.stable = stable

    def __call__(self, x, y, stable=None):
        """
        Method __call__ for the Goguen implication operator.

        Parameters
        ----------
        x: :class:`torch.Tensor`
            The truth value of the first input.
        y: :class:`torch.Tensor`
            The truth value of the second input.
        stable: :obj:`bool`
            A boolean flag indicating whether it has to be used the stable version of the operator or not.

        Returns
        -----------
        :class:`torch.Tensor`
            The Goguen implication of the two inputs.

        Raises
        ----------
        :class:`ValueError`
            Raises when the values of the operands are not in [0., 1.].
        """
        check_values(x, y)
        stable = self.stable if stable is None else stable
        if stable:
            x = pi_0(x)
        return torch.where(torch.le(x, y), torch.ones_like(x), torch.div(y, x))


class ImpliesLuk(BinaryConnectiveOperator):
    """
    Implementation of the Lukasiewicz fuzzy implication.
    """
    def __call__(self, x, y):
        """
        Method __call__ for the Lukasiewicz implication operator.

        Parameters
        ----------
        x: :class:`torch.Tensor`
            The truth value of the first input.
        y: :class:`torch.Tensor`
            The truth value of the second input.

        Returns
        -----------
        :class:`torch.Tensor`
            The Lukasiewicz implication of the two inputs.

        Raises
        ----------
        :class:`ValueError`
            Raises when the values of the operands are not in [0., 1.].
        """
        check_values(x, y)
        ones = torch.ones_like(x)
        return torch.minimum(1. - x + y, ones)

# EQUIVALENCE


class Equiv(BinaryConnectiveOperator):
    """
    Returns an operator that computes: And(Implies(x,y),Implies(y,x)). In other words, it computes: x -> y AND y -> x.
    """
    def __init__(self, and_op, implies_op):
        """
        This constructor has to be used to set the operator for the conjunction and for the implication of the
        equivalence operator.

        Parameters
        ----------
        and_op: :class:`ltn.fuzzy_ops.BinaryConnectiveOperator`
            Fuzzy operator for the conjunction.
        implies_op: :class:`ltn.fuzzy_ops.BinaryConnectiveOperator`
            Fuzzy operator for the implication.
        """
        self.and_op = and_op
        self.implies_op = implies_op

    def __call__(self, x, y):
        """
        Method __call__ for the equivalence operator.

        Parameters
        ----------
        x: :class:`torch.Tensor`
            The truth value of the first input.
        y: :class:`torch.Tensor`
            The truth value of the second input.

        Returns
        -----------
        :class:`torch.Tensor`
            The fuzzy equivalence of the two inputs.

        Raises
        ----------
        :class:`ValueError`
            Raises when the values of the operands are not in [0., 1.].
        """
        check_values(x, y)
        return self.and_op(self.implies_op(x, y), self.implies_op(y, x))

# AGGREGATORS FOR QUANTIFIERS - only the aggregators introduced in the LTN paper are implemented


class AggregationOperator:
    """
    Abstract class for aggregation operators.

    Every aggregation operator implemented in LTN must inherit from this class and implement the __call__ method.

    Raises
    -----------
    :class:`NotImplementedError`
        Raised when __call__ is not implemented in the sub-class.
    """
    def __call__(self, *args, **kwargs):
        """
        Call method for the aggregation operator.

        This method implements the behavior of the aggregation operator, which is usually an aggregation operation, for
        example the mean.
        """
        raise NotImplementedError()


class AggregMin(AggregationOperator):
    """
    Implementation of the min aggregator operator.
    """
    def __call__(self, xs, dim=None, keepdim=False):
        """
        Method __call__ for the min aggregator operator. Notice the use of torch.where(). This has to be used
        because the guarded quantification is implemented in PyTorch by putting NaN values where the grounding of the
        formula does not satisfy the guarded condition. Therefore, if we aggregate on a tensor with NaN values, it is
        highly probable that we will obtain NaN as the output of the aggregation. For this reason, the aggregation do
        not have to consider the NaN values contained in the input tensor.

        Parameters
        ----------
        xs: :class:`torch.Tensor`
            The truth values (grounding of formula) for which the aggregation has to be computed.
        dim: :obj:`tuple`
            Tuple containing the dimensions on which the aggregation has to be performed.
        keepdim: :obj:`bool`
            Boolean flag indicating whether the output has to keep the same dimensions as the input after
            the aggregation.

        Returns
        ----------
        :class:`torch.Tensor`
            The result of the min aggregation. The shape of the result depends on the variables that are used
            in the quantification (namely, the dimensions across which the aggregation has been computed).

        Raises
        ----------
        :class:`ValueError`
            Raises when the truth values given in input are not in the range [0., 1.].
        """
        # we have to put 1 where there are NaN values, since 1 is the maximum value for a truth value. By doing so,
        # this modification will not affect the minimum computation
        check_values(xs)
        xs = torch.where(torch.isnan(xs), 1., xs.double())
        out = torch.amin(xs.float(), dim=dim, keepdim=keepdim)
        return out


class AggregMean(AggregationOperator):
    """
    Implementation of the mean aggregator operator. Notice the use of torch.where(). This has to be used
    because the guarded quantification is implemented in PyTorch by putting NaN values where the grounding of the
    formula does not satisfy the guarded condition. Therefore, if we aggregate on a tensor with NaN values, it is
    highly probable that we will obtain NaN as the output of the aggregation. For this reason, the aggregation do
    not have to consider the NaN values contained in the input tensor.
    """
    def __call__(self, xs, dim=None, keepdim=False):
        """
        Method __call__ for the mean aggregator operator.

        Parameters
        ----------
        xs: :class:`torch.Tensor`
            The truth values (grounding of formula) for which the aggregation has to be computed.
        dim: :obj:`tuple`
            Tuple containing the dimensions on which the aggregation has to be performed.
        keepdim: :obj:`bool`
            Boolean flag indicating whether the output has to keep the same dimensions as the input after
            the aggregation.

        Returns
        ----------
        :class:`torch.Tensor`
            The result of the mean aggregation. The shape of the result depends on the variables that are used
            in the quantification (namely, the dimensions across which the aggregation has been computed).

        Raises
        ----------
        :class:`ValueError`
            Raises when the truth values given in input are not in the range [0., 1.].
        """
        check_values(xs)
        numerator = torch.sum(torch.where(torch.isnan(xs), torch.zeros_like(xs), xs), dim=dim, keepdim=keepdim)
        denominator = torch.sum(~torch.isnan(xs), dim=dim, keepdim=keepdim)
        return torch.div(numerator, denominator)


class AggregPMean(AggregationOperator):
    """
    Implementation of the p-mean aggregator operator. This has been selected as an approximation of the existential
    quantifier with parameter p equal to or greater than 1. If p tends to infinity, the p-mean aggregator tends to the
    maximum of the input values (approximation of fuzzy existential quantification).
    """
    def __init__(self, p=2, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the p-mean aggregator or not. Also, it is possible to set the value of the parameter p.

        Parameters
        ----------
        p: :obj:`int`
            Value of the parameter p.
        stable: :obj:`bool`
            A boolean flag indicating whether it has to be used the stable version of the aggregator or not.
        """
        self.p = p
        self.stable = stable

    def __call__(self, xs, dim=None, keepdim=False, p=None, stable=None):
        """
        Method __call__ for the p-mean aggregator operator. Notice the use of torch.where(). This has to be used
        because the guarded quantification is implemented in PyTorch by putting NaN values where the grounding of the
        formula does not satisfy the guarded condition. Therefore, if we aggregate on a tensor with NaN values, it is
        highly probable that we will obtain NaN as the output of the aggregation. For this reason, the aggregation do
        not have to consider the NaN values contained in the input tensor.

        Parameters
        ----------
        xs: :class:`torch.Tensor`
            The truth values (grounding of formula) for which the aggregation has to be computed.
        dim: :obj:`tuple`
            Tuple containing the dimensions on which the aggregation has to be performed.
        keepdim: :obj:`bool`
            Boolean flag indicating whether the output has to keep the same dimensions as the input after
            the aggregation.
        stable: :obj:`bool`
            A boolean flag indicating whether it has to be used the stable version of the aggregator or not.

        Returns
        ----------
        :class:`torch.Tensor`
            The result of the p-mean aggregation. The shape of the result depends on the variables that are used
            in the quantification (namely, the dimensions across which the aggregation has been computed).

        Raises
        ----------
        :class:`ValueError`
            Raises when the truth values given in input are not in the range [0., 1.].
        """
        check_values(xs)
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable
        if stable:
            xs = pi_0(xs)
        xs = torch.pow(xs, p)
        numerator = torch.sum(torch.where(torch.isnan(xs), torch.zeros_like(xs), xs), dim=dim, keepdim=keepdim)
        denominator = torch.sum(~torch.isnan(xs), dim=dim, keepdim=keepdim)
        return torch.pow(torch.div(numerator, denominator), 1 / p)


class AggregPMeanError(AggregationOperator):
    """
    Implementation of the p-mean error aggregator operator. This has been selected as an approximation of the universal
    quantifier with parameter p equal to or greater than 1. If p tends to infinity, the p-mean error aggregator tends
    to the minimum of the input values (approximation of fuzzy universal quantification).
    """
    def __init__(self, p=2, stable=True):
        """
        This constructor has to be used to set whether it has to be used the stable version (it avoids gradient
        problems) of the p-mean error aggregator or not. Also, it is possible to set the value of the parameter p.

        Parameters
        ----------
        p: :obj:`int`
            Value of the parameter p.
        stable: :obj:`bool`
            A boolean flag indicating whether it has to be used the stable version of the aggregator or not.
        """
        self.p = p
        self.stable = stable

    def __call__(self, xs, dim=None, keepdim=False, p=None, stable=None):
        """
        Method __call__ for the p-mean error aggregator operator. Notice the use of torch.where(). This has to be used
        because the guarded quantification is implemented in PyTorch by putting NaN values where the grounding of the
        formula does not satisfy the guarded condition. Therefore, if we aggregate on a tensor with NaN values, it is
        highly probable that we will obtain NaN as the output of the aggregation. For this reason, the aggregation do
        not have to consider the NaN values contained in the input tensor.

        Parameters
        ----------
        xs: :class:`torch.Tensor`
            The truth values (grounding of formula) for which the aggregation has to be computed.
        dim: :obj:`tuple`
            Tuple containing the dimensions on which the aggregation has to be performed.
        keepdim: :obj:`bool`
            Boolean flag indicating whether the output has to keep the same dimensions as the input after
            the aggregation.
        stable: :obj:`bool`
            A boolean flag indicating whether it has to be used the stable version of the aggregator or not.

        Returns
        ----------
        :class:`torch.Tensor`
            The result of the p-mean error aggregation. The shape of the result depends on the variables that are used
            in the quantification (namely, the dimensions across which the aggregation has been computed).

        Raises
        ----------
        :class:`ValueError`
            Raises when the truth values given in input are not in the range [0., 1.].
        """
        check_values(xs)
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable
        if stable:
            xs = pi_1(xs)
        xs = torch.pow(1. - xs, p)
        numerator = torch.sum(torch.where(torch.isnan(xs), torch.zeros_like(xs), xs), dim=dim, keepdim=keepdim)
        denominator = torch.sum(~torch.isnan(xs), dim=dim, keepdim=keepdim)
        return 1. - torch.pow(torch.div(numerator, denominator), 1 / p)


class SatAgg:
    """
    Implementation of the SatAgg aggregator operator.

    This operator takes as input the truth values of some closed formulas and returns the aggregation of those values
    using the selected aggregation operator. Specifically, it takes care of aggregating the truth values
    of the formulas contained in a knowledge base.
    """
    def __init__(self, agg_op=AggregPMeanError(p=2)):
        """
        This is the constructor of the SatAgg operator.

        It takes as input an aggregation operator which define the behavior of SatAgg.

        Parameters
        ----------
        agg_op: :class:`AggregationOperator`
            Aggregation operator which implements the SatAgg aggregation. By default is the pMeanError with p=2.

        Raises
        ----------
        :class:`TypeError`
            Raises when the type of the input parameter is not correct.
        """
        if not isinstance(agg_op, AggregationOperator):
            raise TypeError("An AggregationOperator is expected in input.")
        self.agg_op = agg_op

    def __call__(self, *truth_values):
        """
        Method __call__ for the SatAgg aggregator operator.

        It simply applies the selected aggregator (`agg_op` attribute) to the truth values given in input.

        Parameters
        ----------
        truth_values: :obj:`list` or :obj:`tuple`
            List or tuple of truth values (LTNObject) of closed formulas for which the aggregation has to be computed.

        Returns
        ----------
        :class:`torch.Tensor`
            The result of the SatAgg aggregation. Note that the result is a scalar. It is the satisfaction level of
            the knowledge based composed of the closed formulas given in input.

        Raises
        ----------
        :class:`TypeError`
            Raises when the type of the input parameter is not correct.

        :class:`ValueError`
            Raises when the truth values given in input are not in the range [0., 1.].
            Raises when the truth values given in input are not scalars, namely some formulas given in input are not
            closed formulas. The closed formulas are identifiable since they are just scalar because all the variables
            have been quantified (i.e., all dimensions have been aggregated).
        """
        truth_values = list(truth_values)
        if not all(isinstance(x, LTNObject) for x in truth_values):
            raise TypeError("The input must be a list of LTNObject.")
        truth_values = [o.value for o in truth_values]
        if not all([f.shape == torch.Size([]) for f in truth_values]):
            raise ValueError("Each element in truth_values should be a scalar. Only closed formulas are accepted.")
        truth_values = torch.stack(truth_values, dim=0)

        return self.agg_op(truth_values, dim=0)
