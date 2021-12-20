import copy

import pytest
import numpy as np
import ltn
from ltn import LTNObject, Constant, Variable, process_ltn_objects, Predicate, Function, LambdaModel, diag, undiag, \
    Connective, Quantifier
import torch
torch.manual_seed(2020)
torch.set_printoptions(precision=4)


def test_LTNObject():
    wrong_value = [1, 2, 3, 4]
    wrong_var_labels = "x"
    wrong_var_labels_2 = ["x", 2]
    good_value = torch.tensor([1, 2, 3, 4])
    good_var_labels = ["x"]

    # an LTNObject wants PyTorch tensors as value
    with pytest.raises(TypeError):
        obj = LTNObject(wrong_value, wrong_var_labels)

    # the labels must be contained in a list of strings
    with pytest.raises(TypeError):
        obj = LTNObject(good_value, wrong_var_labels)

    # the labels must be contained in a list of strings
    with pytest.raises(TypeError):
        obj = LTNObject(good_value, wrong_var_labels_2)

    obj = LTNObject(good_value, good_var_labels)

    assert hasattr(obj, "value"), "An LTNObject should have a value attribute"
    assert hasattr(obj, "free_vars"), "An LTNObject should have a free_vars attribute"
    assert torch.equal(obj.value, good_value), "The value should be as same as the parameter"
    assert obj.free_vars == good_var_labels, "The free_vars should be as same as the parameter"

    assert obj.shape() == good_value.shape, "The shape should be as same as the shape of the tensor given in input"


def test_Constant():
    wrong_value = [1, 2, 3, 4]
    good_value = torch.tensor([1, 2, 3, 4])

    # a constant only accepts PyTorch tensors as values
    with pytest.raises(TypeError):
        const = Constant(wrong_value)

    # test with trainable False
    const = Constant(good_value)
    assert hasattr(const, "value"), "Constant should have a value attribute"
    assert hasattr(const, "free_vars"), "Constant should have a free_vars attribute"
    assert const.free_vars == [], "The free_vars should be an empty list"
    assert torch.equal(const.value, good_value), "The value should be as same as the parameter"
    assert const.shape() == good_value.shape, "The shape should be as same as the shape of the tensor given in input"
    assert const.value.requires_grad is False, "Since trainable parameter has default value to False, required_grad " \
                                               "should be False"
    assert const.value.device == ltn.device, "The device where the constant is should be as same as the device " \
                                             "detected by LTN"

    # test with trainable True
    const = Constant(good_value, trainable=True)
    assert isinstance(const.value, torch.FloatTensor), "If trainable is set to True, the system should convert the " \
                                                       "tensor (value of the constant) to float"
    assert const.value.requires_grad is True, "Since trainable has been set to True, required_grad should be True"


def test_Variable():
    wrong_value = [1, 2, 3, 4]
    good_value_one_dim = torch.DoubleTensor([1, 2, 3, 4])
    good_value_more_dims = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    wrong_label_int = 1
    wrong_label_diag = "diag_x"
    good_label = "x"

    # var labels must be of type String
    with pytest.raises(TypeError):
        var = Variable(wrong_label_int, good_value_more_dims)
    # var labels can't start with diag since it is reserved
    with pytest.raises(ValueError):
        var = Variable(wrong_label_diag, good_value_more_dims)
    # a variable value must be a PyTorch tensor
    with pytest.raises(TypeError):
        var = Variable(good_label, wrong_value)

    # test with add_batch_dim to True
    var = Variable(good_label, good_value_one_dim)
    assert hasattr(var, "value"), "The variable should have a value attribute"
    assert hasattr(var, "free_vars"), "The variable should have a free_vars attribute"
    assert hasattr(var, "latent_var"), "The variable should have a latent_var attribute"
    assert torch.equal(var.value, torch.unsqueeze(good_value_one_dim.float(), 1)), "The value should be as same " \
                                                                                   "as the parameter, but" \
                                                                                   " with an added dimension, " \
                                                                                   "since add_batch_dim is True."
    assert var.free_vars == [good_label], "free_vars should be a list which contains the var labels given to" \
                                          "the Variable."
    assert var.latent_var == good_label, "latent_var should be equal to the given var label."

    assert isinstance(var.value, torch.FloatTensor), "Since the value passed to the Variable is double, LTN should " \
                                                     "convert it to float to avoid type incompatibilities."
    assert var.shape() == torch.Size([4, 1]), "add_batch_dim is set to True and the shape of the Variable is [1]." \
                                              "The shape should become [4, 1] since the Variable contains 4 " \
                                              "individuals and we have decided to add the batch dimension."
    assert var.value.device == ltn.device, "The Variable should be in the same device as the device detected by LTN."

    # test with add_batch_dim to True but shape different from 1 -> the batch dim should not be added
    var = Variable(good_label, good_value_more_dims)
    assert var.shape() == good_value_more_dims.shape, "No dimension should be added, so the shape should remain " \
                                                      "the same. This because the passed value has already a batch dim."

    # test with add_batch_dim to False
    var = Variable(good_label, good_value_one_dim, add_batch_dim=False)
    assert var.shape() == good_value_one_dim.shape, "No dimension should be added, so the shape should remain " \
                                                    "the same. This because add_batch_dim is set to False."


def test_process_ltn_objects():
    torch.manual_seed(2020)
    # test of function with one LTN Object and something which is not an LTN Object -> exception expected
    c1 = Constant(torch.tensor([1, 2, 3, 4]))
    wrong_object = torch.tensor([3, 4, 5, 6])

    # process_ltn_objects only accepts list of LTNObject
    with pytest.raises(TypeError):
        process_ltn_objects([c1, wrong_object])

    # test of constant with constant
    c2 = Constant(torch.tensor([[2, 4, 3, 2], [4, 3, 2, 8]]))
    proc_objs, vars, n_individuals_per_var = process_ltn_objects([c1, c2])
    assert vars == [], "There should not be variables since the function has been called with two constants."
    assert n_individuals_per_var == [], "Since there are not variables, n_individuals_per_var should be an empty list."
    assert torch.equal(proc_objs[0].value, torch.tensor([[1, 2, 3, 4]])), "The constant should remain untouched."
    assert torch.equal(proc_objs[1].value.squeeze_(), torch.tensor([[2, 4, 3, 2], [4, 3, 2, 8]])), "The constant " \
                                                                                                   "should remain " \
                                                                                                   "untouched."
    assert proc_objs[0] is not c1, "A deep copy should have been performed since c1 has not grad_fn."
    assert proc_objs[1] is not c2, "A deep copy should have been performed since c2 has not grad_fn."

    assert proc_objs[0].free_vars == proc_objs[1].free_vars == [], "The two LTN objects should now share " \
                                                                      "the same variables. In this case no variables " \
                                                                   "since they are constants."


    # test of Variable and Constant
    v1 = Variable("x", torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]))  # variable with two individuals

    # test with constant which is not trainable -> deep copy should be performed by the function
    proc_objs, vars, n_individuals_per_var = process_ltn_objects([c1, v1])
    assert vars == ["x"], "The only variable passed to the function is v1, so there should be only its label in vars."
    assert n_individuals_per_var == [2], "Since v1 is the only variable and has only 2 individuals, " \
                                         "n_individuals_per_var should be the list [2]."
    assert torch.equal(proc_objs[0].value, torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])), "Since variable v1 has 2 " \
                                                                                        "individuals, the function " \
                                                                                        "should have expanded the " \
                                                                                        "constant to match the variable"
    assert torch.equal(proc_objs[1].value, torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])), "The variable " \
                                                                                                       "should have " \
                                                                                                       "been left " \
                                                                                                       "untouched"
    assert proc_objs[0] is not c1, "A deep copy should have been performed since c1 has not grad_fn."
    assert proc_objs[1] is not v1, "A deep copy should have been performed since v1 has not grad_fn."

    assert proc_objs[0].free_vars == proc_objs[1].free_vars == ['x'], "The two LTN objects should now share " \
                                                                      "the same variable."

    # same test but with a constant that is now trainable -> no deep copy should be performed by the function
    c1_t = Constant(torch.tensor([1, 2, 3, 4]), trainable=True)
    # put a gradient into c1_t to test a functionality
    c1_t.value = torch.unsqueeze(c1_t.value, 1)
    proc_objs, _, _ = process_ltn_objects([c1_t, v1])

    assert proc_objs[0] is not c1_t, "A deep copy should be performed, since c1 is a constant with grad_fn."
    assert proc_objs[1] is not v1, "A deep copy should be performed since v1 has not grad_fn."

    # same test but with a variable that has a torch operation on it, so it has grad_fn
    v1_ = Variable("x", torch.tensor([[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.]]))
    c1_t = Constant(torch.tensor([1, 2, 3, 4]), trainable=True)
    # put a gradient into c1_t to test a functionality
    c1_t.value = torch.unsqueeze(c1_t.value, 1)
    # put a gradient into v1_ to test a functionality
    v1_.value.requires_grad = True
    v1_.value = torch.unsqueeze(v1_.value, 1)

    proc_objs, _, _ = process_ltn_objects([c1_t, v1_])

    assert proc_objs[0] is not c1_t, "A deep copy should be performed, since c1 is a constant with grad_fn."
    assert proc_objs[1] is v1_, "A deep copy should not be performed since v1_ has grad_fn."

    # test of the function with two variables in input

    # same number of individuals in the two variables
    v2 = Variable("y", torch.tensor([[1, 2], [3, 4]]))
    proc_objs, vars, n_individuals_per_var = process_ltn_objects([v1, v2])

    assert vars == ["x", "y"], "vars should contain the list ['x', 'y'] since v1 has label x and v2 label y."
    assert n_individuals_per_var == [2, 2], "n_individuals_per_var should be [2, 2] since both vars have 2 individuals."

    assert proc_objs[0].free_vars == proc_objs[1].free_vars == ["x", "y"], "The two LTN object should now share " \
                                                                           "the variables."

    assert torch.equal(proc_objs[0].value,
                       torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6],
                                     [7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12]])), "Both individuals of v1 should " \
                                                                                      "have been repeated twice " \
                                                                                      "and in the order specified in " \
                                                                                      "the assert."

    assert torch.equal(proc_objs[1].value, torch.tensor([[1, 2], [3, 4],
                                                         [1, 2], [3, 4]])), "Both individuals of v2 should " \
                                                                                      "have been repeated twice " \
                                                                                      "and in the order specified in " \
                                                                                      "the assert."

    # different number of individuals in the two variables
    v3 = Variable("z", torch.tensor([[1, 2], [3, 4], [5, 6]]))
    proc_objs, vars, n_individuals_per_var = process_ltn_objects([v1, v3])

    assert vars == ["x", "z"], "vars should contain the list ['x', 'z'] since v1 has label x and v3 label z."
    assert n_individuals_per_var == [2, 3], "n_individuals_per_var should be [2, 3] since v1 has 2 " \
                                            "individuals and v2 3 individuals."

    assert proc_objs[0].free_vars == proc_objs[1].free_vars == ["x", "z"], "The two LTN object should now share " \
                                                                           "the variables."

    assert torch.equal(proc_objs[0].value,
                       torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6],
                                     [7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12]])), "Both " \
                                                                                     "individuals of v1 should " \
                                                                                      "have been repeated three times " \
                                                                                      "and in the order specified in " \
                                                                                      "the assert."

    assert torch.equal(proc_objs[1].value, torch.tensor([[1, 2], [3, 4], [5, 6],
                                                         [1, 2], [3, 4], [5, 6]])), "The individuals of v3 should " \
                                                                            "have been repeated twice " \
                                                                            "and in the order specified in " \
                                                                            "the assert."

    # test with diagonal quantification
    v1 = Variable("x", torch.randn((3, 4)))
    v2 = Variable("y", torch.randn((3, 6)))
    v1, v2 = diag(v1, v2)

    proc_objs, vars, n_individuals_per_var = process_ltn_objects([v1, v2])

    assert torch.equal(proc_objs[0].value, v1.value), "The value should be the same since we are in diagonal setting."
    assert torch.equal(proc_objs[1].value, v2.value), "The value should be the same since we are in diagonal setting."
    assert vars == ["diag_x_y"] == v1.free_vars == v2.free_vars, "The free vars of both variables and the vars " \
                                                                 "variable should be the same."
    assert n_individuals_per_var == [3], "Since we are diagonal setting, it is like we have only one variable with 3" \
                                         "individuals."

def test_Predicate():
    torch.manual_seed(2020)
    # create simple models
    class PredicateModel(torch.nn.Module):
        def __init__(self):
            super(PredicateModel, self).__init__()

        def forward(self, x):
            return torch.nn.Sigmoid()(torch.sum(x, dim=1))

    def f(x):
        return torch.nn.Sigmoid()(torch.sum(x, dim=1))

    l = lambda x: torch.nn.Sigmoid()(torch.sum(x, dim=1))

    # the predicate needs at least one construction strategies
    with pytest.raises(ValueError):
        p1 = Predicate()

    # the func parameter wants a function instance
    with pytest.raises(TypeError):
        p1 = Predicate(func=PredicateModel())

    # the model parameter wants a PyTorch model instance
    with pytest.raises(TypeError):
        p1 = Predicate(f)

    # it is not possible to give both construction strategies for the predicate
    with pytest.raises(ValueError):
        p1 = Predicate(PredicateModel(), l)

    # check exception when predicate output is not in the range [0., 1.]

    wrong_predicate = Predicate(func=lambda x: torch.sum(x, dim=1))  # the output will not be in [0., 1.]
    v = Variable("x", torch.randn((4, 7)))

    with pytest.raises(ValueError):
        wrong_predicate(v)

    # unary predicate

    # model

    m = PredicateModel()
    p1 = Predicate(m)
    assert p1.model == m, "The model should be as same as the parameter."
    assert isinstance(p1.model, torch.nn.Module), "The model should be of type torch.nn.Module."
    v = Variable("x", torch.randn((4, 3)))
    t = torch.tensor([1, 2, 3, 4])

    # an LTN predicate takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        p1(t)

    out = p1(v)

    toy_out = torch.nn.Sigmoid()(torch.sum(v.value, dim=1))

    assert torch.equal(out.value.detach(), toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4]), "The output should be a vector of 4 values since the predicate is unary" \
                                           "and it is applied to a variable with 4 individuals."
    assert out.free_vars == ["x"], "The free_vars in the output should be the free_vars contained in v, which " \
                                   "is only 'x'"
    assert isinstance(out, LTNObject), "The output of an LTN predicate should always be an LTN object."

    # lambda func

    p1 = Predicate(func=l)
    assert isinstance(p1.model, LambdaModel), "The model should be of type LambdaModel."

    # an LTN predicate takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        p1(t)

    out = p1(v)

    assert torch.equal(out.value.detach(), toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4]), "The output should be a vector of 4 values since the predicate is unary" \
                                           "and it is applied to a variable with 4 individuals."
    assert out.free_vars == ["x"], "The free_vars in the output should be the free_vars contained in v, which " \
                                   "is only 'x'"
    assert isinstance(out, LTNObject), "The output of an LTN predicate should always be an LTN object."

    # normal func

    p1 = Predicate(func=f)
    assert isinstance(p1.model, LambdaModel), "The model should be of type LambdaModel."

    # an LTN predicate takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        p1(t)

    out = p1(v)

    assert torch.equal(out.value.detach(), toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4]), "The output should be a vector of 4 values since the predicate is unary" \
                                           "and it is applied to a variable with 4 individuals."
    assert out.free_vars == ["x"], "The free_vars in the output should be the free_vars contained in v, which " \
                                   "is only 'x'"
    assert isinstance(out, LTNObject), "The output of an LTN predicate should always be an LTN object."

    # application of predicate to a constant

    class PredicateModelConst(torch.nn.Module):
        def __init__(self):
            super(PredicateModelConst, self).__init__()

        def forward(self, x):
            return torch.nn.Sigmoid()(torch.sum(x))

    m = PredicateModelConst()
    c = Constant(torch.randn((3, 2)))
    p1 = Predicate(m)
    assert p1.model == m, "The model should be as same as the parameter."
    assert isinstance(p1.model, torch.nn.Module), "The model should be of type torch.nn.Module."

    out = p1(c)

    toy_out = torch.nn.Sigmoid()(torch.sum(c.value))

    assert torch.equal(out.value.detach(), toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([]), "Since the input is a constant, the output of the predicate should be " \
                                           "a scalar."
    assert out.free_vars == [], "The predicate has been applied to a constant, which has not free variables, so " \
                                "free_vars should be empty."
    assert isinstance(out, LTNObject), "The output of an LTN predicate should always be an LTN object."

    # binary predicate

    # create simple models
    class PredicateModelBinary(torch.nn.Module):
        def __init__(self):
            super(PredicateModelBinary, self).__init__()

        def forward(self, x, y):
            return torch.nn.Sigmoid()(torch.sum(torch.cat([x, y], dim=1), dim=1))

    # model

    # variable with variable

    m = PredicateModelBinary()
    p1 = Predicate(m)
    assert p1.model == m, "The model should be as same as the parameter."
    assert isinstance(p1.model, torch.nn.Module), "The model should be of type torch.nn.Module."
    v1 = Variable("x", torch.randn((4, 3)))
    v2 = Variable("y", torch.randn((5, 7)))
    t = torch.tensor([1, 2, 3, 4])

    # an LTN predicate takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        p1(t, t)

    out = p1(v1, v2)

    proc_objs, _, _ = process_ltn_objects([v1, v2])

    toy_out = torch.nn.Sigmoid()(torch.sum(torch.cat([proc_objs[0].value,
                                                      proc_objs[1].value], dim=1), dim=1)).view(v1.value.shape[0],
                                                                                                v2.value.shape[0])

    assert torch.equal(out.value.detach(), toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4, 5]), "The output should be a matrix of 4x5 values since the predicate " \
                                              "is binary and it is applied to a variable with 4 individuals and " \
                                              "one with 5 individuals."
    assert out.free_vars == ["x", "y"], "The free_vars in the output should be the free_vars contained in v1 and " \
                                        "v2, which are 'x' for v1 and 'y' for v2."
    assert isinstance(out, LTNObject), "The output of an LTN predicate should always be an LTN object."

    # variable with constant

    class PredicateModelBinaryC(torch.nn.Module):
        def __init__(self):
            super(PredicateModelBinaryC, self).__init__()

        def forward(self, x, y):
            y = torch.flatten(y, start_dim=1)
            return torch.nn.Sigmoid()(torch.sum(torch.cat([x, y], dim=1), dim=1))

    m = PredicateModelBinaryC()
    p1 = Predicate(m)
    assert p1.model == m, "The model should be as same as the parameter."
    assert isinstance(p1.model, torch.nn.Module), "The model should be of type torch.nn.Module."
    v1 = Variable("x", torch.randn((4, 3)))
    c = Constant(torch.randn((5, 7)))
    t = torch.tensor([1, 2, 3, 4])

    # an LTN predicate takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        p1(t, t)

    out = p1(v1, c)

    proc_objs, _, _ = process_ltn_objects([v1, c])

    toy_out = torch.nn.Sigmoid()(torch.sum(torch.cat([proc_objs[0].value,
                                                      torch.flatten(proc_objs[1].value, start_dim=1)],
                                                     dim=1), dim=1))

    assert torch.equal(out.value.detach(), toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4]), "The output should be a vector of 4 values since the predicate " \
                                              "is binary and it is applied to a variable with 4 individuals and " \
                                              "a constant."
    assert out.free_vars == ["x"], "The free_vars in the output should be the free_vars contained in v1, " \
                                        "which is only 'x'."
    assert isinstance(out, LTNObject), "The output of an LTN predicate should always be an LTN object."

    # lambda func

    l = lambda x, y: torch.nn.Sigmoid()(torch.sum(torch.cat([x, y], dim=1), dim=1))

    # variable with variable

    p1 = Predicate(func=l)
    assert isinstance(p1.model, LambdaModel), "The model should be of type LambdaModel."
    v1 = Variable("x", torch.randn((4, 3)))
    v2 = Variable("y", torch.randn((5, 7)))
    t = torch.tensor([1, 2, 3, 4])

    # an LTN predicate takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        p1(t, t)

    out = p1(v1, v2)

    proc_objs, _, _ = process_ltn_objects([v1, v2])

    toy_out = torch.nn.Sigmoid()(torch.sum(torch.cat([proc_objs[0].value,
                                                      proc_objs[1].value], dim=1), dim=1)).view(v1.value.shape[0],
                                                                                                v2.value.shape[0])

    assert torch.equal(out.value.detach(), toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4, 5]), "The output should be a matrix of 4x5 values since the predicate " \
                                              "is binary and it is applied to a variable with 4 individuals and " \
                                              "one with 5 individuals."
    assert out.free_vars == ["x", "y"], "The free_vars in the output should be the free_vars contained in v1 and " \
                                        "v2, which are 'x' for v1 and 'y' for v2."
    assert isinstance(out, LTNObject), "The output of an LTN predicate should always be an LTN object."

    # variable with constant

    l = lambda x, y: torch.nn.Sigmoid()(torch.sum(torch.cat([x, torch.flatten(y, start_dim=1)], dim=1), dim=1))
    p1 = Predicate(func=l)
    assert isinstance(p1.model, LambdaModel), "The model should be of type LamdaModel."
    v1 = Variable("x", torch.randn((4, 3)))
    c = Constant(torch.randn((5, 7)))
    t = torch.tensor([1, 2, 3, 4])

    # an LTN predicate takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        p1(t, t)

    out = p1(v1, c)

    proc_objs, _, _ = process_ltn_objects([v1, c])

    toy_out = torch.nn.Sigmoid()(torch.sum(torch.cat([proc_objs[0].value,
                                                      torch.flatten(proc_objs[1].value, start_dim=1)],
                                                     dim=1), dim=1))

    assert torch.equal(out.value.detach(), toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4]), "The output should be a vector of 4 values since the predicate " \
                                           "is binary and it is applied to a variable with 4 individuals and " \
                                           "a constant."
    assert out.free_vars == ["x"], "The free_vars in the output should be the free_vars contained in v1, " \
                                   "which is only 'x'."
    assert isinstance(out, LTNObject), "The output of an LTN predicate should always be an LTN object."

    # normal func

    def f(x, y):
        return torch.nn.Sigmoid()(torch.sum(torch.cat([x, y], dim=1), dim=1))

    # variable with variable

    p1 = Predicate(func=f)
    assert isinstance(p1.model, LambdaModel), "The model should be of type LambdaModel."
    v1 = Variable("x", torch.randn((4, 3)))
    v2 = Variable("y", torch.randn((5, 7)))
    t = torch.tensor([1, 2, 3, 4])

    # an LTN predicate takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        p1(t, t)

    out = p1(v1, v2)

    proc_objs, _, _ = process_ltn_objects([v1, v2])

    toy_out = torch.nn.Sigmoid()(torch.sum(torch.cat([proc_objs[0].value,
                                                      proc_objs[1].value], dim=1), dim=1)).view(v1.value.shape[0],
                                                                                                v2.value.shape[0])

    assert torch.equal(out.value.detach(), toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4, 5]), "The output should be a matrix of 4x5 values since the predicate " \
                                              "is binary and it is applied to a variable with 4 individuals and " \
                                              "one with 5 individuals."
    assert out.free_vars == ["x", "y"], "The free_vars in the output should be the free_vars contained in v1 and " \
                                        "v2, which are 'x' for v1 and 'y' for v2."
    assert isinstance(out, LTNObject), "The output of an LTN predicate should always be an LTN object."

    # variable with constant

    def f(x, y):
        return torch.nn.Sigmoid()(torch.sum(torch.cat([x, torch.flatten(y, start_dim=1)], dim=1), dim=1))

    p1 = Predicate(func=f)
    assert isinstance(p1.model, LambdaModel), "The model should be of type LamdaModel."
    v1 = Variable("x", torch.randn((4, 3)))
    c = Constant(torch.randn((5, 7)))
    t = torch.tensor([1, 2, 3, 4])

    # an LTN predicate takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        p1(t, t)

    out = p1(v1, c)

    proc_objs, _, _ = process_ltn_objects([v1, c])

    toy_out = torch.nn.Sigmoid()(torch.sum(torch.cat([proc_objs[0].value,
                                                      torch.flatten(proc_objs[1].value, start_dim=1)],
                                                     dim=1), dim=1))

    assert torch.equal(out.value.detach(), toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4]), "The output should be a vector of 4 values since the predicate " \
                                           "is binary and it is applied to a variable with 4 individuals and " \
                                           "a constant."
    assert out.free_vars == ["x"], "The free_vars in the output should be the free_vars contained in v1, " \
                                   "which is only 'x'."
    assert isinstance(out, LTNObject), "The output of an LTN predicate should always be an LTN object."


def test_Function():
    torch.manual_seed(2020)
    # the function changes the values of the features of each individual by adding two new ones
    # by doing so, we test the fact that the domain in input is different from the domain in output

    # create function model
    class FunctionModel(torch.nn.Module):
        def __init__(self):
            super(FunctionModel, self).__init__()

        def forward(self, x):
            t = torch.ones((x.shape[0], 2))
            return torch.cat([x, t], dim=1)

    def f(x):
        t = torch.ones((x.shape[0], 2))
        return torch.cat([x, t], dim=1)

    l = lambda x: torch.cat([x, torch.ones((x.shape[0], 2))], dim=1)

    # check assertions

    # one strategy for creating the function has to be given
    with pytest.raises(ValueError):
        f1 = Function()

    # the function parameter wants a function instance
    with pytest.raises(TypeError):
        f1 = Function(func=FunctionModel())

    # the model parameter wants a PyTorch model instance
    with pytest.raises(TypeError):
        f1 = Function(f)

    # it is not possible to give both construction strategies
    with pytest.raises(ValueError):
        f1 = Function(FunctionModel(), l)

    # unary function

    # model

    m = FunctionModel()
    f1 = Function(m)
    assert f1.model == m, "The model should be as same as the parameter."
    assert isinstance(f1.model, torch.nn.Module), "The model should be of type torch.nn.Module."
    v = Variable("x", torch.randn((4, 3)))
    t = torch.tensor([1, 2, 3, 4])

    # an LTN function takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        f1(t)

    out = f1(v)

    toy_out = torch.cat([v.value, torch.ones((v.shape()[0], 2))], dim=1)

    assert torch.equal(out.value.detach(),
                       toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4, 5]), "The output should be a matrix of 4x5 values since the function " \
                                           "has been applied to a variable with 4 individuals with 3 features, and" \
                                              "2 features have been added to each of them."
    assert out.free_vars == ["x"], "The free_vars in the output should be the free_vars contained in v, which " \
                                   "is only 'x'"
    assert isinstance(out, LTNObject), "The output of an LTN function should always be an LTN object."

    # lambda func

    f1 = Function(func=l)
    assert isinstance(f1.model, LambdaModel), "The model should be of type LambdaModel."

    # an LTN function takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        f1(t)

    out = f1(v)

    assert torch.equal(out.value.detach(),
                       toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4, 5]), "The output should be a matrix of 4x5 values since the function " \
                                              "has been applied to a variable with 4 individuals with 3 features, and" \
                                              "2 features have been added to each of them."
    assert out.free_vars == ["x"], "The free_vars in the output should be the free_vars contained in v, which " \
                                   "is only 'x'"
    assert isinstance(out, LTNObject), "The output of an LTN function should always be an LTN object."

    # normal func

    f1 = Function(func=f)
    assert isinstance(f1.model, LambdaModel), "The model should be of type LambdaModel."

    # an LTN function takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        f1(t)

    out = f1(v)

    assert torch.equal(out.value.detach(),
                       toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4, 5]), "The output should be a matrix of 4x5 values since the function " \
                                              "has been applied to a variable with 4 individuals with 3 features, and" \
                                              "2 features have been added to each of them."
    assert out.free_vars == ["x"], "The free_vars in the output should be the free_vars contained in v, which " \
                                   "is only 'x'"
    assert isinstance(out, LTNObject), "The output of an LTN function should always be an LTN object."

    # different function with only one scalar for each individual in input

    f1 = Function(func=lambda x: torch.mean(x, dim=1))

    assert isinstance(f1.model, LambdaModel), "The model should be of type LambdaModel."

    # an LTN function takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        f1(t)

    out = f1(v)

    toy_out = torch.mean(v.value, dim=1)

    assert torch.equal(out.value.detach(),
                       toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4]), "The output should be a vector of 4 values since the function " \
                                              "has been applied to a variable with 4 individuals and needs to perform" \
                                              "the mean of their features."
    assert out.free_vars == ["x"], "The free_vars in the output should be the free_vars contained in v, which " \
                                   "is only 'x'"
    assert isinstance(out, LTNObject), "The output of an LTN function should always be an LTN object."

    # application of predicate to a constant

    class FunctionModelConst(torch.nn.Module):
        def __init__(self):
            super(FunctionModelConst, self).__init__()

        def forward(self, x):
            return torch.mean(x)

    m = FunctionModelConst()
    c = Constant(torch.randn((3, 2)))
    f1 = Function(m)
    assert f1.model == m, "The model should be as same as the parameter."
    assert isinstance(f1.model, torch.nn.Module), "The model should be of type torch.nn.Module."

    out = f1(c)

    toy_out = torch.mean(c.value)

    assert torch.equal(out.value.detach(),
                       toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([]), "Since the input is a constant, the output of the predicate should be " \
                                          "a scalar."
    assert out.free_vars == [], "The predicate has been applied to a constant, which has not free variables, so " \
                                "free_vars should be empty."
    assert isinstance(out, LTNObject), "The output of an LTN function should always be an LTN object."

    # binary function

    # create simple models
    # this function perform the concatenation of the inputs
    class FunctionModelBinary(torch.nn.Module):
        def __init__(self):
            super(FunctionModelBinary, self).__init__()

        def forward(self, x, y):
            return torch.cat([x, y], dim=1)

    # model

    # variable with variable

    m = FunctionModelBinary()
    f1 = Function(m)
    assert f1.model == m, "The model should be as same as the parameter."
    assert isinstance(f1.model, torch.nn.Module), "The model should be of type torch.nn.Module."
    v1 = Variable("x", torch.randn((4, 3)))
    v2 = Variable("y", torch.randn((5, 7)))
    t = torch.tensor([1, 2, 3, 4])

    # an LTN function takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        f1(t, t)

    out = f1(v1, v2)

    proc_objs, _, _ = process_ltn_objects([v1, v2])

    toy_out = torch.cat([proc_objs[0].value, proc_objs[1].value], dim=1).view(v1.value.shape[0], v2.value.shape[0], 10)

    assert torch.equal(out.value.detach(),
                       toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4, 5, 10]), "The output should be a tensor of 4x5x10 values since the function " \
                                              "is binary and it is applied to a variable with 4 individuals and " \
                                              "one with 5 individuals. Finally, it concatenates their features, " \
                                                  "which are 3 and 7, so the final dimension should be 10."
    assert out.free_vars == ["x", "y"], "The free_vars in the output should be the free_vars contained in v1 and " \
                                        "v2, which are 'x' for v1 and 'y' for v2."
    assert isinstance(out, LTNObject), "The output of an LTN function should always be an LTN object."

    # variable with constant

    class FunctionModelBinaryC(torch.nn.Module):
        def __init__(self):
            super(FunctionModelBinaryC, self).__init__()

        def forward(self, x, y):
            y = torch.flatten(y, start_dim=1)
            return torch.cat([x, y], dim=1)

    m = FunctionModelBinaryC()
    f1 = Function(m)
    assert f1.model == m, "The model should be as same as the parameter."
    assert isinstance(f1.model, torch.nn.Module), "The model should be of type torch.nn.Module."
    v1 = Variable("x", torch.randn((4, 3)))
    c = Constant(torch.randn((5, 7)))
    t = torch.tensor([1, 2, 3, 4])

    # an LTN function takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        f1(t, t)

    out = f1(v1, c)

    proc_objs, _, _ = process_ltn_objects([v1, c])

    toy_out = torch.cat([proc_objs[0].value, torch.flatten(proc_objs[1].value, start_dim=1)], dim=1)

    assert torch.equal(out.value.detach(),
                       toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4, 38]), "The output should a matrix of 4x38 values since the function " \
                                           "is binary and it is applied to a variable with 4 individuals and " \
                                           "a constant. The variable has 3 features for each individual, and the" \
                                               "flatten shape of the constant is 35. 35 + 3 = 38."
    assert out.free_vars == ["x"], "The free_vars in the output should be the free_vars contained in v1, " \
                                   "which is only 'x'."
    assert isinstance(out, LTNObject), "The output of an LTN function should always be an LTN object."

    # lambda func

    l = lambda x, y: torch.cat([x, y], dim=1)

    # variable with variable

    f1 = Function(func=l)
    assert isinstance(f1.model, LambdaModel), "The model should be of type LambdaModel."
    v1 = Variable("x", torch.randn((4, 3)))
    v2 = Variable("y", torch.randn((5, 7)))
    t = torch.tensor([1, 2, 3, 4])

    # an LTN function takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        f1(t, t)

    out = f1(v1, v2)

    proc_objs, _, _ = process_ltn_objects([v1, v2])

    toy_out = torch.cat([proc_objs[0].value, proc_objs[1].value], dim=1).view(4, 5, 10)

    assert torch.equal(out.value.detach(),
                       toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4, 5, 10]), "The output should be a tensor of 4x5x10 values since the function " \
                                              "is binary and it is applied to a variable with 4 individuals and " \
                                              "one with 5 individuals. It concatenates the features that are 3 + 7."
    assert out.free_vars == ["x", "y"], "The free_vars in the output should be the free_vars contained in v1 and " \
                                        "v2, which are 'x' for v1 and 'y' for v2."
    assert isinstance(out, LTNObject), "The output of an LTN function should always be an LTN object."

    # variable with constant

    l = lambda x, y: torch.cat([x, torch.flatten(y, start_dim=1)], dim=1)
    f1 = Function(func=l)
    assert isinstance(f1.model, LambdaModel), "The model should be of type LamdaModel."
    v1 = Variable("x", torch.randn((4, 3)))
    c = Constant(torch.randn((5, 7)))
    t = torch.tensor([1, 2, 3, 4])

    # an LTN function takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        f1(t, t)

    out = f1(v1, c)

    proc_objs, _, _ = process_ltn_objects([v1, c])

    toy_out = torch.cat([proc_objs[0].value, torch.flatten(proc_objs[1].value, start_dim=1)], dim=1)

    assert torch.equal(out.value.detach(),
                       toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4, 38]), "The output should be a matrix of 4x38 values since the function " \
                                           "is binary and it is applied to a variable with 4 individuals and " \
                                           "a constant that has 35 flatten features. 35 + 3 (features of v1) = 38."
    assert out.free_vars == ["x"], "The free_vars in the output should be the free_vars contained in v1, " \
                                   "which is only 'x'."
    assert isinstance(out, LTNObject), "The output of an LTN function should always be an LTN object."

    # normal func

    def f(x, y):
        return torch.cat([x, y], dim=1)

    # variable with variable

    f1 = Function(func=f)
    assert isinstance(f1.model, LambdaModel), "The model should be of type LambdaModel."
    v1 = Variable("x", torch.randn((4, 3)))
    v2 = Variable("y", torch.randn((5, 7)))
    t = torch.tensor([1, 2, 3, 4])

    # an LTN function takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        f1(t, t)

    out = f1(v1, v2)

    proc_objs, _, _ = process_ltn_objects([v1, v2])

    toy_out = torch.cat([proc_objs[0].value, proc_objs[1].value], dim=1).view(4, 5, 10)

    assert torch.equal(out.value.detach(),
                       toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4, 5, 10]), "The output should be a tensor of 4x5x10 values since the function " \
                                                  "is binary and it is applied to a variable with 4 individuals and " \
                                                  "one with 5 individuals. It concatenates the features that are 3 + 7."
    assert out.free_vars == ["x", "y"], "The free_vars in the output should be the free_vars contained in v1 and " \
                                        "v2, which are 'x' for v1 and 'y' for v2."
    assert isinstance(out, LTNObject), "The output of an LTN function should always be an LTN object."

    # variable with constant

    def f(x, y):
        return torch.cat([x, torch.flatten(y, start_dim=1)], dim=1)

    f1 = Function(func=f)
    assert isinstance(f1.model, LambdaModel), "The model should be of type LamdaModel."
    v1 = Variable("x", torch.randn((4, 3)))
    c = Constant(torch.randn((5, 7)))
    t = torch.tensor([1, 2, 3, 4])

    # an LTN function takes as input a list of LTNObjects and not just PyTorch tensors
    with pytest.raises(TypeError):
        f1(t, t)

    out = f1(v1, c)

    proc_objs, _, _ = process_ltn_objects([v1, c])

    toy_out = torch.cat([proc_objs[0].value, torch.flatten(proc_objs[1].value, start_dim=1)], dim=1)

    assert torch.equal(out.value.detach(),
                       toy_out), "Since seed has been set, the result should be always this one."

    assert out.shape() == torch.Size([4, 38]), "The output should be a matrix of 4x38 values since the function " \
                                               "is binary and it is applied to a variable with 4 individuals and " \
                                               "a constant that has 35 flatten features. 35 + 3 (features of v1) = 38."
    assert out.free_vars == ["x"], "The free_vars in the output should be the free_vars contained in v1, " \
                                   "which is only 'x'."
    assert isinstance(out, LTNObject), "The output of an LTN function should always be an LTN object."


def test_LambdaModel():
    m = LambdaModel(lambda x: torch.sum(x))
    t = torch.tensor([1, 2, 3, 4, 5])

    assert torch.sum(t) == m(t), "The result should be the same if the LambdaModel is well implemented."


def test_diag():
    torch.manual_seed(2020)
    # check if exceptions are raised
    v1 = Variable('v1', torch.tensor([[1., 2.], [3., 4.]]))  # 2 individuals
    v2 = Variable('v2', torch.tensor([[5., 6.], [7., 8.]]))  # 2 individuals
    v3 = Variable('v3', torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))  # 3 individuals
    v4 = Variable('v4', torch.tensor([[5., 6., 7.], [7., 8., 9.]]))  # 2 individuals but three features
    c = Constant(torch.tensor([1., 2., 3., 4.]))  # LTN constant

    # diagonal quantification only accepts variables
    with pytest.raises(TypeError):
        diag(v1, c)

    # diagonal quantification only accepts a list of more than one variable
    with pytest.raises(ValueError):
        diag(v1)

    # diagonal quantification requires variables with the same number of individuals
    with pytest.raises(ValueError):
        diag(v1, v3)

    v1, v2, v4 = diag(v1, v2, v4)

    assert v1.free_vars == v2.free_vars == v4.free_vars == ["diag_v1_v2_v4"]

    # test diagonal quantification behavior on function

    function = Function(func=lambda x, y, z: torch.sum(torch.cat([x, y, z], dim=1), dim=1))

    out = function(v1, v2, v4)
    toy_out = torch.sum(torch.cat([v1.value, v2.value, v4.value], dim=1), dim=1)

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert out.free_vars == ["diag_v1_v2_v4"] == v1.free_vars == v2.free_vars == v4.free_vars, "Since we are in " \
                                                                                               "diagonal setting, the " \
                                                                                               "only free var is " \
                                                                                               "this one."
    assert out.shape() == torch.Size([2]), "Since there are only 2 individuals per variable and we are in diagonal" \
                                           "setting, the size should be 2."

    # now undiag and check again

    v1, v2, v4 = undiag(v1, v2, v4)

    out = function(v1, v2, v4)
    proc_obs, _, _ = process_ltn_objects([v1, v2, v4])
    toy_out = torch.sum(torch.cat([proc_obs[0].value, proc_obs[1].value, proc_obs[2].value], dim=1), dim=1)\
        .view(2, 2, 2)

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert out.free_vars == ["v1", "v2", "v4"], "Since we are not in diagonal setting, all the variables are free."
    assert out.shape() == torch.Size([2, 2, 2]), "Since there are only 2 individuals per variable and we are not in " \
                                                 "diagonal setting, the shape must be 2x2x2."

    # test diagonal quantification on predicate

    predicate = Predicate(func=lambda x, y, z: torch.nn.Sigmoid()(torch.sum(torch.cat([x, y, z], dim=1), dim=1)))

    v1 = Variable("x", torch.randn((3, 4)))
    v2 = Variable("y", torch.randn((3, 6)))
    v3 = Variable("z", torch.randn((3, 10)))

    v1, v2, v3 = diag(v1, v2, v3)

    out = predicate(v1, v2, v3)
    toy_out = torch.nn.Sigmoid()(torch.sum(torch.cat([v1.value, v2.value, v3.value], dim=1), dim=1))

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert out.free_vars == ["diag_x_y_z"] == v1.free_vars == v2.free_vars == v3.free_vars, "Since we are in " \
                                                                                               "diagonal setting, the " \
                                                                                               "only free var is " \
                                                                                               "this one."
    assert out.shape() == torch.Size([3]), "Since there are only 3 individuals per variable and we are in diagonal" \
                                           "setting, the size should be 3."

    # now undiag and check again

    v1, v2, v3 = undiag(v1, v2, v3)

    out = predicate(v1, v2, v3)
    proc_obs, _, _ = process_ltn_objects([v1, v2, v3])
    toy_out = torch.nn.Sigmoid()(torch.sum(torch.cat([proc_obs[0].value, proc_obs[1].value, proc_obs[2].value], dim=1),
                                           dim=1).view(3, 3, 3))

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert out.free_vars == ["x", "y", "z"], "Since we are not in diagonal setting, all the variables are free."
    assert out.shape() == torch.Size([3, 3, 3]), "Since there are only 3 individuals per variable and we are not in " \
                                                 "diagonal setting, the shape must be 3x3x3."


def test_undiag():
    # check if exceptions are raised
    v1 = Variable('v1', torch.tensor([[1, 2], [3, 4]]))  # 2 individuals
    v2 = Variable('v2', torch.tensor([[5, 6], [7, 8]]))  # 2 individuals
    v4 = Variable('v4', torch.tensor([[5, 6, 7], [7, 8, 9]]))  # 2 individuals but three features
    c = Constant(torch.tensor([1, 2, 3, 5]))

    # undiagonal quantification only accepts LTN variables
    with pytest.raises(TypeError):
        undiag(v1, c)

    # undiag of a single variable without diag
    v1 = undiag(v1)[0]

    assert v1.free_vars == ['v1'], "free vars should be kept untouched and should be " \
                                                    "equal to latent_var, which is v1."

    v1, v2, v4 = diag(v1, v2, v4)
    assert v1.free_vars == v2.free_vars == v4.free_vars == ["diag_v1_v2_v4"]
    v1, v2, v4 = undiag(v1, v2, v4)

    assert v1.free_vars == ['v1'], "free vars should be kept untouched and should be " \
                                                    "equal to latent_var, which is v1"
    assert v2.free_vars == ['v2'], "free vars should be kept untouched and should be " \
                                                    "equal to latent_var, which is v2"
    assert v4.free_vars == ['v4'], "free vars should be kept untouched and should be " \
                                                    "equal to latent_var, which is v4"

def test_Connective():
    torch.manual_seed(2020)

    # check implementation exceptions

    # ConnectiveOperator
    with pytest.raises(NotImplementedError):
        class StrangeConn(ltn.fuzzy_ops.ConnectiveOperator):
            def __init__(self):
                pass

        conn = StrangeConn()
        conn()

    # unary
    with pytest.raises(NotImplementedError):
        class StrangeConn(ltn.fuzzy_ops.UnaryConnectiveOperator):
            def __init__(self):
                pass

        conn = StrangeConn()
        conn()

    # binary
    with pytest.raises(NotImplementedError):
        class StrangeConn(ltn.fuzzy_ops.BinaryConnectiveOperator):
            def __init__(self):
                pass

        conn = StrangeConn()
        conn()

    # here, we test all the connectives available in LTN
    # check TypeError during construction of the operator
    agg = ltn.fuzzy_ops.AggregMean()

    # it is not possible to construct a Connective with an aggregation operator, only connective operators are allowed
    with pytest.raises(TypeError):
        Connective(agg)

    # create fake LTN objects to test the operands
    # we need a pair of LTN objects with different variables
    # in this case we test the call to the process_ltn_objects function
    op1 = LTNObject(torch.rand((3, 4)), ["x", "y"])
    op2 = LTNObject(torch.rand((5, 6)), ["u", "z"])
    op3 = torch.tensor([1, 2, 3, 4])

    # test behavior with a simple and intuitive connective, for example the AndMin
    not_s = Connective(ltn.fuzzy_ops.NotStandard())
    and_min = Connective(ltn.fuzzy_ops.AndMin())

    # no more than one operand is allowed for unary connectives
    with pytest.raises(ValueError):
        not_s(op1, op2)

    # at least one operand has to be passed to the connective
    with pytest.raises(ValueError):
        not_s()

    # at least one operand has to be passed to the connective
    with pytest.raises(ValueError):
        and_min()

    # maximum two operands have to be passed to the connective
    with pytest.raises(ValueError):
        and_min(op1, op2, op3)

    # LTN connectives accept only LTN objects and not just tensors
    with pytest.raises(TypeError):
        and_min(op1, op3)

    # the LTNObjects in input must contains values in [0., 1.]
    with pytest.raises(ValueError):
        and_min(op1, LTNObject(torch.randn((3, 3)), ["x"]))


    proc_objs, _, _ = process_ltn_objects([op1, op2])

    # torch minimum is how the AndMin connective is implemented internally
    toy_out = torch.minimum(proc_objs[0].value, proc_objs[1].value)

    out = and_min(op1, op2)

    assert torch.equal(torch.flatten(out.value), toy_out), "The output should be the same if the operator has done" \
                                                           " the things correctly."

    assert out.free_vars == ["x", "y", "u", "z"], "The free variables in the output should be x, y, u and z, since " \
                                                  "op1 has variables x and y, while op2 variables u and z."

    assert out.shape() == torch.Size([3, 4, 5, 6]), "The output should be a tensor with " \
                                                    "dimensions associated to the free variables involved " \
                                                    "in the connective operation."

    # definition of all other connectives and simple test of them
    not_standard = Connective(ltn.fuzzy_ops.NotStandard())
    not_godel = Connective(ltn.fuzzy_ops.NotGodel())
    and_prod = Connective(ltn.fuzzy_ops.AndProd())
    and_prod_not_stable = Connective(ltn.fuzzy_ops.AndProd(stable=False))
    and_luk = Connective(ltn.fuzzy_ops.AndLuk())
    or_max = Connective(ltn.fuzzy_ops.OrMax())
    or_prob = Connective(ltn.fuzzy_ops.OrProbSum())
    or_prob_not_stable = Connective(ltn.fuzzy_ops.OrProbSum(stable=False))
    or_luk = Connective(ltn.fuzzy_ops.OrLuk())
    i_kd = Connective(ltn.fuzzy_ops.ImpliesKleeneDienes())
    i_godel = Connective(ltn.fuzzy_ops.ImpliesGodel())
    i_r = Connective(ltn.fuzzy_ops.ImpliesReichenbach())
    i_r_not_stable = Connective(ltn.fuzzy_ops.ImpliesReichenbach(stable=False))
    i_gougen = Connective(ltn.fuzzy_ops.ImpliesGoguen())
    i_gougen_not_stable = Connective(ltn.fuzzy_ops.ImpliesGoguen(stable=False))
    i_luk = Connective(ltn.fuzzy_ops.ImpliesLuk())
    equiv = Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))

    # we test the connective with simple LTN objects based on the same variables, for simplicity
    # the case with different variables has already been tested above

    op1 = LTNObject(torch.rand((5,)), ["x"])
    op2 = LTNObject(torch.rand((5,)), ["x"])
    wrong_values = LTNObject(torch.randn((5,)), ["x"])

    # not standard

    toy_out = 1. - op1.value
    out = not_standard(op1)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # check exception with wrong input
    with pytest.raises(ValueError):
        not_standard(wrong_values)

    # not godel

    toy_out = torch.eq(op1.value, 0.).float()
    out = not_godel(op1)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # check exception with wrong input
    with pytest.raises(ValueError):
        not_godel(wrong_values)

    # and min

    toy_out = torch.minimum(op1.value, op2.value)
    out = and_min(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # check exception with wrong input
    with pytest.raises(ValueError):
        and_min(op1, wrong_values)

    # and prod - no stable

    toy_out = torch.mul(op1.value, op2.value)
    out = and_prod_not_stable(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # check exception with wrong input
    with pytest.raises(ValueError):
        and_prod_not_stable(op1, wrong_values)

    # and prod - stable

    toy_out = torch.mul(ltn.fuzzy_ops.pi_0(op1.value), ltn.fuzzy_ops.pi_0(op2.value))
    out = and_prod(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # and lukasiewicz

    toy_out = torch.maximum(op1.value + op2.value - 1., torch.zeros_like(op1.value))
    out = and_luk(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # check exception with wrong input
    with pytest.raises(ValueError):
        and_luk(op1, wrong_values)

    # or max

    toy_out = torch.maximum(op1.value, op2.value)
    out = or_max(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # check exception with wrong input
    with pytest.raises(ValueError):
        or_max(op1, wrong_values)

    # or prob sum - not stable

    toy_out = op1.value + op2.value - torch.mul(op1.value, op2.value)
    out = or_prob_not_stable(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # check exception with wrong input
    with pytest.raises(ValueError):
        or_prob_not_stable(op1, wrong_values)

    # or prob sum - stable

    toy_out = ltn.fuzzy_ops.pi_1(op1.value) + ltn.fuzzy_ops.pi_1(op2.value) - \
              torch.mul(ltn.fuzzy_ops.pi_1(op1.value), ltn.fuzzy_ops.pi_1(op2.value))
    out = or_prob(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # or luk

    toy_out = torch.minimum(op1.value + op2.value, torch.ones_like(op1.value))
    out = or_luk(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # check exception with wrong input
    with pytest.raises(ValueError):
        or_luk(op1, wrong_values)

    # implies kleene dienes

    toy_out = torch.maximum(1. - op1.value, op2.value)
    out = i_kd(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # check exception with wrong input
    with pytest.raises(ValueError):
        i_kd(op1, wrong_values)

    # implies godel

    toy_out = torch.where(torch.le(op1.value, op2.value), torch.ones_like(op1.value), op2.value)
    out = i_godel(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # check exception with wrong input
    with pytest.raises(ValueError):
        i_godel(op1, wrong_values)

    # implies Reichenbach - not stable

    toy_out = 1. - op1.value + torch.mul(op1.value, op2.value)
    out = i_r_not_stable(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # check exception with wrong input
    with pytest.raises(ValueError):
        i_r_not_stable(op1, wrong_values)

    # implies Reichenbach - stable

    toy_out = 1. - ltn.fuzzy_ops.pi_0(op1.value) + torch.mul(ltn.fuzzy_ops.pi_0(op1.value),
                                                             ltn.fuzzy_ops.pi_1(op2.value))
    out = i_r(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # implies goguen - not stable

    toy_out = torch.where(torch.le(op1.value, op2.value), torch.ones_like(op1.value), torch.div(op2.value, op1.value))
    out = i_gougen_not_stable(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # check exception with wrong input
    with pytest.raises(ValueError):
        i_gougen_not_stable(op1, wrong_values)

    # implies goguen - stable

    toy_out = torch.where(torch.le(ltn.fuzzy_ops.pi_0(op1.value), op2.value),
                          torch.ones_like(ltn.fuzzy_ops.pi_0(op1.value)),
                          torch.div(op2.value, ltn.fuzzy_ops.pi_0(op1.value)))
    out = i_gougen(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # implies lukasiewicz

    toy_out = torch.minimum(1. - op1.value + op2.value, torch.ones_like(op1.value))
    out = i_luk(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # check exception with wrong input
    with pytest.raises(ValueError):
        i_luk(op1, wrong_values)

    # test of pi functions

    eps = 1e-4

    # pi_0

    x = torch.rand((1,))
    assert torch.equal((1 - eps) * x + eps, ltn.fuzzy_ops.pi_0(x))

    # pi_1

    assert torch.equal((1 - eps) * x, ltn.fuzzy_ops.pi_1(x))

    # equiv

    # equiv has been initialized with and prod and implies goguen strong stable
    toy_out = torch.mul(ltn.fuzzy_ops.pi_0(1. - ltn.fuzzy_ops.pi_0(op1.value) + torch.mul(ltn.fuzzy_ops.pi_0(op1.value),
                                                                                          ltn.fuzzy_ops.pi_1(op2.value))),
                        ltn.fuzzy_ops.pi_0(1. - ltn.fuzzy_ops.pi_0(op2.value) + torch.mul(ltn.fuzzy_ops.pi_0(op2.value),
                                                                                          ltn.fuzzy_ops.pi_1(op1.value))))
    out = equiv(op1, op2)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == ["x"], "The only free variable should be x."
    assert out.shape() == torch.Size([5]), "The shape should be 5."

    # check exception with wrong input
    with pytest.raises(ValueError):
        equiv(op1, wrong_values)

    # test check_values function

    # single operand

    with pytest.raises(ValueError):
        ltn.fuzzy_ops.check_values(torch.randn((3, 3)))

    # two operands

    # the second operand has wrong values since they are not in [0., 1.]
    with pytest.raises(ValueError):
        ltn.fuzzy_ops.check_values(torch.rand((3, 3)), torch.randn((3, 4)))


def test_Quantifier():
    torch.manual_seed(2020)

    # check implementation exceptions

    with pytest.raises(NotImplementedError):
        class StrangeAgg(ltn.fuzzy_ops.AggregationOperator):
            def __init__(self):
                pass

        agg = StrangeAgg()
        agg()

    # check exceptions in quantifier construction
    conn = ltn.fuzzy_ops.OrLuk()
    correct_q = ltn.fuzzy_ops.AggregMean()

    # type error if quantifier is build from a connective
    with pytest.raises(TypeError):
        Quantifier(conn, "e")

    # only 'e' and 'f' are possible keywords to decide the quantifiers
    with pytest.raises(ValueError):
        Quantifier(correct_q, "a")

    # create LTN object to test the formula on
    x = Variable("x", torch.randn((2, 3)))
    y = Variable("y", torch.randn((4, 2)))
    c = Constant(torch.tensor([1, 2, 3, 4]))
    wrong_formula = torch.tensor([1, 2, 3])

    p = Predicate(func=lambda x, y: torch.nn.Sigmoid()(torch.sum(torch.cat([x, y], dim=1), dim=1)))

    exists = Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), "e")
    forall = Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), "f")

    # error if vars parameter do not contain only LTN variables, in this case it contains a variables plus constant
    with pytest.raises(TypeError):
        forall([x, c], p(x, c))

    # error if vars parameter do not contain only LTN variables, in this case it contains a constant
    with pytest.raises(TypeError):
        forall(c, p(x, c))

    # formula must be LTNObject and not tensor
    with pytest.raises(TypeError):
        forall([x, y], wrong_formula)

    # if condition variables are included, a condition has to be set
    with pytest.raises(ValueError):
        forall([x, y], p(x, y), cond_vars=[x, y])

    # if condition is set, condition variables have to be set
    with pytest.raises(ValueError):
        forall([x, y], p(x, y), cond_fn=lambda x, y: torch.logical_and(torch.sum(x.value, dim=1) > 5,
                                                                       torch.sum(y.value, dim=1) < 5))

    # formula parameter must contains values in [0., 1.]
    with pytest.raises(ValueError):
        forall([x, y], LTNObject(torch.tensor([[1., 2., 3.], [3., 4., 3.]]), ["x", "y"]))

    # check with a single variable quantified and without guarded quantification

    out = forall(x, p(x, y))
    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(p(x, y).value, dim=0)

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert isinstance(out, LTNObject), "The output of a quantification operation is always an LTNObject."
    assert out.free_vars == ["y"], "Since the quantification is on x, the output will have only y in the free vars."
    assert out.shape() == torch.Size([4]), "x has been quantified, so the output should have shape 4, namely the " \
                                           "number of individuals in y."

    # reverse quantified variable an check again

    out = forall(y, p(x, y))
    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(p(x, y).value, dim=1)

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert isinstance(out, LTNObject), "The output of a quantification operation is always an LTNObject."
    assert out.free_vars == ["x"], "Since the quantification is on y, the output will have only x in the free vars."
    assert out.shape() == torch.Size([2]), "y has been quantified, so the output should have shape 2, namely the " \
                                           "number of individuals in x."

    # quantification on both variables, with same quantifier

    out = forall([x, y], p(x, y))
    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(p(x, y).value, dim=(0, 1))

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert isinstance(out, LTNObject), "The output of a quantification operation is always an LTNObject."
    assert out.free_vars == [], "Since the quantification is on x and y, the output will not have free vars."
    assert out.shape() == torch.Size([]), "x and y have been quantified, so the output should have an empty shape."

    # quantification on both variables with different quantifiers

    out = forall(x, exists(y, p(x, y)))
    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(ltn.fuzzy_ops.AggregPMean(p=2)(p(x, y).value, dim=1), dim=0)

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert isinstance(out, LTNObject), "The output of a quantification operation is always an LTNObject."
    assert out.free_vars == [], "Since the quantification is on x and y, the output will not have free vars."
    assert out.shape() == torch.Size([]), "x and y have been quantified, so the output should have an empty shape."

    # guarded quantification on single variable

    # only first individual in x has a sum greater than 1 on its features
    # only first and last individuals in y have a sum greater than 0. on their features
    out = forall(x, p(x, y), x, lambda x: torch.sum(x.value, dim=1) > 1.)
    mask = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])  # the mask should be this one
    masked_p = torch.where(mask > 0., p(x, y).value.double(), np.nan)
    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(masked_p, dim=0)

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert isinstance(out, LTNObject), "The output of a quantification operation is always an LTNObject."
    assert out.free_vars == ["y"], "Since the quantification is on x, the output should have only y in the free vars."
    assert out.shape() == torch.Size([4]), "x has been quantified, so the output should have a shape of 4, since y" \
                                           "has 4 individuals."

    # guarded quantification on multiple variables

    out = forall(x, p(x, y), [x, y], lambda x, y: torch.logical_and(torch.sum(x.value, dim=1) > 1.,
                                                                    torch.sum(y.value, dim=1) > 0.))
    mask = torch.tensor([[1, 0, 0, 1], [0, 0, 0, 0]])  # the mask should be this one
    masked_p = torch.where(mask > 0., p(x, y).value.double(), np.nan)
    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(masked_p, dim=0)
    # replace of remaining Nan values after the aggregation
    # since quantifier is forall, we replace with 1.
    toy_out = torch.where(
        torch.isnan(toy_out),
        1.,
        toy_out
    )

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert isinstance(out, LTNObject), "The output of a quantification operation is always an LTNObject."
    assert out.free_vars == ["y"], "Since the quantification is on x, the output should have only y in the free vars."
    assert out.shape() == torch.Size([4]), "x has been quantified, so the output should have a shape of 4, since y" \
                                           "has 4 individuals."

    # guarded quantification on multiple variables - exists

    out = exists(x, p(x, y), [x, y], lambda x, y: torch.logical_and(torch.sum(x.value, dim=1) > 1.,
                                                                    torch.sum(y.value, dim=1) > 0.))
    mask = torch.tensor([[1, 0, 0, 1], [0, 0, 0, 0]])  # the mask should be this one
    masked_p = torch.where(mask > 0., p(x, y).value.double(), np.nan)
    toy_out = ltn.fuzzy_ops.AggregPMean(p=2)(masked_p, dim=0)
    # replace of remaining Nan values after the aggregation
    # since quantifier is forall, we replace with 0.
    toy_out = torch.where(
        torch.isnan(toy_out),
        0.,
        toy_out
    )

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert isinstance(out, LTNObject), "The output of a quantification operation is always an LTNObject."
    assert out.free_vars == ["y"], "Since the quantification is on x, the output should have only y in the free vars."
    assert out.shape() == torch.Size([4]), "x has been quantified, so the output should have a shape of 4, since y" \
                                           "has 4 individuals."

    # guarded quantification on multiple variables and all variables quantified

    out = forall([x, y], p(x, y), [x, y], lambda x, y: torch.logical_and(torch.sum(x.value, dim=1) > 1.,
                                                                    torch.sum(y.value, dim=1) > 0.))
    mask = torch.tensor([[1, 0, 0, 1], [0, 0, 0, 0]])  # the mask should be this one
    masked_p = torch.where(mask > 0., p(x, y).value.double(), np.nan)
    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(masked_p, dim=(0, 1))
    # replace of remaining Nan values after the aggregation
    # since quantifier is forall, we replace with 1.
    toy_out = torch.where(
        torch.isnan(toy_out),
        1.,
        toy_out
    )

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert isinstance(out, LTNObject), "The output of a quantification operation is always an LTNObject."
    assert out.free_vars == [], "Since the quantification is on x and y, the output should not have free vars."
    assert out.shape() == torch.Size([]), "x and y have been quantified, so the output should have an empty shape."

    # guarded quantification on a single variable but both quantified

    out = forall([x, y], p(x, y), x, lambda x: torch.sum(x.value, dim=1) > 1.)
    mask = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])  # the mask should be this one
    masked_p = torch.where(mask > 0., p(x, y).value.double(), np.nan)
    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(masked_p, dim=(0, 1))

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert isinstance(out, LTNObject), "The output of a quantification operation is always an LTNObject."
    assert out.free_vars == [], "Since the quantification is on both variables, the output should not have free vars."
    assert out.shape() == torch.Size([]), "Both variables have been quantified, so the output should be a scalar."

    # guarded quantification on a signle variable which is also different from the quantified variable

    out = forall(y, p(x, y), x, lambda x: torch.sum(x.value, dim=1) > 1.)
    mask = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])  # the mask should be this one
    masked_p = torch.where(mask > 0., p(x, y).value.double(), np.nan)
    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(masked_p, dim=1)
    # replace of remaining Nan values after the aggregation
    # since quantifier is forall, we replace with 1.
    toy_out = torch.where(
        torch.isnan(toy_out),
        1.,
        toy_out
    )

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert isinstance(out, LTNObject), "The output of a quantification operation is always an LTNObject."
    assert out.free_vars == ["x"], "Since the quantification is on y, the output should have x free var."
    assert out.shape() == torch.Size([2]), "y has been quantified, so the output should have x dimension (2)."

    # check exceptions on condition

    # only LTN variables are accepted as condition variables, this case there is a constant
    with pytest.raises(TypeError):
        forall(x, p(x, y), c, lambda x: torch.sum(x) > 5)

    # only LTN variables are accepted as condition variables, this case there is a variable + constant
    with pytest.raises(TypeError):
        forall(x, p(x, y), [x, c], lambda x, c: torch.sum(x) > 5)

    # the cond_fn parameter must be a function
    with pytest.raises(TypeError):
        forall(x, p(x, y), x, torch.tensor([1, 2, 3]))

    # diagonal quantification

    v1 = Variable("v1", torch.randn((3, 5)))
    v2 = Variable("v2", torch.randn((3, 6)))

    l = lambda x, y: torch.nn.Sigmoid()(torch.sum(torch.cat([x, y], dim=1), dim=1))

    out = forall(diag(v1, v2), p(v1, v2))
    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(l(v1.value, v2.value), dim=0)

    assert torch.equal(out.value, toy_out), "Since we are in diagonal setting, the output should be the same"
    assert out.free_vars == [], "We have quantified both variables, so there should not be free vars."
    assert isinstance(out, LTNObject), "The output of a quantification is always an LTNObject."
    assert out.shape() == torch.Size([]), "The output should be a scalar."

    # check the undiag has been performed after the quantification
    assert v1.free_vars == ["v1"], "The free vars in v1 should be v1."
    assert v2.free_vars == ["v2"], "The free vars in v2 should be v2."

    # diagonal quantification + guarded quantification
    # condition: consider only when |sum(v1) - sum(v2)| < 0.2

    out = forall(diag(v1, v2), p(v1, v2), [v1, v2], lambda x, y: torch.abs(torch.sum(x.value, dim=1) -
                                                                           torch.sum(y.value, dim=1)) < 0.2)
    assert v1.free_vars == ["v1"], "The free vars in v1 should be v1."
    assert v2.free_vars == ["v2"], "The free vars in v2 should be v2."

    out_p = l(v1.value, v2.value)
    mask = torch.tensor([0, 1, 0])  # the mask should be this one
    masked_p = torch.where(mask > 0., out_p.double(), np.nan)
    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(masked_p, dim=0)

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == [], "We have quantified both variables, so there should not be free vars."
    assert isinstance(out, LTNObject), "The output of a quantification is always an LTNObject."
    assert out.shape() == torch.Size([]), "The output should be a scalar."

    # check the same condition but without diagonal quantification to see changes

    out = forall([v1, v2], p(v1, v2), [v1, v2], lambda x, y: torch.abs(torch.sum(x.value, dim=1) -
                                                                           torch.sum(y.value, dim=1)) < 2.)
    proc_objs, _, _ = process_ltn_objects([v1, v2])
    out_p = l(proc_objs[0].value, proc_objs[1].value).view(3, 3)
    mask = torch.tensor([[0, 0, 0], [0, 1, 1], [0, 0, 1]])  # the mask should be this one
    masked_p = torch.where(mask > 0., out_p.double(), np.nan)
    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(masked_p, dim=(0, 1))

    assert torch.equal(out.value, toy_out), "The output should be the same."
    assert out.free_vars == [], "We have quantified both variables, so there should not be free vars."
    assert isinstance(out, LTNObject), "The output of a quantification is always an LTNObject."
    assert out.shape() == torch.Size([]), "The output should be a scalar."

    # compute_mask
    # it takes formula, cond_vars, cond_fn, agg_vars
    # test to do:
    # - all variable in cond and also aggregated
    # - one variable in cond and the other aggregated
    # - both aggregated but one in cond

    # one variable in cond and the other aggregated

    x = Variable("x", torch.tensor([[0.3, 4.5], [1.6, 7.1]]))
    y = Variable("y", torch.tensor([[0.2, 0.3, 0.1], [5.4, 7.6, 8.3], [1.6, 4.1, 5.6]]))
    toy_formula = LTNObject(torch.rand((2, 3)), ["x", "y"])

    f, m = Quantifier.compute_mask(copy.deepcopy(toy_formula), cond_vars=[y],
                                   cond_fn=lambda x: torch.sum(x.value, dim=1) < 5,
                                   aggregation_vars=[x])

    assert torch.equal(f.value, toy_formula.value.permute(1, 0)), "The formula should be transposed."
    assert toy_formula.free_vars != f.free_vars, "The free vars should be changed in ordering due to transposition."
    assert f.free_vars == ["y", "x"], "The free vars should be transposed."
    assert m.shape() == f.shape(), "Formula and mask should have the same shape."
    assert m.free_vars == ["y", "x"], "The free vars should be the same as the formula."
    assert torch.equal(m.value, torch.tensor([[True, False, False], [True, False, False]]).permute(1, 0)), "The value " \
                                                                                                           "should " \
                                                                                                           "be this one."

    # all variables in cond and also aggregated

    f, m = Quantifier.compute_mask(copy.deepcopy(toy_formula), cond_vars=[x, y],
                                   cond_fn=lambda x, y: torch.sum(x.value, dim=1) < torch.sum(y.value, dim=1),
                                   aggregation_vars=[x, y])

    assert torch.equal(f.value, toy_formula.value), "The formula should not be transposed."
    assert toy_formula.free_vars == f.free_vars, "The free vars should not be changed in ordering."
    assert f.free_vars == ["x", "y"], "The free vars should not be transposed."
    assert m.shape() == f.shape(), "Formula and mask should have the same shape."
    assert m.free_vars == ["x", "y"], "The free vars should be the same as the formula."
    assert torch.equal(m.value, torch.tensor([[False, True, True], [False, True, True]])), "The value should " \
                                                                                            "be this one."

    # both aggregated but one in cond - invariant to first case

    f, m = Quantifier.compute_mask(copy.deepcopy(toy_formula), cond_vars=[y],
                                   cond_fn=lambda x: torch.sum(x.value, dim=1) < 5,
                                   aggregation_vars=[x, y])

    assert torch.equal(f.value, toy_formula.value.permute(1, 0)), "The formula should be transposed."
    assert toy_formula.free_vars != f.free_vars, "The free vars should be changed in ordering due to transposition."
    assert f.free_vars == ["y", "x"], "The free vars should be transposed."
    assert m.shape() == f.shape(), "Formula and mask should have the same shape."
    assert m.free_vars == ["y", "x"], "The free vars should be the same as the formula."
    assert torch.equal(m.value, torch.tensor([[True, False, False], [True, False, False]]).permute(1, 0)), "The value " \
                                                                                                           "should " \
                                                                                                           "be this one."

    # only one variable

    toy_formula = LTNObject(torch.randn((2,)), ["x"])

    f, m = Quantifier.compute_mask(copy.deepcopy(toy_formula), cond_vars=[x],
                                   cond_fn=lambda x: torch.sum(x.value, dim=1) < 5,
                                   aggregation_vars=[x])

    assert torch.equal(f.value, toy_formula.value), "The formula should be untouched."
    assert toy_formula.free_vars == f.free_vars, "The free vars should be the same."
    assert f.free_vars == ["x"], "The free vars should be only x."
    assert m.shape() == f.shape(), "Formula and mask should have the same shape."
    assert m.free_vars == ["x"], "The free vars should be the same as the formula."
    assert torch.equal(m.value, torch.tensor([True, False])), "The value should be this one."

    # transpose_vars

    # same order

    formula = LTNObject(torch.rand((3, 4)), ["x", "y"])
    out = Quantifier.transpose_vars(formula, ["x", "y"])

    assert torch.equal(formula.value, out.value), "Nothing should be changed."
    assert formula.free_vars == out.free_vars, "Free vars should be the same."

    # different order

    out = Quantifier.transpose_vars(copy.deepcopy(formula), ["y", "x"])

    assert torch.equal(formula.value, out.value.permute(1, 0)), "The formula should be transposed."
    assert formula.free_vars != out.free_vars, "Free vars should be different."
    assert out.free_vars == ["y", "x"], "Free vars should be transposed."

    # all quantifiers with simple inputs and without condition only to check they are correct

    min_agg = ltn.fuzzy_ops.AggregMin()
    mean_agg = ltn.fuzzy_ops.AggregMean()
    p_mean_agg = ltn.fuzzy_ops.AggregPMean()
    p_mean_error_agg = ltn.fuzzy_ops.AggregPMeanError()

    truth_values = torch.rand((3, 5, 6))
    truth_values_nan = torch.where(truth_values < 0.1, np.nan, truth_values.double())

    # min

    out_0 = min_agg(truth_values, dim=0)
    out_1 = min_agg(truth_values, dim=1)
    out_2 = min_agg(truth_values, dim=2)
    out_0_1 = min_agg(truth_values, dim=(0, 1))
    out_0_2 = min_agg(truth_values, dim=(0, 2))
    out_1_2 = min_agg(truth_values, dim=(1, 2))

    assert torch.equal(out_0, torch.amin(truth_values, dim=0)), "The min should implement this behavior."
    assert torch.equal(out_1, torch.amin(truth_values, dim=1)), "The min should implement this behavior."
    assert torch.equal(out_2, torch.amin(truth_values, dim=2)), "The min should implement this behavior."
    assert torch.equal(out_0_1, torch.amin(truth_values, dim=(0, 1))), "The min should implement this behavior."
    assert torch.equal(out_0_2, torch.amin(truth_values, dim=(0, 2))), "The min should implement this behavior."
    assert torch.equal(out_1_2, torch.amin(truth_values, dim=(1, 2))), "The min should implement this behavior."

    # check with NaN values
    out_nan = min_agg(truth_values_nan, dim=0)
    assert torch.equal(out_nan, torch.amin(torch.where(torch.isnan(truth_values_nan), 1.,
                                                       truth_values_nan.double()).float(), dim=0)), "The min should " \
                                                                                                    "implement this " \
                                                                                                    "behavior with " \
                                                                                                    "NaN values."

    # check keepdim parameter

    out_k_d = min_agg(truth_values, dim=0, keepdim=True)
    assert torch.equal(out_k_d, torch.amin(truth_values, dim=0, keepdim=True)), "With the keepdim this should " \
                                                                                "be the behavior."

    # mean

    out_0 = mean_agg(truth_values, dim=0)
    out_1 = mean_agg(truth_values, dim=1)
    out_2 = mean_agg(truth_values, dim=2)
    out_0_1 = mean_agg(truth_values, dim=(0, 1))
    out_0_2 = mean_agg(truth_values, dim=(0, 2))
    out_1_2 = mean_agg(truth_values, dim=(1, 2))

    assert torch.equal(out_0, torch.mean(truth_values, dim=0)), "The mean should implement this behavior."
    assert torch.equal(out_1, torch.mean(truth_values, dim=1)), "The mean should implement this behavior."
    assert torch.equal(out_2, torch.mean(truth_values, dim=2)), "The mean should implement this behavior."
    assert torch.equal(out_0_1, torch.mean(truth_values, dim=(0, 1))), "The mean should implement this behavior."
    assert torch.equal(out_0_2, torch.mean(truth_values, dim=(0, 2))), "The mean should implement this behavior."
    assert torch.equal(out_1_2, torch.mean(truth_values, dim=(1, 2))), "The mean should implement this behavior."

    # check with NaN values
    out_nan = mean_agg(truth_values_nan, dim=0)
    numerator = torch.sum(torch.where(torch.isnan(truth_values_nan), torch.zeros_like(truth_values_nan),
                                      truth_values_nan), dim=0)
    denominator = torch.sum(~torch.isnan(truth_values_nan), dim=0)
    assert torch.equal(out_nan, torch.div(numerator, denominator)), "The mean should implement this behavior."

    # check keepdim parameter

    out_k_d = mean_agg(truth_values, dim=0, keepdim=True)
    assert torch.equal(out_k_d, torch.mean(truth_values, dim=0, keepdim=True)), "With the keepdim this should " \
                                                                                "be the behavior."

    # p mean

    out_0 = p_mean_agg(truth_values, dim=0)
    out_1 = p_mean_agg(truth_values, dim=1)
    out_2 = p_mean_agg(truth_values, dim=2)
    out_0_1 = p_mean_agg(truth_values, dim=(0, 1))
    out_0_2 = p_mean_agg(truth_values, dim=(0, 2))
    out_1_2 = p_mean_agg(truth_values, dim=(1, 2))

    assert torch.equal(out_0, torch.pow(torch.mean(torch.pow(
        ltn.fuzzy_ops.pi_0(truth_values), 2), dim=0), 1 / 2)), "The mean should implement this behavior."
    assert torch.equal(out_1, torch.pow(torch.mean(torch.pow(
        ltn.fuzzy_ops.pi_0(truth_values), 2), dim=1), 1 / 2)), "The mean should implement this behavior."
    assert torch.equal(out_2, torch.pow(torch.mean(torch.pow(
        ltn.fuzzy_ops.pi_0(truth_values), 2), dim=2), 1 / 2)), "The mean should implement this behavior."
    assert torch.equal(out_0_1, torch.pow(torch.mean(torch.pow(
        ltn.fuzzy_ops.pi_0(truth_values), 2), dim=(0, 1)), 1 / 2)), "The mean should implement this behavior."
    assert torch.equal(out_0_2, torch.pow(torch.mean(torch.pow(
        ltn.fuzzy_ops.pi_0(truth_values), 2), dim=(0, 2)), 1 / 2)), "The mean should implement this behavior."
    assert torch.equal(out_1_2, torch.pow(torch.mean(torch.pow(
        ltn.fuzzy_ops.pi_0(truth_values), 2), dim=(1, 2)), 1 / 2)), "The mean should implement this behavior."

    # check with NaN values

    out_nan = p_mean_agg(truth_values_nan, dim=0)
    numerator = torch.sum(torch.where(torch.isnan(truth_values_nan), torch.zeros_like(truth_values_nan),
                                      torch.pow(ltn.fuzzy_ops.pi_0(truth_values_nan), 2)), dim=0)
    denominator = torch.sum(~torch.isnan(truth_values_nan), dim=0)
    assert torch.equal(out_nan, torch.pow(torch.div(numerator, denominator), 1/2)), "The mean should implement " \
                                                                                    "this behavior."

    # check keepdim parameter

    out_k_d = p_mean_agg(truth_values, dim=0, keepdim=True)
    assert torch.equal(out_k_d, torch.pow(torch.mean(torch.pow(ltn.fuzzy_ops.pi_0(truth_values), 2),
                                                     dim=0, keepdim=True), 1/2)), "With the keepdim this should " \
                                                                                "be the behavior."

    # different value of p

    out_p = p_mean_agg(truth_values, dim=0, p=3)
    assert torch.equal(out_p, torch.pow(torch.mean(torch.pow(ltn.fuzzy_ops.pi_0(truth_values), 3),
                                                   dim=0), 1 / 3)), "With the keepdim this should " \
                                                                                    "be the behavior."

    # not stable

    out_p = p_mean_agg(truth_values, dim=0, stable=False)
    assert torch.equal(out_p, torch.pow(torch.mean(torch.pow(truth_values, 2), dim=0), 1/2)), "With the keepdim this " \
                                                                                              "should be the behavior."

    # p mean error

    out_0 = p_mean_error_agg(truth_values, dim=0)
    out_1 = p_mean_error_agg(truth_values, dim=1)
    out_2 = p_mean_error_agg(truth_values, dim=2)
    out_0_1 = p_mean_error_agg(truth_values, dim=(0, 1))
    out_0_2 = p_mean_error_agg(truth_values, dim=(0, 2))
    out_1_2 = p_mean_error_agg(truth_values, dim=(1, 2))

    assert torch.equal(out_0, 1. - torch.pow(torch.mean(torch.pow(
        1. - ltn.fuzzy_ops.pi_1(truth_values), 2), dim=0), 1 / 2)), "The mean should implement this behavior."
    assert torch.equal(out_1, 1. - torch.pow(torch.mean(torch.pow(
        1. - ltn.fuzzy_ops.pi_1(truth_values), 2), dim=1), 1 / 2)), "The mean should implement this behavior."
    assert torch.equal(out_2, 1. - torch.pow(torch.mean(torch.pow(
        1. - ltn.fuzzy_ops.pi_1(truth_values), 2), dim=2), 1 / 2)), "The mean should implement this behavior."
    assert torch.equal(out_0_1, 1. - torch.pow(torch.mean(torch.pow(
        1. - ltn.fuzzy_ops.pi_1(truth_values), 2), dim=(0, 1)), 1 / 2)), "The mean should implement this behavior."
    assert torch.equal(out_0_2, 1. - torch.pow(torch.mean(torch.pow(
        1. - ltn.fuzzy_ops.pi_1(truth_values), 2), dim=(0, 2)), 1 / 2)), "The mean should implement this behavior."
    assert torch.equal(out_1_2, 1. - torch.pow(torch.mean(torch.pow(
        1. - ltn.fuzzy_ops.pi_1(truth_values), 2), dim=(1, 2)), 1 / 2)), "The mean should implement this behavior."

    # check with NaN values

    out_nan = p_mean_error_agg(truth_values_nan, dim=0)
    numerator = torch.sum(torch.where(torch.isnan(truth_values_nan), torch.zeros_like(truth_values_nan),
                                      torch.pow(1. - ltn.fuzzy_ops.pi_1(truth_values_nan), 2)), dim=0)
    denominator = torch.sum(~torch.isnan(truth_values_nan), dim=0)
    assert torch.equal(out_nan, 1. - torch.pow(torch.div(numerator, denominator), 1 / 2)), "The mean should implement " \
                                                                                      "this behavior."

    # check keepdim parameter

    out_k_d = p_mean_error_agg(truth_values, dim=0, keepdim=True)
    assert torch.equal(out_k_d, 1. - torch.pow(torch.mean(torch.pow(1. - ltn.fuzzy_ops.pi_1(truth_values), 2),
                                                          dim=0, keepdim=True), 1 / 2)), "With the keepdim this should " \
                                                                                    "be the behavior."

    # different value of p

    out_p = p_mean_error_agg(truth_values, dim=0, p=3)
    assert torch.equal(out_p, 1. - torch.pow(torch.mean(torch.pow(1. - ltn.fuzzy_ops.pi_1(truth_values), 3),
                                                        dim=0), 1 / 3)), "With the keepdim this should " \
                                                                    "be the behavior."

    # not stable

    out_p = p_mean_error_agg(truth_values, dim=0, stable=False)
    assert torch.equal(out_p, 1. - torch.pow(torch.mean(torch.pow(1. - truth_values, 2),
                                                        dim=0), 1 / 2)), "With the keepdim this should be the behavior."

    # guarded quantification with variables which are not in the formula

    # predicate measuring similarity between two points
    Eq = ltn.Predicate(func=lambda x, y: torch.exp(-torch.norm(x - y, dim=1)))

    points = torch.rand((50, 2))  # 3 values in [0,1]^2
    x_ = ltn.Variable("x", points)
    y_ = ltn.Variable("y", points)
    d = ltn.Variable("d", torch.tensor([.1, .2, .3, .4, .5, .6, .7, .8, .9]))

    # function measuring euclidean distance
    dist = lambda x, y: torch.unsqueeze(torch.norm(x.value - y.value, dim=1), 1)

    out = exists(d,
                 forall([x_, y_],
                        Eq(x_, y_),
                        cond_vars=[x_, y_, d],
                        cond_fn=lambda x, y, d: dist(x, y) < d.value
                        ))

    # compute mask
    proc_objs, _, _ = process_ltn_objects([x_, y_, d])
    mask = dist(proc_objs[0], proc_objs[1]) < proc_objs[2].value
    mask = mask.view(50, 50, 9)
    toy_out = Eq(x_, y_)
    toy_out.value = toy_out.value.view(50, 50, 1)
    toy_out.value = toy_out.value.expand(50, 50, 9)
    masked_p = torch.where(mask > 0., toy_out.value.double(), np.nan)
    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(masked_p.permute((2, 0, 1)), dim=(1, 2))
    # replace of remaining Nan values after the aggregation
    # since quantifier is forall, we replace with 1.
    toy_out = torch.where(
        torch.isnan(toy_out),
        1.,
        toy_out
    )
    toy_out = ltn.fuzzy_ops.AggregPMean(p=2)(toy_out, dim=0)

    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert isinstance(out, LTNObject), "The output of a quantification operation is always an LTNObject."
    assert out.free_vars == [], "Since the quantification is on all variables, the output should not have free vars."
    assert out.shape() == torch.Size([]), "All variables are quantified, so the output should be a scalar."

    # check compute_mask on the same case

    f, m = Quantifier.compute_mask(Eq(x_, y_), [x_, y_, d], lambda x, y, d: dist(x, y) < d.value, ["x", "y"])
    formula = Eq(x_, y_)
    formula.value = formula.value.view(50, 50, 1)
    formula.value = formula.value.expand(50, 50, 9)
    formula.value = formula.value.permute([2, 0, 1])
    formula.free_vars = ["d", "x", "y"]
    assert torch.equal(f.value, formula.value), "The formula should be expanded and transposed."
    assert formula.free_vars == f.free_vars, "The free vars should be the same."
    assert f.free_vars == ["d", "x", "y"], "The free vars should be d, x and y."
    assert m.shape() == f.shape(), "Formula and mask should have the same shape."
    assert m.free_vars == ["d", "x", "y"], "The free vars should be the same as the formula."
    assert torch.equal(m.value, mask.permute((2, 0, 1))), "The value should be this one."

    # guarded quantification with variables which are not in the formula and with diagonal quantification

    # predicate measuring similarity between two points
    Eq = ltn.Predicate(func=lambda x, y: torch.exp(-torch.norm(x - y, dim=1)))

    points1 = torch.rand((20, 2))  # 3 values in [0,1]^2
    points2 = torch.rand((20, 2))
    x_ = ltn.Variable("x", points1)
    y_ = ltn.Variable("y", points2)
    d = ltn.Variable("d", torch.tensor([.1, .2, .3, .4, .5, .6, .7, .8, .9]))

    # function measuring euclidean distance
    dist = lambda x, y: torch.unsqueeze(torch.norm(x.value - y.value, dim=1), 1)

    out = exists(d,
                 forall(ltn.diag(x_, y_),
                        Eq(x_, y_),
                        cond_vars=[x_, y_, d],
                        cond_fn=lambda x, y, d: dist(x, y) < d.value
                        ))

    # compute mask
    x_, y_ = ltn.diag(x_, y_)
    proc_objs, _, _ = process_ltn_objects([x_, y_, d])
    mask = dist(proc_objs[0], proc_objs[1]) < proc_objs[2].value
    mask = mask.view(20, 9)
    toy_out = Eq(x_, y_)
    toy_out.value = toy_out.value.view(20, 1)
    toy_out.value = toy_out.value.expand(20, 9)
    masked_p = torch.where(mask > 0., toy_out.value.double(), np.nan)
    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(masked_p, dim=0)
    # replace of remaining Nan values after the aggregation
    # since quantifier is forall, we replace with 1.
    toy_out = torch.where(
        torch.isnan(toy_out),
        1.,
        toy_out
    )
    toy_out = ltn.fuzzy_ops.AggregPMean(p=2)(toy_out, dim=0)

    # here, there is a floating point problem, so we see if the difference between the two in under a very low threshold
    assert torch.equal(out.value, toy_out), "The output should be the same if everything is correct."
    assert isinstance(out, LTNObject), "The output of a quantification operation is always an LTNObject."
    assert out.free_vars == [], "Since the quantification is on all variables, the output should not have free vars."
    assert out.shape() == torch.Size([]), "All variables are quantified, so the output should be a scalar."

    # test SatAgg

    SatAgg = ltn.fuzzy_ops.SatAgg()

    # test exception in construction

    # the constructor only accepts AggregationOperator instances
    with pytest.raises(TypeError):
        ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AndMin())

    # test exceptions with call

    t = torch.tensor([1, 2, 3])

    # the call accepts list of LTNObjects and a list with one tensor and one list is given
    with pytest.raises(TypeError):
        SatAgg(t, [1, 2, 3])

    # now, only one is an LTNObject
    with pytest.raises(TypeError):
        SatAgg(LTNObject(torch.tensor([1, 2, 3]), []), [1, 2, 3])

    # only scalars are accepted as truth values
    with pytest.raises(ValueError):
        SatAgg(LTNObject(torch.tensor([1, 2, 3]), []))

    # only values in [0., 1.] are allowed
    with pytest.raises(ValueError):
        SatAgg(LTNObject(torch.tensor(0.1), []), LTNObject(torch.tensor(1.4), []))

    # now, test the behavior
    l = [LTNObject(torch.tensor(0.1), []), LTNObject(torch.tensor(0.34), []), LTNObject(torch.tensor(0.90), [])]

    out = SatAgg(*l)

    toy_out = ltn.fuzzy_ops.AggregPMeanError(p=2)(torch.stack([o.value for o in l], dim=0), dim=0)

    assert torch.equal(out, toy_out), "The output should be the same."
