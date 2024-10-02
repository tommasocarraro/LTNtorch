# Copyright (c) 2021-2024 Tommaso Carraro
# Licensed under the MIT License. See LICENSE file in the project root for details.

from ltn.core import Variable, Predicate, Constant, Function, Connective, diag, undiag, Quantifier, \
    LTNObject, process_ltn_objects, LambdaModel
import torch
import ltn.fuzzy_ops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")