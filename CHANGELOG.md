# Changelog

## 1.1.0

This release makes LTNtorch faster and lighter on memory. The results of the formulas are unchanged (up to float32
rounding), but a few behaviors that custom code may rely on have changed: see
[Behavior changes](#behavior-changes) before upgrading.

### Performance

- Connectives no longer repeat their operands in memory. A new function, `ltn.broadcast_ltn_objects`, only aligns
  the axes of the operands (one dimension per variable, of size 1 where the operand does not contain the
  variable), and the fuzzy operator relies on the PyTorch broadcasting. For example, in `And(P(x), Q(y))` the operands
  have shapes `(n, 1)` and `(1, m)`, and the `(n, m)` output is the only tensor of that size which is allocated.
- Inputs are no longer cloned. Predicates, functions and connectives used to copy the values of every input before
  processing it, even when nothing had to be repeated. Now the model of a predicate on one variable receives the
  variable's tensor directly.
- Predicates and functions repeat their inputs with a single copy. The repetition is built with `torch.expand`
  (a view) and materialized only once, by the final flattening, instead of one copy per missing variable plus a
  safety copy.
- Truth values are checked once per formula. Outputs of predicates, connectives and quantifiers are marked as
  checked and are not checked again downstream. The checks can be disabled with `ltn.set_validation(False)`, which is
  useful on GPU, where each check forces a synchronization.
- Guarded quantification computes in float32 instead of float64 (see below).
- The Lukasiewicz operators (`AndLuk`, `OrLuk`, `ImpliesLuk`) and the masked aggregators (`AggregMin`, `AggregMean`,
  `AggregPMean`, `AggregPMeanError`) no longer allocate temporary tensors of zeros or ones.

Measured on CPU (4 threads, torch 2.14.1), forward plus backward where a loss exists:

| What is measured                                                              | 1.0.2    | 1.1.0    | Speed-up |
|-------------------------------------------------------------------------------|----------|----------|----------|
| `And(P(x), Q(y))` on a 2000 x 2000 grid, connective only                      | 71.9 ms  | 15.5 ms  | 4.6x     |
| Predicate reading one column of a 20000 x 512 variable                        | 11.8 ms  | 1.0 ms   | 12.3x    |
| Binary predicate on a 1000 x 1000 grid, 64 features each                      | 221.2 ms | 140.0 ms | 1.6x     |
| `Forall([x, y], Implies(And(P(x), Q(y)), Or(P(x), Not(Q(y)))))`, 1500 x 1500, training step | 145.4 ms | 78.3 ms | 1.9x |
| `Forall(x, Exists(y, And(P(x), Q(y)), guarded))`, 1500 x 1500, training step  | 141.6 ms | 89.9 ms  | 1.6x     |
| Guarded `Forall` on a 1500 x 1500 x 40 grid, mask on one variable only        | 5.24 s   | 2.95 s   | 1.8x     |

The gains of the validation switch and of float32 are expected to be larger on GPU, where they were not measured.

### New features

- `ltn.set_validation(enabled)` and `ltn.validation_enabled()` turn the checks on the truth values on and off, and
  read the current setting. Validation is on by default.
- `ltn.broadcast_ltn_objects(objects, expand=True)` makes the shapes of a list of LTN objects compatible without
  copying their values. With `expand=False` the dimensions of the missing variables are left with size 1.
- `LTNObject.truth_checked` is an internal flag which tells whether the values of an object have been already checked
  to be in [0., 1.].

### Bug fixes

- LTN objects given to predicates, functions and connectives are never modified. Before, the output of a trainable
  predicate (an object with a `grad_fn`) was changed in place when it was used in a connective, so a sub-formula such
  as `px = P(x)` could not be reused: after `And(px, Q(y))`, `px` had the shape and the variables of the result.
- The variables missing from an object are now added in a deterministic order.
- The error messages raised by `Quantifier` when only one of `cond_vars` and `cond_fn` is set named the wrong
  parameter.
- The `TypeError` raised by `Quantifier` when `vars` or `cond_vars` is not a variable lost its message when the
  argument was not a list.
- Fixed several typos in the formulas of the documentation pages on broadcasting and quantification.
- The docstrings that contain LaTeX formulas are now raw strings, so importing LTNtorch no longer raises
  `SyntaxWarning: invalid escape sequence` on Python 3.12 and later.
- The examples run again with recent versions of pandas and scikit-learn: the labels of the crabs dataset are
  converted to integers explicitly (example 3), the RMSE is computed without the removed `squared` argument of
  `mean_squared_error` (example 5), and `display.precision` replaces the removed `precision` option (example 7).

### Behavior changes

- **Guarded quantification returns float32.** It used to return float64, and the type spread to everything computed
  afterwards, loss included. Code that relied on float64 has to be updated.
- **Custom connective operators receive broadcastable operands.** They used to receive flat vectors of the same
  length. They now receive tensors with one dimension per variable of the formula, where a dimension can have size 1,
  and they have to return the broadcast result. Operators written with element-wise torch functions work unchanged.
- **Models may receive a view of a variable's tensor.** When nothing has to be repeated (for example, a predicate on
  one variable), the model receives the tensor of the variable itself instead of a copy. A model that modifies its
  input in place (for example, `nn.ReLU(inplace=True)` as first layer) would now modify the variable.
- **Outputs of connectives and quantifiers are trusted.** A custom connective operator or aggregator that returns
  values outside [0., 1.] is no longer caught by the next connective or quantifier, but only by `SatAgg`.
