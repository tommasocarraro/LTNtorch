# LTNtorch

PyTorch implementation of Logic Tensor Networks, published on PyPI as `LTNtorch` (current version in `setup.py`).

## Layout

- `ltn/core.py`: `LTNObject`, `Constant`, `Variable`, `Predicate`, `Function`, `Connective`, `Quantifier`, `diag`/`undiag`, and the two functions that make shapes compatible (`broadcast_ltn_objects`, `process_ltn_objects`).
- `ltn/fuzzy_ops.py`: fuzzy connective operators, aggregators, `SatAgg`, `check_values`, and the validation switch (`set_validation`, `validation_enabled`).
- `tests/tests.py`: the whole test suite, one function per component.
- `docsrc/`: Sphinx sources (the API pages use `automodule`, so new public functions appear automatically). `docs/` is the built HTML.
- `examples/`, `tutorials/`: notebooks.

## Commands

These are the same commands the CI runs (`.github/workflows`):

```bash
python -m pytest tests/tests.py --cov=ltn
python -m doctest -v ltn/core.py
python -m doctest -v ltn/fuzzy_ops.py
```

The docstring examples are doctests, so their printed outputs have to stay exact when behavior changes.

## Conventions

- Supports Python >= 3.7 and torch >= 1.9.0. Avoid torch APIs newer than that (for example, use `masked_fill` instead of `torch.where` with a Python scalar).
- Docstrings follow the NumPy style with Sphinx roles (`:class:`, `:ref:`, `:obj:`).
- Comments are dense and written in the first person plural, in lowercase, explaining why each step is done ("we need to ...", "note that ...", "in such a way ...", "namely ..."). New code should be commented in the same voice and with the same density.
- Commit messages are short lowercase sentences in the past tense ("fixed issue with ...").

## How shapes are made compatible

- `broadcast_ltn_objects(objects, expand=True)` only aligns the axes (adds dimensions of size 1 for missing variables, permutes to a common order) and never copies values. With `expand=False` the new dimensions stay at size 1. Used by connectives (`expand=False`) and by `Quantifier.compute_mask`.
- `process_ltn_objects(objects)` calls the function above and then flattens the variable dimensions into one batch dimension. This is the only step that copies values. Used by predicates, functions, and the condition of a guarded quantification, since user models expect a flat batch with features on dimension 1.
- Neither function modifies the objects it receives.
- `LTNObject.truth_checked` marks outputs of predicates, connectives and quantifiers as already validated, so they are not checked again downstream. Objects built by hand are always checked.

## TODO

### Guarded quantification only on the selected pairs

Today the formula is evaluated on every combination of individuals and the mask discards values afterwards. When the formula contains a predicate that takes the guarded variables together (for example `R(x, y)`) and the guard is selective, most of the model's work is wasted. In a standalone prototype (small MLP, 1000 x 1000 grid, CPU) evaluating only the selected pairs was about 3 times faster with 30% of the pairs kept and about 17 times faster with 5% kept.

This needs a change to the public interface, because `Quantifier.__call__` receives a formula that has already been evaluated. Sketch of the design:

- accept the formula lazily (for example as a callable) alongside the current form, so existing code keeps working;
- compute the mask first from `cond_vars` and `cond_fn`, gather the selected individuals, and evaluate the formula only on them (the selected pairs behave like variables in diagonal quantification);
- when a variable stays free (for example "for all x, exists y such that ..."), aggregate per group with a flat vector of values plus a group index (`index_add` and `bincount`, or `scatter_reduce` for the minimum). Empty groups get the default directly (1 for forall, 0 for exists), with no NaN replacement;
- it does not help for formulas made only of unary predicates combined by connectives (the grid appears only in the connective, which is a cheap broadcast) or for guards that keep most pairs, so it should stay optional.

Nested tensors (`torch.nested`) are not an option: on torch 2.14 the backward of their reductions is not implemented.

### Changelog for the next version

Help write the changelog when the next version is prepared. The source is `efficiency_report.pdf` in the repo root (not tracked by git), which has the before/after code and the measurements. Its content in short:

- connectives use broadcasting instead of repeating their operands (new `ltn.broadcast_ltn_objects`);
- inputs are no longer cloned, and LTN objects are never modified by the library, so a sub-formula such as `px = P(x)` can be reused;
- predicates and functions repeat their inputs with a single copy;
- truth values are checked once per formula, and the checks can be disabled with `ltn.set_validation(False)`;
- guarded quantification returns float32 instead of float64;
- the Lukasiewicz operators and the masked aggregators no longer allocate temporary tensors.

Behavior changes to call out for users:

- guarded quantification now returns float32;
- custom connective operators receive broadcastable operands (one dimension per variable, some of size 1) instead of two flat vectors;
- when nothing has to be repeated, a model receives a view of the variable's tensor, so it must not modify its input in place;
- outputs of connectives and quantifiers are trusted, so a custom operator returning values outside [0, 1] is caught only by `SatAgg`.
