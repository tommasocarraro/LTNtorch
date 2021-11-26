# Logic Tensor Networks (LTN)

Welcome to the PyTorch implementation of Logic Tensor Networks!

Logic Tensor Network (LTN) is a Neural-Symbolic (NeSy) framework which supports learning of neural networks using the
satisfaction of a first-order logic knowledge base as an objective. In other words, LTN uses logical reasoning on the
knowledge base to guide the learning of a potentially deep neural network. 

The idea of the framework is simple: 
- we have a first-order logic knowledge base containing a set of axioms;
- we have some predicates, functions, or logical constants appearing in these axioms that we want to learn;
- we have some data available that we can use to learn the parameters of those symbols.

The idea is to use the logical axioms as a loss function for our Logic Tensor Network. The objective is to find solutions
in the hypothesis space that maximally satisfy all the axioms contained in our knowledge base.

In LTN, the learnable parameters are contained in the predicates, functions, and possibly learnable logical constants
that appear in the logical axioms of the knowledge base. 

During the forward step of the back-propagation algorithm, LTN
computes the truth values of the logical formulas contained in the knowledge base, using the available data to ground
(or instantiate) the logical formulas. As we have already said, these formulas 
will contain some predicates and functions which are represented as learnable models. 

At the end of the forward phase,
the truth values computed for the formulas are aggregated and used in the loss function. Our objective is to maximize
the aggregated truth value, namely maximally satisfy all the axioms.

During the backward step, the
learnable parameters of predicates, functions, and possibly learnable logical constants are changed in such a way to move
towards a solution in the hypothesis space which better satisfies all the axioms in the knowledge base.

At the end of the training, the parameters of predicates, functions, and constants will have been adapted in such a way the 
logical formulas in the knowledge base are maximally satisfied. In particular, the parameters will have been learned by using both
data (to ground the formulas) and logical reasoning (loss function).

After learning, it is possible to query predicates and functions on new data which was not available during training. Also,
it is possible to query the truth values of new formulas which were not included in the knowledge base during training. In addition,
if some logical constants have been learned, their parameters can be interpreted as embeddings.

To make this learning possible, LTN uses a differentiable first-order logic language, called Real Logic, which enable 
the incorporation of data and logic.

Real Logic defines the concept of `grounding` (different from the grounding of logic), which is a mapping from the logical domain 
to tensors in the Real field, or operations (mathematical functions, neural networks, etc.) based on tensors. In other words,
a `grounding`, denoted as <img src="https://render.githubusercontent.com/render/math?math=\mathcal{G}"/>, is a function which maps a logical symbol into a real tensor or an operation on tensors.

In particular, the grounding is defined as follows. Let us assume that *c* is a constant, *x* is a logical 
variable, *P* is a predicate, *f* is a logical function:
- <img src="https://render.githubusercontent.com/render/math?math=\mathcal{G}(c) = \mathbb{R}^{d_1 \times \dots \times d_n}"/>: a logical constant is grounded into a tensor (individual) of any 
order (e.g., <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^4"/> or <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^{5 \times 4 \times 4}"/>);
- <img src="https://render.githubusercontent.com/render/math?math=\mathcal{G}(x) = \mathbb{R}^{m \times d}"/>: a logical variable is grounded into a sequence of *m* tensors (individuals) of 
the same shape *d*;
- <img src="https://render.githubusercontent.com/render/math?math=\mathcal{G}(f \mid \theta) = x \mapsto MLP_{\theta}(x)"/>: a logical function is grounded into a (learnable)
mathematical function which take as input some tensors (individuals) and return a tensor. In this definition, <img src="https://render.githubusercontent.com/render/math?math=\theta"/> are the learnable parameters
of the function, <img src="https://render.githubusercontent.com/render/math?math=MLP_{\theta}"/> is the neural network representing the function and parametrized by <img src="https://render.githubusercontent.com/render/math?math=\theta"/>. Note that a function 
could take multiple tensors (individuals) as input;
- <img src="https://render.githubusercontent.com/render/math?math=\mathcal{G}(P \mid $\theta$) = x \mapsto \sigma (MLP_{\theta}(x))"/>: a logical formula (atomic or not) 
is grounded into a mathematical function which take as input some tensors (individuals) and return a value in [0., 1.]. In this case,
the logistic function assure the output to be in the range [0., 1.], resulting in a value which can be interpreted as a
fuzzy truth value. Note that a predicate of formula could take multiple tensors (individuals) as input.

The `grounding` defines also how the logical connectives (<img src="https://render.githubusercontent.com/render/math?math=\land, \lor, \lnot, \implies, \leftrightarrow"/>) and quantifiers
(<img src="https://render.githubusercontent.com/render/math?math=\forall, \exists" />) have to be grounded. In particular, logical connectives are grounded using fuzzy logic semantics, while
quantifiers are grounded using fuzzy aggregators. Please, carefully read the paper if you have some doubts on these notions.

Examples of possible groundings are showed in the following figure. In the figure, *friend(Mary, John)* is an
atomic formula (predicate), while <img src="https://render.githubusercontent.com/render/math?math=\forall x (friend(John, x) \implies friend(Mary, x))"/> is a closed formula (all the variables are
quantified). The letter <img src="https://render.githubusercontent.com/render/math?math=\mathcal{G}"/> is the grounding, the function which maps the logical domain into the Real domain.

![Grounding_illustration](../LTNtorch/images/framework_grounding.png)

In practice, LTN converts Real Logic formulas (e.g. <img src="https://render.githubusercontent.com/render/math?math=\forall x (cat(x) \implies \exists y (partOf(x,y) \land tail(y)))"/> into [PyTorch](https://www.pytorch.org/) 
computational graphs. Such formulas can express complex queries about the data, prior knowledge to satisfy during 
learning, statements to prove, etc. An example on how LTN converts such formulas into PyTorch computational graphs is 
showed in the following figure.

![Computational_graph_illustration](../LTNtorch/images/framework_computational_graph.png)

Using LTN, one can represent and effectively compute some of the most important tasks of deep learning. Examples of such 
tasks are classification, regression, clustering, and so on.

The [Getting Started](#getting-started) section of the README links to tutorials and examples to learn how to code Logic
Tensor Networks in PyTorch.

We suggest to carefully read the [paper](https://arxiv.org/pdf/2012.13635.pdf) before going through the tutorials and examples.


## Installation

Clone the LTN repository and install it using `pip install -e <local project path>`.

Following are the dependencies used for development (similar versions should run fine):
- python 3.9
- torch >= 1.9.0
- numpy >= 1.21.1
- matplotlib >= 3.4.2
- pandas >= 1.3.0
- scikit-learn >= 0.24.2
- torchvision >= 0.10.0

## Structure of repository

- `ltn/core.py`: the core module contains the implementation of the LTN framework. In particular, it contains the definition 
of constants, variables, predicates, functions, connectives and quantifiers;
- `ltn/fuzzy_ops.py`: the fuzzy_ops module contains the implementation of common fuzzy logic semantics using PyTorch primitives;
- `tutorials/`: the tutorial folder contains some important tutorial to getting started with coding in LTN;
- `examples/`: the examples folder contains various problems approached using LTN and presented in the "Reach of Logic Tensor Networks" section of the paper;
- `tests/`: the tests folder contains all the test that have been used to test the `core` and `fuzzy_ops` modules.

## Tests

The `core` and `fuzzy_ops` modules of this repository have been entirely tested using `pytest`, with a coverage of 100%.

## Documentation

The documentation for this project is still under development, and it will be release soon. However, the majority of functionalities
are already well documented. In the future, a GitHub documentation will be released.

## Getting Started

### Tutorials

`tutorials/` contains some important tutorial to getting started with coding in LTN. We suggest completing the tutorials in order.
The tutorials cover the following topics:
1. [Grounding in LTN (part 1)](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/main/tutorials/1-grounding_non_logical_symbols.ipynb): Real Logic, constants, predicates, functions, variables;
2. [Grounding in LTN (part 2)](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/main/tutorials/2-grounding_connectives.ipynb): connectives and quantifiers (+ [complement](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/main/tutorials/2b-operators_and_gradients.ipynb): choosing appropriate operators for learning),
3. [Learning in LTN](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/main/tutorials/3-knowledgebase-and-learning.ipynb): using satisfiability of LTN formulas as a training objective.

The tutorials are implemented using jupyter notebooks.

### Examples

`examples/` contains the series of example included in the "Reach of Logic Tensor Networks" section of the paper. Their objective 
is to show how the language of Real Logic can be used to specify a number of tasks that involve learning from data and 
reasoning about logical knowledge. 

The examples covered are the following:
1. [Binary classification](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/main/examples/1-binary_classification.ipynb): illustrates in the simplest setting how to ground a binary classifier as a predicate in LTN, and how to feed batches of data during training;
2. [Multi-class single-label classification](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/main/examples/2-multi_class_single_label_classification.ipynb): illustrate how to ground predicates that can classify samples in several mutually exclusive classes;
3. [Multi-class multi-label classification](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/main/examples/3-multi_class_multi_label_classification.ipynb): illustrate how to ground predicates that can classify samples in several classes which are not mutually exclusive;
4. [Semi-supervised pattern recognition](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/main/examples/4-semi-supervised_pattern_recognition.ipynb): showcases the power of LTN in dealing with semi-supervised learning tasks;
5. [Regression](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/main/examples/5-regression.ipynb): illustrates how to ground a regressor as a function symbol in LTN;
6. [Clustering](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/main/examples/5-clustering.ipynb): illustrates how LTN can solve a task using first-order constraints only, without any label being given through supervision;
7. [Learning embeddings with LTN](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/main/examples/7-learning_embeddings_with_LTN.ipynb): illustrates how LTN can learn embeddings for individuals based on fuzzy ground truths and first-order constraints.

The examples are presented using jupyter notebooks.

## License

This project is licensed under the MIT License - see the [LICENSE](https://nbviewer.jupyter.org/github/bmxitalia/LTNtorch/blob/original-implementation/LICENSE) file for details.

## Acknowledgements

LTN has been developed thanks to active contributions and discussions with the following people (in alphabetical order):
- Alessandro Daniele (FBK)
- Artur d’Avila Garcez (City)
- Benedikt Wagner (City)
- Emile van Krieken (VU Amsterdam)
- Francesco Giannini (UniSiena)
- Giuseppe Marra (UniSiena)
- Ivan Donadello (FBK)
- Lucas Bechberger (UniOsnabruck)
- Luciano Serafini (FBK)
- Marco Gori (UniSiena)
- Michael Spranger (Sony AI)
- Michelangelo Diligenti (UniSiena)
- Samy Badreddine (Sony AI)
- Tommaso Carraro (FBK)