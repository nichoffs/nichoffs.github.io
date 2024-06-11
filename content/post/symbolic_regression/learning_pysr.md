+++
title = 'Learning PySr for Symbolic Regression'
date = 2024-06-09T13:17:27+02:00
draft = false
+++

# What is PySr?

A library for symbolic regression made by one of my favorite researchers, Miles Cranmer.
The backend genetic algorithm(optimization method for finding symbolic expression) runs on Julia, so it's really fast. It's also extremely flexible.

## Installing and Running PySr

I'm going to use a Jupyter notebook with a `virtualenv` environment for all my PySr experiments.

```sh
virtualenv venv
source venv/bin/activate
pip install numpy pysr
```

# Vector->Scalar Fit

The first import of pysr will take much longer than the next as Julia dependencies are getting set up.

```python
from pysr import PySRRegressor
import numpy as np
```

This is function $f:\R^5\rightarrow\R$ that we'll be fitting symbolically:
```python
X = np.random.randn(100, 5)
y = np.sum(X * X, axis=-1)
```

`X` has shape `(100,5)`; `y`, `(100,)`.

Now instantiate a new regressor model. Running `fit` on `PySrRegressor` will create lots of useful (but currently unecessary) files that we'll route to the `temp` folder:
```python
model = PySRRegressor(niterations=200, binary_operators=["*", '+'],tempdir='./temp/',temp_equation_file=True)
model.fit(X, y)
```

![model output](/pysr_simplevsfit.png)

To get the best model's equation as a rendered latex string, you can call the `sympy` method. The `latex` method returns the corresponding unrendered latex string.

```python
model.sympy()
```

$$x_{0}^{2} + x_{1}^{2} + x_{2}^{2} + x_{3}^{2} + x_{4}^{2}$$

As you can see, the model learned the exact ground-truth equation.

# Vector->Vector fit

PySr can also fit Vector->Vector functions by outputting a list with different symbolic expressions for each.

```python
model = PySRRegressor(niterations=200, binary_operators=["*", '+'],tempdir='./temp/',temp_equation_file=True)
model.fit(X, y)
model.sympy()
```

```sh
[x0**2, x1**2, x2**2, x3**2, x4**2]
```

Since the output is a list, we can't display each as latex.
