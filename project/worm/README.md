Python implementation of numerical method for visco-elastic rods presented in

| Thomas Ranner . A stable finite element method for low inertia undulatory locomotion in three dimensions. Applied Numerical Mathematics 156 (2020) 422–44, 2020. https://doi.org/10.1016/j.apnum.2020.05.009

The paper describes all the method and all key terminology. This method is applied to simulate the undulatory locomotion of worms.

# Installation

First you will need a Python 3.9 environment. You can create one with [Conda](https://conda.io/docs/) using the environment file provided:

```bash
conda env create -f environment.yml
```

If you want to run the inverse modelling code, you will need to use `environment_inv.yml` as this contains additional requirements:

```bash
conda env create -f environment_inv.yml
```

Alternatively you can install into an existing conda or pip environment using pip. 
Either install directly from the repository:

```bash
pip install git+https://gitlab.com/tom-ranner/simple-worm.git@master
```

Or if you want to play around with the code then clone the repository to your machine and install in `editable` mode (picking the appropriate install command):

```bash
git clone git@gitlab.com:tom-ranner/simple-worm.git
cd simple-worm
# forward model only:
pip install -e . 
# forward and inverse models:
pip install -e .[inv] 
```


# Usage

The main interface is provided by the `Worm` class in `worm.py`. It is initialized with two arguments which specifies how many mesh points to use and the time step. It is recommended to use around 100 points (`N=101`) for accurate simulations and a time step around 1/1000 (`dt=1e-3`).

After initialization the method `update` assembles and solves the matrices with its argument used as the internal forcing. This is the prescribed curvature for the above paper. Currently this is only implemented for `numpy` arrays.

The output of the `Worm` object is summarized as follows:

- `x` :: positions of mid-line points.
- (`e1`, `e2`) :: components of the orthogonormal frame attached to the mid-line.
- (`alpha`, `beta`, `gamma`) :: an intrinsic representation of the body. `alpha` is curvature in the `e1` direction, `beta` is curvature in the `e2` direction and `gamma` describes the twist of the frame about the mid-line.

# Inverse modelling

There is some work in progress on inverse modelling also available. This work requires more testing.
