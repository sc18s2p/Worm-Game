from typing import List
from typing import Union

import numpy as np
from fenics import *

try:
    from fenics_adjoint import *
except ModuleNotFoundError:
    # This optional import is only needed if derivatives are being taken.
    pass


def f2n(var: Union[Function, List[Function]]) -> np.ndarray:
    """
    Fenics to numpy
    Returns a numpy array containing fenics function values
    """
    if type(var) == list:
        return np.stack([f2n(v) for v in var])

    fs = var.function_space()
    dof_maps = _dof_maps(fs)
    vec = var.vector().get_local()
    arr = np.zeros_like(dof_maps, dtype=np.float64)
    for i in np.ndindex(dof_maps.shape):
        arr[i] = vec[dof_maps[i]]

    return arr


def v2f(
        val: Union[np.ndarray, Expression, Function],
        var: Function = None,
        fs: FunctionSpace = None,
        name: str = None
) -> Function:
    """
    Value (mixed) to fenics
    Set a value to a new or existing fenics variable.
    """
    assert var is not None or fs is not None
    if var is None:
        var = Function(fs, name=name)

    # If numpy array passed, set these as the function values
    if isinstance(val, np.ndarray):
        _set_vals_from_numpy(var, val)

    # If an expression is passed, interpolate on the space
    elif isinstance(val, Expression):
        var.assign(interpolate(val, var.function_space()))

    # If a function is passed, just assign
    elif isinstance(val, Function):
        var.assign(val)

    return var


def _set_vals_from_numpy(var: Function, values: np.ndarray):
    """
    Sets the vertex-values (or between-vertex-values) of a variable from a numpy array
    """
    fs = var.function_space()
    dof_maps = _dof_maps(fs)
    assert values.shape == dof_maps.shape, f'shapes don\'t match!  values: {values.shape}. dof_maps: {dof_maps.shape}'
    vec = var.vector()
    for i in np.ndindex(dof_maps.shape):
        vec[dof_maps[i]] = values[i]


def _dof_maps(fs: FunctionSpace) -> np.ndarray:
    """
    Returns a numpy array for the dof maps of the function space
    """
    n_sub = fs.num_sub_spaces()
    if n_sub > 0:
        dof_map = np.array([_dof_maps(fs.sub(d)) for d in range(n_sub)])
    else:
        dof_map = np.array(fs.dofmap().dofs())

    return dof_map

