import numpy as np
from scipy.special import binom


__all__ = [
    'gradient_of_angle',
    'amplitude',
    'instantaneous_frequency',
    'instantaneous_bandwidth',
    'instantaneous_curvature',
    'instantaneous_moments']


# like np.average, but ignores nan
def _nanaverage(arr, axis=None, weights=None, returned=False, keepdims=False):
    summed_weights = np.nansum(weights, axis=axis, keepdims=keepdims)
    result = np.nansum(arr * weights, axis=axis, keepdims=keepdims) / summed_weights
    if returned:
        return result, summed_weights
    return result


def gradient_of_angle(x, edge_order=1, axis=-1, discont=np.pi):
    """
    Specialized version of np.gradient which better handles angle unwrapping
    Args:
        x: angle to compute gradient of (in radians)
        edge_order: How the edges are handled during the gradient computation. See np.gradient
        discont: the discontinuity parameter for unwrap
        axis: which axis to take the diff along
    Returns:
        The diff of the phase-angle of x
    """
    # based on the np.gradient code
    slice_interior_source_even = [slice(None)] * x.ndim
    slice_interior_source_odd = [slice(None)] * x.ndim
    slice_interior_dest_of_source_even = [slice(None)] * x.ndim
    slice_interior_dest_of_source_odd = [slice(None)] * x.ndim
    slice_begin_source = [slice(None)] * x.ndim
    slice_end_source = [slice(None)] * x.ndim
    slice_begin_dest = [slice(None)] * x.ndim
    slice_end_dest = [slice(None)] * x.ndim

    is_even = x.shape[axis] // 2 * 2 == x.shape[axis]

    slice_interior_source_even[axis] = slice(None, None, 2)
    slice_interior_source_odd[axis] = slice(1, None, 2)
    if is_even:
        slice_interior_dest_of_source_even[axis] = slice(1, -1, 2)
        slice_interior_dest_of_source_odd[axis] = slice(2, None, 2)
    else:
        slice_interior_dest_of_source_even[axis] = slice(1, None, 2)
        slice_interior_dest_of_source_odd[axis] = slice(2, -1, 2)

    result = np.empty_like(x)

    result[tuple(slice_interior_dest_of_source_even)] = np.diff(np.unwrap(
        x[tuple(slice_interior_source_even)], axis=axis, discont=discont), axis=axis) / 2
    result[tuple(slice_interior_dest_of_source_odd)] = np.diff(np.unwrap(
        x[tuple(slice_interior_source_odd)], axis=axis, discont=discont), axis=axis) / 2

    slice_begin_dest[axis] = slice(0, 1)
    slice_end_dest[axis] = slice(-1, None)
    if edge_order == 1:
        slice_begin_source[axis] = slice(0, 2)
        slice_end_source[axis] = slice(-2, None)
        result[tuple(slice_begin_dest)] = np.diff(np.unwrap(
            x[tuple(slice_begin_source)], axis=axis, discont=discont), axis=axis)
        result[tuple(slice_end_dest)] = np.diff(np.unwrap(
            x[tuple(slice_end_source)], axis=axis, discont=discont), axis=axis)
    elif edge_order == 2:
        slice_begin_source[axis] = slice(1, 2)
        slice_end_source[axis] = slice(-2, -1)
        begin_source = np.concatenate([x[tuple(slice_end_dest)], x[tuple(slice_begin_source)]], axis=axis)
        end_source = np.concatenate([x[tuple(slice_end_source)], x[tuple(slice_begin_dest)]], axis=axis)
        result[tuple(slice_begin_dest)] = np.diff(np.unwrap(
            begin_source, axis=axis, discont=discont), axis=axis) / 2
        result[tuple(slice_end_dest)] = np.diff(np.unwrap(
            end_source, axis=axis, discont=discont), axis=axis) / 2
    else:
        raise ValueError('Unexpected edge_order: {}'.format(edge_order))

    return result


def _bell_polynomial(*x):
    m = [1]
    for n in range(1, len(x) + 1):
        m.append(0)
        for p in range(n):
            m[n] = m[n] + binom(n - 1, p) * x[n - p - 1] * m[p]
    m = m[1:]
    return m


def amplitude(x, variable_axis=None, keepdims=False):
    """
    The amplitude of the analytic signal x. If variable-axis is specified, x is treated as multivariate with its
        components along variable_axis.
    Args:
        x: An array with shape (..., time)
        variable_axis: If specified, this axis is treated as the components of a multivariate x
        keepdims: If True, the variable_axis will be retained (and of size 1). Ignored if variable_axis is None.
    Returns:
        The amplitude of the signal. If variable_axis is specified and keepdims is False, the result will be
            shape x.shape[:variable_axis] + x.shape[variable_axis + 1:]
    """
    result = np.abs(x)
    if variable_axis is not None:
        return np.sqrt(np.nanmean(np.abs(result) ** 2, axis=variable_axis, keepdims=keepdims))
    return result


def instantaneous_frequency(x, dt=1, edge_order=1, variable_axis=None, keepdims=False):
    """
    The instantaneous frequency of the analytic signal x. If variable-axis is specified, x is treated as
        multivariate with its components along variable_axis.
    Args:
        x: An array with shape (..., time)
        dt: The difference between steps on the time axis
        edge_order: How edges are handled when computing the gradient. See np.gradient
        variable_axis: If specified, this axis is treated as the components of a multivariate x
        keepdims: If True, the variable_axis will be retained (and of size 1). Ignored if variable_axis is None.
    Returns:
        The instantaneous frequency of the signal. If variable_axis is specified and keepdims is False,
            the result will be shape x.shape[:variable_axis] + x.shape[variable_axis + 1:]
    """
    result = gradient_of_angle(np.angle(x), edge_order=edge_order) / dt
    if variable_axis is not None:
        return _nanaverage(result, axis=variable_axis, weights=np.square(np.abs(x)), keepdims=keepdims)
    return result


def instantaneous_bandwidth(x, dt=1, edge_order=1, variable_axis=None, keepdims=False):
    """
    The instantaneous bandwidth of the analytic signal x. If variable-axis is specified, x is treated as
        multivariate with its components along variable_axis.
    Args:
        x: An array with shape (..., time)
        dt: The difference between steps on the time axis
        edge_order: How edges are handled when computing the gradient. See np.gradient
        variable_axis: If specified, this axis is treated as the components of a multivariate x
        keepdims: If True, the variable_axis will be retained (and of size 1). Ignored if variable_axis is None.
    Returns:
        The instantaneous bandwidth of the signal. If variable_axis is specified and keepdims is False,
            the result will be shape x.shape[:variable_axis] + x.shape[variable_axis + 1:]
    """
    result = np.gradient(np.log(np.abs(x)), edge_order=edge_order, axis=-1) / dt
    if variable_axis is not None:
        frequency = instantaneous_frequency(x, dt=dt, edge_order=edge_order)
        multi_frequency = instantaneous_frequency(
            x, dt=dt, edge_order=edge_order, variable_axis=variable_axis, keepdims=True)
        result = np.abs(result + 1j * (frequency - multi_frequency)) ** 2
        return np.sqrt(_nanaverage(result, axis=variable_axis, weights=np.square(np.abs(x)), keepdims=keepdims))
    return result


def instantaneous_curvature(x, dt=1, edge_order=1, variable_axis=None, keepdims=False):
    """
    The instantaneous curvature of the analytic signal x. If variable-axis is specified, x is treated as
        multivariate with its components along variable_axis.
    Args:
        x: An array with shape (..., time)
        dt: The difference between steps on the time axis
        edge_order: How edges are handled when computing the gradient. See np.gradient
        variable_axis: If specified, this axis is treated as the components of a multivariate x
        keepdims: If True, the variable_axis will be retained (and of size 1). Ignored if variable_axis is None.
    Returns:
        The instantaneous curvature of the signal. If variable_axis is specified and keepdims is False,
            the result will be shape x.shape[:variable_axis] + x.shape[variable_axis + 1:]
    """
    moments = instantaneous_moments(x, max_order=3, dt=dt, edge_order=edge_order)
    if variable_axis is not None:
        _, frequency, bandwidth, curvature = moments
        multi_frequency = instantaneous_frequency(
            x, dt=dt, edge_order=edge_order, variable_axis=variable_axis, keepdims=True)
        temp = curvature + 2 * 1j * bandwidth * (frequency - multi_frequency) - (frequency - multi_frequency) ** 2
        return np.sqrt(
            _nanaverage(np.abs(temp) ** 2, axis=variable_axis, weights=np.square(np.abs(x)), keepdims=keepdims))
    return moments[3]


def instantaneous_moments(x, max_order=0, dt=1, edge_order=1, variable_axis=None, keepdims=False):
    """
    The instantaneous moments of the analytic signal x up to and including the max_order moment. If variable-axis
        is specified, x is treated as multivariate with its components along variable_axis. max_order must be no
        more than 3 if variable_axis is given (other moments are not defined). The first 4 moments (up to max order 3)
        are the same as calling amplitude, instantaneous_frequency, instantaneous_bandwidth, and
        instantaneous_curvature respectively.
    Args:
        x: An array with shape (..., time)
        max_order: The moments up to and including max_order are returned. Order 0 is the amplitude, order 1 is the
            instantaneous_frequency, order 2 is the instantaneous_bandwidth, and order 3 is the instantaneous_curvature.
            Additional moments can also be computed if variable_axis is None, but max_order must be no more than 3 when
            variable_axis is given.
        dt: The difference between steps on the time axis
        edge_order: How edges are handled when computing the gradient. See np.gradient
        variable_axis: If specified, this axis is treated as the components of a multivariate x. max_order must be no
            more than 3 if variable_axis is given.
        keepdims: If True, the variable_axis will be retained (and of size 1). Ignored if variable_axis is None.
    Returns:
        The first (max_order + 1) instantaneous moments of the signal. If variable_axis is specified and keepdims is
            False, each moment will be shape x.shape[:variable_axis] + x.shape[variable_axis + 1:]
    """

    if max_order < 0:
        raise ValueError('max_order < 0: {}'.format(max_order))
    result = [amplitude(x, variable_axis=variable_axis, keepdims=keepdims)]
    if max_order > 0:
        result.append(instantaneous_frequency(
            x, dt=dt, edge_order=edge_order, variable_axis=variable_axis, keepdims=keepdims))
    if max_order > 1:
        result.append(instantaneous_bandwidth(
            x, dt=dt, edge_order=edge_order, variable_axis=variable_axis, keepdims=keepdims))
    if max_order > 2:
        if variable_axis is not None:
            result.append(instantaneous_curvature(
                x, dt=dt, edge_order=edge_order, variable_axis=variable_axis, keepdims=keepdims))
            if max_order > 3:
                raise ValueError('max_order cannot be greater than 3 for multivariate moments')
            return result
        eta_diff = np.gradient(result[1] - 1j * result[2], edge_order=edge_order) / dt
        poly_args = [result[2]]
        for current_order in range(2, max_order + 1):
            poly_args.append(1j * eta_diff)
            eta_diff = np.gradient(eta_diff, edge_order=edge_order) / dt
        assert(len(poly_args) == max_order)
        result.extend(_bell_polynomial(*poly_args))
        assert(len(result) == max_order + 1)
    return result
