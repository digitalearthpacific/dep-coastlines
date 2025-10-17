import numpy as np
import xarray as xr
from dep_coastlines.raster.cleaning import smooth_gaussian, remove_disconnected_land


def test_smooth_gaussian_basic():
    """Test that a simple 3x3 array is smoothed without errors."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "y": np.arange(3),
            "x": np.arange(3),
        },
    )
    result = smooth_gaussian(da)

    # Result should be same shape
    assert result.shape == da.shape
    # Values should be float
    assert np.issubdtype(result.dtype, np.floating)
    # Smoothing should change values
    assert not np.allclose(result, da)


def test_smooth_gaussian_nan_handling():
    """Test that NaNs are handled correctly."""
    data = np.array([[1, np.nan, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "y": np.arange(3),
            "x": np.arange(3),
        },
    )
    result = smooth_gaussian(da)

    # Output should have same shape
    assert result.shape == da.shape
    # Output should not be NaN
    assert not np.isnan(result[1, 1])


def test_smooth_gaussian_sigma_effect():
    """Test that sigma affects smoothing."""
    data = np.random.rand(5, 5)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "y": np.arange(5),
            "x": np.arange(5),
        },
    )

    result_low_sigma = smooth_gaussian(da, sigma=0.1)
    result_high_sigma = smooth_gaussian(da, sigma=1.0)

    # Higher sigma should produce more smoothed (less variance) output
    assert np.var(result_high_sigma) < np.var(result_low_sigma)


def test_smooth_gaussian_constant_array():
    """Smoothing a constant array should return the same value."""
    data = np.ones((3, 3))
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "y": np.arange(3),
            "x": np.arange(3),
        },
    )
    result = smooth_gaussian(da)

    assert result[1, 1] == 1


def test_remove_disconnected_land_basic():
    da = (
        xr.DataArray(
            np.ones((3, 3)),
            dims=("y", "x"),
            coords={
                "y": np.arange(3),
                "x": np.arange(3),
            },
        )
        == 1
    )
    result = remove_disconnected_land(da, da)

    assert result.shape == da.shape
    assert np.issubdtype(result.dtype, bool)
    assert isinstance(result, xr.DataArray)


def test_remove_disconnected_land():
    certain_land = np.array(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    certain_land_da = (
        xr.DataArray(
            certain_land,
            dims=("y", "x"),
            coords={
                "y": np.arange(5),
                "x": np.arange(5),
            },
        )
        == 1
    )

    candidate_land = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ]
    )

    candidate_land_da = (
        xr.DataArray(
            candidate_land,
            dims=("y", "x"),
            coords={
                "y": np.arange(5),
                "x": np.arange(5),
            },
        )
        == 1
    )

    presumed_result = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    presumed_result_da = (
        xr.DataArray(
            presumed_result,
            dims=("y", "x"),
            coords={
                "y": np.arange(5),
                "x": np.arange(5),
            },
        )
        == 1
    )

    result = remove_disconnected_land(certain_land_da, candidate_land_da)
    assert (result == presumed_result_da).all()
