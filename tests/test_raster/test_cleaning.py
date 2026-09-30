import numpy as np
import xarray as xr
from dep_coastlines.raster.cleaning import (
    fill_with_nearby_dates,
    load_land_additions,
    smooth_gaussian,
    remove_disconnected_land,
)


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

    expected_result_da = (
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
    xr.testing.assert_equal(result, expected_result_da)


def test_fill_with_nearby_dates_dataarray():
    nan = float("nan")
    data = xr.DataArray(
        np.array(
            [
                [nan, 1, nan, nan],
                [2, nan, nan, nan],
                [3, nan, 3, nan],
            ]
        ),
        dims=("z", "year"),
        coords=dict(z=np.arange(4), year=[2001, 2002, 2003]),
    )

    expected_output = xr.DataArray(
        np.array([[2, 1, nan, nan], [2, 1, 3, nan], [3, nan, nan, nan]]),
        dims=("z", "year"),
        coords=dict(z=np.arange(4), year=[2001, 2002, 2003]),
    )

    xr.testing.assert_equal(fill_with_nearby_dates(data), expected_output)


def _water_index_around(x_center: float, y_center: float) -> xr.DataArray:
    """A small (year, y, x) array in EPSG:3832 with 30 m pixels."""
    offsets = np.arange(-50, 50) * 30.0
    water_index = xr.DataArray(
        np.zeros((2, 100, 100)),
        dims=("year", "y", "x"),
        coords=dict(year=["2020", "2021"], y=y_center - offsets, x=x_center + offsets),
    )
    return water_index.rio.write_crs(3832)


def test_load_land_additions_without_additions():
    # Jaluit Atoll, where the file has no polygons
    water_index = _water_index_around(2_180_000, 660_000)
    result = load_land_additions(water_index)
    assert result.dims == ("y", "x")
    assert not result.any()


def test_load_land_additions_with_additions():
    # Hunga Tonga-Hunga Ha'apai, where the file has a polygon
    water_index = _water_index_around(3_852_700, -2_322_600)
    result = load_land_additions(water_index)
    assert result.shape == (100, 100)
    assert result.any()
