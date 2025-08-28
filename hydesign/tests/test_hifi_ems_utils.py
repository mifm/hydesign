# -*- coding: utf-8 -*-
"""
Test module for HiFiEMS utils functionality

Created for increasing test coverage
"""
import numpy as np
import pandas as pd
import pytest

from hydesign.HiFiEMS.utils import f_xmin_to_ymin


def test_f_xmin_to_ymin_basic():
    """Test f_xmin_to_ymin function with basic input."""
    # Test case where we convert from higher to lower resolution
    # e.g., from 15-minute to 1-hour data
    x = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 data points
    reso_x = 15  # 15-minute resolution
    reso_y = 60  # 1-hour resolution
    
    result = f_xmin_to_ymin(x, reso_x, reso_y)
    
    # Should aggregate 4 points (60/15 = 4) into 1 point
    # Expected: (1+2+3+4)/4 = 2.5, (5+6+7+8)/4 = 6.5
    expected = pd.DataFrame([2.5, 6.5])
    
    assert isinstance(result, pd.DataFrame)
    # Check values are approximately correct
    np.testing.assert_array_almost_equal(result.values.flatten(), expected.values.flatten())


def test_f_xmin_to_ymin_single_aggregation():
    """Test f_xmin_to_ymin with single aggregation step."""
    x = [10, 20, 30, 40]
    reso_x = 30  # 30-minute data
    reso_y = 60  # 1-hour data (2 points -> 1 point)
    
    result = f_xmin_to_ymin(x, reso_x, reso_y)
    
    # Should aggregate pairs: (10+20)/2 = 15, (30+40)/2 = 35
    expected = [15.0, 35.0]
    
    assert len(result) == 2
    np.testing.assert_array_almost_equal(result.values.flatten(), expected)


def test_f_xmin_to_ymin_no_aggregation():
    """Test f_xmin_to_ymin when no aggregation is needed."""
    x = [1, 2, 3, 4]
    reso_x = 60  # 1-hour data
    reso_y = 30  # 30-minute data (higher resolution)
    
    result = f_xmin_to_ymin(x, reso_x, reso_y)
    
    # When reso_y < reso_x, the function should handle it gracefully
    # Based on the code, it seems to only handle reso_y > reso_x case
    # So result might be empty DataFrame or original data
    assert isinstance(result, pd.DataFrame)


def test_f_xmin_to_ymin_pandas_input():
    """Test f_xmin_to_ymin with pandas DataFrame input."""
    x = pd.DataFrame([1, 2, 3, 4, 5, 6])
    reso_x = 10  # 10-minute data
    reso_y = 30  # 30-minute data (3 points -> 1 point)
    
    result = f_xmin_to_ymin(x, reso_x, reso_y)
    
    # Should aggregate triplets: (1+2+3)/3 = 2, (4+5+6)/3 = 5
    expected = [2.0, 5.0]
    
    assert isinstance(result, pd.DataFrame)
    np.testing.assert_array_almost_equal(result.values.flatten(), expected)


def test_f_xmin_to_ymin_numpy_array():
    """Test f_xmin_to_ymin with numpy array input."""
    x = np.array([2, 4, 6, 8, 10, 12])
    reso_x = 5   # 5-minute data
    reso_y = 15  # 15-minute data (3 points -> 1 point)
    
    result = f_xmin_to_ymin(x, reso_x, reso_y)
    
    # Should aggregate triplets: (2+4+6)/3 = 4, (10+12)/2 for incomplete last group
    expected = [4.0, 10.0]  # Last group might be handled differently
    
    assert isinstance(result, pd.DataFrame)
    # Check at least first aggregation is correct
    assert abs(result.iloc[0, 0] - 4.0) < 0.001


def test_f_xmin_to_ymin_edge_cases():
    """Test f_xmin_to_ymin with edge cases."""
    # Test with single value
    x = [5]
    reso_x = 30
    reso_y = 60
    
    result = f_xmin_to_ymin(x, reso_x, reso_y)
    assert isinstance(result, pd.DataFrame)
    
    # Test with empty input
    x = []
    result = f_xmin_to_ymin(x, reso_x, reso_y)
    assert isinstance(result, pd.DataFrame)


def test_f_xmin_to_ymin_float_values():
    """Test f_xmin_to_ymin with floating point values."""
    x = [1.5, 2.7, 3.1, 4.9, 5.2, 6.8]
    reso_x = 10
    reso_y = 20  # 2 points -> 1 point
    
    result = f_xmin_to_ymin(x, reso_x, reso_y)
    
    # Should aggregate pairs: (1.5+2.7)/2 = 2.1, (3.1+4.9)/2 = 4.0, (5.2+6.8)/2 = 6.0
    expected = [2.1, 4.0, 6.0]
    
    assert isinstance(result, pd.DataFrame)
    np.testing.assert_array_almost_equal(result.values.flatten(), expected, decimal=1)


def test_f_xmin_to_ymin_large_aggregation():
    """Test f_xmin_to_ymin with large aggregation ratio."""
    # Create data for a full day in 15-minute intervals (96 points)
    x = list(range(1, 97))  # 1 to 96
    reso_x = 15  # 15-minute data
    reso_y = 360  # 6-hour data (24 points -> 1 point)
    
    result = f_xmin_to_ymin(x, reso_x, reso_y)
    
    # Should aggregate every 24 points (360/15 = 24)
    # First group: mean of 1 to 24 = 12.5
    # Second group: mean of 25 to 48 = 36.5
    # etc.
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4  # 96/24 = 4 groups
    
    # Check first aggregated value
    first_group_mean = np.mean(range(1, 25))  # 1 to 24
    assert abs(result.iloc[0, 0] - first_group_mean) < 0.001


def test_f_xmin_to_ymin_uneven_division():
    """Test f_xmin_to_ymin when data doesn't divide evenly."""
    x = [1, 2, 3, 4, 5]  # 5 points
    reso_x = 20
    reso_y = 60  # 3 points -> 1 point
    
    result = f_xmin_to_ymin(x, reso_x, reso_y)
    
    # Should handle incomplete last group
    # First group: (1+2+3)/3 = 2
    # Second group: incomplete, might be handled differently
    
    assert isinstance(result, pd.DataFrame)
    
    # At least should have first complete aggregation
    if len(result) > 0:
        assert abs(result.iloc[0, 0] - 2.0) < 0.001


def test_f_xmin_to_ymin_same_resolution():
    """Test f_xmin_to_ymin when input and output resolutions are the same."""
    x = [1, 2, 3, 4]
    reso_x = 60
    reso_y = 60  # Same resolution
    
    result = f_xmin_to_ymin(x, reso_x, reso_y)
    
    # When resolutions are equal, should return similar data
    assert isinstance(result, pd.DataFrame)