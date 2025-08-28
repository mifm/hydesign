# -*- coding: utf-8 -*-
"""
Additional tests for utils.py functions to increase coverage

Created for increasing test coverage
"""
import numpy as np
import openmdao.api as om
import pytest

from hydesign.utils import get_weights, hybridization_shifted, sample_mean


def test_get_weights_different_grids():
    """Test get_weights function with different grid configurations."""
    # Test with symmetric grid
    grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    xtgt = 0.0
    maxorder = 2
    
    weights = get_weights(grid, xtgt, maxorder)
    
    # Should return a matrix with shape (len(grid), maxorder + 1)
    assert weights.shape == (5, 3)
    
    # Check that central difference weights for first derivative are reasonable
    # For symmetric grid centered at 0, first derivative should have pattern like [-a, -b, 0, b, a]
    first_deriv_weights = weights[:, 1]
    assert abs(first_deriv_weights[2]) < 1e-10  # Center point should be 0 for first derivative


def test_get_weights_asymmetric_grid():
    """Test get_weights with asymmetric grid."""
    grid = np.array([0.0, 0.5, 1.0, 2.0])
    xtgt = 1.0
    maxorder = 1
    
    weights = get_weights(grid, xtgt, maxorder)
    
    assert weights.shape == (4, 2)
    # Function weights (0th derivative) should sum to 1
    assert abs(np.sum(weights[:, 0]) - 1.0) < 1e-10


def test_get_weights_off_center_target():
    """Test get_weights with target point not at grid point."""
    grid = np.array([0.0, 1.0, 2.0])
    xtgt = 0.5  # Between grid points
    maxorder = 1
    
    weights = get_weights(grid, xtgt, maxorder)
    
    assert weights.shape == (3, 2)
    # Function values should sum to 1
    assert abs(np.sum(weights[:, 0]) - 1.0) < 1e-10


def test_get_weights_single_point():
    """Test get_weights with single grid point."""
    grid = np.array([1.0])
    xtgt = 1.0
    maxorder = 0
    
    weights = get_weights(grid, xtgt, maxorder)
    
    assert weights.shape == (1, 1)
    assert abs(weights[0, 0] - 1.0) < 1e-10


def test_get_weights_higher_order():
    """Test get_weights with higher order derivatives."""
    grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    xtgt = 0.0
    maxorder = 4
    
    weights = get_weights(grid, xtgt, maxorder)
    
    assert weights.shape == (5, 5)
    
    # Check that function interpolation weights sum to 1
    assert abs(np.sum(weights[:, 0]) - 1.0) < 1e-10


def test_hybridization_shifted_initialization():
    """Test hybridization_shifted component initialization."""
    N_limit = 5
    life_y = 25
    N_time = 8760
    life_h = 365 * 24 * life_y
    
    comp = hybridization_shifted(
        N_limit=N_limit,
        life_y=life_y,
        N_time=N_time,
        life_h=life_h
    )
    
    assert comp.N_limit == N_limit
    assert comp.life_y == life_y
    assert comp.N_time == N_time
    assert comp.life_h == life_h


def test_hybridization_shifted_setup():
    """Test hybridization_shifted component setup."""
    comp = hybridization_shifted(
        N_limit=5,
        life_y=25,
        N_time=8760,
        life_h=365 * 24 * 25
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('hyb', comp)
    prob.setup()
    
    # Check inputs and outputs
    inputs = prob.model.hyb.list_inputs(out_stream=None)
    outputs = prob.model.hyb.list_outputs(out_stream=None)
    
    input_names = [inp[0] for inp in inputs]
    output_names = [out[0] for out in outputs]
    
    assert 'hyb.delta_life' in input_names
    assert 'hyb.SoH' in input_names
    assert 'hyb.SoH_shifted' in output_names


def test_hybridization_shifted_compute():
    """Test hybridization_shifted component computation."""
    N_limit = 5
    life_y = 2  # Small for testing
    N_time = 48
    life_h = 365 * 24 * life_y
    
    comp = hybridization_shifted(
        N_limit=N_limit,
        life_y=life_y,
        N_time=N_time,
        life_h=life_h
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('hyb', comp)
    prob.setup()
    
    # Set inputs
    delta_life = 1  # Shift by 1 year
    prob.set_val('hyb.delta_life', delta_life)
    
    # Create SoH time series (decreasing battery health)
    soh_values = np.linspace(1.0, 0.8, life_h)
    prob.set_val('hyb.SoH', soh_values)
    
    # Run computation
    prob.run_model()
    
    # Get output
    soh_shifted = prob.get_val('hyb.SoH_shifted')
    
    # Check output length
    expected_length = life_h
    assert len(soh_shifted) == expected_length
    
    # Check that first year is zeros (due to delta_life = 1)
    first_year_hours = 365 * 24
    assert np.allclose(soh_shifted[:first_year_hours], 0.0)
    
    # Check that second year has original data
    second_year_data = soh_shifted[first_year_hours:2*first_year_hours]
    original_first_year = soh_values[:first_year_hours]
    assert np.allclose(second_year_data, original_first_year)


def test_hybridization_shifted_zero_delta():
    """Test hybridization_shifted with zero delta_life."""
    comp = hybridization_shifted(
        N_limit=3,
        life_y=1,
        N_time=24,
        life_h=365 * 24
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('hyb', comp)
    prob.setup()
    
    # Set zero shift
    prob.set_val('hyb.delta_life', 0)
    
    # Simple SoH data
    soh_values = np.ones(365 * 24) * 0.9
    prob.set_val('hyb.SoH', soh_values)
    
    prob.run_model()
    
    soh_shifted = prob.get_val('hyb.SoH_shifted')
    
    # With zero shift, first year should match original data
    original_year_length = 365 * 24
    assert np.allclose(soh_shifted[:original_year_length], soh_values[:original_year_length])


def test_hybridization_shifted_max_delta():
    """Test hybridization_shifted with maximum delta_life."""
    N_limit = 3
    comp = hybridization_shifted(
        N_limit=N_limit,
        life_y=1,
        N_time=24,
        life_h=365 * 24
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('hyb', comp)
    prob.setup()
    
    # Set maximum shift
    prob.set_val('hyb.delta_life', N_limit - 1)  # N_limit - 1 should be maximum
    
    soh_values = np.ones(365 * 24) * 0.85
    prob.set_val('hyb.SoH', soh_values)
    
    prob.run_model()
    
    soh_shifted = prob.get_val('hyb.SoH_shifted')
    
    # Most of the output should be zeros with max shift
    zero_hours = (N_limit - 1) * 365 * 24
    assert np.allclose(soh_shifted[:zero_hours], 0.0)


def test_sample_mean_basic():
    """Test sample_mean function with basic input."""
    # Test with 2D array
    samples = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    
    result = sample_mean(samples)
    
    # Should return mean along axis 0
    expected = np.array([4, 5, 6])  # Mean of each column
    np.testing.assert_array_equal(result, expected)


def test_sample_mean_single_sample():
    """Test sample_mean with single sample."""
    samples = np.array([[1, 2, 3]])
    
    result = sample_mean(samples)
    
    # With single sample, should return that sample
    expected = np.array([1, 2, 3])
    np.testing.assert_array_equal(result, expected)


def test_sample_mean_3d_array():
    """Test sample_mean with 3D array."""
    samples = np.array([[[1, 2], [3, 4]],
                       [[5, 6], [7, 8]]])
    
    result = sample_mean(samples)
    
    # Should average along first axis
    expected = np.array([[3, 4], [5, 6]])
    np.testing.assert_array_equal(result, expected)


def test_sample_mean_float_values():
    """Test sample_mean with floating point values."""
    samples = np.array([[1.1, 2.2],
                       [3.3, 4.4],
                       [5.5, 6.6]])
    
    result = sample_mean(samples)
    
    expected = np.array([3.3, 4.4])  # Mean of each column
    np.testing.assert_array_almost_equal(result, expected)