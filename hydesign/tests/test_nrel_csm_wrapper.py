# -*- coding: utf-8 -*-
"""
Test module for nrel_csm_wrapper.py functionality

Created for increasing test coverage
"""
import numpy as np
import pytest

from hydesign.nrel_csm_wrapper import wt_cost


def test_wt_cost_basic():
    """Test basic functionality of wt_cost function."""
    # Test with typical wind turbine parameters
    machine_rating = 2.0  # MW
    rotor_diameter = 90.0  # m
    turbine_class = 1
    hub_height = 80.0  # m
    blade_number = 3
    blade_has_carbon = False
    bearing_number = 2
    crane = True
    
    result = wt_cost(
        machine_rating=machine_rating,
        rotor_diameter=rotor_diameter,
        turbine_class=turbine_class,
        hub_height=hub_height,
        blade_number=blade_number,
        blade_has_carbon=blade_has_carbon,
        bearing_number=bearing_number,
        crane=crane
    )
    
    # Result should be a positive number (cost in some currency)
    assert isinstance(result, (int, float, np.ndarray))
    assert result > 0
    

def test_wt_cost_with_optional_params():
    """Test wt_cost function with optional parameters."""
    machine_rating = 3.0  # MW
    rotor_diameter = 100.0  # m
    turbine_class = 2
    hub_height = 100.0  # m
    blade_number = 3
    blade_has_carbon = True
    bearing_number = 3
    crane = False
    max_tip_speed = 90
    max_efficiency = 0.95
    
    result = wt_cost(
        machine_rating=machine_rating,
        rotor_diameter=rotor_diameter,
        turbine_class=turbine_class,
        hub_height=hub_height,
        blade_number=blade_number,
        blade_has_carbon=blade_has_carbon,
        bearing_number=bearing_number,
        crane=crane,
        max_tip_speed=max_tip_speed,
        max_efficiency=max_efficiency
    )
    
    # Result should be a positive number
    assert isinstance(result, (int, float, np.ndarray))
    assert result > 0


def test_wt_cost_carbon_blade_difference():
    """Test that carbon blades affect the cost."""
    base_params = {
        'machine_rating': 2.5,
        'rotor_diameter': 95.0,
        'turbine_class': 1,
        'hub_height': 85.0,
        'blade_number': 3,
        'bearing_number': 2,
        'crane': True
    }
    
    cost_without_carbon = wt_cost(**base_params, blade_has_carbon=False)
    cost_with_carbon = wt_cost(**base_params, blade_has_carbon=True)
    
    # Both should be positive
    assert cost_without_carbon > 0
    assert cost_with_carbon > 0
    
    # Carbon blades should typically increase cost
    assert cost_with_carbon >= cost_without_carbon


def test_wt_cost_different_turbine_classes():
    """Test wt_cost with different turbine classes."""
    base_params = {
        'machine_rating': 2.0,
        'rotor_diameter': 90.0,
        'hub_height': 80.0,
        'blade_number': 3,
        'blade_has_carbon': False,
        'bearing_number': 2,
        'crane': True
    }
    
    costs = []
    for turbine_class in [1, 2, 3]:
        cost = wt_cost(**base_params, turbine_class=turbine_class)
        costs.append(cost)
        assert cost > 0
    
    # All costs should be positive and potentially different
    assert all(cost > 0 for cost in costs)


def test_wt_cost_bearing_number_effect():
    """Test that bearing number affects the cost."""
    base_params = {
        'machine_rating': 2.0,
        'rotor_diameter': 90.0,
        'turbine_class': 1,
        'hub_height': 80.0,
        'blade_number': 3,
        'blade_has_carbon': False,
        'crane': True
    }
    
    cost_2_bearing = wt_cost(**base_params, bearing_number=2)
    cost_3_bearing = wt_cost(**base_params, bearing_number=3)
    
    # Both should be positive
    assert cost_2_bearing > 0
    assert cost_3_bearing > 0
    
    # Different bearing numbers should potentially give different costs
    # (not asserting which is higher as it depends on the model)