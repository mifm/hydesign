# -*- coding: utf-8 -*-
"""
Test module for costs functionality

Created for increasing test coverage
"""
import numpy as np
import openmdao.api as om
import pytest

from hydesign.costs.costs import wpp_cost


def test_wpp_cost_initialization():
    """Test wpp_cost component initialization."""
    wind_turbine_cost = 1200000  # Euro/MW
    wind_civil_works_cost = 300000  # Euro/MW
    wind_fixed_onm_cost = 25000  # Euro/MW/year
    wind_variable_onm_cost = 10  # Euro/MWh
    d_ref = 90.0  # m
    hh_ref = 80.0  # m
    p_rated_ref = 2.0  # MW
    N_time = 8760  # hours in a year
    
    comp = wpp_cost(
        wind_turbine_cost=wind_turbine_cost,
        wind_civil_works_cost=wind_civil_works_cost,
        wind_fixed_onm_cost=wind_fixed_onm_cost,
        wind_variable_onm_cost=wind_variable_onm_cost,
        d_ref=d_ref,
        hh_ref=hh_ref,
        p_rated_ref=p_rated_ref,
        N_time=N_time
    )
    
    # Check that initialization was successful
    assert comp.wind_turbine_cost == wind_turbine_cost
    assert comp.wind_civil_works_cost == wind_civil_works_cost
    assert comp.wind_fixed_onm_cost == wind_fixed_onm_cost
    assert comp.wind_variable_onm_cost == wind_variable_onm_cost
    assert comp.d_ref == d_ref
    assert comp.hh_ref == hh_ref
    assert comp.p_rated_ref == p_rated_ref
    assert comp.N_time == N_time


def test_wpp_cost_setup():
    """Test wpp_cost component setup method."""
    comp = wpp_cost(
        wind_turbine_cost=1200000,
        wind_civil_works_cost=300000,
        wind_fixed_onm_cost=25000,
        wind_variable_onm_cost=10,
        d_ref=90.0,
        hh_ref=80.0,
        p_rated_ref=2.0,
        N_time=8760
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('cost', comp)
    prob.setup()
    
    # Check that inputs and outputs are properly set up
    inputs = prob.model.cost.list_inputs(out_stream=None)
    outputs = prob.model.cost.list_outputs(out_stream=None)
    
    # Verify expected inputs exist
    input_names = [inp[0] for inp in inputs]
    expected_inputs = ['Nwt', 'Awpp', 'hh', 'd', 'p_rated', 'wind_t']
    for expected in expected_inputs:
        assert f'cost.{expected}' in input_names
    
    # Verify expected outputs exist
    output_names = [out[0] for out in outputs]
    expected_outputs = ['CAPEX_w', 'OPEX_w']
    for expected in expected_outputs:
        assert f'cost.{expected}' in output_names


def test_wpp_cost_compute():
    """Test wpp_cost component computation."""
    comp = wpp_cost(
        wind_turbine_cost=1200000,
        wind_civil_works_cost=300000,
        wind_fixed_onm_cost=25000,
        wind_variable_onm_cost=10,
        d_ref=90.0,
        hh_ref=80.0,
        p_rated_ref=2.0,
        N_time=24  # Small time series for testing
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('cost', comp)
    prob.setup()
    
    # Set input values
    prob.set_val('cost.Nwt', 5)  # 5 wind turbines
    prob.set_val('cost.Awpp', 10.0)  # 10 km^2
    prob.set_val('cost.hh', 85.0)  # 85 m hub height
    prob.set_val('cost.d', 95.0)  # 95 m diameter
    prob.set_val('cost.p_rated', 2.5)  # 2.5 MW rated power
    
    # Create a simple power time series (24 hours)
    wind_power = np.array([1.0, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1, 0.0,
                          0.0, 0.2, 0.5, 0.8, 1.2, 1.5, 1.8, 2.0,
                          2.2, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6])
    prob.set_val('cost.wind_t', wind_power)
    
    # Run the model
    prob.run_model()
    
    # Check outputs
    capex = prob.get_val('cost.CAPEX_w')
    opex = prob.get_val('cost.OPEX_w')
    
    # CAPEX should be positive
    assert capex > 0
    # OPEX should be positive  
    assert opex > 0
    
    # Basic sanity checks
    # CAPEX should scale with number of turbines
    assert isinstance(capex, (int, float, np.ndarray))
    assert isinstance(opex, (int, float, np.ndarray))


def test_wpp_cost_different_intervals_per_hour():
    """Test wpp_cost with different intervals_per_hour."""
    comp = wpp_cost(
        wind_turbine_cost=1200000,
        wind_civil_works_cost=300000,
        wind_fixed_onm_cost=25000,
        wind_variable_onm_cost=10,
        d_ref=90.0,
        hh_ref=80.0,
        p_rated_ref=2.0,
        N_time=48,
        intervals_per_hour=2  # 30-minute intervals
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('cost', comp)
    prob.setup()
    
    # Set input values
    prob.set_val('cost.Nwt', 3)
    prob.set_val('cost.Awpp', 5.0)
    prob.set_val('cost.hh', 80.0)
    prob.set_val('cost.d', 90.0)
    prob.set_val('cost.p_rated', 2.0)
    
    # Create power time series for 48 intervals (24 hours * 2 intervals/hour)
    wind_power = np.random.rand(48) * 2.0  # Random power between 0 and 2 MW
    prob.set_val('cost.wind_t', wind_power)
    
    # Run the model
    prob.run_model()
    
    # Check outputs are reasonable
    capex = prob.get_val('cost.CAPEX_w')
    opex = prob.get_val('cost.OPEX_w')
    
    assert capex > 0
    assert opex > 0


def test_wpp_cost_scaling_with_turbine_count():
    """Test that costs scale appropriately with number of turbines."""
    def run_cost_model(n_turbines):
        comp = wpp_cost(
            wind_turbine_cost=1200000,
            wind_civil_works_cost=300000,
            wind_fixed_onm_cost=25000,
            wind_variable_onm_cost=10,
            d_ref=90.0,
            hh_ref=80.0,
            p_rated_ref=2.0,
            N_time=24
        )
        
        prob = om.Problem()
        prob.model.add_subsystem('cost', comp)
        prob.setup()
        
        prob.set_val('cost.Nwt', n_turbines)
        prob.set_val('cost.Awpp', 10.0)
        prob.set_val('cost.hh', 80.0)
        prob.set_val('cost.d', 90.0)
        prob.set_val('cost.p_rated', 2.0)
        prob.set_val('cost.wind_t', np.ones(24) * 1.5)  # Constant 1.5 MW
        
        prob.run_model()
        
        return prob.get_val('cost.CAPEX_w'), prob.get_val('cost.OPEX_w')
    
    # Test with different turbine counts
    capex_1, opex_1 = run_cost_model(1)
    capex_2, opex_2 = run_cost_model(2)
    
    # CAPEX should roughly scale with number of turbines
    assert capex_2 > capex_1
    # OPEX should also scale with number of turbines
    assert opex_2 > opex_1


def test_wpp_cost_zero_power_time_series():
    """Test wpp_cost with zero power time series."""
    comp = wpp_cost(
        wind_turbine_cost=1200000,
        wind_civil_works_cost=300000,
        wind_fixed_onm_cost=25000,
        wind_variable_onm_cost=10,
        d_ref=90.0,
        hh_ref=80.0,
        p_rated_ref=2.0,
        N_time=24
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('cost', comp)
    prob.setup()
    
    prob.set_val('cost.Nwt', 2)
    prob.set_val('cost.Awpp', 5.0)
    prob.set_val('cost.hh', 80.0)
    prob.set_val('cost.d', 90.0)
    prob.set_val('cost.p_rated', 2.0)
    prob.set_val('cost.wind_t', np.zeros(24))  # No wind power
    
    prob.run_model()
    
    capex = prob.get_val('cost.CAPEX_w')
    opex = prob.get_val('cost.OPEX_w')
    
    # CAPEX should still be positive (installation costs)
    assert capex > 0
    # OPEX might be lower but should still be positive (fixed costs)
    assert opex >= 0