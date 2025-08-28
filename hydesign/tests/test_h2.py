# -*- coding: utf-8 -*-
"""
Test module for h2 functionality (hydrogen production)

Created for increasing test coverage
"""
import numpy as np
import openmdao.api as om
import pytest

from hydesign.h2.h2 import BiogasH2


def test_biogas_h2_initialization():
    """Test BiogasH2 component initialization."""
    N_time = 24
    heat_mwht_per_kg_h2 = 0.05  # MW*h/kg
    biogas_h2_mass_ratio = 2.5  # kg biogas per kg H2
    water_h2_mass_ratio = 9.0  # kg water per kg H2
    co2_h2_mass_ratio = 22.0  # kg CO2 per kg H2
    
    comp = BiogasH2(
        N_time=N_time,
        heat_mwht_per_kg_h2=heat_mwht_per_kg_h2,
        biogas_h2_mass_ratio=biogas_h2_mass_ratio,
        water_h2_mass_ratio=water_h2_mass_ratio,
        co2_h2_mass_ratio=co2_h2_mass_ratio
    )
    
    # Check initialization
    assert comp.N_time == N_time
    assert comp.heat_mwht_per_kg_h2 == heat_mwht_per_kg_h2
    assert comp.biogas_h2_mass_ratio == biogas_h2_mass_ratio
    assert comp.water_h2_mass_ratio == water_h2_mass_ratio
    assert comp.co2_h2_mass_ratio == co2_h2_mass_ratio


def test_biogas_h2_setup():
    """Test BiogasH2 component setup method."""
    comp = BiogasH2(
        N_time=24,
        heat_mwht_per_kg_h2=0.05,
        biogas_h2_mass_ratio=2.5,
        water_h2_mass_ratio=9.0,
        co2_h2_mass_ratio=22.0
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('h2', comp)
    prob.setup()
    
    # Check that inputs and outputs are properly set up
    inputs = prob.model.h2.list_inputs(out_stream=None)
    outputs = prob.model.h2.list_outputs(out_stream=None)
    
    # Verify expected inputs exist
    input_names = [inp[0] for inp in inputs]
    expected_inputs = ['max_solar_flux_biogas_h2_t']
    for expected in expected_inputs:
        assert f'h2.{expected}' in input_names
    
    # Verify expected outputs exist  
    output_names = [out[0] for out in outputs]
    expected_outputs = [
        'biogas_h2_mass_ratio', 'water_h2_mass_ratio', 
        'co2_h2_mass_ratio', 'heat_mwht_per_kg_h2',
        'max_solar_flux_dni_reactor_biogas_h2_t'
    ]
    for expected in expected_outputs:
        assert f'h2.{expected}' in output_names


def test_biogas_h2_compute():
    """Test BiogasH2 component computation."""
    N_time = 12
    heat_mwht_per_kg_h2 = 0.055
    biogas_h2_mass_ratio = 2.8
    water_h2_mass_ratio = 9.5
    co2_h2_mass_ratio = 22.5
    
    comp = BiogasH2(
        N_time=N_time,
        heat_mwht_per_kg_h2=heat_mwht_per_kg_h2,
        biogas_h2_mass_ratio=biogas_h2_mass_ratio,
        water_h2_mass_ratio=water_h2_mass_ratio,
        co2_h2_mass_ratio=co2_h2_mass_ratio
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('h2', comp)
    prob.setup()
    
    # Set input values - solar flux time series
    solar_flux = np.array([0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 
                          1.5, 1.2, 1.0, 0.8, 0.5, 0.2])
    prob.set_val('h2.max_solar_flux_biogas_h2_t', solar_flux)
    
    # Run the model
    prob.run_model()
    
    # Check outputs
    biogas_ratio_out = prob.get_val('h2.biogas_h2_mass_ratio')
    water_ratio_out = prob.get_val('h2.water_h2_mass_ratio')
    co2_ratio_out = prob.get_val('h2.co2_h2_mass_ratio')
    heat_out = prob.get_val('h2.heat_mwht_per_kg_h2')
    solar_flux_out = prob.get_val('h2.max_solar_flux_dni_reactor_biogas_h2_t')
    
    # Check that outputs match expected values
    assert biogas_ratio_out == biogas_h2_mass_ratio
    assert water_ratio_out == water_h2_mass_ratio
    assert co2_ratio_out == co2_h2_mass_ratio
    assert heat_out == heat_mwht_per_kg_h2
    
    # Check that solar flux passthrough works correctly
    np.testing.assert_array_equal(solar_flux_out, solar_flux)


def test_biogas_h2_different_time_series_lengths():
    """Test BiogasH2 with different time series lengths."""
    for N_time in [8, 24, 48, 168]:  # Different time periods
        comp = BiogasH2(
            N_time=N_time,
            heat_mwht_per_kg_h2=0.05,
            biogas_h2_mass_ratio=2.5,
            water_h2_mass_ratio=9.0,
            co2_h2_mass_ratio=22.0
        )
        
        prob = om.Problem()
        prob.model.add_subsystem('h2', comp)
        prob.setup()
        
        # Create random solar flux time series of correct length
        solar_flux = np.random.rand(N_time) * 2.0  # 0 to 2 MW
        prob.set_val('h2.max_solar_flux_biogas_h2_t', solar_flux)
        
        # Run the model
        prob.run_model()
        
        # Check outputs are reasonable
        solar_flux_out = prob.get_val('h2.max_solar_flux_dni_reactor_biogas_h2_t')
        assert len(solar_flux_out) == N_time
        np.testing.assert_array_equal(solar_flux_out, solar_flux)


def test_biogas_h2_zero_solar_flux():
    """Test BiogasH2 with zero solar flux."""
    comp = BiogasH2(
        N_time=24,
        heat_mwht_per_kg_h2=0.05,
        biogas_h2_mass_ratio=2.5,
        water_h2_mass_ratio=9.0,
        co2_h2_mass_ratio=22.0
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('h2', comp)
    prob.setup()
    
    # Set zero solar flux
    prob.set_val('h2.max_solar_flux_biogas_h2_t', np.zeros(24))
    
    # Run the model
    prob.run_model()
    
    # Check that outputs are still reasonable
    biogas_ratio_out = prob.get_val('h2.biogas_h2_mass_ratio')
    water_ratio_out = prob.get_val('h2.water_h2_mass_ratio')
    co2_ratio_out = prob.get_val('h2.co2_h2_mass_ratio')
    heat_out = prob.get_val('h2.heat_mwht_per_kg_h2')
    solar_flux_out = prob.get_val('h2.max_solar_flux_dni_reactor_biogas_h2_t')
    
    # Mass ratios should still be valid
    assert biogas_ratio_out > 0
    assert water_ratio_out > 0
    assert co2_ratio_out > 0
    assert heat_out > 0
    
    # Solar flux output should be zero
    np.testing.assert_array_equal(solar_flux_out, np.zeros(24))


def test_biogas_h2_high_solar_flux():
    """Test BiogasH2 with high solar flux values."""
    comp = BiogasH2(
        N_time=12,
        heat_mwht_per_kg_h2=0.05,
        biogas_h2_mass_ratio=2.5,
        water_h2_mass_ratio=9.0,
        co2_h2_mass_ratio=22.0
    )
    
    prob = om.Problem()
    prob.model.add_subsystem('h2', comp)
    prob.setup()
    
    # Set high solar flux values
    high_flux = np.ones(12) * 10.0  # 10 MW constant
    prob.set_val('h2.max_solar_flux_biogas_h2_t', high_flux)
    
    # Run the model
    prob.run_model()
    
    # Check that high flux passes through correctly
    solar_flux_out = prob.get_val('h2.max_solar_flux_dni_reactor_biogas_h2_t')
    np.testing.assert_array_equal(solar_flux_out, high_flux)
    
    # Other outputs should be unchanged
    assert prob.get_val('h2.biogas_h2_mass_ratio') == 2.5
    assert prob.get_val('h2.water_h2_mass_ratio') == 9.0
    assert prob.get_val('h2.co2_h2_mass_ratio') == 22.0
    assert prob.get_val('h2.heat_mwht_per_kg_h2') == 0.05


def test_biogas_h2_varying_parameters():
    """Test BiogasH2 with different parameter values."""
    test_cases = [
        {
            'heat_mwht_per_kg_h2': 0.03,
            'biogas_h2_mass_ratio': 2.0,
            'water_h2_mass_ratio': 8.0,
            'co2_h2_mass_ratio': 20.0
        },
        {
            'heat_mwht_per_kg_h2': 0.08,
            'biogas_h2_mass_ratio': 3.0,
            'water_h2_mass_ratio': 10.0,
            'co2_h2_mass_ratio': 25.0
        },
        {
            'heat_mwht_per_kg_h2': 0.1,
            'biogas_h2_mass_ratio': 4.0,
            'water_h2_mass_ratio': 12.0,
            'co2_h2_mass_ratio': 30.0
        }
    ]
    
    for params in test_cases:
        comp = BiogasH2(N_time=6, **params)
        
        prob = om.Problem()
        prob.model.add_subsystem('h2', comp)
        prob.setup()
        
        # Set simple input
        prob.set_val('h2.max_solar_flux_biogas_h2_t', np.ones(6))
        
        # Run the model
        prob.run_model()
        
        # Check that outputs match parameters
        assert prob.get_val('h2.heat_mwht_per_kg_h2') == params['heat_mwht_per_kg_h2']
        assert prob.get_val('h2.biogas_h2_mass_ratio') == params['biogas_h2_mass_ratio']
        assert prob.get_val('h2.water_h2_mass_ratio') == params['water_h2_mass_ratio']
        assert prob.get_val('h2.co2_h2_mass_ratio') == params['co2_h2_mass_ratio']