# -*- coding: utf-8 -*-
"""
Test module for assembly functionality

Created for increasing test coverage
"""
import os
import tempfile
import pytest
import numpy as np
import yaml

from hydesign.assembly.hpp_assembly import hpp_base, hpp_model


def test_hpp_base_initialization():
    """Test hpp_base initialization with minimal configuration."""
    # Create a temporary yaml file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump({
            'altitude': 100.0,
            'latitude': 55.0,
            'longitude': 12.0,
            'work_dir': './',
            'verbose': True
        }, f)
        temp_file = f.name
    
    try:
        # Test initialization
        hpp = hpp_base(
            sim_pars_fn=temp_file,
            N_time=24,
            wind_deg=True,
            wind_deg_yr=0.5,
            price=50.0
        )
        
        # Check that attributes are set correctly
        assert hpp.sim_pars_fn == temp_file
        assert 'altitude' in hpp.sim_pars
        assert 'latitude' in hpp.sim_pars
        assert 'longitude' in hpp.sim_pars
        assert hpp.sim_pars['altitude'] == 100.0
        assert hpp.sim_pars['latitude'] == 55.0
        assert hpp.sim_pars['longitude'] == 12.0
        
    finally:
        os.unlink(temp_file)


def test_hpp_base_get_defaults():
    """Test hpp_base get_defaults method."""
    # Create minimal temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump({
            'altitude': 100.0,
            'latitude': 55.0,
            'longitude': 12.0
        }, f)
        temp_file = f.name
    
    try:
        hpp = hpp_base(
            sim_pars_fn=temp_file,
            N_time=24,
            wind_deg=True,
            wind_deg_yr=0.5,
            price=50.0
        )
        
        defaults = hpp.get_defaults()
        
        # Check default values
        assert 'work_dir' in defaults
        assert 'max_num_batteries_allowed' in defaults
        assert 'ems_type' in defaults
        assert defaults['work_dir'] == './'
        assert defaults['max_num_batteries_allowed'] == 3
        assert defaults['ems_type'] == 'cplex'
        assert defaults['verbose'] is True
        
    finally:
        os.unlink(temp_file)


def test_hpp_base_check_inputs_valid():
    """Test hpp_base check_inputs method with valid inputs."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump({
            'altitude': 100.0,
            'latitude': 55.0,
            'longitude': 12.0
        }, f)
        temp_file = f.name
    
    try:
        hpp = hpp_base(
            sim_pars_fn=temp_file,
            N_time=24,
            wind_deg=True,
            wind_deg_yr=0.5,
            price=50.0
        )
        
        # Should not raise any exception
        assert hpp.sim_pars is not None
        
    finally:
        os.unlink(temp_file)


def test_hpp_base_check_inputs_missing_required():
    """Test hpp_base check_inputs method with missing required variables."""
    # Create config missing required variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump({
            'altitude': 100.0,
            'latitude': 55.0,
            # Missing longitude
        }, f)
        temp_file = f.name
    
    try:
        with pytest.raises(ValueError) as excinfo:
            hpp_base(
                sim_pars_fn=temp_file,
                N_time=24,
                wind_deg=True,
                wind_deg_yr=0.5,
                price=50.0
            )
        
        assert "longitude" in str(excinfo.value)
        
    finally:
        os.unlink(temp_file)


def test_hpp_base_check_inputs_none_values():
    """Test hpp_base check_inputs method with None values."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump({
            'altitude': None,  # None value should cause error
            'latitude': 55.0,
            'longitude': 12.0
        }, f)
        temp_file = f.name
    
    try:
        with pytest.raises(ValueError) as excinfo:
            hpp_base(
                sim_pars_fn=temp_file,
                N_time=24,
                wind_deg=True,
                wind_deg_yr=0.5,
                price=50.0
            )
        
        assert "altitude" in str(excinfo.value)
        assert "cannot be provided as None" in str(excinfo.value)
        
    finally:
        os.unlink(temp_file)


def test_hpp_base_parameter_override():
    """Test that parameters can be overridden during initialization."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump({
            'altitude': 100.0,
            'latitude': 55.0,
            'longitude': 12.0,
            'max_num_batteries_allowed': 2  # Will be overridden
        }, f)
        temp_file = f.name
    
    try:
        hpp = hpp_base(
            sim_pars_fn=temp_file,
            N_time=24,
            wind_deg=True,
            wind_deg_yr=0.5,
            price=50.0,
            max_num_batteries_allowed=5  # Override value
        )
        
        # Should use overridden value
        assert hpp.sim_pars['max_num_batteries_allowed'] == 5
        
    finally:
        os.unlink(temp_file)


def test_hpp_base_defaults_override():
    """Test that defaults can be overridden."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump({
            'altitude': 100.0,
            'latitude': 55.0,
            'longitude': 12.0
        }, f)
        temp_file = f.name
    
    try:
        custom_defaults = {'ems_type': 'pyomo', 'verbose': False}
        hpp = hpp_base(
            sim_pars_fn=temp_file,
            N_time=24,
            wind_deg=True,
            wind_deg_yr=0.5,
            price=50.0,
            defaults=custom_defaults
        )
        
        # Should use custom defaults
        assert hpp.sim_pars['ems_type'] == 'pyomo'
        assert hpp.sim_pars['verbose'] is False
        
    finally:
        os.unlink(temp_file)


def test_hpp_base_print_design():
    """Test hpp_base print_design method."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump({
            'altitude': 100.0,
            'latitude': 55.0,
            'longitude': 12.0
        }, f)
        temp_file = f.name
    
    try:
        hpp = hpp_base(
            sim_pars_fn=temp_file,
            N_time=24,
            wind_deg=True,
            wind_deg_yr=0.5,
            price=50.0
        )
        
        # Add some dummy attributes for testing
        hpp.list_vars = ['var1', 'var2']
        hpp.list_out_vars = ['out1', 'out2']
        
        x_opt = [1.5, 2.7]
        outs = [10.2, 15.8]
        
        # Should not raise an exception (just prints to stdout)
        hpp.print_design(x_opt, outs)
        
    finally:
        os.unlink(temp_file)


def test_hpp_base_evaluation_in_csv():
    """Test hpp_base evaluation_in_csv method."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump({
            'altitude': 100.0,
            'latitude': 55.0,
            'longitude': 12.0
        }, f)
        temp_file = f.name
    
    try:
        hpp = hpp_base(
            sim_pars_fn=temp_file,
            N_time=24,
            wind_deg=True,
            wind_deg_yr=0.5,
            price=50.0
        )
        
        # Add dummy attributes
        hpp.list_vars = ['var1', 'var2']
        hpp.list_out_vars = ['out1', 'out2']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, 'test_output')
            
            hpp.evaluation_in_csv(
                name_file=output_file,
                longitude=12.0,
                latitude=55.0,
                altitude=100.0,
                x_opt=[1.5, 2.7],
                outs=[10.2, 15.8]
            )
            
            # Check that CSV file was created
            assert os.path.exists(f'{output_file}.csv')
            
    finally:
        os.unlink(temp_file)


def test_hpp_model_initialization():
    """Test hpp_model initialization."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump({
            'altitude': 100.0,
            'latitude': 55.0,
            'longitude': 12.0,
            'work_dir': './',
            'verbose': True
        }, f)
        temp_file = f.name
    
    try:
        model = hpp_model(sim_pars_fn=temp_file)
        
        # Should inherit from hpp_base
        assert hasattr(model, 'sim_pars')
        assert model.sim_pars['altitude'] == 100.0
        assert model.sim_pars['latitude'] == 55.0
        assert model.sim_pars['longitude'] == 12.0
        
    finally:
        os.unlink(temp_file)


def test_hpp_model_with_kwargs():
    """Test hpp_model initialization with kwargs."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump({
            'altitude': 100.0,
            'latitude': 55.0,
            'longitude': 12.0
        }, f)
        temp_file = f.name
    
    try:
        model = hpp_model(
            sim_pars_fn=temp_file,
            max_num_batteries_allowed=4,
            ems_type='pyomo'
        )
        
        # Check that kwargs were applied
        assert model.sim_pars['max_num_batteries_allowed'] == 4
        assert model.sim_pars['ems_type'] == 'pyomo'
        
    finally:
        os.unlink(temp_file)


def test_hpp_base_with_complex_config():
    """Test hpp_base with more complex configuration."""
    complex_config = {
        'altitude': 100.0,
        'latitude': 55.0,
        'longitude': 12.0,
        'work_dir': './',
        'max_num_batteries_allowed': 5,
        'ems_type': 'cplex',
        'weeks_per_season_per_year': 2,
        'seed': 42,
        'verbose': False,
        'name': 'test_hpp',
        'ppa_price': 75.0
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(complex_config, f)
        temp_file = f.name
    
    try:
        hpp = hpp_base(
            sim_pars_fn=temp_file,
            N_time=8760,
            wind_deg=True,
            wind_deg_yr=0.8,
            price=60.0
        )
        
        # Check that all config values were loaded
        for key, value in complex_config.items():
            assert hpp.sim_pars[key] == value
            
        # Check that additional parameters were set
        assert hpp.sim_pars['N_time'] == 8760
        assert hpp.sim_pars['wind_deg'] is True
        assert hpp.sim_pars['wind_deg_yr'] == 0.8
        assert hpp.sim_pars['price'] == 60.0
        
    finally:
        os.unlink(temp_file)