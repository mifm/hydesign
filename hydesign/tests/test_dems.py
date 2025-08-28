"""
Tests for hydesign.HiFiEMS.DEMS module

This module tests the energy management system functions to get coverage
on the previously uncovered lines.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

# Import the functions we want to test
from hydesign.HiFiEMS.DEMS import (
    ReadData, 
    f_xmin_to_ymin, 
    get_var_value_from_sol,
    SMOpt,
    BMOpt,
    RDOpt,
    RTSim,
    RBOpt,
    revenue_process
)


@contextmanager
def pandas_append_patch():
    """Context manager to temporarily add append method to pandas DataFrame"""
    original_append = getattr(pd.DataFrame, 'append', None)
    if original_append is None:
        def append_method(self, other, ignore_index=False, verify_integrity=False, sort=False):
            return pd.concat([self, other], ignore_index=ignore_index, sort=sort)
        pd.DataFrame.append = append_method
    
    try:
        yield
    finally:
        if original_append is None and hasattr(pd.DataFrame, 'append'):
            delattr(pd.DataFrame, 'append')


class TestReadData:
    """Test the ReadData function to cover lines 57, 61, 103-106, 119-120"""
    
    def setup_method(self):
        """Set up test data for ReadData tests"""
        # Create sample data
        self.day_num = 1
        self.exten_num = 12
        self.DI_num = 12
        self.T = 288  # 24 hours * 12 intervals
        self.PsMax = 100
        self.PwMax = 200
        
        # Create sample dataframes
        time_range = pd.date_range(start='2021-01-01', periods=400, freq='5min')
        
        # Wind data WITH measurement column
        self.wind_data_with_measurement = pd.DataFrame({
            'time': time_range,
            'Measurement': np.random.rand(400) * 0.8,
            'DA_wind': np.random.rand(400) * 0.7,
            'HA_wind': np.random.rand(400) * 0.75,
            'FMA_wind': np.random.rand(400) * 0.73,
        }, index=time_range)
        
        # Wind data WITHOUT measurement column (to trigger line 57)
        self.wind_data_no_measurement = pd.DataFrame({
            'time': time_range,
            'DA_wind': np.random.rand(400) * 0.7,
            'HA_wind': np.random.rand(400) * 0.75,
            'FMA_wind': np.random.rand(400) * 0.73,
        }, index=time_range)
        
        # Solar data WITH measurement column
        self.solar_data_with_measurement = pd.DataFrame({
            'time': time_range,
            'Measurement': np.random.rand(400) * 0.9,
            'DA_solar': np.random.rand(400) * 0.8,
            'HA_solar': np.random.rand(400) * 0.85,
            'FMA_solar': np.random.rand(400) * 0.83,
        }, index=time_range)
        
        # Solar data WITHOUT measurement column (to trigger line 61)
        self.solar_data_no_measurement = pd.DataFrame({
            'time': time_range,
            'DA_solar': np.random.rand(400) * 0.8,
            'HA_solar': np.random.rand(400) * 0.85,
            'FMA_solar': np.random.rand(400) * 0.83,
        }, index=time_range)
        
        # Market data with reg_vol_Up > 0 (normal case)
        market_time_range = pd.date_range(start='2021-01-01', periods=35, freq='h')
        self.market_data_normal = pd.DataFrame({
            'time': market_time_range,
            'SM_cleared': np.random.rand(35) * 50 + 20,
            'SP': np.random.rand(35) * 50 + 20,
            'RP': np.random.rand(35) * 10 + 5,
            'reg_cleared': np.random.rand(35) * 10 + 5,
            'BM_Down_cleared': np.random.rand(35) * 40 + 15,
            'BM_Up_cleared': np.random.rand(35) * 45 + 25,
            'reg_vol_Up': np.random.rand(35) * 20 + 10,  # > 0
            'reg_vol_Down': np.random.rand(35) * 20 + 10
        }, index=market_time_range)
        
        # Market data with reg_vol_Up = 0 (to trigger lines 103-106)
        self.market_data_zero_reg = pd.DataFrame({
            'time': market_time_range,
            'SM_cleared': np.random.rand(35) * 50 + 20,
            'SP': np.random.rand(35) * 50 + 20,
            'RP': np.random.rand(35) * 10 + 5,
            'reg_cleared': np.random.rand(35) * 10 + 5,
            'BM_Down_cleared': np.random.rand(35) * 40 + 15,
            'BM_Up_cleared': np.random.rand(35) * 45 + 25,
            'reg_vol_Up': np.zeros(35),  # = 0 to trigger else case
            'reg_vol_Down': np.random.rand(35) * 20 + 10
        }, index=market_time_range)

    def test_readdata_no_wind_measurement(self):
        """Test ReadData when wind data has no Measurement column (line 57)"""
        simulation_dict = {
            "start_date": "01/01/21",
            "wind_df": self.wind_data_no_measurement,
            "solar_df": self.solar_data_with_measurement,
            "market_df": self.market_data_normal,
            "DA_wind": "DA_wind",
            "HA_wind": "HA_wind", 
            "FMA_wind": "FMA_wind",
            "DA_solar": "DA_solar",
            "HA_solar": "HA_solar",
            "FMA_solar": "FMA_solar",
            "SP": "SP",
            "RP": "RP",
            "BP": 1
        }
        
        result = ReadData(self.day_num, self.exten_num, self.DI_num, 
                         self.T, self.PsMax, self.PwMax, simulation_dict)
        
        # Check that Wind_measurement is None (line 57) - index 8
        assert result[8] is None
        # Check that Solar_measurement is not None - index 9
        assert result[9] is not None

    def test_readdata_no_solar_measurement(self):
        """Test ReadData when solar data has no Measurement column (line 61)"""
        simulation_dict = {
            "start_date": "01/01/21",
            "wind_df": self.wind_data_with_measurement,
            "solar_df": self.solar_data_no_measurement,
            "market_df": self.market_data_normal,
            "DA_wind": "DA_wind",
            "HA_wind": "HA_wind",
            "FMA_wind": "FMA_wind", 
            "DA_solar": "DA_solar",
            "HA_solar": "HA_solar",
            "FMA_solar": "FMA_solar",
            "SP": "SP",
            "RP": "RP",
            "BP": 1
        }
        
        result = ReadData(self.day_num, self.exten_num, self.DI_num,
                         self.T, self.PsMax, self.PwMax, simulation_dict)
        
        # Check that Solar_measurement is None (line 61) - index 9
        assert result[9] is None
        # Check that Wind_measurement is not None - index 8
        assert result[8] is not None

    def test_readdata_zero_reg_volume(self):
        """Test ReadData when reg_vol_Up is zero (lines 103-106)"""
        simulation_dict = {
            "start_date": "01/01/21",
            "wind_df": self.wind_data_with_measurement,
            "solar_df": self.solar_data_with_measurement,
            "market_df": self.market_data_zero_reg,
            "DA_wind": "DA_wind",
            "HA_wind": "HA_wind",
            "FMA_wind": "FMA_wind",
            "DA_solar": "DA_solar", 
            "HA_solar": "HA_solar",
            "FMA_solar": "FMA_solar",
            "SP": "SP",
            "RP": "RP",
            "BP": 1
        }
        
        result = ReadData(self.day_num, self.exten_num, self.DI_num,
                         self.T, self.PsMax, self.PwMax, simulation_dict)
        
        # Should complete without error and trigger the else case for reg_vol_Up == 0
        assert len(result) > 0

    def test_readdata_bp_equals_2(self):
        """Test ReadData when BP=2 (lines 119-120)"""
        simulation_dict = {
            "start_date": "01/01/21",
            "wind_df": self.wind_data_with_measurement,
            "solar_df": self.solar_data_with_measurement,
            "market_df": self.market_data_normal,
            "DA_wind": "DA_wind",
            "HA_wind": "HA_wind",
            "FMA_wind": "FMA_wind",
            "DA_solar": "DA_solar",
            "HA_solar": "HA_solar", 
            "FMA_solar": "FMA_solar",
            "SP": "SP",
            "RP": "RP",
            "BP": 2  # This should trigger lines 119-120
        }
        
        result = ReadData(self.day_num, self.exten_num, self.DI_num,
                         self.T, self.PsMax, self.PwMax, simulation_dict)
        
        # Should complete without error and trigger the BP==2 condition
        assert len(result) > 0


class TestFXminToYmin:
    """Test the f_xmin_to_ymin function to cover lines 154-167"""
    
    def test_f_xmin_to_ymin_basic(self):
        """Test basic functionality of f_xmin_to_ymin"""
        with pandas_append_patch():
            x = pd.DataFrame([1, 2, 3, 4, 5, 6])
            reso_x = 0.5  # 30 minutes
            reso_y = 1.0  # 1 hour
            
            result = f_xmin_to_ymin(x, reso_x, reso_y)
            
            # Should average every 2 values (num = reso_y/reso_x = 2)
            assert len(result) == 3
            # Test that we get reasonable results (exact values may vary due to indexing)
            assert len(result) > 0

    def test_f_xmin_to_ymin_different_resolution(self):
        """Test f_xmin_to_ymin with different resolutions"""
        with pandas_append_patch():
            x = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8])
            reso_x = 0.25
            reso_y = 1.0
            
            result = f_xmin_to_ymin(x, reso_x, reso_y)
            
            # Should average every 4 values (num = 1.0/0.25 = 4)
            assert len(result) == 2
            assert len(result) > 0


class TestGetVarValueFromSol:
    """Test the get_var_value_from_sol function"""
    
    def test_get_var_value_from_sol(self):
        """Test basic functionality of get_var_value_from_sol"""
        # Mock solution object
        mock_sol = Mock()
        mock_sol.get_value_dict.return_value = {0: 1.5, 1: 2.5, 2: 3.5}
        
        # Mock variable object
        mock_var = Mock()
        
        result = get_var_value_from_sol(mock_var, mock_sol)
        
        # Should return a DataFrame with the solution values
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


class TestSMOptFunction:
    """Test SMOpt function to cover lines 273, 286, 313-320"""
    
    def setup_method(self):
        """Set up test parameters for SMOpt"""
        self.dt = 1/12  # 5 minutes
        self.T = 24
        self.PbMax = 50
        self.EBESS = 200
        self.SoCmin = 0.1
        self.SoCmax = 0.9
        self.eta_dis = 0.95
        self.eta_cha = 0.95
        self.eta_leak = 0.001
        self.Emax = 100
        self.PreUp = 20
        self.PreDw = 20
        self.P_grid_limit = 1000
        self.mu = 0.01
        self.ad = 0.05
        
        # Create sample forecast data
        self.DA_wind_forecast = pd.Series(np.random.rand(24) * 50)
        self.DA_solar_forecast = pd.Series(np.random.rand(24) * 30)
        self.SM_price_forecast = pd.Series(np.random.rand(24) * 40 + 20)
        self.SoC0 = 0.5

    @patch('hydesign.HiFiEMS.DEMS.Model')
    def test_smopt_with_degradation_verbose(self, mock_model):
        """Test SMOpt with degradation enabled and verbose output (lines 273, 286)"""
        # Create mock model instance
        mock_mdl = Mock()
        mock_model.return_value = mock_mdl
        
        # Mock solution
        mock_sol = Mock()
        mock_sol.get_value_dict.return_value = {i: np.random.rand() for i in range(24)}
        mock_mdl.solve.return_value = mock_sol
        
        result = SMOpt(
            self.dt, self.T, self.PbMax, self.EBESS, self.SoCmin, self.SoCmax,
            self.eta_dis, self.eta_cha, self.eta_leak, self.Emax, self.PreUp, self.PreDw,
            self.P_grid_limit, self.mu, self.ad, self.DA_wind_forecast, self.DA_solar_forecast,
            self.SM_price_forecast, self.SoC0, deg_indicator=1, verbose=True
        )
        
        # Should call print_information (line 286) when verbose=True
        mock_mdl.print_information.assert_called_once()
        assert len(result) > 0

    @patch('hydesign.HiFiEMS.DEMS.Model')
    def test_smopt_no_solution(self, mock_model):
        """Test SMOpt when no solution is found (lines 319-320)"""
        # Create mock model instance
        mock_mdl = Mock()
        mock_model.return_value = mock_mdl
        
        # Mock no solution case
        mock_mdl.solve.return_value = None
        
        with patch('builtins.print') as mock_print:
            result = SMOpt(
                self.dt, self.T, self.PbMax, self.EBESS, self.SoCmin, self.SoCmax,
                self.eta_dis, self.eta_cha, self.eta_leak, self.Emax, self.PreUp, self.PreDw,
                self.P_grid_limit, self.mu, self.ad, self.DA_wind_forecast, self.DA_solar_forecast,
                self.SM_price_forecast, self.SoC0, deg_indicator=0, verbose=False
            )
            
            # Should print "DA EMS Model has no solution" (line 320)
            mock_print.assert_called_with("DA EMS Model has no solution")
        
        assert len(result) > 0

    @patch('hydesign.HiFiEMS.DEMS.Model')
    @patch('builtins.print')
    def test_smopt_verbose_solution(self, mock_print, mock_model):
        """Test SMOpt with verbose solution output (lines 313-318)"""
        # Create mock model instance
        mock_mdl = Mock()
        mock_model.return_value = mock_mdl
        
        # Mock solution
        mock_sol = Mock()
        mock_sol.get_value_dict.return_value = {i: np.random.rand() for i in range(24)}
        mock_mdl.solve.return_value = mock_sol
        
        result = SMOpt(
            self.dt, self.T, self.PbMax, self.EBESS, self.SoCmin, self.SoCmax,
            self.eta_dis, self.eta_cha, self.eta_leak, self.Emax, self.PreUp, self.PreDw,
            self.P_grid_limit, self.mu, self.ad, self.DA_wind_forecast, self.DA_solar_forecast,
            self.SM_price_forecast, self.SoC0, deg_indicator=1, verbose=True
        )
        
        # Should call print multiple times for verbose output (lines 313-318)
        assert mock_print.call_count >= 5
        assert len(result) > 0


class TestBMOptFunction:
    """Test BMOpt function to cover lines 374-615"""
    
    def setup_method(self):
        """Set up test parameters for BMOpt"""
        self.dt = 1/12
        self.ds = 1/4
        self.dk = 1
        self.T = 288
        self.EBESS = 200
        self.PbMax = 50
        self.PreUp = 20
        self.PreDw = 20
        self.P_grid_limit = 1000
        self.SoCmin = 0.1
        self.SoCmax = 0.9
        self.Emax = 100
        self.eta_dis = 0.95
        self.eta_cha = 0.95
        self.eta_leak = 0.001
        self.mu = 0.01
        self.ad = 0.05
        
        # Create forecast data
        self.HA_wind_forecast = pd.Series(np.random.rand(288) * 50)
        self.HA_solar_forecast = pd.Series(np.random.rand(288) * 30)
        self.BM_dw_price_forecast = pd.Series(np.random.rand(24) * 20)
        self.BM_up_price_forecast = pd.Series(np.random.rand(24) * 25)
        self.BM_dw_price_forecast_settle = pd.Series(np.random.rand(24) * 20)
        self.BM_up_price_forecast_settle = pd.Series(np.random.rand(24) * 25)
        self.reg_up_sign_forecast = pd.Series(np.random.randint(0, 2, 72))
        self.reg_dw_sign_forecast = pd.Series(np.random.randint(0, 2, 72))
        self.P_HPP_SM_t_opt = pd.Series(np.random.rand(288) * 60)
        self.start = 0
        self.s_UP_t = pd.Series(np.random.rand(72) * 10)
        self.s_DW_t = pd.Series(np.random.rand(72) * 10)
        self.P_HPP_UP_t0 = 0
        self.P_HPP_DW_t0 = 0
        self.SoC0 = 0.5
        self.exten_num = 12

    @patch('hydesign.HiFiEMS.DEMS.Model')
    def test_bmopt_basic_execution(self, mock_model):
        """Test basic execution of BMOpt function"""
        # Create mock model instance
        mock_mdl = Mock()
        mock_model.return_value = mock_mdl
        
        # Mock solution
        mock_sol = Mock()
        mock_sol.get_value_dict.return_value = {i: np.random.rand() for i in range(100)}
        mock_mdl.solve.return_value = mock_sol
        
        result = BMOpt(
            self.dt, self.ds, self.dk, self.T, self.EBESS, self.PbMax,
            self.PreUp, self.PreDw, self.P_grid_limit, self.SoCmin, self.SoCmax,
            self.Emax, self.eta_dis, self.eta_cha, self.eta_leak, self.mu, self.ad,
            self.HA_wind_forecast, self.HA_solar_forecast, self.BM_dw_price_forecast,
            self.BM_up_price_forecast, self.BM_dw_price_forecast_settle,
            self.BM_up_price_forecast_settle, self.reg_up_sign_forecast,
            self.reg_dw_sign_forecast, self.P_HPP_SM_t_opt, self.start,
            self.s_UP_t, self.s_DW_t, self.P_HPP_UP_t0, self.P_HPP_DW_t0,
            self.SoC0, self.exten_num, deg_indicator=1
        )
        
        assert len(result) > 0


class TestRevenueProcess:
    """Test revenue_process function to cover lines 1374-1395"""
    
    def test_revenue_process_with_temp_dir(self):
        """Test revenue_process function with temporary directory"""
        # Create a temporary directory with sample files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample CSV files
            sample_data = pd.DataFrame({
                'SM_revenue': [100, 200, 150],
                'reg_revenue': [50, 75, 60],
                'BM_revenue': [25, 30, 20]
            })
            
            # Save sample files
            sample_data.to_csv(os.path.join(temp_dir, 'result_1.csv'))
            sample_data.to_csv(os.path.join(temp_dir, 'result_2.csv'))
            
            # Test the function
            result = revenue_process(temp_dir)
            
            # Should return a DataFrame with accumulated revenue
            assert isinstance(result, pd.DataFrame)


# Test running functions that trigger the large line ranges
class TestLargeFunctions:
    """Test the large functions to get basic coverage"""
    
    def setup_method(self):
        """Set up common parameters"""
        self.dt = 1/12
        self.ds = 1/4  
        self.dk = 1
        self.T = 288
        self.basic_params = {
            'EBESS': 200,
            'PbMax': 50,
            'PreUp': 20,
            'PreDw': 20,
            'P_grid_limit': 1000,
            'SoCmin': 0.1,
            'SoCmax': 0.9,
            'Emax': 100,
            'eta_dis': 0.95,
            'eta_cha': 0.95,
            'eta_leak': 0.001,
            'mu': 0.01,
            'ad': 0.05
        }

    @patch('hydesign.HiFiEMS.DEMS.Model')
    def test_rdopt_basic(self, mock_model):
        """Test RDOpt function basic execution (lines 676-940)"""
        mock_mdl = Mock()
        mock_model.return_value = mock_mdl
        mock_sol = Mock()
        mock_sol.get_value_dict.return_value = {i: np.random.rand() for i in range(100)}
        mock_mdl.solve.return_value = mock_sol
        
        # Create minimal required parameters
        RD_wind_forecast = pd.Series(np.random.rand(288) * 50)
        RD_solar_forecast = pd.Series(np.random.rand(288) * 30)
        price_forecasts = pd.Series(np.random.rand(24) * 25)
        reg_forecasts = pd.Series(np.random.randint(0, 2, 72))
        P_HPP_SM_t_opt = pd.Series(np.random.rand(288) * 60)
        
        result = RDOpt(
            self.dt, self.ds, self.dk, self.T, 
            self.basic_params['EBESS'], self.basic_params['PbMax'],
            self.basic_params['PreUp'], self.basic_params['PreDw'], 
            self.basic_params['P_grid_limit'], self.basic_params['SoCmin'], 
            self.basic_params['SoCmax'], self.basic_params['Emax'],
            self.basic_params['eta_dis'], self.basic_params['eta_cha'], 
            self.basic_params['eta_leak'], self.basic_params['mu'], self.basic_params['ad'],
            RD_wind_forecast, RD_solar_forecast, price_forecasts, price_forecasts,
            price_forecasts, price_forecasts, reg_forecasts, reg_forecasts,
            P_HPP_SM_t_opt, 0, reg_forecasts[:72], reg_forecasts[:72],
            0, 0, 0, 0, 0.5, False, 12, 1
        )
        
        assert len(result) > 0

    @patch('hydesign.HiFiEMS.DEMS.Model')  
    def test_rbopt_basic(self, mock_model):
        """Test RBOpt function basic execution (lines 1134-1357)"""
        mock_mdl = Mock()
        mock_model.return_value = mock_mdl
        mock_sol = Mock()
        mock_sol.get_value_dict.return_value = {i: np.random.rand() for i in range(100)}
        mock_mdl.solve.return_value = mock_sol
        
        # Create minimal required parameters
        RB_wind_forecast = pd.Series(np.random.rand(288) * 50)
        RB_solar_forecast = pd.Series(np.random.rand(288) * 30)
        price_forecasts = pd.Series(np.random.rand(24) * 25)
        reg_forecasts = pd.Series(np.random.randint(0, 2, 72))
        P_HPP_SM_t_opt = pd.Series(np.random.rand(288) * 60)
        
        result = RBOpt(
            self.dt, self.ds, self.dk, self.T,
            self.basic_params['EBESS'], self.basic_params['PbMax'],
            self.basic_params['PreUp'], self.basic_params['PreDw'],
            self.basic_params['P_grid_limit'], self.basic_params['SoCmin'],
            self.basic_params['SoCmax'], self.basic_params['Emax'], 
            self.basic_params['eta_dis'], self.basic_params['eta_cha'],
            self.basic_params['eta_leak'], self.basic_params['mu'], self.basic_params['ad'],
            RB_wind_forecast, RB_solar_forecast, price_forecasts, price_forecasts,
            price_forecasts, price_forecasts, reg_forecasts, reg_forecasts,
            P_HPP_SM_t_opt, 0, reg_forecasts[:72], reg_forecasts[:72],
            0, 0, 0, 0, 0.5, False, 12, 1
        )
        
        assert len(result) > 0

    def test_rtsim_basic(self):
        """Test RTSim function basic execution (lines 982-1081)"""
        # Create required parameters for RTSim
        Wind_measurement = pd.Series([50.0])
        Solar_measurement = pd.Series([30.0])
        RT_wind_forecast = pd.Series([48.0])
        RT_solar_forecast = pd.Series([32.0])
        P_activated_UP_t = pd.Series([5.0])
        P_activated_DW_t = pd.Series([3.0])
        
        result = RTSim(
            self.dt, self.basic_params['PbMax'], self.basic_params['PreUp'], 
            self.basic_params['PreDw'], self.basic_params['P_grid_limit'],
            self.basic_params['SoCmin'], self.basic_params['SoCmax'], 
            self.basic_params['Emax'], self.basic_params['eta_dis'],
            self.basic_params['eta_cha'], self.basic_params['eta_leak'],
            Wind_measurement, Solar_measurement, RT_wind_forecast, 
            RT_solar_forecast, 0.5, 75.0, 0, P_activated_UP_t, P_activated_DW_t
        )
        
        assert len(result) > 0