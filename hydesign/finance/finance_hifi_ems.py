# basic libraries
import numpy as np
import numpy_financial as npf
import openmdao.api as om
import pandas as pd
import scipy as sp

from hydesign.HiFiEMS.utils import _revenue_calculation
from hydesign.openmdao_wrapper import ComponentWrapper
from hydesign.costmodels import (
    calculate_NPV_IRR,
    calculate_WACC,
    calculate_revenues,
    calculate_break_even_PPA_price,
    calculate_CAPEX_phasing,
    get_inflation_index,
)


class finance:
    """Hybrid power plant financial model to estimate the overall profitability of the hybrid power plant.
    It considers different weighted average costs of capital (WACC) for wind, PV and battery. The model calculates
    the yearly cashflow as a function of the average revenue over the year, the tax rate and WACC after tax
    ( = weighted sum of the wind, solar, battery, and electrical infrastracture WACC). Net present value (NPV)
    and levelized cost of energy (LCOE) is then be calculated using the calculates WACC as the discount rate, as well
    as the internal rate of return (IRR).
    """

    def __init__(
        self,
        parameter_dict,
        # N_time,
        # Depreciation curve
        depreciation_yr,
        depreciation,
        # Inflation curve
        inflation_yr,
        inflation,
        ref_yr_inflation,
        # Early paying or CAPEX Phasing
        phasing_yr,
        phasing_CAPEX,
        life_y=25,
        intervals_per_hour=4,
    ):
        """Initialization of the HPP finance model

        Parameters
        ----------
        N_time : Number of hours in the representative dataset
        life_h : Lifetime of the plant in hours
        """
        # super().__init__()
        self.parameter_dict = parameter_dict
        self.life_y = life_y
        self.intervals_per_hour = intervals_per_hour
        self.life_h = 365 * 24 * life_y
        self.life_intervals = self.life_h * intervals_per_hour

        # Depreciation curve
        self.depreciation_yr = depreciation_yr
        self.depreciation = depreciation

        # Inflation curve
        self.inflation_yr = inflation_yr
        self.inflation = inflation
        self.ref_yr_inflation = ref_yr_inflation

        # Early paying or CAPEX Phasing
        self.phasing_yr = phasing_yr
        self.phasing_CAPEX = phasing_CAPEX

    # def setup(self):

    # def setup_partials(self):
    #     self.declare_partials('*', '*', dependent=False, val=0)

    # def compute_partials(self, inputs, partials):
    #     pass

    def compute(self, **inputs):
        """Calculating the financial metrics of the hybrid power plant project.

        Parameters
        ----------
        price_t_ext : Electricity price time series [Eur]
        hpp_t_with_deg : HPP power time series [MW]
        penalty_t : penalty for not reaching expected energy productin at peak hours [Eur]
        CAPEX_w : CAPEX of the wind power plant
        OPEX_w : OPEX of the wind power plant
        CAPEX_s : CAPEX of the solar power plant
        OPEX_s : OPEX of solar power plant
        CAPEX_b : CAPEX of the battery
        OPEX_b : OPEX of the battery
        CAPEX_sh :  CAPEX of the shared electrical infrastracture
        OPEX_sh : OPEX of the shared electrical infrastracture
        wind_WACC : After tax WACC for onshore WT
        solar_WACC : After tax WACC for solar PV
        battery_WACC: After tax WACC for stationary storge li-ion batteries
        tax_rate : Corporate tax rate

        Returns
        -------
        CAPEX : Total capital expenditure costs of the HPP
        OPEX : Operational and maintenance costs of the HPP
        NPV : Net present value
        IRR : Internal rate of return
        NPV_over_CAPEX : NPV over CAPEX
        mean_AEP : Mean annual energy production
        LCOE : Levelized cost of energy
        penalty_lifetime : total penalty
        """
        outputs = {}
        parameter_dict = self.parameter_dict
        parameter_dict.update(
            {
                # hpp parameters
                "hpp_grid_connection": float(inputs["G_MW"][0]),  # in MW
                # hpp wind parameters
                "wind_capacity": float(inputs["wind_MW"][0]),  # in MW
                # hpp solar parameters
                "solar_capacity": float(inputs["solar_MW"][0]),  # in MW
                # hpp battery parameters
                "battery_energy_capacity": float(inputs["b_E"][0]),  # in MWh
                "battery_power_capacity": float(inputs["b_P"][0]),  # in MW
                "battery_minimum_SoC": 1
                - float(inputs["battery_depth_of_discharge"][0]),
            }
        )

        intervals_per_year = 365 * 24 * self.intervals_per_hour
        life_intervals = self.life_y * intervals_per_year
        life_yr = self.life_y

        depreciation_yr = self.depreciation_yr
        depreciation = self.depreciation

        inflation_yr = self.inflation_yr
        inflation = self.inflation
        ref_yr_inflation = self.ref_yr_inflation

        phasing_yr = self.phasing_yr
        phasing_CAPEX = self.phasing_CAPEX

        df = pd.DataFrame()

        df["hpp_t"] = inputs["P_HPP_ts"]

        df["i_year"] = np.hstack(
            [np.array([ii] * intervals_per_year) for ii in range(life_yr)]
        )[:life_intervals]

        revenues = calculate_revenues(
            self.parameter_dict,
            inputs["P_HPP_SM_t_opt"],
            inputs["P_HPP_ts"],
            inputs["P_HPP_RT_refs"],
            inputs["SM_price_cleared"],
            inputs["BM_dw_price_cleared"],
            inputs["BM_up_price_cleared"],
            inputs["P_HPP_UP_bid_ts"],
            inputs["P_HPP_DW_bid_ts"],
            inputs["s_UP_t"],
            inputs["s_DW_t"],
            df,
        )

        CAPEX = (
            inputs["CAPEX_w"]
            + inputs["CAPEX_s"]
            + inputs["CAPEX_b"]
            + inputs["CAPEX_el"]
        )
        OPEX = (
            inputs["OPEX_w"] + inputs["OPEX_s"] + inputs["OPEX_b"] + inputs["OPEX_el"]
        )

        # Discount rate
        hpp_discount_factor = calculate_WACC(
            inputs["CAPEX_w"],
            inputs["CAPEX_s"],
            inputs["CAPEX_b"],
            inputs["CAPEX_el"],
            inputs["wind_WACC"],
            inputs["solar_WACC"],
            inputs["battery_WACC"],
        )

        # Apply CAPEX phasing using the inflation index for all years before the start of the project (t=0).
        inflation_index_phasing = get_inflation_index(
            yr=phasing_yr,
            inflation_yr=inflation_yr,
            inflation=inflation,
            ref_yr_inflation=ref_yr_inflation,
        )
        CAPEX_eq = calculate_CAPEX_phasing(
            CAPEX=CAPEX,
            phasing_yr=phasing_yr,
            phasing_CAPEX=phasing_CAPEX,
            discount_rate=hpp_discount_factor,
            inflation_index=inflation_index_phasing,
        )

        # len of revenues = years of life
        iy = (
            np.arange(len(revenues)) + 1
        )  # Plus becasue the year zero is added externally in the NPV and IRR calculations

        # Compute inflation, all cahsflow are in nominal prices
        inflation_index = get_inflation_index(
            yr=np.arange(
                len(revenues) + 1
            ),  # It includes t=0, to compute the reference
            inflation_yr=inflation_yr,
            inflation=inflation,
            ref_yr_inflation=ref_yr_inflation,
        )

        revenues = revenues.values.flatten()
        outputs["CAPEX"] = CAPEX
        outputs["OPEX"] = OPEX
        outputs["revenues"] = revenues.mean()

        # We need to add DEVEX
        DEVEX = 0

        # Calculate the
        NPV, IRR = calculate_NPV_IRR(
            Net_revenue_t=revenues,
            investment_cost=CAPEX_eq,  # includes early paying of CAPEX, CAPEX-phasing
            maintenance_cost_per_year=OPEX,
            tax_rate=inputs["tax_rate"],
            discount_rate=hpp_discount_factor,
            depreciation_yr=depreciation_yr,
            depreciation=depreciation,
            development_cost=DEVEX,
            inflation_index=inflation_index,
        )

        break_even_PPA_price = np.maximum(
            0,
            calculate_break_even_PPA_price(
                df=df,
                CAPEX=CAPEX_eq,
                OPEX=OPEX,
                tax_rate=inputs["tax_rate"],
                discount_rate=hpp_discount_factor,
                depreciation_yr=depreciation_yr,
                depreciation=depreciation,
                DEVEX=DEVEX,
                inflation_index=inflation_index,
                parameter_dict=self.parameter_dict,
                P_HPP_SM_t_opt=inputs["P_HPP_SM_t_opt"],
                P_HPP_ts=inputs["P_HPP_ts"],
                P_HPP_RT_refs=inputs["P_HPP_RT_refs"],
                SM_price_cleared=inputs["SM_price_cleared"],
                BM_dw_price_cleared=inputs["BM_dw_price_cleared"],
                BM_up_price_cleared=inputs["BM_up_price_cleared"],
                P_HPP_UP_bid_ts=inputs["P_HPP_UP_bid_ts"],
                P_HPP_DW_bid_ts=inputs["P_HPP_DW_bid_ts"],
                s_UP_t=inputs["s_UP_t"],
                s_DW_t=inputs["s_DW_t"],
            ),
        )
        outputs["NPV"] = NPV
        outputs["IRR"] = IRR
        outputs["NPV_over_CAPEX"] = NPV / CAPEX

        level_costs = np.sum(OPEX / (1 + hpp_discount_factor) ** iy) + CAPEX
        AEP_per_year = (
            df.groupby("i_year").hpp_t.mean() * 365 * 24 * self.intervals_per_hour
        )
        level_AEP = np.sum(AEP_per_year / (1 + hpp_discount_factor) ** iy)

        mean_AEP_per_year = np.mean(AEP_per_year)
        if level_AEP > 0:
            outputs["LCOE"] = level_costs / (level_AEP)  # in Euro/MWh
        else:
            outputs["LCOE"] = 1e6

        outputs["mean_AEP"] = mean_AEP_per_year

        # outputs['penalty_lifetime'] = df['penalty_t'].sum()
        outputs["break_even_PPA_price"] = break_even_PPA_price
        out_keys = [
            "CAPEX",
            "OPEX",
            "NPV",
            "IRR",
            "NPV_over_CAPEX",
            "mean_AEP",
            "LCOE",
            "revenues",
            "break_even_PPA_price",
        ]
        return [outputs[key] for key in out_keys]


class finance_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = finance(**insta_inp)
        super().__init__(
            inputs=[
                ("G_MW", dict(units="MW", desc="Grid size")),
                ("wind_MW", dict(units="MW", desc="Wind plant nominal size")),
                ("solar_MW", dict(units="MW", desc="Solar plant nominal size")),
                ("b_E", dict(desc="Battery energy storage capacity")),
                (
                    "battery_depth_of_discharge",
                    dict(desc="battery depth of discharge", units="MW"),
                ),
                ("b_P", dict(desc="Battery power capacity", units="MW")),
                # ('hpp_t_with_deg',
                #                dict(desc="HPP power time series",
                #                units='MW',
                #                shape=[model.life_intervals])),
                ("CAPEX_w", dict(desc="CAPEX wpp")),
                ("OPEX_w", dict(desc="OPEX wpp")),
                ("CAPEX_s", dict(desc="CAPEX solar pvp")),
                ("OPEX_s", dict(desc="OPEX solar pvp")),
                ("CAPEX_b", dict(desc="CAPEX battery")),
                ("OPEX_b", dict(desc="OPEX battery")),
                ("CAPEX_el", dict(desc="CAPEX electrical infrastructure")),
                ("OPEX_el", dict(desc="OPEX electrical infrastructure")),
                ("wind_WACC", dict(desc="After tax WACC for onshore WT")),
                ("solar_WACC", dict(desc="After tax WACC for solar PV")),
                (
                    "battery_WACC",
                    dict(desc="After tax WACC for stationary storge li-ion batteries"),
                ),
                ("tax_rate", dict(desc="Corporate tax rate")),
                (
                    "P_HPP_SM_t_opt",
                    dict(
                        desc="",
                        shape=[model.life_intervals],
                    ),
                ),
                (
                    "SM_price_cleared",
                    dict(
                        desc="",
                        shape=[model.life_h],
                    ),
                ),
                (
                    "BM_dw_price_cleared",
                    dict(
                        desc="",
                        shape=[model.life_h],
                    ),
                ),
                (
                    "BM_up_price_cleared",
                    dict(
                        desc="",
                        shape=[model.life_h],
                    ),
                ),
                (
                    "P_HPP_RT_refs",
                    dict(
                        desc="",
                        shape=[model.life_intervals],
                    ),
                ),
                (
                    "P_HPP_UP_bid_ts",
                    dict(
                        desc="",
                        shape=[model.life_intervals],
                    ),
                ),
                (
                    "P_HPP_DW_bid_ts",
                    dict(
                        desc="",
                        shape=[model.life_intervals],
                    ),
                ),
                (
                    "s_UP_t",
                    dict(
                        desc="",
                        shape=[model.life_intervals],
                    ),
                ),
                (
                    "s_DW_t",
                    dict(
                        desc="",
                        shape=[model.life_intervals],
                    ),
                ),
                (
                    "residual_imbalance",
                    dict(
                        desc="",
                        shape=[model.life_intervals],
                    ),
                ),
                (
                    "P_HPP_ts",
                    dict(
                        desc="",
                        shape=[model.life_intervals],
                    ),
                ),
                (
                    "P_curtailment_ts",
                    dict(
                        desc="",
                        shape=[model.life_intervals],
                    ),
                ),
                (
                    "P_charge_discharge_ts",
                    dict(
                        desc="",
                        shape=[model.life_intervals],
                    ),
                ),
                (
                    "E_SOC_ts",
                    dict(
                        desc="",
                        shape=[model.life_intervals + 1],
                    ),
                ),
            ],
            outputs=[
                ("CAPEX", dict(desc="CAPEX")),
                ("OPEX", dict(desc="OPEX")),
                ("NPV", dict(desc="NPV")),
                ("IRR", dict(desc="IRR")),
                ("NPV_over_CAPEX", dict(desc="NPV/CAPEX")),
                ("mean_AEP", dict(desc="mean AEP")),
                ("LCOE", dict(desc="LCOE")),
                ("revenues", dict(desc="Revenues")),
                (
                    "break_even_PPA_price",
                    dict(
                        desc="PPA price of electricity that results in NPV=0 with the given hybrid power plant configuration and operation",
                        val=0,
                    ),
                ),
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )




# -----------------------------------------------------------------------
# Auxiliar functions for financial modelling
# -----------------------------------------------------------------------
# 
# Note: Core financial functions have been moved to hydesign.costmodels
# Import them from there for consistency across all finance modules.
