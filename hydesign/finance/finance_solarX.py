# import os
# import time

# basic libraries
import numpy as np

# from numpy import newaxis as na
import numpy_financial as npf
import pandas as pd

# import openmdao.api as om
import scipy as sp

# import openpyxl
from hydesign.openmdao_wrapper import ComponentWrapper
from hydesign.costmodels import (
    calculate_NPV_IRR,
    calculate_revenues,
    calculate_break_even_PPA_price,
    calculate_CAPEX_phasing,
    get_inflation_index,
)


class finance_solarX:
    """
    Financial model to evaluate the profitability of a hybrid power plant (HPP).

    This model calculates annual cash flows, using parameters like Weighted Average Cost of Capital (WACC),
    tax rates, depreciation schedules, and inflation indices. Financial metrics like Net Present Value (NPV),
    Internal Rate of Return (IRR), Levelized Cost of Energy (LCOE), and other performance indicators are computed.
    """

    def __init__(
        self,
        N_time,
        life_h,
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
    ):
        """
        Initializes the HPP financial model with parameters for depreciation, inflation, and CAPEX phasing.

        Parameters
        ----------
        N_time : int
            Number of hours in the representative dataset.
        life_h : int, optional
            Plant's operational lifetime in hours (default is 25 years).
        depreciation_yr : array-like
            Years for which depreciation is calculated.
        depreciation : array-like
            Values of depreciation at specified years.
        inflation_yr : array-like
            Years for which inflation rates are provided.
        inflation : array-like
            Inflation values corresponding to specified years.
        ref_yr_inflation : int
            Reference year for the inflation index.
        phasing_yr : array-like
            Years for CAPEX phasing.
        phasing_CAPEX : array-like
            CAPEX values associated with each phasing year.
        """

        # super().__init__()
        self.N_time = int(N_time)
        self.life_h = int(life_h)

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
        """
        Defines inputs and outputs for the financial model in OpenMDAO.

        Inputs represent various financial and operational parameters like CAPEX, OPEX, and prices.
        Outputs include computed financial metrics such as CAPEX, OPEX, NPV, IRR, and break-even PPA prices.
        """
        # inputs
        # hpp
        self.inputs = [
            (
                "hpp_t_ext",
                dict(desc="HPP power time series", units="MW", shape=[self.life_h]),
            ),
            (
                "hpp_curt_t_ext",
                dict(
                    desc="HPP curtailment power time series",
                    units="MW",
                    shape=[self.life_h],
                ),
            ),
            # cpv
            (
                "cpv_t_ext",
                dict(desc="cpv power time series", units="MW", shape=[self.life_h]),
            ),
            # cst
            (
                "p_st_t_ext",
                dict(desc="CSP power time series", units="MW", shape=[self.life_h]),
            ),
            (
                "q_t_ext",
                dict(desc="produced heat time series", units="MW", shape=[self.life_h]),
            ),
            # biogas_h2
            (
                "h2_t_ext",
                dict(
                    desc="H2 production time series", units="kg/h", shape=[self.life_h]
                ),
            ),
            (
                "biogas_t_ext",
                dict(
                    desc="Consumed biogas time series",
                    units="kg/h",
                    shape=[self.life_h],
                ),
            ),
            # prices
            (
                "price_el_t_ext",
                dict(desc="Electricity price time series", shape=[self.life_h]),
            ),
            (
                "price_h2_t_ext",
                dict(desc="Hydrogen price time series", shape=[self.life_h]),
            ),
            (
                "price_biogas_t_ext",
                dict(desc="Biogas price time series", shape=[self.life_h]),
            ),
            # capex and opex
            ("CAPEX_sf", dict(desc="CAPEX solar field")),
            ("OPEX_sf", dict(desc="OPEX solar field")),
            ("CAPEX_cpv", dict(desc="CAPEX cpv")),
            ("OPEX_cpv", dict(desc="OPEX cpv")),
            ("CAPEX_cst", dict(desc="CAPEX CST")),
            ("OPEX_cst", dict(desc="OPEX CSP")),
            ("CAPEX_h2", dict(desc="CAPEX Biogas_H2")),
            ("OPEX_h2", dict(desc="OPEX H2")),
            ("CAPEX_sh", dict(desc="CAPEX electrical infrastructure")),
            ("OPEX_sh", dict(desc="OPEX electrical infrastructure")),
            # other
            (
                "penalty_t_ext",
                dict(
                    desc="penalty for not reaching expected energy production at peak hours",
                    shape=[self.life_h],
                ),
            ),
            (
                "penalty_q_t_ext",
                dict(
                    desc="penalty for not reaching expected heat production",
                    shape=[self.life_h],
                ),
            ),
            ("discount_rate", dict(desc="discount rate")),
            ("tax_rate", dict(desc="Corporate tax rate")),
        ]
        self.outputs = [
            # outputs
            ("CAPEX", dict(desc="CAPEX")),
            ("OPEX", dict(desc="OPEX")),
            ("NPV", dict(desc="NPV")),
            ("IRR", dict(desc="IRR")),
            ("NPV_over_CAPEX", dict(desc="NPV/CAPEX")),
            ("mean_AEP", dict(desc="mean AEP")),
            ("mean_AH2P", dict(desc="mean annual H2 production")),
            ("LCOE", dict(desc="LCOE")),
            ("revenues", dict(desc="Revenues")),
            ("penalty_lifetime", dict(desc="penalty_lifetime")),
            (
                "break_even_PPA_price",
                dict(
                    desc="PPA price of electricity that results in NPV=0 with the given hybrid power plant configuration and operation"
                ),
            ),
            (
                "break_even_PPA_price_h2",
                dict(
                    desc="PPA price of hydrogen that results in NPV=0 with the given hybrid power plant configuration and operation"
                ),
            ),
            (
                "break_even_PPA_price_q",
                dict(
                    desc="PPA price of heat that results in NPV=0 with the given hybrid power plant configuration and operation"
                ),
            ),
            ("lcove", dict(desc="cost of valued energy")),
        ]

    def compute(self, **inputs):
        """Computes financial metrics based on revenue, CAPEX, OPEX, and other financial parameters.

        The method calculates cash flow using inflation-adjusted revenues, CAPEX phasing, and
        depreciation. It computes financial indicators like NPV, IRR, LCOE, and additional metrics for
        profitability evaluation.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values, such as CAPEX, OPEX, price time series, and tax rate.
        outputs : dict
            Dictionary of computed financial outputs, including CAPEX, OPEX, NPV, IRR, etc.
        """
        outputs = {}
        # Extract inputs and setup time-based parameters
        N_time = self.N_time
        life_h = self.life_h
        life_yr = int(np.ceil(life_h / N_time))

        # Preparing data for financial calculations including revenue and depreciation
        depreciation_yr = self.depreciation_yr
        depreciation = self.depreciation
        inflation_yr = self.inflation_yr
        inflation = self.inflation
        ref_yr_inflation = self.ref_yr_inflation
        phasing_yr = self.phasing_yr
        phasing_CAPEX = self.phasing_CAPEX
        df = pd.DataFrame()
        df["hpp_t"] = inputs["hpp_t_ext"]
        df["p_cpv_t"] = inputs["cpv_t_ext"]
        df["p_st_t"] = inputs["p_st_t_ext"]
        df["h2_t"] = inputs["h2_t_ext"]
        df["q_t"] = inputs["q_t_ext"]
        df["biogas_t"] = inputs["biogas_t_ext"]
        df["penalty_t"] = inputs["penalty_t_ext"]
        df["penalty_q_t"] = inputs["penalty_q_t_ext"]
        df["i_year"] = np.hstack([np.array([ii] * N_time) for ii in range(life_yr)])[
            :life_h
        ]
        df["price_el_t"] = inputs["price_el_t_ext"]
        df["price_h2_t"] = inputs["price_h2_t_ext"]
        df["price_biogas_t"] = inputs["price_biogas_t_ext"]

        # Calculate yearly revenues and cash flow
        revenues = calculate_revenues(df)
        CAPEX = (
            inputs["CAPEX_sf"]
            + inputs["CAPEX_cpv"]
            + inputs["CAPEX_cst"]
            + inputs["CAPEX_sh"]
            + inputs["CAPEX_h2"]
        )
        OPEX = (
            inputs["OPEX_sf"] * life_yr
            + inputs["OPEX_cpv"] * life_yr
            + inputs["OPEX_cst"] * life_yr
            + inputs["OPEX_sh"] * life_yr
            + inputs["OPEX_h2"]
        )

        # Calculate present value of CAPEX based on phasing and inflation adjustment
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
            discount_rate=inputs["discount_rate"],  # hpp_discount_factor,
            inflation_index=inflation_index_phasing,
        )

        # Final calculations for NPV, IRR, and LCOE based on revenues and annualized costs
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
        outputs["revenues"] = revenues.sum()

        # DEVEX
        DEVEX = 0

        opex_per_year = OPEX / life_yr

        # Calculate the
        NPV, IRR = calculate_NPV_IRR(
            Net_revenue_t=revenues,
            investment_cost=CAPEX_eq,  # includes early paying of CAPEX, CAPEX-phasing
            maintenance_cost_per_year=opex_per_year,
            tax_rate=inputs["tax_rate"],
            discount_rate=inputs["discount_rate"],  # hpp_discount_factor,
            depreciation_yr=depreciation_yr,
            depreciation=depreciation,
            development_cost=DEVEX,
            inflation_index=inflation_index,
        )

        break_even_price_el, break_even_price_h2, break_even_price_q = (
            calculate_break_even_PPA_price(
                df=df,
                CAPEX=CAPEX_eq,
                OPEX=OPEX,
                tax_rate=inputs["tax_rate"],
                discount_rate=inputs["discount_rate"],
                depreciation_yr=depreciation_yr,
                depreciation=depreciation,
                DEVEX=DEVEX,
                inflation_index=inflation_index,
            )
        )

        # Ensure that the break-even prices are not negative
        break_even_PPA_price_el = np.maximum(0, break_even_price_el)
        break_even_PPA_price_h2 = np.maximum(0, break_even_price_h2)
        break_even_PPA_price_q = np.maximum(0, break_even_price_q)
        outputs["break_even_PPA_price_q"] = break_even_PPA_price_q

        outputs["NPV"] = NPV
        outputs["IRR"] = IRR
        outputs["NPV_over_CAPEX"] = NPV / CAPEX

        level_costs = (
            np.sum(opex_per_year / (1 + inputs["discount_rate"]) ** iy) + CAPEX
        )
        AEP_per_year = df.groupby("i_year").hpp_t.mean() * 365 * 24
        level_AEP = np.sum(AEP_per_year / (1 + inputs["discount_rate"]) ** iy)
        mean_AEP_per_year = np.mean(AEP_per_year)

        # Excluding H2 cost and consumpsion for LCOE
        CAPEX_no_h2 = (
            inputs["CAPEX_sf"]
            + inputs["CAPEX_cpv"]
            + inputs["CAPEX_cst"]
            + inputs["CAPEX_sh"]
        )
        OPEX_no_h2 = (
            inputs["OPEX_sf"] * life_yr
            + inputs["OPEX_cpv"] * life_yr
            + inputs["OPEX_cst"] * life_yr
            + inputs["OPEX_sh"] * life_yr
        )
        opex_no_h2_per_year = OPEX_no_h2 / life_yr
        level_costs_no_h2 = (
            np.sum(opex_no_h2_per_year / (1 + inputs["discount_rate"]) ** iy)
            + CAPEX_no_h2
        )
        AEP_per_year_no_h2 = (
            (df.groupby("i_year").p_cpv_t.mean() + df.groupby("i_year").p_st_t.mean())
            * 365
            * 24
        )
        level_AEP_no_h2 = np.sum(
            AEP_per_year_no_h2 / (1 + inputs["discount_rate"]) ** iy
        )

        if level_AEP > 0:
            outputs["LCOE"] = level_costs_no_h2 / (level_AEP_no_h2)  # in Euro/MWh
        else:
            outputs["LCOE"] = 1e6

        outputs["mean_AEP"] = mean_AEP_per_year

        outputs["penalty_lifetime"] = df["penalty_t"].sum()
        outputs["break_even_PPA_price"] = break_even_PPA_price_el
        outputs["break_even_PPA_price_h2"] = break_even_PPA_price_h2

        # AH2P
        AH2P_per_year = df.groupby("i_year").h2_t.mean() * 365 * 24
        mean_AH2P_per_year = np.mean(AH2P_per_year)
        outputs["mean_AH2P"] = mean_AH2P_per_year

        # lcove
        revenues_discount = np.sum(revenues / (1 + inputs["discount_rate"]) ** iy)
        lcove = level_costs / revenues_discount
        outputs["lcove"] = lcove
        out_keys = [
            "CAPEX",
            "OPEX",
            "NPV",
            "IRR",
            "NPV_over_CAPEX",
            "mean_AEP",
            "mean_AH2P",
            "LCOE",
            "revenues",
            "penalty_lifetime",
            "break_even_PPA_price",
            "break_even_PPA_price_h2",
            "break_even_PPA_price_q",
            "lcove",
        ]
        return [outputs[key] for key in out_keys]


class finance_solarX_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = finance_solarX(**insta_inp)
        super().__init__(
            inputs=model.inputs,
            outputs=model.outputs,
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


# -----------------------------------------------------------------------
# Auxiliar functions for financial modelling

# -----------------------------------------------------------------------
# Auxiliar functions for financial modelling
# -----------------------------------------------------------------------
# 
# Note: Core financial functions have been moved to hydesign.costmodels
# Import them from there for consistency across all finance modules.
