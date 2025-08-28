"""
Financial functions for cost modeling in HyDesign.

This module contains centralized financial computation functions 
that were previously duplicated across different finance modules.
"""

import numpy as np
import numpy_financial as npf
import scipy as sp


def calculate_NPV_IRR(
    Net_revenue_t,
    investment_cost,
    maintenance_cost_per_year,
    tax_rate,
    discount_rate,
    depreciation_yr,
    depreciation,
    development_cost,
    inflation_index,
):
    """A function to estimate the yearly cashflow using the net revenue time series, and the yearly OPEX costs.
    It then calculates the NPV and IRR using the yearly cashlow, the CAPEX, the WACC after tax, and the tax rate.

    Parameters
    ----------
    Net_revenue_t : Net revenue time series
    investment_cost : Capital costs
    maintenance_cost_per_year : yearly operation and maintenance costs
    tax_rate : tax rate
    discount_rate : Discount rate
    depreciation_yr : Depreciation curve (x-axis) time in years
    depreciation : Depreciation curve at the given times
    development_cost : DEVEX
    inflation_index : Yearly Inflation index time-sereis

    Returns
    -------
    NPV : Net present value
    IRR : Internal rate of return
    """

    yr = np.arange(
        len(Net_revenue_t) + 1
    )  # extra year to start at 0 and end at end of lifetime.
    depre = np.interp(yr, depreciation_yr, depreciation)

    # EBITDA: earnings before interest and taxes in nominal prices
    EBITDA = (Net_revenue_t - maintenance_cost_per_year) * inflation_index[1:]

    # EBIT taxable income
    depreciation_on_each_year = np.diff(investment_cost * depre)
    EBIT = EBITDA - depreciation_on_each_year

    # Taxes
    Taxes = EBIT * tax_rate

    Net_income = EBITDA - Taxes
    Cashflow = np.insert(Net_income, 0, -investment_cost - development_cost)
    NPV = npf.npv(discount_rate, Cashflow)
    if NPV > 0:
        IRR = npf.irr(Cashflow)
    else:
        IRR = 0
    return NPV, IRR


def calculate_WACC(
    CAPEX_w,
    CAPEX_s,
    CAPEX_b,
    CAPEX_el,
    wind_WACC,
    solar_WACC,
    battery_WACC,
):
    """This function returns the weighted average cost of capital after tax, using solar, wind, and battery
    WACC. First the shared costs WACC is computed by taking the mean of the WACCs across all technologies.
    Then the WACC after tax is calculated by taking the weighted sum by the corresponding CAPEX.

    Parameters
    ----------
    CAPEX_w : CAPEX of the wind power plant
    CAPEX_s : CAPEX of the solar power plant
    CAPEX_b : CAPEX of the battery
    CAPEX_el : CAPEX of the shared electrical costs
    wind_WACC : After tax WACC for onshore WT
    solar_WACC : After tax WACC for solar PV
    battery_WACC : After tax WACC for stationary storge li-ion batteries

    Returns
    -------
    WACC_after_tax : WACC after tax
    """

    # Weighted average cost of capital
    WACC_after_tax = (
        CAPEX_w * wind_WACC
        + CAPEX_s * solar_WACC
        + CAPEX_b * battery_WACC
        + CAPEX_el * (wind_WACC + solar_WACC + battery_WACC) / 3
    ) / (CAPEX_w + CAPEX_s + CAPEX_b + CAPEX_el)
    return WACC_after_tax


def calculate_revenues(price_el, df):
    """Calculate revenues from price and power time series.

    Parameters
    ----------
    price_el : Electricity price time series
    df : DataFrame with 'hpp_t' and 'penalty_t' columns

    Returns
    -------
    revenues : Annual revenues grouped by year
    """
    df["revenue"] = (
        df["hpp_t"] * np.broadcast_to(price_el, df["hpp_t"].shape) - df["penalty_t"]
    )
    return df.groupby("i_year").revenue.mean() * 365 * 24


def calculate_break_even_PPA_price(
    df,
    CAPEX,
    OPEX,
    tax_rate,
    discount_rate,
    depreciation_yr,
    depreciation,
    DEVEX,
    inflation_index,
):
    """Calculate the break-even PPA price that results in NPV=0.

    Parameters
    ----------
    df : DataFrame with power and penalty data
    CAPEX : Capital expenditure
    OPEX : Operational expenditure
    tax_rate : Corporate tax rate
    discount_rate : Discount rate
    depreciation_yr : Depreciation curve years
    depreciation : Depreciation curve values
    DEVEX : Development expenditure
    inflation_index : Inflation index time series

    Returns
    -------
    break_even_price : PPA price that results in NPV=0
    """
    def fun(price_el):
        revenues = calculate_revenues(price_el, df)
        NPV, _ = calculate_NPV_IRR(
            Net_revenue_t=revenues.values.flatten(),
            investment_cost=CAPEX,
            maintenance_cost_per_year=OPEX,
            tax_rate=tax_rate,
            discount_rate=discount_rate,
            depreciation_yr=depreciation_yr,
            depreciation=depreciation,
            development_cost=DEVEX,
            inflation_index=inflation_index,
        )
        return NPV**2

    out = sp.optimize.minimize(fun=fun, x0=50, method="SLSQP", tol=1e-10)
    return out.x


def calculate_CAPEX_phasing(
    CAPEX,
    phasing_yr,
    phasing_CAPEX,
    discount_rate,
    inflation_index,
):
    """This function calulates the equivalent net present value CAPEX given a early paying "phasing" approach.

    Parameters
    ----------
    CAPEX : CAPEX
    phasing_yr : Yearly early paying of CAPEX curve. x-axis, time in years.
    phasing_CAPEX : Yearly early paying of CAPEX curve. Shares will be normalized to sum the CAPEX.
    discount_rate : Discount rate for present value calculation
    inflation_index : Inflation index time series at the phasing_yr years. Accounts for inflation.

    Returns
    -------
    CAPEX_eq : Present value equivalent CAPEX
    """

    phasing_CAPEX = inflation_index * CAPEX * phasing_CAPEX / np.sum(phasing_CAPEX)
    CAPEX_eq = np.sum(
        [
            phasing_CAPEX[ii] / (1 + discount_rate) ** yr
            for ii, yr in enumerate(phasing_yr)
        ]
    )

    return CAPEX_eq


def calculate_NPV_IRR_hybridized(
    delta_life,
    Net_revenue_t,
    investment_cost,
    maintenance_cost_per_year,
    capex_vector,
    capex_for_depreciation,
    tax_rate,
    discount_rate,
    depreciation_yr,
    depreciation,
    depre_rate,
    development_cost,
    decommissioning_vec,
    inflation_index,
    plot=False,
):
    """A function to estimate the yearly cashflow for hybridized systems using the net revenue time series, and the yearly OPEX costs.
    This is specialized for systems with staged deployment (wind/PV at different times).
    It then calculates the NPV and IRR using the yearly cashlow, the CAPEX, the WACC after tax, and the tax rate.

    Parameters
    ----------
    delta_life : Number of years between the start of operations of the first and second plants
    Net_revenue_t : Net revenue time series
    investment_cost : Capital costs
    maintenance_cost_per_year : yearly operation and maintenance costs
    capex_vector : CAPEX vector over time
    capex_for_depreciation : CAPEX vector for depreciation calculation
    tax_rate : tax rate
    discount_rate : Discount rate
    depreciation_yr : Depreciation curve (x-axis) time in years
    depreciation : Depreciation curve at the given times
    depre_rate : Straight line depreciation rate
    development_cost : DEVEX
    decommissioning_vec : vector containing the decommissioning costs over time
    inflation_index : Yearly Inflation index time-sereis
    plot : Whether to generate cashflow plots

    Returns
    -------
    NPV : Net present value
    IRR : Internal rate of return
    """

    # EBITDA: earnings before interest and taxes in nominal prices
    EBITDA = (Net_revenue_t - maintenance_cost_per_year) * inflation_index

    # EBIT taxable income
    depreciation_on_each_year = depre_rate * capex_for_depreciation
    EBIT = EBITDA - depreciation_on_each_year

    # Taxes
    Taxes = np.zeros(len(EBIT))

    for ii in range(1, len(EBIT)):
        if EBIT[ii] <= 0:
            Taxes[ii] = 0
        else:
            Taxes[ii] = EBIT[ii] * tax_rate

    Net_income = EBITDA - Taxes
    Income_minus_capex = Net_income - capex_vector
    Cashflow = Income_minus_capex - decommissioning_vec

    # Bar plot for the cashflows
    if plot:
        import matplotlib.pyplot as plt
        
        opex = -maintenance_cost_per_year * inflation_index
        revenue = Net_revenue_t * inflation_index
        tax_vec = -Taxes
        capex_vec_w = np.zeros(len(capex_vector))
        capex_vec_w[0] = -capex_vector[0]
        capex_vec_p = np.zeros(len(capex_vector))
        capex_vec_p[delta_life] = -capex_vector[delta_life]

        indices = np.arange(len(Cashflow))
        plt.figure(figsize=(10, 6))
        plt.bar(indices, revenue, color="green", label="Revenues", alpha=0.7)
        plt.bar(indices, opex, color="orange", label="OPEX", alpha=0.7)
        plt.bar(indices, tax_vec, bottom=opex, color="blue", label="Taxes", alpha=0.7)
        plt.bar(indices, capex_vec_w, color="red", label="CAPEX wind", alpha=0.7)
        plt.bar(
            indices,
            capex_vec_p,
            bottom=tax_vec + opex,
            color="magenta",
            label="CAPEX PV and batteries",
            alpha=0.7,
        )
        plt.bar(
            indices,
            -decommissioning_vec,
            bottom=tax_vec + opex,
            color="purple",
            label="Decommissioning of WT",
            alpha=0.7,
        )
        plt.title("Cashflows by Year", fontsize=16)
        plt.xlabel("Year", fontsize=16)
        plt.ylabel("Amount (MEur)", fontsize=16)
        plt.tick_params(axis="both", which="major", labelsize=14)
        plt.legend(fontsize=12)
        plt.grid(axis="y")
        plt.savefig("cashflows.eps", format="eps", bbox_inches="tight")
        plt.show()

    NPV = npf.npv(discount_rate, Cashflow)
    if NPV > 0:
        IRR = npf.irr(Cashflow)
    else:
        IRR = 0
    return NPV, IRR


def get_inflation_index(yr, inflation_yr, inflation, ref_yr_inflation=0):
    """This function calulates the inflation index time series.

    Parameters
    ----------
    yr : Years for eavaluation of the  inflation index
    inflation_yr : Yearly inflation curve. x-axis, time in years. To be used in interpolation.
    inflation : Yearly inflation curve.  To be used in interpolation.
    ref_yr_inflation : Referenece year, at which the inflation index takes value of 1.

    Returns
    -------
    inflation_index : inflation index time series at yr
    """
    infl = np.interp(yr, inflation_yr, inflation)

    ind_ref = np.where(np.array(yr) == ref_yr_inflation)[0]
    inflation_index = np.cumprod(1 + np.array(infl))
    inflation_index = inflation_index / inflation_index[ind_ref]

    return inflation_index