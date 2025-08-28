"""
Cost models package for HyDesign.

This package contains centralized financial computation functions 
used across different finance modules.
"""

from .financial_functions import (
    calculate_NPV_IRR,
    calculate_NPV_IRR_hybridized,
    calculate_WACC,
    calculate_revenues,
    calculate_break_even_PPA_price,
    calculate_CAPEX_phasing,
    get_inflation_index,
)

__all__ = [
    "calculate_NPV_IRR",
    "calculate_NPV_IRR_hybridized",
    "calculate_WACC", 
    "calculate_revenues",
    "calculate_break_even_PPA_price",
    "calculate_CAPEX_phasing",
    "get_inflation_index",
]