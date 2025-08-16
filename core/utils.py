
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ScenarioInputs:
    years: int
    discount_rate: float  # WACC
    capex_initial: float
    opex_annual: float
    capacity_per_year: float  # units/year (e.g., MWh, t)
    product_price: float  # $/unit
    degradation: float = 0.0  # fractional
    availability: float = 0.9
    inflation: float = 0.02
    carbon_price: float = 0.0  # $/tCO2e initial
    carbon_price_growth: float = 0.0
    baseline_intensity: float = 0.8  # tCO2e/unit
    project_intensity: float = 0.2   # tCO2e/unit
    salvage_value: float = 0.0
    construction_years: int = 1

@dataclass
class ScenarioResults:
    npv: float
    irr: float
    payback_year: float
    lcox: float
    co2e_avoided_total: float
    carbon_credit_value_total: float
    cash_flows: List[float]
    annual_summary: List[Dict[str, Any]]
