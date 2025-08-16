
from typing import List, Dict
from .utils import ScenarioInputs

def annual_carbon(p: ScenarioInputs):
    """Return list of dict per year with baseline, project, avoided, credit value."""
    rows = []
    prod = p.capacity_per_year * p.availability
    carbon_price = p.carbon_price
    for t in range(1, p.years+1):
        prod_t = prod * ((1.0 - p.degradation) ** (t-1))
        base = p.baseline_intensity * prod_t
        proj = p.project_intensity * prod_t
        avoided = max(base - proj, 0.0)
        credit_price_t = carbon_price * ((1.0 + p.carbon_price_growth) ** (t-1))
        credit_value = avoided * credit_price_t
        rows.append({
            'year': t,
            'production': prod_t,
            'baseline_tCO2e': base,
            'project_tCO2e': proj,
            'avoided_tCO2e': avoided,
            'carbon_price': credit_price_t,
            'credit_value': credit_value,
        })
    return rows


def totals(rows: List[Dict]):
    co2e_avoided_total = sum(r['avoided_tCO2e'] for r in rows)
    carbon_credit_value_total = sum(r['credit_value'] for r in rows)
    return co2e_avoided_total, carbon_credit_value_total
