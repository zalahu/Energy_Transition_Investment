
from typing import List, Dict
import numpy as np
import numpy_financial as npf
from .utils import ScenarioInputs


def build_cash_flows(p: ScenarioInputs) -> (List[float], List[Dict]):
    ann = []
    cash_flows = [ -p.capex_initial ]
    prod = p.capacity_per_year * p.availability
    price = p.product_price
    opex = p.opex_annual
    for t in range(1, p.years+1):
        prod_t = prod * ((1.0 - p.degradation) ** (t-1))
        revenue_t = prod_t * price
        opex_t = opex * ((1.0 + p.inflation) ** (t-1))
        ebitda = revenue_t - opex_t
        cf = ebitda
        if t == p.years and p.salvage_value:
            cf += p.salvage_value
        cash_flows.append(cf)
        ann.append({
            'year': t,
            'production': prod_t,
            'revenue': revenue_t,
            'opex': opex_t,
            'ebitda': ebitda,
            'cash_flow': cf,
        })
    return cash_flows, ann


def npv_irr_payback(cash_flows: List[float], discount_rate: float):
    years = list(range(len(cash_flows)))
    discounts = [(1/(1+discount_rate))**t for t in years]
    npv = sum(cf * d for cf, d in zip(cash_flows, discounts))
    try:
        irr = npf.irr(cash_flows)
    except Exception:
        irr = float('nan')
    # payback: cumulative CF crosses zero (undiscounted)
    cum = 0.0
    payback_year = float('inf')
    for i, cf in enumerate(cash_flows):
        cum += cf
        if cum >= 0 and i>0:
            # linear interpolation in year i
            prev_cum = cum - cf
            frac = (0 - prev_cum) / cf if cf != 0 else 0
            payback_year = i-1 + frac
            break
    return npv, irr, payback_year


def lcox_like(p: ScenarioInputs, cash_flows: List[float]):
    # Simple levelized cost proxy = PV(costs)/PV(output)
    outputs = []
    prod = p.capacity_per_year * p.availability
    for t in range(1, p.years+1):
        prod_t = prod * ((1.0 - p.degradation) ** (t-1))
        outputs.append(prod_t)
    pv_output = sum( out / ((1+p.discount_rate)**t) for t, out in enumerate(outputs, start=1))
    # PV costs: Capex + Opex discounted
    # Re-create only cost side
    pv_costs = p.capex_initial
    opex = p.opex_annual
    for t in range(1, p.years+1):
        opex_t = opex * ((1+p.inflation)**(t-1))
        pv_costs += opex_t / ((1+p.discount_rate)**t)
    return pv_costs / max(pv_output, 1e-9)


def monte_carlo(p: ScenarioInputs, n:int=2000, price_sigma=0.15, capex_sigma=0.15, opex_sigma=0.10, rate_sigma=0.02):
    rng = np.random.default_rng(42)
    results = []
    for _ in range(n):
        p_mc = ScenarioInputs(**p.__dict__)
        p_mc.product_price *= rng.lognormal(mean=0, sigma=price_sigma)
        p_mc.capex_initial *= rng.lognormal(mean=0, sigma=capex_sigma)
        p_mc.opex_annual *= rng.lognormal(mean=0, sigma=opex_sigma)
        p_mc.discount_rate = max(0.0, rng.normal(p.discount_rate, rate_sigma))
        cfs, _ = build_cash_flows(p_mc)
        npv, irr, pb = npv_irr_payback(cfs, p_mc.discount_rate)
        results.append(npv)
    return np.array(results)


def breakeven_price(p: ScenarioInputs, target_npv=0.0, tol=1e-4, max_iter=100):
    # Bisection on price
    low, high = 0.0, max(5*p.product_price, 1e-6)
    for _ in range(max_iter):
        mid = 0.5*(low+high)
        p_mid = ScenarioInputs(**p.__dict__)
        p_mid.product_price = mid
        cfs, _ = build_cash_flows(p_mid)
        npv, _, _ = npv_irr_payback(cfs, p_mid.discount_rate)
        if abs(npv-target_npv) < tol:
            return mid
        if npv < target_npv:
            low = mid
        else:
            high = mid
    return mid
