
from typing import Dict, Tuple
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pulp
from .utils import ScenarioInputs
from .finance import build_cash_flows, npv_irr_payback
from .carbon import annual_carbon, totals


def optimize_config(base_inputs: ScenarioInputs, weights: Tuple[float,float]=(0.5,0.5)):
    """Choose among discrete options: capacity scale, capture intensity, WACC.
    Maximize weighted normalized (NPV, CO2e_avoided). Returns chosen option and metrics.
    """
    caps = [0.8, 1.0, 1.2]
    intensities = [base_inputs.project_intensity, max(base_inputs.project_intensity-0.1, 0.0)]
    waccs = [max(base_inputs.discount_rate-0.02, 0.01), base_inputs.discount_rate, base_inputs.discount_rate+0.02]

    # enumerate all combinations
    combos = []
    for s in caps:
        for it in intensities:
            for r in waccs:
                p = ScenarioInputs(**base_inputs.__dict__)
                p.capacity_per_year *= s
                p.project_intensity = it
                p.discount_rate = r
                cfs, _ = build_cash_flows(p)
                npv, irr, pb = npv_irr_payback(cfs, p.discount_rate)
                co2_rows = annual_carbon(p)
                co2_total, _ = totals(co2_rows)
                combos.append({'scale': s, 'intensity': it, 'wacc': r, 'npv': npv, 'co2': co2_total})

    npvs = np.array([c['npv'] for c in combos]).reshape(-1,1)
    co2s = np.array([c['co2'] for c in combos]).reshape(-1,1)
    scaler = MinMaxScaler()
    npvs_n = scaler.fit_transform(npvs).flatten()
    co2s_n = MinMaxScaler().fit_transform(co2s).flatten()

    w1, w2 = weights
    scores = w1*npvs_n + w2*co2s_n
    best_idx = int(np.argmax(scores))
    return combos[best_idx], combos
