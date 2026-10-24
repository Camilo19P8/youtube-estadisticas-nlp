"""Avances IV: evaluacion retrospectiva del artefacto final y prueba de estres.

No ajusta hiperparametros con la prueba conocida ni afirma validacion externa.
"""
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from experiment_avances_iii import load_data, hgb, BASE, metrics, candidates
from portable_inference import load_bundle, predict_log, validated_record

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results_iv'

def evaluate_final():
    OUT.mkdir(exist_ok=True)
    X, y, audit = load_data()
    development, known_test = train_test_split(
        np.arange(len(X)), test_size=0.20, random_state=42)
    bundle = load_bundle()
    rows = []
    predictions = {}
    models = {
        'Ensamble congelado III': bundle['champion'],
        'Solo comentarios congelado III': bundle['text_model'],
    }
    for name, model in models.items():
        pred = np.array([predict_log(model, row)
                         for row in X.iloc[known_test].to_dict('records')])
        predictions[name] = pred
        rows.append(dict(model=name, **metrics(y[known_test], pred)))

    # Referencia de la configuracion reportada en Avances II, sin nueva busqueda.
    baseline = hgb(BASE).fit(X.iloc[development], y[development])
    pred = baseline.predict(X.iloc[known_test])
    predictions['HGB referencia II'] = pred
    rows.append(dict(model='HGB referencia II', **metrics(y[known_test], pred)))
    dummy = DummyRegressor(strategy='median').fit(
        np.zeros((len(development), 1)), y[development])
    pred = dummy.predict(np.zeros((len(known_test), 1)))
    rows.append(dict(model='Mediana', **metrics(y[known_test], pred)))
    pd.DataFrame(rows).to_csv(OUT / 'retrospective_metrics.csv', index=False)

    frozen = predictions['Ensamble congelado III']
    observed = y[known_test]
    rng = np.random.default_rng(20261024)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(observed), len(observed))
        boot.append(metrics(observed[idx], frozen[idx])['rmse'])
    paired = []
    for _ in range(2000):
        idx = rng.integers(0, len(observed), len(observed))
        paired.append(metrics(observed[idx], frozen[idx])['rmse'] -
                      metrics(observed[idx], predictions['HGB referencia II'][idx])['rmse'])
    records = X.iloc[known_test][['Video ID', 'Keyword', 'Views']].copy()
    records['observed_log'] = observed
    records['prediction_log'] = frozen
    records['absolute_log_error'] = np.abs(observed - frozen)
    records.to_csv(OUT / 'retrospective_predictions.csv', index=False)
    segments = []
    records['quartile'] = pd.qcut(records['Views'], 4, labels=['Q1','Q2','Q3','Q4'])
    for group, frame in records.groupby('quartile', observed=True):
        segments.append(dict(group=str(group), n=len(frame),
                             **metrics(frame.observed_log, frame.prediction_log)))
    pd.DataFrame(segments).to_csv(OUT / 'segments.csv', index=False)

    # Estres por fecha de publicacion, no un pronostico longitudinal ni prueba externa.
    order = np.argsort(pd.to_datetime(X['Published At']).to_numpy(), kind='stable')
    cut = int(len(order) * .8)
    early, recent = order[:cut], order[cut:]
    ensemble_recipe = candidates()[-1][1]
    text_recipe = candidates()[4][1].set_params(model__alpha=10)
    stress = []
    for name, model in [('HGB fijo', hgb(BASE)),
                        ('Ensamble fijo', ensemble_recipe),
                        ('Texto fijo alpha 10', text_recipe)]:
        fitted = clone(model).fit(X.iloc[early], y[early])
        stress.append(dict(model=name, **metrics(y[recent], fitted.predict(X.iloc[recent]))))
    pd.DataFrame(stress).to_csv(OUT / 'publication_stress.csv', index=False)

    cases = json.loads((ROOT / 'reference/parity_cases.json').read_text(encoding='utf-8'))
    errors = [abs(predict_log(bundle['champion'], c['input']) - c['expected'])
              for c in cases]
    valid = validated_record('Video demo', 'Very useful explanation!', 15000, 800, 365, 'tech')
    invalid = [dict(likes=-1), dict(total_comments=float('nan')),
               dict(age_days=float('inf')), dict(comments='  '),
               dict(title='x' * 2001), dict(likes=True)]
    base = dict(title='Video demo', comments='Useful explanation!', likes=15000,
                total_comments=800, age_days=365, keyword='tech')
    rejected = 0
    for change in invalid:
        try:
            validated_record(**(base | change))
        except ValueError:
            rejected += 1
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        result = predict_log(bundle['champion'], valid)
        latencies.append((time.perf_counter() - start) * 1000)
    assert max(errors) < 1e-8 and rejected == len(invalid) and np.isfinite(result)
    checks = dict(parity_cases=len(cases), max_error_log=max(errors),
                  tolerance=1e-8, invalid_inputs_rejected=rejected,
                  invalid_inputs_tested=len(invalid),
                  latency_ms_median=float(np.median(latencies)),
                  latency_ms_p95=float(np.quantile(latencies, .95)), passed=True)
    (OUT / 'verification.json').write_text(json.dumps(checks, indent=2))
    summary = dict(executed_at=datetime.now(timezone.utc).isoformat(), audit=audit,
                   n_development=len(development), n_known_test=len(known_test),
                   known_test_warning='Retrospectiva: prueba conocida en Avances II; sin seleccion nueva.',
                   rmse_bootstrap95=np.quantile(boot, [.025, .975]).tolist(),
                   difference_vs_hgb_bootstrap95=np.quantile(paired, [.025, .975]).tolist(),
                   temporal_train=len(early), temporal_test=len(recent),
                   temporal_cutoff=str(X.iloc[recent]['Published At'].min()),
                   versions=dict(python=platform.python_version(), numpy=np.__version__,
                                 pandas=pd.__version__, sklearn=sklearn.__version__),
                   model_sha256=hashlib.sha256((ROOT / 'artifacts/model_portable.json.gz.b64').read_bytes()).hexdigest())
    (OUT / 'summary_iv.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print('EVALUACION FINAL COMPLETADA')
    print(pd.DataFrame(rows).to_string(index=False))
    print('\nESTRES POR PUBLICACION')
    print(pd.DataFrame(stress).to_string(index=False))
    print('\nVERIFICACIONES', json.dumps(checks))
    print('\nRESUMEN', json.dumps(summary, ensure_ascii=False))
    return summary

if __name__ == '__main__':
    with threadpool_limits(limits=2):
        evaluate_final()
