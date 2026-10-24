"""Inferencia portable del modelo congelado de Avances III, sin pickle."""
import base64
import gzip
import json
import math
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def load_bundle():
    data = (ROOT / 'artifacts/model_portable.json.gz.b64').read_text()
    return json.loads(gzip.decompress(base64.b64decode(data)))

def transform(prep, record):
    values = []
    if 'num' in prep:
        cfg = prep['num']
        missing = [record.get(c) is None or not np.isfinite(float(record[c]))
                   for c in cfg['cols']]
        raw = [cfg['medians'][i] if missing[i] else float(record[c])
               for i, c in enumerate(cfg['cols'])]
        raw.extend(float(missing[i]) for i in cfg['indicators'])
        values.extend((v - m) / s for v, m, s in
                      zip(raw, cfg['mean'], cfg['scale']))
    values.extend(float(k == record.get('Keyword'))
                  for k in prep.get('categories', []))
    if 'text' in prep:
        cfg = prep['text']
        tokens = re.findall(r'(?u)\b\w\w+\b', str(record.get(cfg['col'], '')).lower())
        terms = tokens + [' '.join(pair) for pair in zip(tokens, tokens[1:])]
        counts = {}
        for term in terms:
            pos = cfg['vocab'].get(term)
            if pos is not None:
                counts[pos] = counts.get(pos, 0) + 1
        vector = np.zeros(len(cfg['idf']))
        for pos, count in counts.items():
            vector[pos] = (1 + math.log(count)) * cfg['idf'][pos]
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        values.extend(vector)
    return np.asarray(values, dtype=float)

def predict_log(model, record):
    if model['kind'] == 'voting':
        return float(np.mean([predict_log(m, record) for m in model['models']]))
    if model['kind'] == 'constant':
        return model['value']
    x = transform(model['prep'], record)
    if model['kind'] == 'ridge':
        return float(model['intercept'] + np.dot(model['coef'], x))
    result = model['baseline']
    for tree in model['trees']:
        node = tree[0]
        while not node['is_leaf']:
            value = x[node['feature_idx']]
            left = bool(node['missing_go_to_left']) if np.isnan(value) else value <= node['num_threshold']
            node = tree[node['left'] if left else node['right']]
        result += node['value']
    return float(result)

def validated_record(title, comments, likes, total_comments, age_days, keyword):
    """Contrato del prototipo; los agregados no disponibles se imputan."""
    if not isinstance(title, str) or not isinstance(comments, str):
        raise ValueError('Titulo y comentarios deben ser texto.')
    if not comments.strip():
        raise ValueError('Se requiere una muestra de comentarios.')
    if len(title) > 2000 or len(comments) > 100000:
        raise ValueError('La entrada excede el limite de longitud.')
    for value in [likes, total_comments, age_days]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Los conteos y la antiguedad deben ser finitos y no negativos.')
    return dict(log_likes=math.log1p(likes), log_comments=math.log1p(total_comments),
                age_days=age_days, title_chars=len(title), title_words=len(title.split()),
                Keyword=keyword, Title=title, comment_text=comments,
                combined_text=title + ' ' + comments)
