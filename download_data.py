"""Descargar el dataset asignado y exigir identidad con Avances III."""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
expected = json.loads((root / 'reference/summary.json').read_text())['sha256']
data = urllib.request.urlopen('https://www.kaggle.com/api/v1/datasets/download/advaypatil/youtube-statistics').read()
(root / 'data').mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    for name, digest in expected.items():
        content = archive.read(name)
        actual = hashlib.sha256(content).hexdigest()
        if actual != digest:
            raise ValueError(f'El dataset cambio: {name}; detener antes de comparar resultados.')
        (root / 'data' / name).write_bytes(content)
        print(name, actual, 'VERIFICADO')
