# Estadísticas de YouTube con procesamiento de lenguaje natural

Proyecto de la Especialización en Inteligencia Artificial, Universidad de Cundinamarca. Avances IV, semana 10, 24 de octubre de 2026.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Camilo19P8/youtube-estadisticas-nlp/blob/main/Avances_IV_YouTube_Colab_Ejecutado.ipynb)

El notebook descarga una revisión fija del código, obtiene el dataset público de Kaggle, verifica sus hashes y ejecuta la evaluación final. Se distinguen resultados históricos, evaluación retrospectiva y prueba de estrés. Los resultados originales de Avances III están en `reference/`.

## Alcance

El modelo estima vistas acumuladas con interacciones ya observadas. No predice crecimiento futuro ni demuestra causalidad. El ensamble combina HGB sin el total de comentarios con Ridge de variables tabulares y TF-IDF del título y comentarios.

Avances III comparó nueve alternativas y 16 configuraciones mediante validación anidada en 1.493 videos. El procedimiento anidado obtuvo RMSE log 0,5865; el ensamble fijo obtuvo RMSE OOF 0,5743. Estas cifras no equivalen a una prueba externa.

## Evaluación final

Se congela el modelo de Avances III y se evalúa retrospectivamente en los 374 videos de la prueba conocida de Avances II, sin nueva selección. Se contrasta con HGB, texto aislado y una mediana. La prueba de estrés por fecha de publicación no equivale a observar vistas futuras. Se comprueba inferencia portable en 50 casos, rechazo de entradas inválidas y latencia de CPU.

## Ejecutar

```bash
python -m pip install -r requirements.txt
python download_data.py
python final_evaluation.py
```

Los datos se descargan de https://www.kaggle.com/datasets/advaypatil/youtube-statistics y se comparan sus hashes con `reference/summary.json`. No se redistribuyen datos originales.

`artifacts/model_portable.json.gz.b64` contiene JSON comprimido, sin objetos pickle. `portable_inference.py` realiza la inferencia con NumPy. `experiment_avances_iii.py` conserva la preparación y los estimadores de la etapa anterior.

## Archivos

- `Avances_IV_YouTube_Colab_Ejecutado.ipynb`: cinco celdas ejecutadas en Google Colab, con salidas reales.
- `Avances_IV_YouTube_Colab.ipynb`: verificación local anterior.
- `final_evaluation.py`: evaluación escrita y guardada en VS Code.
- `portable_inference.py`: contrato de entrada e inferencia congelada.
- `reference/`: resultados y casos de paridad de Avances III.
- `results_iv/`: métricas de la ejecución local inicial.
- `results_colab/`: reproducción remota, verificaciones y procedencia del entorno.
- `evidence/`: capturas reales.
- `docs/`: informe final y documentación.

## Equipo

Cristian Camilo Alarcon Delgadillo, Bryan Arley Saenz Cortes y Jonathan David Lancheros Bello. La autoría del informe no prueba por sí sola aportes individuales al repositorio. La documentación distingue aportes declarados y registros técnicos de esta cuenta; no se fabrican commits ni reuniones de otras personas.

El prototipo de Avances III está alojado con acceso privado en [Prototipo interactivo del proyecto](https://youtube-avances-iii-calar-2026.calarcondelgadillo.chatgpt.site). El monitoreo es local en el navegador, sin observabilidad central de producción.

## Resultados y entorno verificado

El 24 de octubre de 2026 se ejecutaron las cinco celdas de código en un servidor CPU de Google Colab mediante la extensión oficial Google.colab para VS Code. Las salidas remotas están en `Avances_IV_YouTube_Colab_Ejecutado.ipynb`; `results_colab/cloud_provenance.json` registra Linux, Python 3.13.16, NumPy 2.1.3, pandas 2.2.3 y Scikit-learn 1.6.1. El editor es VS Code y el cómputo se realizó en Google Colab. Se conserva la ejecución local inicial por separado.

El ensamble congelado obtuvo RMSE log 0,5593 frente a 0,5831 de HGB. El intervalo bootstrap apareado de la diferencia incluye cero. En el estrés por fecha, HGB obtuvo 0,7023 y el ensamble 0,7497. Los resultados no validan un pronóstico futuro.

## Informe final

[Leer el informe en PDF](docs/Informe_YouTube_Avances_IV.pdf) · [Guía de reproducción](docs/REPRODUCCION.md). El cuaderno incluye salidas ejecutadas y una gráfica.

En Colab, el ensamble congelado conservó RMSE log 0,559311. La referencia HGB reajustada obtuvo 0,580889, frente a 0,583084 en el entorno local. Pasaron los 50 casos de paridad y las seis entradas inválidas; la latencia mediana local del servidor fue 2,35 ms. El informe compara ambos entornos y mantiene la distinción entre prueba conocida y validación externa.
