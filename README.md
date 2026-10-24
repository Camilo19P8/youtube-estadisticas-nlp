# Estadísticas de YouTube con procesamiento de lenguaje natural

Proyecto de la Especialización en Inteligencia Artificial, Universidad de Cundinamarca. Avances IV, semana 10, 24 de octubre de 2026.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Camilo19P8/youtube-estadisticas-nlp/blob/main/Avances_IV_YouTube_Colab.ipynb)

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

- `Avances_IV_YouTube_Colab.ipynb`: reproducción en Colab.
- `final_evaluation.py`: evaluación escrita y guardada en VS Code.
- `portable_inference.py`: contrato de entrada e inferencia congelada.
- `reference/`: resultados y casos de paridad de Avances III.
- `results_iv/`: métricas y verificaciones finales.
- `evidence/`: capturas reales.
- `docs/`: informe final y documentación.

## Equipo

Cristian Camilo Alarcon Delgadillo, Bryan Arley Saenz Cortes y Jonathan David Lancheros Bello. La autoría del informe no prueba por sí sola aportes individuales al repositorio. La documentación distingue aportes declarados y registros técnicos de esta cuenta; no se fabrican commits ni reuniones de otras personas.

El prototipo de Avances III está alojado con acceso privado en https://youtube-avances-iii-calar-2026.calarcondelgadillo.chatgpt.site. El monitoreo es local en el navegador, sin observabilidad central de producción.
