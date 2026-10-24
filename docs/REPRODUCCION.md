# Reproducción y revisión

1. Abra `Avances_IV_YouTube_Colab_Ejecutado.ipynb` desde el botón Colab del README o desde VS Code con un kernel de Google Colab.
2. Ejecute las cinco celdas de código en orden. La primera identifica el entorno real y verifica los hashes de los datos.
3. Revise la tabla retrospectiva, el estrés por publicación, las verificaciones y la figura.
4. Descargue el ZIP de resultados creado por la última celda si ejecuta en Colab.

El cuaderno `Avances_IV_YouTube_Colab_Ejecutado.ipynb` conserva cinco celdas ejecutadas sin errores en un servidor CPU de Google Colab, mediante la extensión oficial Google.colab para VS Code, el 24 de octubre de 2026. La primera salida identifica Linux, Python 3.13.16 y google-colab 1.0.0. El cuaderno sin el sufijo Ejecutado conserva la verificación local anterior como antecedente separado.

La referencia inmutable de las fuentes es `eb793a5282a6b6b5f20f1d67cf48fa965d928135`. El notebook obtiene esa revisión para que actualizaciones de documentación no cambien el experimento. El modelo congelado usa JSON, con casos de paridad en `reference/`.

Los archivos en `results_iv/` corresponden a la ejecución local inicial usada en las Tablas 3 a 7 del PDF. `results_colab/` contiene las métricas de Google Colab y `cloud_provenance.json`, con la revisión de fuentes y el entorno real. La inferencia congelada mantiene sus métricas; los modelos que se reajustan cambian ligeramente entre las versiones de Scikit-learn 1.9.1 (local) y 1.6.1 (Colab). La Tabla 8 del informe compara ambas ejecuciones. Latencia y hora también dependen de cada ejecución.

El informe contrasta el diseño de referencia disponible (orientación del curso y Avances II/III). No se recibió un documento formal de Avances I ni un reparto individual confirmado. Los tres integrantes están confirmados como autores; no se atribuyen commits a otras cuentas.
