# Proyecto Final de Machine Learning — Predicción del éxito académico en educación superior

**Autor:** Miguel Pajuelo Gómez
**Titulación:** Ingeniería Matemática e Inteligencia Artificial
**Curso:** 2025/2026

Este paquete contiene el código y la memoria del Proyecto Final de la asignatura de Aprendizaje Automático. El trabajo aborda tres tareas conectadas sobre el dataset `rendimiento_estudiantes.csv`:

1. **Clasificación multiclase** de la situación final del estudiante (`abandono`, `matriculado`, `graduado`) con un *Linear Discriminant Analysis* implementado desde cero.
2. **Regresión** de la calificación media del segundo semestre (`nota_media_2sem`) con un *Gradient Boosting Regressor* implementado desde cero, evitando *data leakage*.
3. **Aprendizaje no supervisado** (PCA + *K-Means*) para descubrir perfiles latentes de estudiantes.

Toda la metodología, decisiones de preprocesado, formulación matemática, métricas e interpretación están detalladas en la memoria PDF.

---

## 1. Contenido del paquete

```
Proyecto Final ML Miguel Pajuelo/
├── README.md                              # este archivo
├── informe_proyecto_ml_neurips.pdf        # memoria final (7 páginas, formato NeurIPS)
│
├── exploratory_analysis.py                # flujo principal de preprocesado
├── exploratory_functions.py               # mapeos, codificaciones y utilidades
├── lda.py                                 # LDA implementado desde cero
├── boosting.py                            # Decision Tree + Gradient Boosting desde cero
│
├── classification_problem.ipynb           # tarea 1 (clasificación, usa lda.py)
├── regresion_problem.ipynb                # tarea 2 (regresión, usa boosting.py)
└── unsupervised_learning.ipynb            # tarea 3 (PCA + K-Means)
```

> El dataset `rendimiento_estudiantes.csv` no se incluye en el paquete; se asume que está disponible en la misma carpeta antes de ejecutar el código (es el archivo proporcionado por la asignatura).

---

## 2. Requisitos

- **Python ≥ 3.10** (probado con 3.12).
- Librerías: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `jupyter`.

Instalación rápida con pip:

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

> Nota: `lda.py` y `boosting.py` sólo dependen de `numpy`. `scikit-learn` se utiliza únicamente en los notebooks para utilidades auxiliares (split, métricas, PCA, *K-Means* y `StandardScaler`).

---

## 3. Cómo reproducir los resultados

Asumiendo que el directorio de trabajo es la raíz del paquete y que `rendimiento_estudiantes.csv` está presente, los pasos son los siguientes.

### Paso 1 — Generar el dataset procesado

```bash
python exploratory_analysis.py
```

Esto carga `rendimiento_estudiantes.csv`, aplica el preprocesado completo (codificación binaria, mapeos ordinales, agrupación semántica + *one-hot*, ratios académicos y normalización a `[0,1]`) y produce **`rendimiento_estudiantes_procesado.csv`** (4424 filas × 63 columnas) en la misma carpeta. Este archivo es la entrada que esperan los tres notebooks.

### Paso 2 — Ejecutar los tres notebooks

Cada notebook corresponde a una tarea del proyecto y se ejecuta de forma independiente:

```bash
jupyter notebook classification_problem.ipynb     # tarea 1: LDA propio
jupyter notebook regresion_problem.ipynb          # tarea 2: Gradient Boosting propio
jupyter notebook unsupervised_learning.ipynb      # tarea 3: PCA + K-Means
```

Para cada notebook basta con ejecutar todas las celdas en orden (*Cell → Run All* en Jupyter, o el botón equivalente en VS Code). Los notebooks usan `random_state=0` (o `42` para *K-Means*) de forma que los resultados son reproducibles y coinciden con los de la memoria.

### Orden recomendado de lectura/ejecución

1. `exploratory_analysis.py` y `exploratory_functions.py` para entender el preprocesado.
2. `classification_problem.ipynb` — clasificación con LDA propio.
3. `regresion_problem.ipynb` — regresión con Gradient Boosting propio.
4. `unsupervised_learning.ipynb` — PCA y *K-Means*.

---

## 4. Decisiones metodológicas clave (resumen)

Más detalle en la memoria. Aquí queda registrado lo imprescindible para entender cada script.

### Preprocesado (`exploratory_analysis.py` + `exploratory_functions.py`)

- Cada bloque de variables se codifica según su semántica, no de forma genérica.
- **Binarias** → `{0,1}` (`becado`, `deudor`, `desplazado`, `internacional`, `matricula_al_dia`, `genero`, etc.).
- **Ordinales** (cualificaciones del estudiante y de los padres) → escala creciente `0..26` que respeta el orden educativo.
- **Nominales** (modo de solicitud, carreras, ocupaciones) → agrupación semántica + *one-hot* (reduce de docenas de categorías a unas pocas interpretables).
- **Reducción de colinealidad académica:** se construyen dos ratios por semestre y se eliminan los conteos brutos:
  - `ratio_presentacion = evaluadas / matriculadas`
  - `ratio_exito       = aprobadas / evaluadas`
- **Normalización final a `[0,1]`** para que LDA, PCA y *K-Means* no estén dominados por escalas dispares.
- Cada mapeo se valida con `comprobar_mapeo_completo`: si aparece una categoría no contemplada, el preprocesado falla (no errores silenciosos).

### Clasificación (`lda.py`, usado en `classification_problem.ipynb`)

- LDA propio: matrices de dispersión `S_W` y `S_B`, autovectores principales de `S_W^{-1} S_B` (con pseudoinversa por estabilidad), predicción por *score* bayesiano `s_c(x) = -½‖W·x − W·μ_c‖² + log π_c`.
- Split 80/20 con `random_state=0`.
- **Oversampling** únicamente del conjunto de entrenamiento (`duplicate_minority_only`): la clase `matriculado` pasa de 643 a 1286 muestras; el test queda intacto.
- Métricas: *accuracy*, *balanced accuracy*, $F_1$ macro y matriz de confusión.

### Regresión (`boosting.py`, usado en `regresion_problem.ipynb`)

- *Decision Tree Regressor* propio con criterio de reducción de varianza, `max_depth` y `min_samples_split` configurables.
- *Gradient Boosting Regressor* propio con pérdida cuadrática:
  - `y_hat_0 = mean(y)`
  - residuos `r_t = y - y_hat_{t-1}`
  - actualización `y_hat_t = y_hat_{t-1} + lr · h_t(X)`
- Para evitar *data leakage*, se eliminan: `nota_media_2sem` (target), `ratio_presentacion_2sem`, `ratio_exito_2sem` y `objetivo`.
- Hiperparámetros finales: `n_estimators=40`, `max_depth=3`, `learning_rate=0.1` (seleccionados con CV de 3 *folds*).

### No supervisado (`unsupervised_learning.ipynb`)

- Estandarización con `StandardScaler` (media 0, desviación 1).
- PCA completo para inspeccionar la dimensionalidad.
- *K-Means* con `k = 2..8`, 20 inicializaciones, selección por *silhouette*.

---

## 5. Resultados principales

| Tarea | Modelo | Métrica clave |
|---|---|---|
| Clasificación | LDA propio | accuracy = 0.716, balanced acc = 0.599, $F_1$ macro = 0.610 |
| Regresión     | Gradient Boosting propio | RMSE = 0.128, $R^2$ = 0.796 (escala normalizada) |
| Clustering    | K-Means estandarizado    | mejor *silhouette* = 0.129 en $k=4$, 4 perfiles interpretables |

La variable más informativa, coincidiendo en LDA, Gradient Boosting y *K-Means*, es **`ratio_exito_1sem`** (ratio de éxito del primer semestre).

---

## 6. Notas sobre reproducibilidad

- Las particiones train/test usan `random_state=0` para que cualquier ejecución reproduzca exactamente las métricas de la memoria.
- El *K-Means* del notebook no supervisado usa `random_state=42` y `n_init=20`.
- Las cifras pueden variar ligeramente si se cambia la versión de NumPy o scikit-learn (cambios internos de PCA y KMeans), pero el orden de magnitud y las conclusiones se mantienen.
- Si Jupyter genera carpetas auxiliares (`__pycache__/`, `.ipynb_checkpoints/`) se pueden borrar sin problema.

---

## 7. Limitaciones conocidas

Documentadas también en la memoria:

1. **Causalidad:** los modelos detectan asociación, no efectos.
2. **Desbalanceo:** el *oversampling* simple por duplicación no resuelve completamente la clase `matriculado`.
3. **Escalado global:** la normalización a `[0,1]` se calcula sobre el dataset completo y no solo sobre *train*; en producción debería ajustarse al subconjunto de entrenamiento.
4. **Interpretación temporal de la clasificación:** LDA usa variables del segundo semestre, por lo que no debe presentarse como una alerta temprana pura. La regresión, en cambio, sí lo es.
5. **Solapamiento de clusters:** con *silhouette* ≈ 0.13, los grupos no son nítidos y deben interpretarse como tendencias.
