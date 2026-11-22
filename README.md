
# MLOps Churn Lab

Proyecto de predicción de abandono de clientes (*churn*) para una fintech utilizando Machine Learning y la metodología **CRISP-ML(Q)**.

El proyecto implementa validación de datos, preprocesamiento reproducible, comparación de modelos, seguimiento de experimentos con MLflow y evaluación mediante criterios técnicos y de negocio.

## 1. El caso

Una fintech busca identificar clientes con riesgo de abandonar sus servicios para ofrecer campañas de retención.

Se utiliza CRISP-ML(Q) para asegurar que los modelos sean evaluados con criterios técnicos y económicos antes de promoverlos a producción.

El laboratorio se divide en dos partes:

- **Laboratorio 14.1:** Negocio y datos, preparación, modelado y evaluación.
- **Laboratorio 14.2:** Despliegue, monitoreo y reentrenamiento.

## 2. Tecnologías utilizadas

- Python 3.10
- Pandas y NumPy
- Scikit-learn
- XGBoost
- MLflow 2.10
- FastAPI
- Pytest
- GitHub Actions
- Conda

## 3. Configuración del entorno

Clonar el repositorio:

```bash
git clone https://github.com/TU-USUARIO/mlops-churn-lab.git
cd mlops-churn-lab
```

Crear y activar el entorno:

```bash
conda create -n churn-lab python=3.10 -y
conda activate churn-lab
```

Instalar dependencias:

```bash
pip install -r requirements.txt "setuptools<81"
```

## 4. Dataset

Se utiliza un dataset de **300 clientes** de una fintech.

| Variable | Descripción |
|---|---|
| customer_id | Identificador del cliente |
| gender | Género |
| senior_citizen | Indicador de adulto mayor |
| tenure_months | Antigüedad del cliente |
| contract_type | Tipo de contrato |
| internet_service | Servicio de internet |
| monthly_charges | Cargo mensual |
| support_tickets_last_6m | Tickets de soporte recientes |
| churn | Variable objetivo: Yes / No |

El dataset se divide en:

- **Entrenamiento:** 240 registros (80 %).
- **Prueba:** 60 registros (20 %).

Se utiliza una partición estratificada para conservar la proporción de clases.

## 5. Metodología CRISP-ML(Q)

### Fase 1. Negocio y datos

Se establecieron criterios de éxito y un contrato de datos para validar la estructura, los valores y la consistencia del dataset.

```bash
python src/validate_data.py
```

Resultado esperado:

```text
Contrato de datos: 23/23 checks OK
```

### Fase 2. Preparación de datos

Se implementó un `Pipeline` de scikit-learn que integra el preprocesamiento y el modelo.

Esto permite evitar diferencias entre las transformaciones utilizadas durante el entrenamiento y la inferencia (*training-serving skew*).

```bash
python src/data_prep.py
```

Se generan los archivos:

- `data/train.csv`
- `data/test.csv`
- `data/data_prep_summary.json`

### Fase 3. Modelado

Se entrenaron cuatro modelos candidatos y un baseline.

Se utilizó **validación cruzada estratificada de cinco folds**, registrando las métricas y los modelos en MLflow.

```bash
python src/train.py --model all
```

### Fase 4. Evaluación

Se evalúa el modelo seleccionado considerando:

- F1 y ROC-AUC en test.
- Umbral de clasificación según costos de negocio.
- Recall por segmento.
- Cumplimiento de cuatro controles del quality gate.
- Registro mediante los aliases `champion` y `challenger`.

```bash
python src/evaluate_and_register.py
```

El proceso genera:

- `reports/evaluation_report.json`
- `reports/model_card.md` (cuando el modelo es promovido).

## 6. Comparación de modelos — Reto obligatorio

Se incorporó un cuarto modelo candidato, **GradientBoostingClassifier**, para compararlo con los tres modelos originales.

La evaluación se realizó utilizando validación cruzada estratificada de cinco folds.

### Resultados de validación cruzada

| Modelo | F1 CV | ROC-AUC CV |
|---|---:|---:|
| Logistic Regression | 0.6666 | 0.7315 |
| **Random Forest** | **0.6900** | **0.7326** |
| XGBoost | 0.6829 | 0.7100 |
| Gradient Boosting | 0.6476 | 0.6858 |

### Cuarto modelo implementado

Se utilizó `GradientBoostingClassifier` de scikit-learn con la siguiente configuración:

```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

Este algoritmo construye árboles de decisión secuencialmente, buscando corregir los errores de los modelos anteriores.

### Análisis de resultados

Gradient Boosting obtuvo:

- **F1 CV:** 0.6476
- **ROC-AUC CV:** 0.6858

Random Forest obtuvo:

- **F1 CV:** 0.6900
- **ROC-AUC CV:** 0.7326

Gradient Boosting presentó un rendimiento inferior a los modelos originales.

### Conclusión

La incorporación del cuarto modelo no modificó la selección del mejor algoritmo.

**Random Forest continúa siendo el modelo ganador**, porque obtuvo el mayor F1 promedio de validación cruzada, que es el criterio establecido para seleccionar el modelo.

Los resultados muestran que incorporar un modelo adicional no garantiza mejorar el rendimiento predictivo.

## 7. Pipeline CI/CD con GitHub Actions

Se configuró un workflow que automatiza las fases principales de CRISP-ML(Q).

El pipeline contiene cinco jobs dependientes:

```text
Tests
  |
  v
Data Validation
  |
  v
Data Preparation
  |
  v
Modeling
  |
  v
Evaluation
```

Cada fase debe completarse correctamente antes de ejecutar la siguiente.

Archivo de configuración:

`.github/workflows/mlops.yml`

### Función de cada job

| Job | Función |
|---|---|
| Tests | Ejecuta las pruebas unitarias con Pytest |
| Data Validation | Comprueba el contrato de datos |
| Data Preparation | Prepara los conjuntos de entrenamiento y prueba |
| Modeling | Entrena y compara los modelos mediante MLflow |
| Evaluation | Evalúa el modelo y aplica los quality gates |

## 8. MLflow

MLflow permite registrar y comparar los experimentos realizados durante el entrenamiento.

Iniciar la interfaz:

```bash
mlflow ui --backend-store-uri mlruns --port 5001
```

Acceder desde:

http://localhost:5001

Experimento utilizado:

`churn-prediction-fintech`

Se registran cinco runs por entrenamiento completo:

1. baseline
2. logistic
3. random_forest
4. xgboost
5. gradient_boosting

Los modelos se comparan mediante las métricas:

- `cv_f1_mean`
- `cv_roc_auc_mean`
- `cv_precision_mean`
- `cv_recall_mean`

## 9. Estructura del proyecto

```text
mlops-churn-lab/
├── app/
│   └── main.py
├── configs/
│   └── churn.yaml
├── data/
│   └── telco_customer_churn_mlops.csv
├── docs/
│   └── crisp-mlq.md
├── reports/
│   ├── evaluation_report.json
│   └── model_card.md
├── src/
│   ├── common.py
│   ├── validate_data.py
│   ├── features.py
│   ├── data_prep.py
│   ├── train.py
│   └── evaluate_and_register.py
├── tests/
├── .github/
│   └── workflows/
│       └── mlops.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

Los modelos entrenados y los archivos generados se almacenan localmente o como artefactos del pipeline, según la configuración.

## 10. Ejecución del laboratorio

Activar el entorno:

```bash
conda activate churn-lab
```

Ejecutar las pruebas unitarias:

```bash
pytest tests/ -q
```

Validar el dataset:

```bash
python src/validate_data.py
```

Preparar los datos:

```bash
python src/data_prep.py
```

Entrenar los modelos:

```bash
python src/train.py --model all
```

Evaluar y registrar el mejor modelo:

```bash
python src/evaluate_and_register.py
```

## 11. Resultados y conclusiones

- Se implementó la metodología CRISP-ML(Q) en las fases 1 a 4.
- Se incorporó un contrato de datos para verificar la calidad del dataset.
- Se integró el preprocesamiento con el modelo mediante un Pipeline de scikit-learn.
- Se entrenaron cuatro modelos candidatos y un baseline.
- Se utilizó validación cruzada estratificada de cinco folds.
- Se registraron los experimentos en MLflow.
- Se incorporó Gradient Boosting como cuarto modelo obligatorio.
- Random Forest obtuvo el mejor F1 de validación cruzada (0.6900).
- El cuarto modelo no modificó la selección del algoritmo ganador.
- Se configuró un pipeline CI/CD de cinco jobs en GitHub Actions.

## 12. Próxima etapa — Laboratorio 14.2

El Laboratorio 14.2 continuará con las fases de despliegue y monitoreo, incluyendo:

- Actualización de la API FastAPI.
- Pruebas de consistencia entre entrenamiento e inferencia.
- Despliegue mediante Docker.
- Detección de drift.
- Monitoreo del rendimiento.
- Reentrenamiento automático.
```
