# RF Device Fingerprinting Pipeline

Pipeline local para inventariar datasets RF, entrenar modelos de identificacion de dispositivos, validar resultados, comparar tecnicas y ejecutar prediccion sobre muestras reales I/Q.

## Datasets soportados

- `kri_wifi`: SigMF WiFi con transmisores USRP X310 (`3123D52`, `3123D54`, etc.), dtype `cf32`.
- `uav_lightbridge`: bursts DJI/UAV Lightbridge, dtype `cf16_le`.
- `ieee_cbrs`: SigMF IEEE/CBRS, dtype `cf32_le`; puede usarse con `ieee_target=band` u otros objetivos si no hay suficientes identidades de dispositivo.
- `wifi_dat_day`: archivos `.dat` crudos en `WiFi-Dataset/Indoor/Day_1/Device_*`, dtype `cf32`.

## Arranque web

```powershell
python rf_ai_pipeline\web_app.py --host 127.0.0.1 --port 8080
```

Abrir:

```text
http://127.0.0.1:8080
```

La interfaz permite seleccionar dataset, modelo, ventana I/Q, seleccion de ventanas, presupuesto por GB o porcentaje, numero de predicciones de control y objetivo de etiquetas IEEE.

## Logica de botones

### Train from scratch

`Train from scratch` crea una nueva version desde cero con los parametros actuales: dataset, split, features, ventanas, presupuesto, modelo y seed.

Siempre se conserva una version trazable en:

```text
models/versions/<dataset>/<modelo>/<dataset>_<modelo>_<version>.joblib
models/versions/<dataset>/<modelo>/<dataset>_<modelo>_<version>.pt
```

Tambien se actualiza el alias operativo usado por prediccion rapida:

```text
models/<dataset>_<modelo>_model.joblib
models/<dataset>_<modelo>_model.pt
```

### Retrain

`Retrain` vuelve a entrenar un modelo existente con nueva seed, nuevos datos, mas ventanas, otro presupuesto o configuracion distinta.

El reporte registra:

- `training_action = retrain`;
- `base_model_path`;
- `base_report_path`;
- `base_metrics`;
- `model_version`;
- `random_state`;
- `trained_at`;
- metricas comparables contra la version anterior.

### Compare

`Comparar tecnicas IA` por defecto solo evalua modelos ya entrenados. No entrena ni reentrena en silencio.

El campo `Modo de comparacion` tiene tres opciones:

- `Solo evaluar modelos ya entrenados`: no modifica modelos ni reportes de validacion.
- `Reentrenar copia experimental`: toma el modelo actual como base, entrena una version experimental y no pisa el alias operativo.
- `Train from scratch + retrain experimental`: crea un train desde cero y un retrain experimental para medir estabilidad; tampoco pisa el alias operativo.

El modo normal ejecuta:

1. cargar modelos existentes;
2. leer metricas originales guardadas;
3. ejecutar predicciones de control;
4. contar aciertos y fallos;
5. medir tiempo de inferencia;
6. generar ranking global y ranking por familia.

Score normal:

```text
0.40 * macro_f1_original
+ 0.25 * balanced_accuracy_original
+ 0.20 * accuracy_original
+ 0.15 * prediction_accuracy
```

Si se usa un modo experimental, el benchmark queda marcado como:

```text
benchmark_train_from_scratch
benchmark_retrain_existing_stability
benchmark_retrain_stability
```

Score con estabilidad:

```text
0.35 * macro_f1_retrain
+ 0.25 * balanced_accuracy_retrain
+ 0.20 * accuracy_retrain
+ 0.15 * prediction_accuracy
+ 0.05 * (1 - stability_gap_macro_f1)
```

## Modelos disponibles

Modelos clasicos y baselines:

- `knn`
- `svm_linear`
- `svm_rbf`
- `random_forest`
- `extra_trees`
- `logistic_regression`
- `mlp`
- `hist_gradient_boosting`

Modelos deep learning sobre I/Q crudo:

- `cnn1d_iq`
- `lstm_iq`
- `transformer_iq`

Los modelos clasicos se guardan como `.joblib`. Los modelos deep learning se guardan como `.pt` de PyTorch.

## Familias cientificas de modelos

El benchmark normaliza cada tecnica con `model_family`:

```text
machine_learning_classical: knn, svm_linear, svm_rbf, random_forest, extra_trees
linear_baseline: logistic_regression
boosting: hist_gradient_boosting
dense_neural_network: mlp
deep_learning_iq: cnn1d_iq, lstm_iq, transformer_iq
```

Por familia se calcula:

- numero de modelos;
- mejor modelo;
- mejor score;
- media de score;
- media de Macro-F1;
- media de balanced accuracy;
- media de prediction accuracy;
- tiempo total medio;
- tiempo total acumulado.

La pestana `Comparacion cientifica` muestra ranking global, ranking por familia, mejor modelo por familia y una alerta si `deep_learning_iq` no supera al mejor ML clasico.

## Reportes generados

Por entrenamiento:

```text
reports/<dataset>_<modelo>_validation.json
reports/<dataset>_validation.json
reports/<dataset>_<modelo>_history.json
reports/model_training_history.json
```

Por benchmark:

```text
reports/<dataset>_benchmark.json
reports/<dataset>_benchmark.md
```

Campos importantes en los reportes:

- `training_action`;
- `model_version`;
- `versioned_model_path`;
- `model_path`;
- `model_size_bytes`;
- `random_state`;
- `trained_at`;
- `dataset`;
- `model_kind`;
- `model_family`;
- `requested_max_data_percent`;
- `requested_max_data_gb`;
- `requested_max_data_bytes`;
- `windows`;
- `records_used`;
- `estimated_iq_gb_read`;
- `referenced_data_size_bytes`;
- `holdout_accuracy`;
- `balanced_accuracy`;
- `macro_f1`;
- `confusion_matrix`;
- `sample_predictions`;
- `base_metrics` si es reentrenamiento.

## Metricas mostradas

La interfaz muestra:

- accuracy holdout;
- balanced accuracy;
- Macro-F1;
- precision/recall por clase;
- matriz de confusion;
- aciertos/fallos de prediccion de control;
- confianza de predicciones;
- tiempo medio de inferencia;
- tiempo total de benchmark;
- peso de modelo;
- GB referenciados;
- GB leidos estimados;
- historial de aprendizaje por modelo.

## Historial de aprendizaje

Cada entrenamiento o reentrenamiento anade una entrada a:

```text
reports/<dataset>_<modelo>_history.json
reports/model_training_history.json
```

La pestana `Metricas` dibuja curvas de:

- Macro-F1 por ejecucion;
- accuracy holdout por ejecucion;
- balanced accuracy por ejecucion;
- loss por epoca para `cnn1d_iq`, `lstm_iq` y `transformer_iq`.

Esto ayuda a ver si el modelo mejora realmente con nuevas configuraciones o si solo cambia por azar del split/seed.

## Presupuesto por GB o porcentaje

La interfaz calcula el peso total del dataset y permite limitar el entrenamiento con:

```text
Peso max entrenamiento GB
Porcentaje dataset %
```

Reglas:

- Si se define porcentaje y GB queda vacio, se calcula el presupuesto a partir del peso total del dataset.
- Si se definen GB y porcentaje, manda el valor en GB.
- Si ambos quedan vacios, se usan todos los archivos elegibles.
- El presupuesto por GB/% tiene prioridad sobre `Limite total` y `Max archivos por clase`.

El limite se aplica a archivos IQ completos, no a bytes sueltos dentro del archivo. Por eso el reporte separa:

- `Presupuesto`: cantidad pedida.
- `GB referenciados`: peso completo de los archivos seleccionados.
- `GB leidos est.`: bytes realmente leidos para extraer ventanas I/Q.

En datasets con archivos grandes, como `kri_wifi`, es normal leer pocos MB aunque los archivos referenciados pesen varios GB, porque el pipeline entrena con ventanas y no con todo el archivo.

## Dataset `.dat`: WiFi-Dataset

Ruta esperada:

```text
WiFi-Dataset/Indoor/Day_1/Device_1/tx_1_iq.dat
WiFi-Dataset/Indoor/Day_1/Device_2/tx_1_iq (1).dat
WiFi-Dataset/Indoor/Day_1/Device_3/tx_1_iq (2).dat
WiFi-Dataset/Indoor/Day_1/Device_4/tx_1_iq (3).dat
WiFi-Dataset/Indoor/Day_1/Device_5/tx_1_iq (4).dat
WiFi-Dataset/Indoor/Day_1/Device_6/tx_1_iq (5).dat
```

Formato inferido:

```text
dataset = wifi_dat_day
extension = .dat
dtype = cf32
label = Device_1 ... Device_6
```

`cf32` significa pares `float32` intercalados como I/Q little-endian.

Como hay pocos archivos por dispositivo, la validacion de `wifi_dat_day` es estratificada por ventanas dentro del archivo. Es util para probar el lector `.dat`, pero es menos estricta que separar capturas completas entre train/test.

Si varios dispositivos estan dentro del mismo `.dat`, crear:

```text
WiFi-Dataset/labels.csv
```

Columnas minimas:

```csv
data_path,label,start_sample,end_sample,dtype
Indoor/Day_1/tx_1_iq.dat,device_001,0,5000000,cf32
Indoor/Day_1/tx_1_iq.dat,device_002,5000000,10000000,cf32
Indoor/Day_1/tx_1_iq.dat,device_003,10000000,15000000,cf32
```

Tambien se aceptan `path` o `file` en vez de `data_path`, y `device_id`, `device` o `tx` en vez de `label`.

## Recomendaciones accionables

Si el reporte detecta Macro-F1 bajo, baja prediccion de control o inestabilidad, la interfaz muestra un bloque de medidas recomendadas.

Botones:

- `Adoptar mejoras en parametros`: cambia el formulario para el siguiente train, retrain o compare.
- `Adoptar y comparar otra vez`: aplica cambios y lanza de nuevo la comparacion.

Ejemplo para `kri_wifi` con Macro-F1 bajo:

```text
Seleccion de ventanas = energia RF
Ventana IQ = 32768 o 65536
Ventanas por archivo >= 6
Predicciones control >= 20
```

## CLI basico

Inventario:

```powershell
python rf_ai_pipeline\rf_device_pipeline.py discover --dataset kri_wifi
python rf_ai_pipeline\rf_device_pipeline.py discover --dataset uav_lightbridge
python rf_ai_pipeline\rf_device_pipeline.py discover --dataset ieee_cbrs --limit-files 500
```

Entrenamiento:

```powershell
python rf_ai_pipeline\rf_device_pipeline.py train --dataset kri_wifi --model-kind extra_trees --window-strategy energy --window-size 32768 --windows-per-file 6
python rf_ai_pipeline\rf_device_pipeline.py train --dataset uav_lightbridge --model-kind random_forest --max-files-per-class 300 --window-size 4096 --windows-per-file 2
python rf_ai_pipeline\rf_device_pipeline.py train --dataset wifi_dat_day --model-kind cnn1d_iq --window-size 32768 --windows-per-file 3
```

Prediccion:

```powershell
python rf_ai_pipeline\rf_device_pipeline.py predict --model models\kri_wifi_extra_trees_model.joblib --meta neu_m044q5210\KRI-16Devices-RawData\14ft\WiFi_air_X310_3123D52_14ft_run1.sigmf-meta
```

Resumen:

```powershell
python rf_ai_pipeline\rf_device_pipeline.py summary
```

## Nota cientifica

Un resultado alto con split aleatorio o por ventanas no demuestra necesariamente fingerprinting robusto. Para justificar identificacion real de dispositivos hay que controlar fuga por captura, sesion, dia, receptor, distancia, canal y SNR.

`Compare` normal sirve para evaluar modelos guardados. `Compare with retraining stability test` sirve para medir estabilidad experimental, pero debe reportarse como reentrenamiento experimental, no como simple evaluacion.
