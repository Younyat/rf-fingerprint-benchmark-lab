# RF Device Fingerprinting Pipeline

Este flujo crea tres modelos supervisados, uno por dataset:

- `kri_wifi`: identifica el USRP X310 transmisor desde archivos SigMF WiFi (`3123D52`, `3123D54`, etc.).
- `uav_lightbridge`: identifica el UAV/dispositivo DJI M100 desde bursts `cf16_le`.
- `ieee_cbrs`: intenta identificar emisores/receptores desde metadatos SigMF; si solo hay una identidad de dispositivo, se puede entrenar por `--ieee-target band` como control de pipeline.

## Comandos

Inventario:

```powershell
python rf_ai_pipeline\rf_device_pipeline.py discover --dataset kri_wifi
python rf_ai_pipeline\rf_device_pipeline.py discover --dataset uav_lightbridge
python rf_ai_pipeline\rf_device_pipeline.py discover --dataset ieee_cbrs --limit-files 500
```

Entrenamiento y validacion con muestras reales:

```powershell
python rf_ai_pipeline\rf_device_pipeline.py train --dataset kri_wifi --model-kind random_forest --window-size 4096 --windows-per-file 3
python rf_ai_pipeline\rf_device_pipeline.py train --dataset uav_lightbridge --model-kind svm_rbf --max-files-per-class 120 --window-size 4096 --windows-per-file 2
python rf_ai_pipeline\rf_device_pipeline.py train --dataset ieee_cbrs --model-kind extra_trees --max-files-per-class 200 --ieee-target band --window-size 4096 --windows-per-file 2
```

Modelos disponibles para comparar:

- `knn`: baseline de vecinos, como en trabajos con fingerprints manuales.
- `svm_linear`: separabilidad lineal de features RF.
- `svm_rbf`: SVM no lineal, fuerte en datasets medianos.
- `random_forest`: ensemble de arboles para features tabulares.
- `extra_trees`: ensemble mas aleatorizado.
- `logistic_regression`: baseline probabilistico lineal.
- `mlp`: red neuronal densa ligera sobre features IQ/espectrales.
- `hist_gradient_boosting`: boosting no lineal para features tabulares.

Ejemplo de comparacion manual:

```powershell
python rf_ai_pipeline\rf_device_pipeline.py train --dataset uav_lightbridge --model-kind knn --max-files-per-class 80 --window-size 4096 --windows-per-file 2
python rf_ai_pipeline\rf_device_pipeline.py train --dataset uav_lightbridge --model-kind svm_rbf --max-files-per-class 80 --window-size 4096 --windows-per-file 2
python rf_ai_pipeline\rf_device_pipeline.py train --dataset uav_lightbridge --model-kind mlp --max-files-per-class 80 --window-size 4096 --windows-per-file 2
python rf_ai_pipeline\rf_device_pipeline.py summary
```

Prediccion:

```powershell
python rf_ai_pipeline\rf_device_pipeline.py predict --model models\kri_wifi_device_model.joblib --meta neu_m044q5210\KRI-16Devices-RawData\14ft\WiFi_air_X310_3123D52_14ft_run1.sigmf-meta
```

Reentrenamiento:

```powershell
python rf_ai_pipeline\rf_device_pipeline.py retrain --dataset kri_wifi --window-size 4096 --windows-per-file 3
```

Resumen:

```powershell
python rf_ai_pipeline\rf_device_pipeline.py summary
```

Los modelos se guardan en `models/` y los reportes de validacion en `reports/`.

## Interfaz web local

Arranque:

```powershell
python rf_ai_pipeline\web_app.py --host 127.0.0.1 --port 8080
```

Abra:

```text
http://127.0.0.1:8080
```

La interfaz permite:

- seleccionar `kri_wifi`, `uav_lightbridge` o `ieee_cbrs`;
- seleccionar tecnica IA concreta o comparar todas las tecnicas disponibles;
- generar inventario del dataset con clases, dtype, sample_count y ejemplos;
- entrenar y reentrenar modelos con parametros controlados;
- revisar accuracy holdout, balanced accuracy, macro-F1, precision, recall, matriz de confusion y predicciones de control;
- seleccionar una muestra real del dataset y ejecutar prediccion contra el modelo guardado;
- ver jobs, errores y JSON de resultados para reproducibilidad.

## Comparacion rigurosa de tecnicas IA

La pestaña `Comparacion cientifica` esta dedicada solo a comparar modelos. El boton `Comparar tecnicas IA` ejecuta un protocolo completo por cada modelo:

1. Entrena el modelo con los parametros seleccionados.
2. Reentrena el mismo modelo con otra semilla para medir estabilidad.
3. Ejecuta predicciones sobre muestras reales etiquetadas del dataset.
4. Cuenta aciertos y fallos de prediccion.
5. Mide tiempo de entrenamiento, reentrenamiento, inferencia media y tiempo total.
6. Calcula un ranking ponderado.
7. Guarda el benchmark en `reports/<dataset>_benchmark.json`.

Score usado en el ranking:

```text
0.35 * macro_f1_retrain
+ 0.25 * balanced_accuracy_retrain
+ 0.20 * accuracy_retrain
+ 0.15 * prediction_accuracy
+ 0.05 * (1 - stability_gap_macro_f1)
```

Esto evita comparar solo por accuracy. El ranking favorece modelos que funcionan bien por clase, aguantan el reentrenamiento y aciertan predicciones reales de control.

La tabla comparativa muestra por modelo:

- ranking y score;
- aciertos/fallos de prediccion sobre muestras reales;
- accuracy de prediccion de control;
- macro-F1 en train inicial y retrain;
- balanced accuracy del retrain;
- gap de estabilidad entre train y retrain;
- tiempo de entrenamiento;
- tiempo de reentrenamiento;
- tiempo medio de prediccion;
- tiempo total del benchmark.

La pagina tambien incluye:

- graficas de barras para score, prediccion, Macro-F1 y coste temporal;
- lectura critica automatica cuando hay contradicciones entre metricas;
- aviso si el mejor modelo tiene Macro-F1 bajo o poca estabilidad;
- predicciones de control tomadas del holdout del reentrenamiento, no de muestras arbitrarias del dataset;
- split estratificado por clase y separado por archivo/captura para reducir fuga de informacion y mantener representacion de dispositivos.

## Modelos guardados y prediccion

Cada modelo queda guardado por dataset y tecnica:

```text
models/<dataset>_<model_kind>_model.joblib
reports/<dataset>_<model_kind>_validation.json
```

La interfaz carga esos modelos desde `/api/models?dataset=<dataset>` y muestra sus ultimas metricas en `Metricas > Modelos entrenados guardados`.

La prediccion usa el modelo seleccionado en `Modelo IA`. Si existe `models/<dataset>_<model_kind>_model.joblib`, se usa ese archivo. Si no existe, se muestra en la tabla que no hay modelo para esa tecnica.

Cada dataset mantiene su propia serie de modelos. Los archivos se guardan con el patron:

```text
models/<dataset>_<modelo>_model.joblib
reports/<dataset>_<modelo>_validation.json
```

La serie mostrada en `Metricas` es:

```text
hist_gradient_boosting
random_forest
extra_trees
mlp
svm_rbf
knn
svm_linear
logistic_regression
```

Cambiar de dataset no sobreescribe modelos de otro dataset. Entrenar `kri_wifi_random_forest` no toca `uav_lightbridge_random_forest`.

## Dataset `.dat` local: WiFi-Dataset

Se detecto un dataset crudo en:

```text
WiFi-Dataset/Indoor/Day_1/Device_1/tx_1_iq.dat
WiFi-Dataset/Indoor/Day_1/Device_2/tx_1_iq (1).dat
WiFi-Dataset/Indoor/Day_1/Device_3/tx_1_iq (2).dat
WiFi-Dataset/Indoor/Day_1/Device_4/tx_1_iq (3).dat
WiFi-Dataset/Indoor/Day_1/Device_5/tx_1_iq (4).dat
WiFi-Dataset/Indoor/Day_1/Device_6/tx_1_iq (5).dat
```

Cada archivo pesa 400 MB. El formato inferido es `cf32` little-endian: pares `float32` intercalados como I/Q. Cada archivo contiene aproximadamente 50.000.000 muestras complejas.

El pipeline lo registra como:

```text
dataset = wifi_dat_day
dtype = cf32
extension = .dat
label = Device_1 ... Device_6
```

En esta estructura, cada carpeta `Device_*` se usa como etiqueta de dispositivo. Como solo hay un archivo por dispositivo, el pipeline usa validacion estratificada por ventanas dentro de cada archivo para este dataset. Es una validacion util para probar el lector `.dat`, pero menos estricta que separar capturas completas entre train/test.

Si en el futuro varios dispositivos estan dentro del mismo `.dat`, hace falta un manifiesto de segmentos. El pipeline soporta:

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

Tambien se aceptan nombres equivalentes: `path` o `file` en vez de `data_path`, y `device_id`, `device` o `tx` en vez de `label`.

Sin `labels.csv`, el pipeline solo puede etiquetar por `Day_1 ... Day_6` o por `tx_1` inferido del nombre del archivo. Para fingerprinting real de dispositivos dentro de `Day_1`, el `labels.csv` debe indicar que rango de muestras pertenece a cada dispositivo.

Para KRI WiFi, si el Macro-F1 es bajo, no basta con reentrenar igual. Pruebe primero:

```powershell
python rf_ai_pipeline\rf_device_pipeline.py train --dataset kri_wifi --model-kind extra_trees --window-strategy energy --window-size 32768 --windows-per-file 6
```

En la interfaz use `Seleccion de ventanas = energia RF`, `Ventana IQ = 32768` o `65536`, y aumente `Ventanas por archivo`.

## Entrenamiento por peso o porcentaje del dataset

La interfaz muestra el peso total de cada dataset y permite limitar el entrenamiento con `Peso max entrenamiento GB` o `Porcentaje dataset %`.

Ejemplos:

```text
Peso max entrenamiento GB = 1
Peso max entrenamiento GB = 2
Peso max entrenamiento GB = 3
Porcentaje dataset % = 5
Porcentaje dataset % = 10
```

Si el campo GB queda vacio y se define un porcentaje, la interfaz calcula automaticamente cuantos MB/GB representa sobre el total del dataset. Si se rellenan GB y porcentaje a la vez, manda el valor en GB. Si ambos quedan vacios, se usan todos los archivos elegibles.

Cuando se usa presupuesto por GB o por porcentaje, ese presupuesto tiene prioridad sobre `Limite total` y `Max archivos por clase`. Esto evita pedir, por ejemplo, 10% del dataset y que el experimento se corte accidentalmente por `Limite total = 10`.

El limite se aplica a archivos IQ completos, no a bytes sueltos dentro de un archivo. Por eso el reporte separa:

- `Presupuesto`: cantidad pedida por GB o porcentaje.
- `GB referenciados`: peso completo de los archivos seleccionados.
- `GB leidos est.`: bytes realmente leidos al extraer ventanas IQ.

En datasets con archivos grandes, como `kri_wifi`, es normal ver pocos MB/GB leidos aunque los archivos referenciados pesen varios GB: el pipeline no entrena con el archivo entero, sino con ventanas extraidas de cada captura. Esto evita tiempos enormes, pero debe documentarse en el reporte para justificar el experimento.

Para `uav_lightbridge`, el peso total no parece enorme, pero hay 13.893 archivos pequenos. El coste fuerte es abrir miles de JSON/binarios y extraer ventanas/features de cada burst. Para pruebas rapidas use:

```text
Max archivos por clase = 100
Peso max entrenamiento GB = vacio
Ventanas por archivo = 1 o 2
```

Para una comparacion mas seria pero todavia razonable:

```text
Max archivos por clase = 300
Ventanas por archivo = 2
```

Evite usar todo `uav_lightbridge` en `Comparar tecnicas IA` salvo que quiera esperar mucho tiempo, porque compara 8 modelos y cada uno hace train + retrain + predicciones.

## Recomendaciones accionables

La pestaña `Comparacion cientifica` no solo muestra una lectura critica. Si el reporte detecta Macro-F1 bajo, baja prediccion de control o inestabilidad entre train y retrain, la interfaz genera un bloque de medidas recomendadas.

Botones disponibles:

- `Adoptar mejoras en parametros`: cambia el formulario para el siguiente entrenamiento, reentrenamiento o comparacion.
- `Adoptar y comparar otra vez`: aplica los cambios y lanza de nuevo la comparacion cientifica.

Ejemplo para `kri_wifi` con Macro-F1 bajo: se activa seleccion por energia RF, ventana IQ mayor, mas ventanas por archivo y mas predicciones de control.

Equivalente por CLI:

```powershell
python rf_ai_pipeline\rf_device_pipeline.py train --dataset uav_lightbridge --model-kind extra_trees --max-data-gb 1 --window-size 4096 --windows-per-file 2
```

Nota cientifica: `ieee_cbrs` no expone multiples identidades de dispositivo con `--ieee-target auto` en las muestras verificadas; para ese dataset el front permite usar `band` u otros objetivos de metadatos como control de pipeline, pero eso no debe reportarse como identificacion unica de dispositivo.
