# Proyecto Final IAO: Clasificador de Setas

Este repositorio contiene el código y los recursos del trabajo final para la asignatura de Inteligencia Artificial y Optimización (IAO). El objetivo principal del proyecto es desarrollar un sistema capaz de identificar el género de una seta a partir de una imagen.

## Descripción del Proyecto

Para resolver el problema de clasificación de setas, hemos implementado y comparado dos arquitecturas de Deep Learning diferentes:

1.  **MobileNetV2**: Utilizando técnicas de Transfer Learning sobre la red preentrenada en ImageNet. Se ha realizado un fine-tuning en dos fases para adaptar la red a nuestro dataset.
2.  **YOLOv8**: Utilizando el modelo YOLOv8m-cls orientado a tareas de clasificación de imágenes en tiempo real.

El sistema está diseñado para distinguir entre 9 géneros de setas y cuenta con una categoría más ("otras") para descartar imágenes con baja confianza.

## Contenido del Repositorio y Estructura

Qué es cada archivo y carpeta?
 **Nota importante:** La carpeta `Mushrooms` (con las imágenes) no se incluye en la entrega por tamaño, pero es necesaria para que todo funcione.

### Estructura de Carpetas
Para que los scripts funcionen, la carpeta del proyecto debería ser algo así:

*   `Mushrooms/`: **(descargar dataset de Kaggle: https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images)**. Dentro debe haber una carpeta por cada tipo de seta (`Agaricus`, `Amanita`, etc.) con sus fotos. Cuidado porque descargar el dataset de Kaggle puede crear dos carpetas `mushrooms/Mushrooms/Amanita`.
*   `Mushrooms_YOLO/`: al crea YOLO a partir de la anterior
*   `imagenet/`:  donde el modelo MobileNet guarda sus cosas (modelos entrenados, gráficas, logs...).
*   `runs/`:  donde YOLO guarda sus entrenamientos.
*   `test_results/`: donde aparecen las fotos con las predicciones cuando ejecutas los tests.
*   `scraped_mushrooms/`: Carpeta donde se descargan las imágenes raw si usas el web scraper.

### Descripción de Archivos
*   `imagenet.py`: Script principal para entrenar el modelo MobileNetV2 (TensorFlow).
*   `yolov8.py`: Script principal para entrenar el modelo YOLOv8 (PyTorch).
*   `test_yolo.py` y `test_imagenet.py`: Scripts para probar los modelos ya entrenados con imágenes nuevas.
*   `mushroom_cnn_train.py`: Nuestro primer intento con una CNN propia desde cero (para aprender).
*   `scrape_mushrooms.py` y `config_mushrooms.py`: Scripts de Web Scraping para bajar fotos de First Nature.
*   `data_augmentation_GAN.py`: Código de GANs para generar imágenes sintéticas (no usado en el modelo final, usamos mixup y otras técnicas en imagenet y yolo en su lugar).

## Requisitos e Instalación

El proyecto utiliza dos entornos de Conda separados para evitar conflictos entre TensorFlow y PyTorch. A continuación se explica para la reproducibilidad para cada uno con las versiones exactas que hemos utilizado.

### 1. Entorno para MobileNetV2 (`iao_tf`)

Este entorno utiliza **Python 3.9** y **TensorFlow 2.10.1** (versión específica para soporte GPU nativo en Windows).

```bash
conda create -n iao_tf python=3.9
conda activate iao_tf

# Instalar librerías de CUDA necesarias para TensorFlow
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# Instalar TensorFlow y librerías de procesamiento de datos
pip install tensorflow==2.10.1
pip install matplotlib seaborn scikit-learn pandas pillow
```

Para ejecutar el entrenamiento:
```bash
python imagenet.py
```

### 2. Entorno para YOLOv8 (`iao`)

Este entorno usa **Python 3.8** y **PyTorch**.

```bash
conda create -n iao python=3.8
conda activate iao

# Instalar PyTorch (puede que se necesite una versión específica segun la gráfica)
pip install torch torchvision

# Instalar YOLO y el resto de librerías necesarias
pip install ultralytics matplotlib seaborn scikit-learn pandas opencv-python
```

Para ejecutar el entrenamiento:
```bash
python yolov8.py
```

### 3. Web Scraping

Se recomienda usar un entorno con Python 3.10+ con las siguientes librerías:

```bash
pip install requests beautifulsoup4
```

Para ejecutar la recolección de imagenes:
```bash
python scrape_mushrooms.py
```

Las imagenes obtenidas se almacenarán en la carpeta ./scraped_mushrooms/

### 4. GAN

Se recomienda usar un entorno con Python 3.8 con las siguientes librerías:
```bash
pip install tensorflow numpy matplotlib
```
Es necesario que la carpeta con las imagenes y el archivo del GAN se encuentren
en el mismo directorio.

Para ejecutar la generación de imágenes mediante GANs:
```bash
python data_augmentation_GAN.py
```
Las imagenes generadas se almacenarán en la carpeta ./mushrooms_augmented/

### 5. Entorno para CNN propia 

Este entorno virtual usa **Python 3.12**.
Se requiere que se instalen las librerias con las versiones que estan dentro de requirements.txt 

```bash
# Ejecutar el entrenamiento
python mushroom_cnn_train.py

# Ejecutar el test
python mushroom_cnn_test.py
```
## Notas sobre el Entrenamiento

*   Ambos scripts están configurados para usar aceleración por GPU si está disponible (CUDA).
*   Se han aplicado técnicas de Data Augmentation (rotación, mixup, cambios de color) para mejorar la generalización.
*   Los resultados (accuracy, loss, matrices de confusión) se generan automáticamente al finalizar la ejecución en sus respectivas carpetas.

## Autores

Trabajo realizado por alumnos de la UC3M para la asignatura de IAO.
