# Proyecto Final IAO: Clasificador de Setas

Este repositorio contiene el código y los recursos del trabajo final para la asignatura de Inteligencia Artificial y Optimización (IAO). El objetivo principal del proyecto es desarrollar un sistema capaz de identificar el género de una seta a partir de una imagen.

## Descripción del Proyecto

Para resolver el problema de clasificación de setas, hemos implementado y comparado dos arquitecturas de Deep Learning diferentes:

1.  **MobileNetV2**: Utilizando técnicas de Transfer Learning sobre la red preentrenada en ImageNet. Se ha realizado un ajuste fino (fine-tuning) en dos fases para adaptar la red a nuestro dataset.
2.  **YOLOv8**: Utilizando el modelo YOLOv8m-cls orientado a tareas de clasificación de imágenes.

El sistema está diseñado para distinguir entre 9 géneros de setas y cuenta con una categoría adicional ("otras") para descartar imágenes con baja confianza.

## Contenido del Repositorio

*   `imagenet.py`: Script principal para el entrenamiento y evaluación del modelo basado en MobileNetV2 (TensorFlow).
*   `yolov8.py`: Script principal para el entrenamiento y evaluación del modelo YOLOv8 (PyTorch).
*   `Mushrooms/`: Directorio que debe contener el dataset de imágenes organizado por carpetas (clases).
*   `imagenet/`: Carpeta donde se guardan los resultados, modelos (.h5) y gráficas del modelo MobileNet.
*   `yolo/`: Carpeta donde se guardan los resultados y gráficas del modelo YOLO.
*   `scraped_mushrooms/`: Carpeta con las imágenes obtenidas mediante web scraping.

## Requisitos y Ejecución

El proyecto utiliza dos entornos diferentes debido a las dependencias de las librerías.

### 1. Modelo MobileNetV2 (TensorFlow)

Se recomienda usar un entorno Conda con Python 3.9:

```bash
conda create -n iao_tf python=3.9
conda activate iao_tf
pip install tensorflow matplotlib seaborn scikit-learn pillow
```

Para ejecutar el entrenamiento:
```bash
python imagenet.py
```

### 2. Modelo YOLOv8 (PyTorch)

Se recomienda usar un entorno Conda con Python 3.8:

```bash
conda create -n iao python=3.8
conda activate iao
pip install ultralytics matplotlib seaborn scikit-learn
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


## Notas sobre el Entrenamiento

*   Ambos scripts están configurados para usar aceleración por GPU si está disponible (CUDA).
*   Se han aplicado técnicas de Data Augmentation (rotación, mixup, cambios de color) para mejorar la generalización.
*   Los resultados (accuracy, loss, matrices de confusión) se generan automáticamente al finalizar la ejecución en sus respectivas carpetas.

## Autores

Trabajo realizado por alumnos de la UC3M para la asignatura de IAO.
