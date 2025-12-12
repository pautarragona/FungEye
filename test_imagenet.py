"""
Test dedicado para el modelo imagenet (MobileNetV2 / TensorFlow)
Uso ejemplos:
    conda activate iao_tf
    python test_imagenet.py --image "Mushrooms/Agaricus/000_ePQknW8cTp8.jpg"
    python test_imagenet.py --folder "imagenes_nuevas" --save
"""

# Importar librerías
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# Configuración
MODEL_PATH = 'imagenet/checkpoints/best_20251126_152956_phase2.h5' # Ruta del modelo
CLASSES = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']
IMG_SIZE = 224
THRESHOLD = 0.7

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow no está instalado. Activa el entorno correcto:")
    print("    conda activate iao_tf")
    raise SystemExit(1)

if not Path(MODEL_PATH).exists():
    print(f"No encuentro el modelo en {MODEL_PATH}")
    raise SystemExit(1)

model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado correctamente")

def predict_image(image_path: Path):
    # Preparar la imagen
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    # Predecir
    preds = model.predict(arr, verbose=0)[0]
    top_idx = np.argmax(preds)
    top_conf = preds[top_idx]
    
    # Umbral de confianza
    if top_conf < THRESHOLD:
        label = 'otras'
    else:
        label = CLASSES[top_idx]
        
    # Sacar top 3
    top3_idx = np.argsort(preds)[-3:][::-1]
    top3 = []
    for i in top3_idx:
        cls_name = CLASSES[i]
        top3.append((cls_name, preds[i]))
    return label, float(top_conf), top3


def show_result(image_path: Path, result, save=False):
    label, conf, top3 = result
    img = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(8,6))
    plt.imshow(img)
    plt.axis('off')
    
    color = '#2ecc71' if conf > 0.8 else '#f39c12' if conf > 0.6 else '#e74c3c'
    plt.title(f"Predicción: {label} ({conf:.1%})", color=color, fontsize=14, fontweight='bold')
    
    y = 0.05
    for cls, c in top3:
        plt.text(0.02, y, f"{cls}: {c:.1%}", transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        y += 0.05
        
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path('test_results')
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{image_path.stem}_imagenet_{timestamp}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"💾 Guardado: {out_path}")

    plt.show()
    plt.close()


def process_folder(folder: Path, save=False):
    exts = {'.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'}
    images = [p for p in folder.iterdir() if p.suffix in exts]
    
    if not images:
        print("No hay imágenes aquí.")
        return
        
    print(f"Procesando {len(images)} imágenes...")
    import pandas as pd
    from collections import Counter
    
    rows = []
    predictions_list = []
    
    for img_path in images:
        label, conf, top3 = predict_image(img_path)
        rows.append({'image': img_path.name, 'prediction': label, 'confidence': conf})
        predictions_list.append(label)
        print(f"{img_path.name}: {label} ({conf:.1%})")
        
        if save:
            show_result(img_path, (label, conf, top3), save=True)
            
    # Generar resumen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path('test_results')
    out_dir.mkdir(exist_ok=True)
    
    print("\n--- Resumen ---")
    print(f"Total: {len(images)}")
    counts = Counter(predictions_list)
    
    summary_lines = [f"Resumen de ejecución - {timestamp}", f"Total imágenes: {len(images)}", "-"*30]
    
    for cls_name, count in counts.most_common():
        line = f"  - {cls_name}: {count} ({count/len(images):.1%})"
        print(line)
        summary_lines.append(line)

    # Guardar CSV
    csv_path = out_dir / f'imagenet_batch_{timestamp}.csv'
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"CSV guardado en {csv_path}")
    
    # Guardar resumen TXT
    txt_path = out_dir / f'imagenet_summary_{timestamp}.txt'
    with open(txt_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"Resumen guardado en {txt_path}")


def main():
    parser = argparse.ArgumentParser(description='Test del modelo imagenet con nuevas imágenes')
    parser.add_argument('--image', type=str, help='Ruta a una imagen')
    parser.add_argument('--folder', type=str, help='Ruta a carpeta con imágenes')
    parser.add_argument('--save', action='store_true', help='Guardar visualizaciones')
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.print_help(); return

    if args.image:
        p = Path(args.image)
        if not p.exists():
            print(f"Imagen no encontrada: {p}"); return
        res = predict_image(p)
        print(f"\nPredicción: {res[0]} ({res[1]:.1%})")
        print("Top-3:")
        for cls, c in res[2]:
            print(f"  - {cls}: {c:.1%}")
        # Guardar siempre la visualización en modo imagen única
        show_result(p, res, save=True)
    else:
        f = Path(args.folder)
        if not f.exists():
            print(f"Carpeta no encontrada: {f}"); return
        process_folder(f, save=args.save)

if __name__ == '__main__':
    main()
