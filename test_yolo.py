"""
Test dedicado para el modelo YOLO (Ultralytics clasificación)
Uso ejemplos:
    conda activate iao
    python test_yolo.py --image "Mushrooms/Agaricus/000_ePQknW8cTp8.jpg"
    python test_yolo.py --folder "imagenes_nuevas" --save
"""

# Importar librerías necesarias
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Configuración básica
MODEL_PATH = 'yolo/weights/best.pt'  # Ruta donde debería estar el modelo
CLASSES = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']
IMG_SIZE = 224
UNKNOWN_THRESHOLD = 0.7

# Intentar importar YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics no está instalado. Ejecuta: pip install ultralytics")
    raise SystemExit(1)

def find_weights():
    # Buscar los pesos si no están en la ruta por defecto
    # 1. Ruta principal
    main_path = Path(MODEL_PATH)
    if main_path.exists():
        return main_path
    # 2. Buscar en runs/classify/*/weights/best.pt
    runs_dir = Path('runs/classify')
    if runs_dir.exists():
        candidates = []
        for exp in runs_dir.glob('*'):
            w = exp / 'weights' / 'best.pt'
            if w.exists():
                candidates.append(w)
        if candidates:
            # Coger el más nuevo
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
    return None

weights_path = find_weights()
if not weights_path:
    print("No encuentro el modelo entrenado.")
    print("Tienes que entrenarlo primero con:")
    print("    conda activate iao")
    print("    python yolov8.py")
    raise SystemExit(1)

print(f"Cargando modelo desde: {weights_path}")
model = YOLO(str(weights_path))


def predict_image(image_path: Path):
    # Hacer la predicción
    results = model.predict(str(image_path), imgsz=IMG_SIZE, verbose=False)
    pred_idx = results[0].probs.top1
    conf = results[0].probs.top1conf.item()
    
    # Si no está muy seguro, decimos que es 'otras'
    if conf < UNKNOWN_THRESHOLD:
        label = 'otras'
    else:
        label = CLASSES[pred_idx]

    # Sacar el top 3 para ver con qué se confunde
    top3_indices = results[0].probs.top5[:3]
    top3 = []
    for i in top3_indices:
        prob = results[0].probs.data[i].item()
        cls_name = CLASSES[i]
        top3.append((cls_name, prob))
    return label, conf, top3


def show_result(image_path: Path, result, save=False):
    # Mostrar la imagen con el resultado
    label, conf, top3 = result
    img = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(8,6))
    plt.imshow(img)
    plt.axis('off')
    
    # Colores según confianza (verde=bien, naranja=regular, rojo=mal)
    color = '#2ecc71' if conf > 0.8 else '#f39c12' if conf > 0.6 else '#e74c3c'
    plt.title(f"YOLO dice: {label} ({conf:.1%})", color=color, fontsize=14, fontweight='bold')
    
    # Poner el top 3 en la imagen
    y = 0.05
    for cls, c in top3:
        plt.text(0.02, y, f"{cls}: {c:.1%}", transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        y += 0.05
        
    if save:
        # Guardar en la carpeta de resultados con timestamp para no borrar anteriores
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path('test_results')
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{image_path.stem}_yolo_{timestamp}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Imagen guardada en: {out_path}")
    plt.show(); plt.close()


def process_folder(folder: Path, save=False):
    # Procesar todas las imágenes de una carpeta
    exts = {'.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'}
    images = [p for p in folder.iterdir() if p.suffix in exts]
    
    if not images:
        print("No hay imágenes en esa carpeta.")
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
        print(f" -> {img_path.name}: {label} ({conf:.1%})")
        
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

    # Guardar resultados en CSV con timestamp
    csv_path = out_dir / f'yolo_batch_{timestamp}.csv'
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nCSV guardado en {csv_path}")
    
    # Guardar resumen en TXT
    txt_path = out_dir / f'yolo_summary_{timestamp}.txt'
    with open(txt_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"Resumen guardado en {txt_path}")


def main():
    parser = argparse.ArgumentParser(description='Test del modelo YOLO con nuevas imágenes')
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
