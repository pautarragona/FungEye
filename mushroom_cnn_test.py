import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
from datetime import datetime

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'Mushrooms'
IMG_SIZE = 224
BATCH_SIZE = 32


# ARQUITECTURA


class MushroomCNN(nn.Module):
    def __init__(self, num_classes):
        super(MushroomCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x


# DATASET


class MushroomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            return self.transform(Image.new('RGB', (IMG_SIZE, IMG_SIZE))), label


# FUNCIONES DE EVALUACIÓN


def cargar_dataset_test():
    """Carga el dataset de test desde el modelo guardado"""
    try:
        # Intentar cargar datos de test del modelo final
        checkpoint = torch.load('mushroom_cnn_final.pth', map_location=DEVICE)
        if 'test_data' in checkpoint:
            X_test = checkpoint['test_data']['X_test']
            y_test = checkpoint['test_data']['y_test']
            class_names = checkpoint['class_names']
            print(" Datos de test cargados desde mushroom_cnn_final.pth")
            return X_test, y_test, class_names
    except:
        pass

    # Si no hay datos guardados, hacer split manual (misma lógica que entrenamiento)
    print(" Generando conjunto de test con la misma división que entrenamiento...")
    image_paths = []
    labels = []

    classes = sorted([d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d))])
    class_names = classes

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(DATA_DIR, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_idx)

    # Dividir igual que en entrenamiento: 70% train, 30% temp
    _, X_temp, _, y_temp = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Dividir temp en val/test: 50%/50% del 30% = 15%/15%
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return X_test, y_test, class_names

def evaluar_modelo(model, dataloader, class_names, device):
    """Evalúa el modelo y retorna predicciones y etiquetas reales"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluando"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)





def plot_class_metrics(report_dict, class_names):
    """Grafica métricas por clase """
    # Extraer métricas usando los nombres de clases como claves
    precision = []
    recall = []
    f1 = []
    support = []

    for class_name in class_names:
        if class_name in report_dict:
            precision.append(report_dict[class_name]['precision'])
            recall.append(report_dict[class_name]['recall'])
            f1.append(report_dict[class_name]['f1-score'])
            support.append(report_dict[class_name]['support'])
        else:
            precision.append(0.0)
            recall.append(0.0)
            f1.append(0.0)
            support.append(0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Precisión por clase
    ax = axes[0, 0]
    bars = ax.barh(class_names, precision, color='skyblue')
    ax.set_xlabel('Precisión', fontsize=12)
    ax.set_title('Precisión por Clase', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    for i, (bar, val) in enumerate(zip(bars, precision)):
        ax.text(val, i, f' {val:.2f}', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # Recall por clase
    ax = axes[0, 1]
    bars = ax.barh(class_names, recall, color='lightcoral')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_title('Recall por Clase', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    for i, (bar, val) in enumerate(zip(bars, recall)):
        ax.text(val, i, f' {int(val)}', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # F1-Score por clase
    ax = axes[1, 0]
    bars = ax.barh(class_names, f1, color='lightgreen')
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score por Clase', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    for i, (bar, val) in enumerate(zip(bars, f1)):
        ax.text(val, i, f' {int(val)}', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # Soporte por clase
    ax = axes[1, 1]
    bars = ax.barh(class_names, support, color='plum')
    ax.set_xlabel('Número de muestras', fontsize=12)
    ax.set_title('Soporte por Clase (muestras de test)', fontsize=14, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, support)):
        ax.text(val, i, f' {val}', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('class_metrics.png', dpi=300, bbox_inches='tight')
    print(" Métricas por clase guardadas en: class_metrics.png")
    plt.show()

def analizar_errores(y_true, y_pred, y_probs, class_names, X_test, top_n=5):
    """Analiza los errores más frecuentes del modelo"""
    print("\n" + "="*70)
    print(" ANÁLISIS DE ERRORES")
    print("="*70)

    # Encontrar errores
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]

    if len(error_indices) == 0:
        print("\n ¡No hay errores! Clasificación perfecta.")
        return

    print(f"\n Total de errores: {len(error_indices)} / {len(y_true)} "
          f"({len(error_indices)/len(y_true)*100:.2f}%)")

    # Errores por clase
    print("\n Errores por clase:")
    for i, class_name in enumerate(class_names):
        class_errors = np.sum((y_true == i) & (y_pred != i))
        class_total = np.sum(y_true == i)
        if class_total > 0:
            print(f"   {class_name}: {class_errors}/{class_total} "
                  f"({class_errors/class_total*100:.1f}%)")

    # Confusiones más comunes
    print(f"\n Top {top_n} confusiones más frecuentes:")
    confusion_pairs = {}
    for true_label, pred_label in zip(y_true[errors], y_pred[errors]):
        pair = (true_label, pred_label)
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    sorted_confusions = sorted(confusion_pairs.items(),
                              key=lambda x: x[1], reverse=True)[:top_n]

    for (true_idx, pred_idx), count in sorted_confusions:
        print(f"   {class_names[true_idx]} → {class_names[pred_idx]}: {count} veces")

    # Predicciones con menor confianza
    confidences = np.max(y_probs, axis=1)
    low_conf_indices = np.argsort(confidences)[:top_n]

    print(f"\n Top {top_n} predicciones con menor confianza:")
    for idx in low_conf_indices:
        true_class = class_names[y_true[idx]]
        pred_class = class_names[y_pred[idx]]
        conf = confidences[idx] * 100
        status = "✓" if y_true[idx] == y_pred[idx] else "✗"
        print(f"   {status} Real: {true_class} | Pred: {pred_class} | "
              f"Confianza: {conf:.2f}%")


# MAIN


def main():
    print("\n" + "="*70)
    print(" "*12 + " EVALUACIÓN DETALLADA CNN CLASIFICADOR DE SETAS")
    print("="*70 + "\n")

    # Cargar modelo
    try:
        checkpoint = torch.load('mushroom_cnn_best.pth', map_location=DEVICE)
        class_names = checkpoint['class_names']
        num_classes = len(class_names)

        model = MushroomCNN(num_classes).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f" Modelo cargado correctamente")
        print(f" Precisión de validación durante entrenamiento: {checkpoint['val_acc']:.2f}%")
    except FileNotFoundError:
        print(" No se encontró mushroom_cnn_best.pth")
        print("   Ejecuta primero: python mushroom_cnn_train.py")
        return

    # Intentar cargar checkpoint final para obtener historial
    try:
        checkpoint_final = torch.load('mushroom_cnn_final.pth', map_location=DEVICE)
        print(" Checkpoint final cargado (incluye historial de entrenamiento)")
    except:
        print(" No se encontró mushroom_cnn_final.pth")
        checkpoint_final = checkpoint  # Usar checkpoint best como fallback

    # Cargar dataset de test
    print("\n Cargando dataset de test...")
    X_test, y_test, class_names_loaded = cargar_dataset_test()

    print(f"   Imágenes de test: {len(X_test)}")
    print(f"   Clases: {len(class_names)}\n")

    # Crear dataset y dataloader
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = MushroomDataset(X_test, y_test, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=2)

    # Evaluar
    print(" Evaluando modelo en conjunto de test...\n")
    y_pred, y_true, y_probs = evaluar_modelo(model, test_loader, class_names, DEVICE)

    # Calcular accuracy general
    accuracy = np.mean(y_pred == y_true) * 100

    print(f"\n{'='*70}")
    print(f" PRECISIÓN GENERAL EN TEST: {accuracy:.2f}%")
    print(f"{'='*70}\n")

    # Reporte de clasificación
    print("\n" + "="*70)
    print(" REPORTE DE CLASIFICACIÓN")
    print("="*70 + "\n")

    report = classification_report(y_true, y_pred,
                                   target_names=class_names,
                                   output_dict=True)

    print(classification_report(y_true, y_pred, target_names=class_names))


    # Métricas por clase
    print("\n Generando métricas detalladas por clase...")
    plot_class_metrics(report, class_names)

    # Análisis de errores
    analizar_errores(y_true, y_pred, y_probs, class_names, X_test)

    print(f"\n{'='*70}")
    print(" Evaluación completada")
    print(" Archivos generados:")
    print("   - class_metrics.png (métricas detalladas por clase)")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()