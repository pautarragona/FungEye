"""
Red Neuronal Convolucional para Clasificación de Setas
Entrenamiento desde cero con arquitectura personalizada
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import numpy as np
from datetime import datetime

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'Mushrooms'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

print(f"Usando: {DEVICE}")


# ARQUITECTURA CNN


class MushroomCNN(nn.Module):
    """CNN para clasificación de setas"""
    def __init__(self, num_classes):
        super(MushroomCNN, self).__init__()

        # Bloque Convolucional 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 224 -> 112
        )

        # Bloque Convolucional 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 112 -> 56
        )

        # Bloque Convolucional 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 56 -> 28
        )

        # Bloque Convolucional 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 28 -> 14
        )

        # Capas Completamente Conectadas
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

# ============================================================================
# DATASET
# ============================================================================

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
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")
            if self.transform:
                return self.transform(Image.new('RGB', (IMG_SIZE, IMG_SIZE))), label
            return Image.new('RGB', (IMG_SIZE, IMG_SIZE)), label


# CARGA DE DATOS


def cargar_dataset(data_dir):
    """Carga las rutas de imágenes y sus etiquetas"""
    image_paths = []
    labels = []
    class_names = []

    classes = sorted([d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d))])

    print(f"\n Clases encontradas: {len(classes)}")
    for i, class_name in enumerate(classes):
        print(f"   {i}: {class_name}")
        class_names.append(class_name)

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_idx)

    print(f"\n Total de imágenes: {len(image_paths)}")
    return image_paths, labels, class_names

# Transformaciones
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ENTRENAMIENTO Y VALIDACIÓN


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena el modelo por una época"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    top3_correct = 0

    all_labels = []
    all_preds = []

    pbar = tqdm(dataloader, desc='Entrenando')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Top-3 accuracy
        _, top3_pred = outputs.topk(3, 1, True, True)
        top3_pred = top3_pred.t()
        correct_top3 = top3_pred.eq(labels.view(1, -1).expand_as(top3_pred))
        top3_correct += correct_top3.sum().item()

        # Guardar para F1-score
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    top3_acc = 100. * top3_correct / total
    train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return running_loss / len(dataloader), 100. * correct / total, top3_acc, train_f1

def validate(model, dataloader, criterion, device):
    """Valida el modelo y calcula métricas completas"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    top3_correct = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validando'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-3 accuracy
            _, top3_pred = outputs.topk(3, 1, True, True)
            top3_pred = top3_pred.t()
            correct_top3 = top3_pred.eq(labels.view(1, -1).expand_as(top3_pred))
            top3_correct += correct_top3.sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    acc = 100. * correct / total
    top3_acc = 100. * top3_correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    all_probs = np.vstack(all_probs)

    return running_loss / len(dataloader), acc, precision, recall, f1, top3_acc, np.array(all_labels), np.array(all_preds), all_probs


# VISUALIZACIÓN AVANZADA


def plot_advanced_training_history(history, true_classes, predicted_classes, predictions, class_names):
    """Grafica análisis avanzado del entrenamiento (8 subplots)"""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig = plt.figure(figsize=(20, 10))

    # 1. Training & Validation Loss
    plt.subplot(2, 4, 1)
    plt.plot(history['train_loss'], label='Train', linewidth=2.5, color='#3498db')
    plt.plot(history['val_loss'], label='Val', linewidth=2.5, color='#e74c3c')
    plt.title('Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Training & Validation Accuracy
    plt.subplot(2, 4, 2)
    plt.plot(history['train_acc'], label='Train', linewidth=2.5, color='#2ecc71')
    plt.plot(history['val_acc'], label='Val', linewidth=2.5, color='#e74c3c')
    plt.title('Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Top-3 Accuracy
    plt.subplot(2, 4, 3)
    plt.plot(history['train_top3_acc'], label='Train', linewidth=2.5, color='#2ecc71')
    plt.plot(history['val_top3_acc'], label='Val', linewidth=2.5, color='#e74c3c')
    plt.title('Top-3 Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Top-3 Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. F1-Score (Train vs Val)
    plt.subplot(2, 4, 4)
    plt.plot(history['train_f1'], label='Train F1', linewidth=2.5, color='#2ecc71')
    plt.plot(history['val_f1'], label='Val F1', linewidth=2.5, color='#e74c3c')
    plt.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='Excellent (0.90)')
    plt.axhline(y=0.80, color='orange', linestyle='--', alpha=0.7, label='Good (0.80)')
    plt.title('F1-Score (Train vs Val)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])

    # 5. Confusion Matrix
    plt.subplot(2, 4, 5)
    cm = confusion_matrix(true_classes, predicted_classes, labels=range(len(class_names)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # 6. Per-class Accuracy
    plt.subplot(2, 4, 6)
    class_acc = []
    for i in range(len(class_names)):
        mask = true_classes == i
        if mask.sum() > 0:
            acc = (predicted_classes[mask] == true_classes[mask]).mean()
        else:
            acc = 0.0
        class_acc.append(acc)

    colors = ['#2ecc71' if a > 0.80 else '#f39c12' if a > 0.65 else '#e74c3c' for a in class_acc]
    plt.barh(class_names, class_acc, color=colors, alpha=0.8)
    plt.xlabel('Accuracy')
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlim([0, 1])
    for i, v in enumerate(class_acc):
        plt.text(v + 0.01, i, f'{v:.1%}', va='center', fontweight='bold', fontsize=9)

    # 7. Confidence Distribution
    plt.subplot(2, 4, 7)
    max_probs = np.max(predictions, axis=1)
    correct = predicted_classes == true_classes
    plt.hist(max_probs[correct], bins=30, alpha=0.7, label='Correct',
             color='#2ecc71', density=True)
    plt.hist(max_probs[~correct], bins=30, alpha=0.7, label='Incorrect',
             color='#e74c3c', density=True)
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title('Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 8. Learning Rate Schedule
    plt.subplot(2, 4, 8)
    epochs = list(range(len(history['learning_rates'])))
    plt.semilogy(epochs, history['learning_rates'], linewidth=2.5, color='#3498db', marker='o')
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log scale)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f'training_results_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nGráficas avanzadas guardadas en: {save_path}")
    plt.show()


# MAIN


def main():
    print("\n" + "="*70)
    print(" "*15 + " ENTRENAMIENTO CNN CLASIFICADOR DE SETAS")
    print("="*70 + "\n")

    # Cargar datos
    image_paths, labels, class_names = cargar_dataset(DATA_DIR)

    # Dividir en train/temp (70% / 30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Dividir temp en validation/test (50% / 50% del 30% = 15% / 15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nDivisión de datos:")
    print(f"   Entrenamiento: {len(X_train)} imágenes ({len(X_train)/len(image_paths)*100:.1f}%)")
    print(f"   Validación: {len(X_val)} imágenes ({len(X_val)/len(image_paths)*100:.1f}%)")
    print(f"   Test: {len(X_test)} imágenes ({len(X_test)/len(image_paths)*100:.1f}%)")

    # Crear datasets
    train_dataset = MushroomDataset(X_train, y_train, train_transform)
    val_dataset = MushroomDataset(X_val, y_val, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

    # Crear modelo
    num_classes = len(class_names)
    model = MushroomCNN(num_classes).to(DEVICE)

    print(f"\nArquitectura del modelo:")
    print(f"   Clases: {num_classes}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parámetros totales: {total_params:,}")

    # Configurar entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Historial
    history = {
        'train_loss': [], 'train_acc': [], 'train_top3_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_top3_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'learning_rates': []
    }

    best_val_acc = 0.0
    final_val_labels = None
    final_val_preds = None
    final_val_probs = None

    print(f"\n Iniciando entrenamiento...")
    print(f"   Épocas: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Dispositivo: {DEVICE}\n")

    # Entrenamiento
    for epoch in range(EPOCHS):
        print(f"\n{'='*70}")
        print(f"Época {epoch+1}/{EPOCHS}")
        print(f"{'='*70}")

        # Entrenar
        train_loss, train_acc, train_top3_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        # Validar
        val_loss, val_acc, val_precision, val_recall, val_f1, val_top3_acc, val_labels, val_preds, val_probs = validate(
            model, val_loader, criterion, DEVICE
        )

        # Guardar para plots finales
        final_val_labels = val_labels
        final_val_preds = val_preds
        final_val_probs = val_probs

        # Actualizar scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        # Guardar historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_top3_acc'].append(train_top3_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top3_acc'].append(val_top3_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['learning_rates'].append(current_lr)

        # Imprimir resultados
        print(f"\nResultados:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Top-3: {train_top3_acc:.2f}%")
        print(f"   Train F1: {train_f1:.4f}")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Top-3: {val_top3_acc:.2f}%")
        print(f"   Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")
        print(f"   Learning Rate: {current_lr:.6f}")

        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, 'mushroom_cnn_best.pth')
            print(f"  Mejor modelo guardado (Val Acc: {val_acc:.2f}%)")

    # Guardar modelo final
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'class_names': class_names,
        'history': history,
        'test_data': {'X_test': X_test, 'y_test': y_test}
    }, 'mushroom_cnn_final.pth')

    # Guardar clases
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)

    print(f"\n{'='*70}")
    print(f" Entrenamiento completado")
    print(f"   Mejor precisión de validación: {best_val_acc:.2f}%")
    print(f"   Modelo guardado en: mushroom_cnn_best.pth")
    print(f"{'='*70}\n")

    # Graficar resultados avanzados
    plot_advanced_training_history(history, final_val_labels, final_val_preds,
                                   final_val_probs, class_names)

if __name__ == "__main__":
    main()