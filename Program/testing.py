import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Models.LeNet import LeNet  # Pastikan Anda mendefinisikan model LeNet di file Models/LeNet.py
from Utils.getData import Data  # Pastikan Anda mendefinisikan kelas Data di file Utils/getData.py

def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for src, trg in data_loader:
            src = src.permute(0, 3, 1, 2).float()
            trg = torch.argmax(trg, dim=1)  # Konversi one-hot encoding ke indeks kelas
            
            outputs = model(src)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(trg.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Konversi label ke one-hot encoding untuk AUC
    all_labels_onehot = np.eye(6)[all_labels]
    auc = roc_auc_score(all_labels_onehot, all_probs, multi_class='ovr', average='weighted')

    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, auc, cm

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """
    Plot heatmap dari confusion matrix dan simpan sebagai gambar.
    :param cm: Confusion Matrix
    :param class_names: Daftar nama kelas
    :param save_path: Path untuk menyimpan heatmap
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def main():
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6  # Sesuaikan dengan jumlah kelas yang ada

    # Paths ke dataset
    aug_path = "D:/DeepLearning/Assasment - Deep Learning/Dataset/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "D:/DeepLearning/Assasment - Deep Learning/Dataset/Original Images/Original Images/FOLDS/"

    # Inisialisasi dataset dan dataloaders
    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)

    # Load data test dari folder original images
    test_data = dataset.dataset_test
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Definisikan model LeNet
    model = LeNet(num_classes=NUM_CLASSES)

    # Memuat model yang telah dilatih (misalnya, dari file)
    model.load_state_dict(torch.load("trained_model4.pth"))
    model.eval()

    # Evaluasi model pada data test
    accuracy, precision, recall, f1, auc, cm = evaluate_model(model, test_loader)

    # Print hasil evaluasi
    print("Evaluasi pada data test:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Visualisasi Confusion Matrix sebagai Heatmap
    class_names = ["Chickenpox", "Cowpox", "Healthy", "HFMD", "Measles", "Monkeypox"]  # Ganti sesuai nama kelas
    plot_confusion_matrix(cm, class_names, save_path="./confusion_matrix.png")
    # print("Confusion Matrix heatmap saved as confusion_matrix.png")

if __name__ == "__main__":
    main()
