import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Models.LeNet import LeNet 
from Utils.getData import Data 

def main():
    BATCH_SIZE = 4
    EPOCH = 50
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6 


    aug_path = "./Dataset/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "./Dataset/Original Images/Original Images/FOLDS/"


    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)

    # augmented dan original data untuk training
    train_data = dataset.dataset_train + dataset.dataset_aug
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Validation data
    valid_data = dataset.dataset_valid
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

    # LeNet model
    model = LeNet(num_classes=NUM_CLASSES)

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    valid_losses = []

    for epoch in range(EPOCH):
        loss_train = 0.0
        correct_train = 0
        total_train = 0
        model.train() 

        for batch_idx, (src, trg) in enumerate(train_loader):
            src = src.permute(0, 3, 1, 2).float()
            trg = torch.argmax(trg, dim=1)


            pred = model(src)
            loss = loss_fn(pred, trg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            _, predicted = torch.max(pred, 1)
            total_train += trg.size(0)
            correct_train += (predicted == trg).sum().item()

        accuracy_train = 100 * correct_train / total_train

        loss_valid = 0.0
        correct_valid = 0
        total_valid = 0
        model.eval()  

        with torch.no_grad():
            for batch_idx, (src, trg) in enumerate(valid_loader):
                src = src.permute(0, 3, 1, 2).float()
                trg = torch.argmax(trg, dim=1)

                pred = model(src)
                loss = loss_fn(pred, trg)

                loss_valid += loss.item()
                _, predicted = torch.max(pred, 1)
                total_valid += trg.size(0)
                correct_valid += (predicted == trg).sum().item()

        accuracy_valid = 100 * correct_valid / total_valid
        print(f"Epoch [{epoch + 1}/{EPOCH}], Train Loss: {loss_train / len(train_loader):.4f}, "
              f"Train Accuracy: {accuracy_train:.2f}%, Validation Loss: {loss_valid / len(valid_loader):.4f}, "
              f"Validation Accuracy: {accuracy_valid:.2f}%")

        train_losses.append(loss_train / len(train_loader))
        valid_losses.append(loss_valid / len(valid_loader))

    torch.save(model.state_dict(), "trained_model.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(range(EPOCH), train_losses, color="#3399e6", label='Training Loss')
    plt.plot(range(EPOCH), valid_losses, color="#FF5733", label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("./training.png")

if __name__ == "__main__":
    main()
