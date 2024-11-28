import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Models.LeNet import LeNet  # Ensure you have defined LeNet model in Models/LeNet.py
from Utils.getData import Data  # Ensure you have defined Data class in Utils/getData.py

def main():
    BATCH_SIZE = 4
    EPOCH = 30
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6  # Adjust with the number of classes

    # Paths to dataset
    aug_path = "D:/DeepLearning/Assasment - Deep Learning/Dataset/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "D:/DeepLearning/Assasment - Deep Learning/Dataset/Original Images/Original Images/FOLDS/"

    # Initialize dataset and dataloaders
    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)

    # Combine augmented and original data for training
    train_data = dataset.dataset_train + dataset.dataset_aug
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Define LeNet model
    model = LeNet(num_classes=NUM_CLASSES)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_losses = []

    # Training loop
    for epoch in range(EPOCH):
        loss_train = 0.0
        correct_train = 0
        total_train = 0
        model.train()  # Set the model to training mode

        for batch_idx, (src, trg) in enumerate(train_loader):

            # Ensure input is in the format (batch_size, channels, height, width)
            src = src.permute(0, 3, 1, 2).float()  # Convert from (batch, height, width, channels) to (batch, channels, height, width)
            trg = torch.argmax(trg, dim=1)  # If trg is one-hot encoded, convert to class indices

            # Forward pass
            pred = model(src)
            loss = loss_fn(pred, trg)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

            _, predicted = torch.max(pred, 1)
            total_train += trg.size(0)
            correct_train += (predicted == trg).sum().item()

        # Accuracy per epoch
        accuracy_train = 100 * correct_train / total_train
        print(f"Epoch [{epoch + 1}/{EPOCH}], Train Loss: {loss_train / len(train_loader):.4f}, Accuracy: {accuracy_train:.2f}%")

        # Append average training loss for this epoch
        train_losses.append(loss_train / len(train_loader))

    # After training is done, save the model
    torch.save(model.state_dict(), "trained_model4.pth")

    # Plot the training loss
    plt.plot(range(EPOCH), train_losses, color="#3399e6", label='Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./training.png")
    # print("Model saved to lenet_trained_model.pth")

if __name__ == "__main__":
    main()
