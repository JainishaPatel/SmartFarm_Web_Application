import os
import random
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv

# =====================
# 1. PATH
# =====================
load_dotenv()
data_dir = os.getenv("PLANT_VILLAGE_DATA")

train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "val")

# =====================
# 2. TRANSFORMS
# =====================
train_transforms = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

valid_transforms = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# =====================
# 3. LIMIT DATA FUNCTION
# =====================
def limit_images_per_class(dataset, max_per_class=100):
    class_indices = {}

    for idx, (_, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    selected_indices = []

    for label, indices in class_indices.items():
        if len(indices) > max_per_class:
            indices = random.sample(indices, max_per_class)
        selected_indices.extend(indices)

    return Subset(dataset, selected_indices)

# =====================
# 4. DATASET
# =====================
if __name__ == "__main__":
    model = models.resnet18(pretrained=True)

    print("Step 1: Starting...")

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    print("Step 2: Train data loaded")


    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    print("Step 3: Valid data loaded")

    # 👉 LIMIT TRAIN DATA ONLY
    train_data = limit_images_per_class(train_data, 100)
    print("Step 4: Data limited")

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    print("Step 5: DataLoader ready")

    valid_loader = DataLoader(valid_data, batch_size=32, num_workers=0)

    print("Classes:", train_data.dataset.classes if isinstance(train_data, Subset) else train_data.classes)
    print("Training images:", len(train_data))

    # =====================
    # 5. DEVICE
    # =====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # =====================
    # 6. MODEL (TRANSFER LEARNING)
    # =====================
    model = models.resnet18(pretrained=True)

    # freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze last layer block
    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    # replace final layer
    num_classes = len(train_data.dataset.classes if isinstance(train_data, Subset) else train_data.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)

    # =====================
    # 7. LOSS & OPTIMIZER
    # =====================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # =====================
    # 8. TRAINING LOOP
    # =====================
    epochs = 5

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # =====================
        # VALIDATION
        # =====================
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Accuracy: {acc:.2f}%")

    # =====================
    # 9. SAVE MODEL
    # =====================
    torch.save(model.state_dict(), "plant_model.pth")

    print("✅ Model saved successfully!")