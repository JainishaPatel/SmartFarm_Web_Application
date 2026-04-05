import os
import json
import random
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv

# =====================
# 1. LOAD PATH
# =====================
load_dotenv()
data_dir = os.getenv("PLANT_VILLAGE_DATA")

# ✅ Check dataset path
if not data_dir or not os.path.exists(data_dir):
    raise FileNotFoundError("❌ PLANT_VILLAGE_DATA path is invalid. Check .env")

train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "val")

# =====================
# 2. MODELS FOLDER
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# =====================
# 3. TRANSFORMS
# =====================
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

valid_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# =====================
# 4. LIMIT DATA FUNCTION
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
# 5. MAIN TRAINING
# =====================
if __name__ == "__main__":

    print("🚀 Training Started...")

    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    print("✅ Data loaded")

    # Limit training data
    train_data = limit_images_per_class(train_data, 100)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=32, num_workers=0)

    # Get class names
    classes = train_data.dataset.classes if isinstance(train_data, Subset) else train_data.classes

    print("📊 Classes:", classes)
    print("📦 Training samples:", len(train_data))

    # =====================
    # DEVICE
    # =====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("💻 Using:", device)

    # =====================
    # MODEL (Transfer Learning)
    # =====================
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last layers
    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    # Replace final layer
    num_classes = len(classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)

    # =====================
    # LOSS & OPTIMIZER
    # =====================
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    # =====================
    # TRAINING LOOP
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

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Accuracy: {acc:.2f}%")

    # =====================
    # SAVE MODEL
    # =====================
    model_path = os.path.join(MODELS_DIR, "plant_model.pth")

    torch.save({
        "model_state": model.state_dict(),
        "num_classes": num_classes
    }, model_path)

    print(f"✅ Model saved at: {model_path}")

    # =====================
    # SAVE CLASS MAP
    # =====================
    class_map_path = os.path.join(MODELS_DIR, "plant_class_map.json")

    with open(class_map_path, "w") as f:
        json.dump(classes, f)

    print(f"📊 Class map saved at: {class_map_path}")

    # =====================
    # SAVE REVERSE MAP
    # =====================
    idx_to_class = {idx: cls for idx, cls in enumerate(classes)}

    reverse_map_path = os.path.join(MODELS_DIR, "plant_class_map_reverse.json")

    with open(reverse_map_path, "w") as f:
        json.dump(idx_to_class, f)

    print(f"🔁 Reverse map saved at: {reverse_map_path}")