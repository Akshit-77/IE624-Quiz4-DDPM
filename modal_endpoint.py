import modal

app = modal.App("ann-gpu-training")

# Container with PyTorch + torchvision
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision")
)

@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=32768,
    ephemeral_disk=530000,
    timeout=1800
)
def train_ann():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="/tmp/mnist",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Simple ANN (no CNN)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluate simple accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return {
        "device_used": str(device),
        "final_accuracy_percent": accuracy
    }


@app.local_entrypoint()
def main():
    result = train_ann.remote()
    print("\nTraining Complete on Modal GPU:")
    print(result)