from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
# from tqdm import tqdm  # Not available, will use simple progress

from bitbybit.utils.models import get_backbone
from bitbybit.utils.data import get_loaders, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
from bitbybit.utils.score import calculate_submission_score
import bitbybit as bb
from bitbybit.config.resnet20 import resnet20_full_patch_config

OUTPUT_DIR = Path(__file__).parent / "submission_checkpoints"

def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def train_model(model, train_loader, test_loader, device, epochs=5, lr=0.001):
    """Train model with fine-tuning for hash kernels"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 50 == 0:  # Print progress every 50 batches
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}: Loss = {loss.item():.4f}')
        
        # Evaluate after each epoch
        test_acc = evaluate_model(model, test_loader, device)

        print(f'Epoch {epoch+1}: Test Accuracy = {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc

    return best_acc


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cifar_10_train_loader, cifar_10_test_loader = get_loaders(
        dataset_name="CIFAR10",
        data_dir=Path(__file__).parent / "data",
        batch_size=64,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        num_workers=2,
        pin_memory=True,
    )

    cifar_100_train_loader, cifar_100_test_loader = get_loaders(
        dataset_name="CIFAR100",
        data_dir=Path(__file__).parent / "data",
        batch_size=64,     
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
        num_workers=2,
        pin_memory=True,
    )

    models = [
        ("cifar10_resnet20", get_backbone("cifar10_resnet20"), cifar_10_train_loader, cifar_10_test_loader),
        ("cifar100_resnet20", get_backbone("cifar100_resnet20"), cifar_100_train_loader, cifar_100_test_loader),
    ]

    # Set device for training with MPS fallback handling
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    for model_name, model, train_loader, test_loader in models:
        print(f"\n=== Processing {model_name} ===")
        
        # Move model to device
        model = model.to(device)
        
        # Evaluate original model accuracy
        original_acc = evaluate_model(model, test_loader, device)
        print(f"Original model accuracy: {original_acc:.2f}%")
        
        # Apply hash-based patching
        hashed_model = bb.patch_model(model, config=resnet20_full_patch_config)
        hashed_model = hashed_model.to(device)
        
        # Evaluate hashed model accuracy (before fine-tuning)
        hashed_acc_before = evaluate_model(hashed_model, test_loader, device)
        print(f"Hashed model accuracy (before training): {hashed_acc_before:.2f}%")
           
        # Calculate accuracy drop and score
        acc_drop = (original_acc - hashed_acc_before) / 100.0
        score = calculate_submission_score(hashed_model, acc_drop)
        print(f"Accuracy drop: {acc_drop:.4f}, Score: {score:.4f}")

        # Fine-tune the hashed model
        print("Fine-tuning hashed model...")
        final_acc = train_model(hashed_model, train_loader, test_loader, device, epochs=3, lr=0.0001)
        print(f"Final accuracy after fine-tuning: {final_acc:.2f}%")
        
        # Calculate accuracy drop and score
        acc_drop = (original_acc - final_acc) / 100.0
        score = calculate_submission_score(hashed_model, acc_drop)
        print(f"Accuracy drop: {acc_drop:.4f}, Score: {score:.4f}")
        
        # Store model
        torch.save(hashed_model.state_dict(), OUTPUT_DIR / f"{model_name}.pth")
        print(f"Model saved to {OUTPUT_DIR / f'{model_name}.pth'}")


if __name__ == "__main__":
    main()