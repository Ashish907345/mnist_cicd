import torch
import pytest
from torchvision import datasets, transforms
from model.network import SimpleCNN
import torch.nn.utils.prune as prune
import numpy as np
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_model_parameters():
    model = SimpleCNN()
    param_count = count_parameters(model)
    print(f"\nModel Parameter Test:")
    print(f"Total parameters: {param_count:,}")
    print(f"Parameter limit: 100,000")
    assert param_count < 100000, f"Model has {param_count} parameters, should be less than 100000"

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_input_output_shape():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"\nShape Test:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_batch_processing():
    model = SimpleCNN()
    batch_size = 32
    test_input = torch.randn(batch_size, 1, 28, 28)
    output = model(test_input)
    print(f"\nBatch Processing Test:")
    print(f"Batch size: {batch_size}")
    print(f"Batch output shape: {output.shape}")
    assert output.shape == (batch_size, 10), f"Batch output shape is {output.shape}, should be ({batch_size}, 10)"

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_output_range():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"\nOutput Range Test:")
    print(f"Min output value: {output.min().item():.4f}")
    print(f"Max output value: {output.max().item():.4f}")
    assert not torch.isnan(output).any(), "Model output contains NaN values"
    assert not torch.isinf(output).any(), "Model output contains infinite values"

@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_model_accuracy():
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Load the latest model
    import glob
    import os
    
    model_files = glob.glob('models/model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    # Load state dict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.load_state_dict(torch.load(latest_model))
    
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    print("\nAccuracy Test:")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Per-class accuracy
            c = (predicted == target).squeeze()
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    overall_accuracy = 100 * correct / total
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print("\nPer-class Accuracy:")
    for i in range(10):
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f'Digit {i}: {class_acc:.2f}%')
    
    assert overall_accuracy > 80, f"Model accuracy is {overall_accuracy}%, should be > 80%"

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_model_gradients():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28, requires_grad=True)
    output = model(test_input)
    loss = output.sum()
    loss.backward()
    
    print("\nGradient Test:")
    has_gradients = all(p.grad is not None for p in model.parameters())
    print(f"All parameters have gradients: {has_gradients}")
    print(f"Input gradient shape: {test_input.grad.shape}")
    assert test_input.grad is not None, "Input gradients are None"
    assert not torch.isnan(test_input.grad).any(), "Gradients contain NaN values"

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_model_memory():
    model = SimpleCNN()
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # Size in MB
    
    print("\nMemory Usage Test:")
    print(f"Model size in memory: {param_size:.2f} MB")
    assert param_size < 100, f"Model size ({param_size:.2f} MB) exceeds 100 MB limit" 