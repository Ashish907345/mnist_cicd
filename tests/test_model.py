import torch
import pytest
from torchvision import datasets, transforms
from model.network import SimpleCNN
import torch.nn.utils.prune as prune

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = SimpleCNN()
    param_count = count_parameters(model)
    assert param_count < 100000, f"Model has {param_count} parameters, should be less than 100000"

def test_input_output_shape():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

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
    
    # Load the latest model safely
    import glob
    import os
    
    model_files = glob.glob('models/model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    # Load state dict with weights_only=True
    model.load_state_dict(torch.load(latest_model, weights_only=True))
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Model accuracy is {accuracy}%, should be > 80%" 