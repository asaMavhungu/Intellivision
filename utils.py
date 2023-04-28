import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.optim import Optimizer

transform_train: transforms.Compose = transforms.Compose([
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(10),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Normalize the test set same as training set without augmentation
transform_test: transforms.Compose = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load MNIST dataset
# Train
trainset: torchvision.datasets.CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=True,
									  download=True, transform=transform_train)
# Test
testset: torchvision.datasets.CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=False,
									  download=True, transform=transform_test)

# Send data to the data loaders
BATCH_SIZE: int = 128
train_loader: DataLoader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, # type: ignore
										  shuffle=True)

test_loader: DataLoader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, # type: ignore
										  shuffle=False)


# Get cpu or gpu device for training.
device: str = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))



# Define the training and testing functions
def train(net: nn.Module, train_loader: DataLoader, criterion: nn.Module , optimizer: Optimizer, device: str) -> float:
	"""
	Function to train a PyTorch neural network.

	Args:
		net (nn.Module): A PyTorch neural network model.

		train_loader (DataLoader): A PyTorch DataLoader object containing the training data.

		criterion (nn.Module): A PyTorch loss function.

		optimizer (Optimizer): A PyTorch optimizer object.
		
		device (str): A string specifying the device to be used for training.

	Returns:
		float: The average training loss over all batches.
	"""

	net.train()  # Set model to training mode.
	running_loss: float = 0.0  # To calculate loss across the batches
	for data in train_loader:
		inputs, labels = data  # Get input and labels for batch
		inputs, labels = inputs.to(device), labels.to(device)  # Send to device
		optimizer.zero_grad()  # Zero out the gradients of the ntwork i.e. reset
		outputs = net(inputs)  # Get predictions
		loss = criterion(outputs, labels)  # Calculate loss
		loss.backward()  # Propagate loss backwards
		optimizer.step()  # Update weights
		running_loss += loss.item()  # Update loss
	return running_loss / len(train_loader)

def test(net: nn.Module, test_loader: DataLoader, device: str) -> float:
	"""
	Function to test a PyTorch neural network.

	Args:
		net (nn.Module): A PyTorch neural network model.

		test_loader (DataLoader): A PyTorch DataLoader object containing the testing data.

		device (str): A string specifying the device to be used for testing.

	Returns:
		float: The accuracy of the model on the testing data.
	"""
	net.eval()  # We are in evaluation mode
	correct: int = 0
	total: int = 0
	with torch.no_grad():  # Don't accumulate gradients
		for data in test_loader:
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device) # Send to device
			outputs = net(inputs)  # Get predictions
			_, predicted = torch.max(outputs.data, 1)  # Get max value
			total += labels.size(0)
			correct += (predicted == labels).sum().item()  # How many are correct?
	return correct / total
