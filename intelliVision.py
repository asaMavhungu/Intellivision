import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load MNIST dataset
# Train
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform_train)
# Test
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform_test)

# Send data to the data loaders
BATCH_SIZE = 128
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, # type: ignore
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, # type: ignore
                                          shuffle=False)


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten() # For flattening the 2D image
        self.fc1 = nn.Linear(32*32*3, 512*2)  # Input is image with shape (32x32x3)
        self.fc2 = nn.Linear(512*2, 512*2)  # First HL
        self.fc3= nn.Linear(512*2, 256*2) # Second HL
        self.fc4= nn.Linear(512, 256) # Third HL
        self.fc5= nn.Linear(256, 10) # Fourth HL
        self.output = nn.LogSoftmax(dim=1)

    def forward(self, x):
      # Batch x of shape (B, C, W, H)
      x = self.flatten(x) # Batch now has shape (B, C*W*H)
      x = F.relu(self.fc1(x))  # First Hidden Layer
      x = F.relu(self.fc2(x))  # Second Hidden Layer
      x = F.relu(self.fc3(x))  # third Hidden Layer
      x = F.relu(self.fc4(x))  # Fourth Hidden Layer
      x = self.fc5(x)  # Output Layer
      x = self.output(x)  # For multi-class classification
      return x  # Has shape (B, 10)
    
# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 3, padding=0) # First Conv Layer
        self.pool = nn.MaxPool2d(2)  # For pooling
        self.flatten = nn.Flatten() # For flattening the 2D image
        self.fc1 = nn.Linear(5*16*16, 120)  # First FC HL
        self.fc2= nn.Linear(120, 10) # Output layer

    def forward(self, x):
      # Batch x of shape (B, C, W, H)
      x = F.relu(self.conv1(x)) # Shape: (B, 5, 32, 32)
      x = self.pool(x)  # Shape: (B, 5, 16, 16)
      x = self.flatten(x) # Shape: (B, 980)
      x = F.relu(self.fc1(x))  # Shape (B, 256)
      x = self.fc2(x)  # Shape: (B, 10)
      return x  

# Identify device
device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() # type: ignore
    else "cpu"
)
print(f"Using {device} device")

# Creat the model and send its parameters to the appropriate device

import torch.optim as optim # Optimizers

# Define the training and testing functions
def train(net, train_loader, criterion, optimizer, device):
    net.train()  # Set model to training mode.
    running_loss = 0.0  # To calculate loss across the batches
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

def test(net, test_loader, device):
    net.eval()  # We are in evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Don't accumulate gradients
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Send to device
            outputs = net(inputs)  # Get predictions
            _, predicted = torch.max(outputs.data, 1)  # Get max value
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # How many are correct?
    return correct / total

mlp = MLP().to(device)
print(mlp)

LEARNING_RATE = 1e-2
MOMENTUM = 0.9

# Define the loss function, optimizer, and learning rate scheduler
#criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# Train the MLP for 5 epochs
for epoch in range(15):
    train_loss = train(mlp, train_loader, criterion, optimizer, device)
    test_acc = test(mlp, test_loader, device)
    print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
