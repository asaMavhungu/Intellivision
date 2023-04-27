import utils

import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions

class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        # input = 3x32x32, output = 6x28x28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,kernel_size=5, stride=1,padding=0)  
        self.bn1 = nn.BatchNorm2d(6)
        # input = 6x28x28, output = 6x14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
        # input = 6x14x14, output = 16x10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  
        self.bn2 = nn.BatchNorm2d(16)
        # input = 16x10x10, output = 16x5x5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  
        # input = 16x5x5, output = 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)  
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)  

    def forward(self, x: utils.torch.Tensor) -> utils.torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x
    
if __name__ == "__main__":
    
	import torch.optim as optim # Optimizers
	from torch.optim.lr_scheduler import StepLR
        
     
	cnn = CNN().to(utils.device)
	print(cnn)

	LEARNING_RATE = 1.5e-2
	MOMENTUM = 0.9
	STEP_SIZE = 10 
	GAMMA = 0.1
	DECAY = 0.001

	print(f"lr={LEARNING_RATE}, m={MOMENTUM}, step={STEP_SIZE}, gamme={GAMMA}, decay={DECAY}")
	# Define the loss function, optimizer, and learning rate scheduler
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=DECAY)
	lr_scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

	# Train the MLP for 5 epochs
	for epoch in range(15):
		train_loss = utils.train(cnn, utils.train_loader, criterion, optimizer, utils.device)
		test_acc = utils.test(cnn, utils.test_loader, utils.device)
		print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}, Learning rate = {optimizer.param_groups[0]['lr']:.4f}")
		lr_scheduler.step()  # apply learning rate decay