import utils

import torch.nn as nn
import torch.nn.functional as F

# help: https://www.youtube.com/watch?v=o_3mboe1jYI
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1, downsample=None)-> None:
        super(ResBlock, self).__init__()
        # input = in_channel, output = in_channel (kernel 3, paddin 1, stride 1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=3, stride=stride,padding=1)  
        # Use batchNorm to re-normalize features
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Pass through ReLu
        self.relu = nn.ReLU(inplace=True)
        # # input = out_channel, output = out_channel (kernel 3, paddin 1, stride 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)  
        # Use batchNorm to re-normalize features
        self.bn2 = nn.BatchNorm2d(out_channels) 
        self.downsample = downsample

    def forward(self, x) -> utils.torch.Tensor:
        # save x to use as identity function later
        identity = x

        # pass x through conv layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # downsample if required
        if self.downsample is not None:
            identity = self.downsample(x)

        # add saved identity features to our current set of features in the network
        # done element wise
        out += identity
        # pass thought activation function
        out = self.relu(out)
        # return as output of our residual block
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # input = 3x32x32, output = 16x32x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,kernel_size=3, stride=1,padding=1)  
        # input = 16x32x32, output = 16x16x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.residual_block = ResBlock(16, 16)
        # input = 16x5x5, output = 120
        self.fc1 = nn.Linear(16 * 16 * 16, 10)    

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.residual_block(x)
        x = x.view(-1, 16**3)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    
	import torch.optim as optim # Optimizers
	from torch.optim.lr_scheduler import StepLR
        
	resNet = ResNet().to(utils.device)
	print(resNet)

	LEARNING_RATE = 1.5e-2
	MOMENTUM = 0.9
	STEP_SIZE = 10
	GAMMA = 0.1
	DECAY = 0.001

	print(f"lr={LEARNING_RATE}, m={MOMENTUM}, step={STEP_SIZE}, gamme={GAMMA}, decay={DECAY}")
	# Define the loss function, optimizer, and learning rate scheduler
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(resNet.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=DECAY)
	lr_scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

	# Train the MLP for 5 epochs
	for epoch in range(15):
		train_loss = utils.train(resNet, utils.train_loader, criterion, optimizer, utils.device)
		test_acc = utils.test(resNet, utils.test_loader, utils.device)
		print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}, Learning rate = {optimizer.param_groups[0]['lr']:.4f}")
		lr_scheduler.step()  # apply learning rate decay