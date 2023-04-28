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
        # input = out_channel, output = out_channel (kernel 3, paddin 1, stride 1)
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
    def __init__(self) -> None:
        super(ResNet, self).__init__()
        # input = 3x32x32, output = 32x32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        # input = 32x32x32, output = 32x16x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        downsample=nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, stride=2))
        # First ResBlock, input size = 32x16x16, output size = 64x8x8
        self.residual_block1 = ResBlock(32, 64, stride=2, downsample=downsample)
        # Second ResBlock , input size = 64x8x8, output size = 64x8x8
        self.residual_block2 = ResBlock(64, 64)
        # input size = 64x8x8, output size = 64x1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # input size = 64, output size = 10
        self.fc = nn.Linear(64, 10)   

    def forward(self, x) -> utils.torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.residual_block1(x)
        x = self.residual_block2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    
	import torch.optim as optim # Optimizers
	from torch.optim.lr_scheduler import StepLR
        
	resNet = ResNet().to(utils.device)
	print(resNet)

	LEARNING_RATE = 1.5e-2
	MOMENTUM = 0.9
	STEP_SIZE = 6
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