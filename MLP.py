import utils

import torch.nn as nn 

class MLP(nn.Module):
	def __init__(self, input_size:int, hidden_sizes:list[int], output_size:int, dropout_rate:float =0.5) -> None:
		"""
		Multilayer perceptron neural network implementation with batch normalization and ReLU activation.
		
		Args:
		- input_size (int): The number of input features.
		- hidden_sizes (list[int]): A list of integers specifying the number of nodes in each hidden layer.
		- output_size (int): The number of output features.
		- dropout_rate (float): The dropout rate (default: 0.5).
		"""
		super(MLP, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		
		layers: list[nn.Module] = []
		prev_size: int = input_size
		for size in hidden_sizes:
			layers.append(nn.Linear(prev_size, size))
			layers.append(nn.BatchNorm1d(size))
			layers.append(nn.ReLU())
			# layers.append(nn.Dropout(p=dropout_rate))
			prev_size = size
		
		layers.append(nn.Linear(prev_size, output_size))
		self.linear_relu_stack = nn.Sequential(*layers)

	def forward(self, x: utils.torch.Tensor) -> utils.torch.Tensor:
		"""
		Forward pass of the MLP model.
		
		Args:
		- x (utils.torch.Tensor): The input tensor.
		
		Returns:
		- utils.torch.Tensor: The output tensor.
		"""
		x = x.view(-1, self.input_size)
		logits = self.linear_relu_stack(x)
		return logits

if __name__ == "__main__":
	
	import sys
	import os

	import torch.optim as optim
	from torch.optim.lr_scheduler import StepLR
		
	input_size: int = 32*32*3
	hidden_size: list[int] = [512*2, 512, 256]
	output_size: int = 10
		
	mlp = MLP(input_size, hidden_size, output_size ).to(utils.device)


	if len(sys.argv) != 1 and len(sys.argv) != 2:
		print("Invalid number of arguments. Usage: python MODEL_NAME.py [-load | -save]")
	
	elif len(sys.argv) == 2 and sys.argv[1] not in ["-load", "-save"]:
		print("Invalid argument. Usage: python MODEL_NAME.py [-load | -save]")

	elif len(sys.argv) == 2 and sys.argv[1] == "-load":
		if os.path.isfile("./mlp.pt"):
			print("Loading model...")
			# load the model parameters
			mlp.load_state_dict(utils.torch.load("./mlp.pt"))
			print("Done!")
			test_acc = utils.test(mlp, utils.test_loader, utils.device)
			print(f"Test accuracy = {test_acc*100:.2f}%")
		else:
			print("Saved model not found!")

			
	elif len(sys.argv) == 2 or len(sys.argv) == 1:
		print(mlp)

		LEARNING_RATE = 1.5e-2
		MOMENTUM = 0.9
		STEP_SIZE = 8
		GAMMA = 0.1
		DECAY = 0.001

		hidden = [512*2, 512, 256]

		print(f"lr={LEARNING_RATE}, m={MOMENTUM}, step={STEP_SIZE}, gamme={GAMMA}, decay={DECAY}")

		# Define the loss function, optimizer, and learning rate scheduler
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=DECAY)
		lr_scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

		# Train the MLP for 15 epochs
		for epoch in range(15):
			train_loss = utils.train(mlp, utils.train_loader, criterion, optimizer, utils.device)
			test_acc = utils.test(mlp, utils.test_loader, utils.device)
			print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}, Learning rate = {optimizer.param_groups[0]['lr']:.4f}")
			lr_scheduler.step()  # apply learning rate decay
		
		if len(sys.argv) == 2:
			if sys.argv[1] == "-save":
				print("Saving model...")
				utils.torch.save(mlp.state_dict(), "./mlp.pt")
				print("Done!")