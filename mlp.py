import utils

import torch.nn as nn 

class MLP(nn.Module):
    def __init__(self, input_size:int, hidden_sizes:list[int], output_size:int, dropout_rate:float =0.5) -> None:
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        layers: list[nn.Module] = []
        prev_size: int = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x: utils.torch.Tensor) -> utils.torch.Tensor:
        x = x.view(-1, self.input_size)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":
    
	import torch.optim as optim # Optimizers
	from torch.optim.lr_scheduler import StepLR
        
	input_size: int = 32*32*3
	hidden_size: list[int] = [512*2, 512, 256]
	output_size: int = 10
        
	mlp = MLP(input_size, hidden_size, output_size ).to(utils.device)

	LEARNING_RATE = 1.5e-2
	MOMENTUM = 0.9
	STEP_SIZE = 10
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