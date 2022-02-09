import config
import create_dataloaders
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import torch
import time


trainTansform = transforms.Compose([
	transforms.RandomResizedCrop(config.image_size),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(90),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.mean, std=config.std)])

valTransform = transforms.Compose([
	transforms.Resize((config.image_size, config.image_size)),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.mean, std=config.std)])

trainDS, trainLoader = create_dataloaders.get_dataloader(config.train,
	transforms=trainTansform,
	batchSize=config.feature_extraction_batch_size)
valDS, valLoader = create_dataloaders.get_dataloader(config.val,
	transforms=valTransform,
	batchSize=config.feature_extraction_batch_size, shuffle=False)
#if __name__ == '__main__':
model = resnet50(pretrained=True)

for param in model.parameters():
	param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(trainDS.classes))
#model = model.to(config.device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=config.lr)

startTime = time.time()

for epoch in tqdm(range(config.epoch)):
	model.train()
	loss_accum = 0
	correct_samples = 0
	total_samples = 0
	correct_samples_val = 0
	total_samples_val = 0
	for i_step, (x, y) in enumerate(trainLoader):
		#(x, y) = (x.to(config.device), y.to(config.device))
		pred = model(x)
		loss_value = loss(pred, y)
		optimizer.zero_grad()
		loss_value.backward()
		optimizer.step()



		correct_samples += (pred.argmax(1) == y).type(torch.float).sum().item()
		total_samples += y.shape[0]
		loss_accum += loss_value

	with torch.no_grad():
		model.eval()
		for x, y in valLoader:
			#(x, y) = (x.to(config.device), y.to(config.device))
			pred = model(x)
			correct_samples_val += (pred.argmax(1) == y).type(torch.float).sum().item()
			total_samples_val += y.shape[0]

	ave_loss = loss_accum / (i_step + 1)
	train_accuracy = float(correct_samples) / total_samples
	val_accuracy = float(correct_samples_val) / total_samples_val
	print(ave_loss)
	print(train_accuracy)
	print(val_accuracy)
torch.save(model, 'model.pth')