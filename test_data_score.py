import config
import create_dataloaders
from torchvision import transforms
import torch



testTransform = transforms.Compose([
	transforms.Resize((config.image_size, config.image_size)),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.mean, std=config.std)
])

testDS, testLoader = create_dataloaders.get_dataloader(config.val,
	transforms=testTransform, batchSize=config.pred_batch_size,
	shuffle=True)

correct_samples_test = 0
total_samples_test = 0
model = torch.load('model.pth')
with torch.no_grad():
    model.eval()
    for x, y in testLoader:
        # (x, y) = (x.to(config.device), y.to(config.device))
        pred = model(x)
        correct_samples_test += (pred.argmax(1) == y).type(torch.float).sum().item()
        total_samples_test += y.shape[0]

test_accuracy = float(correct_samples_test) / total_samples_test
print(test_accuracy)