import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from models.model import LeNet5
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.helper import training_loop

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

IMG_SIZE = 32
N_CLASSES = 10

transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

train_dataset = datasets.MNIST(root="mnist_data", train=True, transform=transforms, download=True)
valid_dataset = datasets.MNIST(root="mnist_data", train=False, transform=transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

torch.manual_seed(RANDOM_SEED)

model = LeNet5(N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)

ROW_IMG = 10
N_ROWS = 5

fig = plt.figure()
for index in range(1, ROW_IMG * N_ROWS + 1):
    plt.subplot(N_ROWS, ROW_IMG, index)
    plt.axis("off")
    plt.imshow(valid_dataset.data[index], cmap="gray_r")

    with torch.no_grad():
        model.eval()
        _, probs = model(valid_dataset[index][0].unsqueeze(0))

    title = f"{torch.argmax(probs)} ({torch.max(probs * 100):.0f}%)"

    plt.title(title, fontsize=7)

fig.suptitle("LeNet-5 - predictions")
