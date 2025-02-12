import argparse
import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from ultralytics import YOLO

from project.pipeline_utils import load_config, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Object Tracking in images")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the JSON config file"
    )
    return parser.parse_args()


args = parse_args()
config = load_config(args.config)
epochs = config.get("epochs", 100)
datadir = config["datadir"]
savedir = config["savedir"]
train_yolo = config.get("train_yolo", False)
train_backbone = config.get("train_backbone", False)

logger = setup_logger()
args = parse_args()
datadir = args.datadir
epochs = args.epochs
savedir = args.savedir
train_yolo = args.train_yolo
yolo_classes = args.yolo_classes
yolo_classes = {int(k): v for k, v in yolo_classes.items()}
train_backbone = args.train_backbone

logger.info(f"Data directory: {datadir}")
logger.info(f"Number of epochs: {epochs}")
logger.info(f"Save directory: {savedir}")
logger.info(f"Train YOLO: {train_yolo}")
logger.info(f"Train Backbone: {train_backbone}")

if train_yolo:
    ###training code from ultalytics
    logger.info("Starting YOLO training...")
    yolo_dir = os.path.join(savedir, "ft_yolo")
    logger.info(f"YOLO save directory: {yolo_dir}")
    yolo_model = YOLO("yolo11s.pt")
    if torch.backends.mps.is_available():
        device = "mps"
    elif not torch.cuda.is_available():
        device = "cpu"
    else:
        device = 0  # Note: This should be updated for multi-gpu training, I just can't test this locally because I don't have access to a gpu rn

    logger.info(f"Using device: {device}")
    results = yolo_model.train(
        data=datadir,
        epochs=epochs,
        save=True,
        project=yolo_dir,
        exist_ok=True,
        classes=list(yolo_classes.keys()),
        device=device,
    )
    logger.info("YOLO training completed.")

if train_backbone:
    logger.info("Starting ResNet18 training...")

    # Define transformations for the training data
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the dataset
    train_dataset = ImageFolder(root=datadir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Load the pre-trained ResNet18 model
    resnet18 = models.resnet18(pretrained=True)

    # Modify the final layer to match the number of classes in the dataset
    num_classes = len(train_dataset.classes)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet18.parameters(), lr=0.001)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18.to(device)

    for epoch in range(epochs):
        resnet18.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        logger.info(
            f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}"
        )

    # Save the fine-tuned model
    backbone_dir = os.path.join(savedir, "ft_backbone")
    os.makedirs(backbone_dir, exist_ok=True)
    torch.save(
        resnet18.state_dict(), os.path.join(backbone_dir, "resnet18_finetuned.pth")
    )
    logger.info("ResNet18 training completed.")
