# leaf.py
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from PIL import Image
import cv2
import numpy as np


class LeafDataset(Dataset):
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.transform = transforms.ToTensor()

        # Get the current script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load JSON file
        json_path = os.path.join(script_dir, folder_name, "_annotations.coco.json")
        with open(json_path, "r") as f:
            self.annotations = json.load(f)
            
        

        # Create image_id to annotations mapping
        self.image_to_anns = {}
        for ann in self.annotations["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_to_anns:
                self.image_to_anns[img_id] = []
            self.image_to_anns[img_id].append(ann)

        self.images = self.annotations["images"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image info
        img_info = self.images[idx]
        img_id = img_info["id"]

        # Load image
        # Get the current script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, self.folder_name, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        # image = self.transform(image)
        
        # Convert PIL image to numpy for edge detection
        image_np = np.array(image)
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)

        # Convert back to PIL and transform to tensor
        edge_image = Image.fromarray(edges).convert("RGB")
        edge_image = self.transform(edge_image)

        # Get annotations
        boxes = []
        labels = []

        # Get annotations for this image
        if img_id in self.image_to_anns:
            for ann in self.image_to_anns[img_id]:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])

        # Handle images with no annotations
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}

        return edge_image, target


def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Create datasets
        print("Loading datasets...")
        train_dataset = LeafDataset("train")
        valid_dataset = LeafDataset("valid")

        print(f"Found {len(train_dataset)} training images")
        print(f"Found {len(valid_dataset)} validation images")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x)),  # Important for variable size boxes
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=2, collate_fn=lambda x: tuple(zip(*x))
        )

        # Load model with newer syntax
        print("Loading model...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        model.to(device)

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

        # Training loop
        print("Starting training...")
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            num_batches = len(train_loader)

            for batch_idx, (images, targets) in enumerate(train_loader, 1):
                # Move to device
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                total_loss += losses.item()

                # Print progress
                if batch_idx % 5 == 0:
                    print(
                        f"Epoch {epoch+1}/{num_epochs} [{batch_idx}/{num_batches}] - Loss: {losses.item():.4f}"
                    )

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs} complete - Avg Loss: {avg_loss:.4f}")

            # Save model
        model_path = os.path.join(os.getcwd(), f"leaf_model_epoch_{num_epochs}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved as {model_path}")

        print("Training completed!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    train()
