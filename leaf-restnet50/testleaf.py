# test_leaf.py
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class_names = [
    "midrib",
    "vein",
]

def load_model(model_path):
    # Load model with the same architecture
    model = fasterrcnn_resnet50_fpn()
    model.load_state_dict(torch.load(model_path, map_location="cpu" ))
    model.eval()
    return model

def evaluate_model(model, image_paths, ground_truth_labels):
    transform = transforms.ToTensor()
    all_preds = []
    all_labels = []

    for image_path, true_label in zip(image_paths, ground_truth_labels):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)[0]
            if len(outputs["scores"]) > 0 and outputs["scores"][0] > 0.5:
                predicted_label = outputs["labels"][0].item()
            else:
                predicted_label = -1  # No valid prediction

        all_preds.append(predicted_label)
        all_labels.append(true_label)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def predict_and_display(model, image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(
        predictions["boxes"], predictions["scores"], predictions["labels"]
    ):

        if score > 0.5:  # Only consider predictions with confidence > 0.5
            x1, y1, x2, y2 = box.tolist()
            class_name = class_names[label - 1]
            color = "red" if class_name == "vein" else "blue"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1), f"{class_name}{score:.2f}", fill=color)

    # Display the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    model_path = ("leaf_model_epoch_10.pth")  # Replace with the actual model path if needed
    test_image_paths = ["AlignTogether-PlantVeinDetection\leaf-restnet50\\test\pepper_healthy_jpg.rf.eddc793bc66e0df1790008b015ff3b7f.jpg"]
    image_path = "AlignTogether-PlantVeinDetection\leaf-restnet50\\test\pepper_healthy_jpg.rf.eddc793bc66e0df1790008b015ff3b7f.jpg"
    ground_truth_labels = [1]

    # Load the model
    model = load_model(model_path)

    # Evaluate the model
    print("Evaluating model performance....")
    evaluate_model(model, test_image_paths, ground_truth_labels)
    
    
    print("Evaluating The result...")
    predict_and_display(model, image_path)
