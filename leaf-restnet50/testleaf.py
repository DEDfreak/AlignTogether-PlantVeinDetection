# test_leaf.py
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
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
            print(label)
            x1, y1, x2, y2 = box.tolist()
            class_name = class_names[label - 1]
            print(class_name, label)
            color = "red" if class_name == "vein" else "blue"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1), f"{class_name}{score:.2f}", fill=color)

    # Display the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    model_path = (
        "leaf_model_epoch_10.pth"  # Replace with the actual model path if needed
    )
    image_path = "AlignTogether-PlantVeinDetection\leaf-restnet50\\test\pepper_healthy5_jpg.rf.7d03f8ceba15348f1ed3dff9dd587f6a.jpg"  # Replace with the path to the test image

    # Load the model
    model = load_model(model_path)

    # Run prediction and display results
    predict_and_display(model, image_path)