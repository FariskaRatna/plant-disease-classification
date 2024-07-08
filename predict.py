import os
from pathlib import Path
from scripts.classifier import Classifier
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import random
import pytorch_lightning as pl
from dataset.dataset import create_data_loaders

model_path = Path(
    "D:\Python\Widya Robotics\Project Week 2 (CNN)\scripts\checkpoints\efficientnet\efficient.ckpt"
)
model = Classifier.load_from_checkpoint(model_path)
classes = os.listdir(
    Path("D:\Python\Widya Robotics\Project Week 2 (CNN)\data\plant_disease")
)

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

train_data_dir = "train-test-splitted/train"
test_data_dir = "train-test-splitted/test"
val_data_dir = "train-test-splitted/val"


train_loader, validation_loader, test_loader, train_data, test_data = (
    create_data_loaders(train_data_dir, test_data_dir, val_data_dir)
)

class_names = classes

# Load trained model here
model = model
model.eval()

# Transform for the input image
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Path to the test images folder
test_folder = "train-test-splitted/val"

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up a subplot for the predictions
fig, axs = plt.subplots(2, 4, figsize=(15, 8))
fig.suptitle("Predictions", fontsize=16)

# Iterate through all subplots
for i in range(2):
    for j in range(4):
        # Calculate the index in the flat iteration
        index = i * 4 + j

        # Skip if all images have been processed
        if index >= len(class_names):
            break

        # Get the class name for the current subplot
        class_name = class_names[index]

        # Get the list of image files for the class
        class_folder = os.path.join(test_folder, class_name)
        image_files = os.listdir(class_folder)

        # Determine the number of images to predict
        num_images_to_predict = 2 if i == 0 and j < 2 else 1

        # Skip if there are not enough images
        if len(image_files) < num_images_to_predict:
            continue

        # Randomly select different images for each prediction
        image_indices = random.sample(range(len(image_files)), num_images_to_predict)

        # Iterate over selected image indices
        for k in range(num_images_to_predict):
            image_index = image_indices[k]
            image_path = os.path.join(class_folder, image_files[image_index])
            img = Image.open(image_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                output = model(img)
                _, predicted = torch.max(output, 1)
                predicted_class = class_names[predicted.item()]

            # Display the image with title
            axs[i, j].imshow(
                img.cpu().squeeze().permute(1, 2, 0).numpy()
            )  # Move to CPU before converting to numpy
            title_color = "green" if predicted_class == class_name else "red"
            axs[i, j].set_title(predicted_class, color=title_color)
            axs[i, j].axis("off")

            if title_color == "red":
                axs[i, j].text(
                    0.5,
                    -0.1,
                    "Incorrect Prediction",
                    color="red",
                    transform=axs[i, j].transAxes,
                )

plt.show()
