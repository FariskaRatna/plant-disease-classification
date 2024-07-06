import os
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import splitfolders


def count_images_per_class(data_path):
    classes = os.listdir(data_path)
    print(classes)

    for i in classes:
        new_loc = os.path.join(data_path, i)
        images_jpg = glob(os.path.join(new_loc, "*.jpg"))
        images_JPG = glob(os.path.join(new_loc, "*.JPG"))

        # images_jpg = glob(new_jpg)
        # images_JPG = glob(new_JPG)

        total_images = len(images_jpg)

        print(f"{i}: {total_images} images")


def perform_data_split(data_path):
    splitfolders.ratio(
        data_path, seed=1337, output="train-test-splitted", ratio=(0.6, 0.2, 0.2)
    )


def create_transforms():
    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0, contrast=0.2, saturation=0, hue=0),
        ]
    )

    transform_tests = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    transform_vals = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    return transform_train, transform_tests, transform_vals


def create_data_loaders(train_data_dir, test_data_dir, val_data_dir):
    transform_train, transform_tests, transform_vals = create_transforms()

    train_data = datasets.ImageFolder(train_data_dir, transform=transform_train)
    test_data = datasets.ImageFolder(test_data_dir, transform=transform_tests)
    val_data = datasets.ImageFolder(val_data_dir, transform=transform_vals)

    train_loader = DataLoader(
        train_data, batch_size=24, drop_last=True, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        test_data, batch_size=24, drop_last=True, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        val_data, batch_size=24, drop_last=True, shuffle=True, num_workers=0
    )

    return train_loader, validation_loader, test_loader, train_data, test_data
