from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, InterpolationMode #, Scale
from torchvision.transforms import RandomResizedCrop, Resize, RandomAffine, RandomRotation, RandomHorizontalFlip, \
    RandomVerticalFlip
import torch.utils.data as data
from os import listdir
from os.path import exists, join
from PIL import Image
import torch
import random

image_size = 512


def myDataset(dest):
    if not exists(dest):
        print("dataset not exist ")
    return dest


def input_transform():  # need to add data augmentation
    # return Compose([ToTensor()])
    return Compose([Resize((image_size, image_size), InterpolationMode.BILINEAR), ToTensor()]) # interpolation=2
                    # ToTensor(), Normalize([0.6640], [0.2056])])  # 100 images


def target_transform():
    return Compose([ToTensor()])


def get_training_set(dest, color_dim=3):
    root_dir = myDataset(dest)
    train_dir = join(root_dir, "train")
    return TrainDatasetFromFolder(train_dir, color_dim,
                                  input_transform=input_transform())


def get_valid_set(dest, color_dim=3):
    root_dir = myDataset(dest)
    train_dir = join(root_dir, "valid")
    return TrainDatasetFromFolder(train_dir, color_dim,
                                  input_transform=input_transform())


def get_test_set(dest, color_dim=3):
    root_dir = myDataset(dest)
    test_dir = join(root_dir, "test")
    return TestDatasetFromFolder(test_dir, color_dim,
                                 input_transform=input_transform())


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath, color_dim=3):
    if color_dim == 1:
        img = Image.open(filepath).convert('L')
    else:
        img = Image.open(filepath).convert('RGB')
    return img


class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, class_dir, color_dim=3, input_transform=None):
        super(TrainDatasetFromFolder, self).__init__()
        self.centroid_synthetic_filenames = [join(class_dir, 'centroid_synthetic', x) for x in sorted(listdir(join(class_dir, 'centroid_synthetic'))) if is_image_file(x)]
        self.synthetic_filenames = [join(class_dir, 'synthetic', x) for x in sorted(listdir(join(class_dir, 'synthetic'))) if is_image_file(x)]
        self.neuron_filenames = [join(class_dir, 'neuron', x) for x in sorted(listdir(join(class_dir, 'neuron'))) if is_image_file(x)]
        self.input_transform = input_transform
        self.image_dir = class_dir
        self.color_dim = color_dim

    def __getitem__(self, index):
        index_synthetic = index % len(self.synthetic_filenames)
        filename_synthetic = self.synthetic_filenames[index_synthetic]
        input_synthetic = load_img(self.synthetic_filenames[index_synthetic], 3)
        input_synthetic_centroid = load_img(self.centroid_synthetic_filenames[index_synthetic], 1)
        filename_centroid_synthetic = self.centroid_synthetic_filenames[index_synthetic]
        index_neuron = random.randint(0, len(self.neuron_filenames) - 1)
        filename_neuron = self.synthetic_filenames[index_neuron]
        input_neuron = load_img(self.neuron_filenames[index_neuron], 3)
        self.filename_synthetic = self.synthetic_filenames[index]
        self.filename_neuron = self.neuron_filenames[index_neuron]

        if self.input_transform:
            input_synthetic = self.input_transform(input_synthetic)
            input_synthetic_centroid = self.input_transform(input_synthetic_centroid)
            input_neuron = self.input_transform(input_neuron)

        return  input_synthetic, input_neuron, input_synthetic_centroid, filename_synthetic, filename_neuron, filename_centroid_synthetic

    def __len__(self):
        return len(self.synthetic_filenames)


class TestDatasetFromFolder(data.Dataset):
    def __init__(self, class_dir, color_dim, input_transform=None):
        super(TestDatasetFromFolder, self).__init__()
        self.synthetic_filenames = [join(class_dir, 'synthetic', x) for x in listdir(join(class_dir, 'synthetic')) if is_image_file(x)]
        self.neuron_filenames = [join(class_dir, 'neuron', x) for x in listdir(join(class_dir, 'neuron')) if is_image_file(x)]
        self.input_transform = input_transform
        self.image_dir = class_dir
        self.color_dim = color_dim

    def __getitem__(self, index):
        input_synthetic = load_img(self.synthetic_filenames[index % len(self.synthetic_filenames)], 3)
        index_neuron = random.randint(0, len(self.neuron_filenames) - 1)
        input_neuron = load_img(self.neuron_filenames[index_neuron], 3)
        filename_synthetic = self.synthetic_filenames[index]
        filename_neuron= self.neuron_filenames[index_neuron]

        if self.input_transform:
            input_synthetic = self.input_transform(input_synthetic)
            input_neuron = self.input_transform(input_neuron)

        return input_synthetic, input_neuron, filename_synthetic, filename_neuron

    def __len__(self):
        return len(self.synthetic_filenames)
