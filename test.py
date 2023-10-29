from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from myDatasets import get_training_set, get_test_set
import torchvision
import os
from os.path import join
import time
from PseudoNeuronGAN import Generator, Discriminator
import utils
import itertools
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dest = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--image_size', type=int, default=512)

# Training options
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--input_nc', type=int, default=3)
parser.add_argument('--output_nc', type=int, default=3)
parser.add_argument('--model_name', default='PseudoNeuronGAN')

args = parser.parse_args()

batch_size = args.batch_size
image_size = args.image_size
epochs = args.epochs
learning_rate = args.learning_rate
input_nc = args.input_nc
output_nc = args.output_nc
model_name = args.model_name

obj = f'{model_name}_epochs{epochs}'


# A - synthetic  B - neuron
netG_A2B = Generator(input_nc, output_nc)
netG_B2A = Generator(output_nc, input_nc)

netG_A2B.to(device)
netG_B2A.to(device)

# Load state dicts
netG_A2B.load_state_dict(torch.load(f'{obj}/netG_A2B.pth', map_location='cpu'))
netG_B2A.load_state_dict(torch.load(f'{obj}/netG_B2A.pth', map_location='cpu'))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_synthetic = Tensor(batch_size, input_nc, image_size, image_size)
input_neuron = Tensor(batch_size, output_nc, image_size, image_size)

# Dataset loader
print('===> Loading testing datasets')
test_set = get_test_set(dest, input_nc)
test_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=batch_size, shuffle=False)

os.makedirs(f'{obj}/B2A', exist_ok=True)
os.makedirs(f'{obj}/A2B', exist_ok=True)

for i, (input_A, input_B, filename_A, filename_B) in enumerate(test_data_loader):
    # Set model input
    real_A = Variable(input_synthetic.copy_(input_A)).to(device)
    real_B = Variable(input_neuron.copy_(input_B)).to(device)

    # Generate output
    fake_B = 0.5 * (netG_A2B(real_A)[0].data + 1.0)
    fake_A = 0.5 * (netG_B2A(real_B)[0].data + 1.0)

    # Save image files
    filename_predB = (filename_A[0].split('.png', 1)[0] + "_predict.png").split("synthetic\\",1)[-1]
    filename_predA = (filename_B[0].split('.png', 1)[0] + "_predict.png").split("neuron\\",1)[-1]
    torchvision.utils.save_image(fake_A, f'{obj}/B2A/{filename_predA}')
    torchvision.utils.save_image(fake_B, f'{obj}/A2B/{filename_predB}')
