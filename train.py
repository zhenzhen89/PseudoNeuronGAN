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
netG_A2B = Generator(input_nc, output_nc).to(device)
netG_B2A = Generator(output_nc, input_nc).to(device)
netD_A = Discriminator(input_nc).to(device)
netD_B = Discriminator(output_nc).to(device)
netD_centroidA = Discriminator(1).to(device)
netD_centroidB = Discriminator(1).to(device)

netG_A2B.apply(utils.weights_init_normal)
netG_B2A.apply(utils.weights_init_normal)
netD_A.apply(utils.weights_init_normal)
netD_B.apply(utils.weights_init_normal)
netD_centroidA.apply(utils.weights_init_normal)
netD_centroidB.apply(utils.weights_init_normal)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_centroid = torch.nn.MSELoss()
criterion_CE = torch.nn.CrossEntropyLoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_centroidA = torch.optim.Adam(netD_centroidA.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_centroidB = torch.optim.Adam(netD_centroidB.parameters(), lr=learning_rate, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=utils.LambdaLR(epochs, 0, int(epochs/2)).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=utils.LambdaLR(epochs, 0, int(epochs/2)).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=utils.LambdaLR(epochs, 0, int(epochs/2)).step)
lr_scheduler_D_CenroidA = torch.optim.lr_scheduler.LambdaLR(optimizer_D_centroidA, lr_lambda=utils.LambdaLR(epochs, 0, int(epochs/2)).step)
lr_scheduler_D_CenroidB = torch.optim.lr_scheduler.LambdaLR(optimizer_D_centroidB, lr_lambda=utils.LambdaLR(epochs, 0, int(epochs/2)).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_synthetic = Tensor(batch_size, input_nc, image_size, image_size)
input_neuron = Tensor(batch_size, output_nc, image_size, image_size)
target_real = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)

fake_A_buffer = utils.ReplayBuffer()
fake_B_buffer = utils.ReplayBuffer()
centroid_same_A_buffer = utils.ReplayBuffer()
centroid_same_B_buffer = utils.ReplayBuffer()
centroid_recoveredA_buffer = utils.ReplayBuffer()
centroid_recoveredB_buffer = utils.ReplayBuffer()
centroid_fakeA_buffer = utils.ReplayBuffer()
centroid_fakeB_buffer = utils.ReplayBuffer()

# Dataset loader
print('===> Loading training datasets')
train_set = get_training_set(dest, input_nc)
training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True)
length = len(training_data_loader.dataset)

# Loss plot
logger = utils.Logger(epochs, len(training_data_loader))
###################################

with open(join(dest,obj+'.csv'),'w') as epoch_log:
    epoch_log.write('epoch,avg_loss_G,avg_loss_G_centroid,avg_loss_G_identity,avg_loss_G_GAN,avg_loss_G_cycle,avg_loss_D_A,avg_loss_D_B,avg_loss_D_centroidA,avg_loss_D_centroidB\n')

###### Training ######
start = time.time()
for epoch in range(epochs):
    avg_loss_G = 0
    avg_loss_G_centroid = 0
    avg_loss_G_identity = 0
    avg_loss_G_GAN = 0
    avg_loss_G_cycle = 0
    avg_loss_D_A = 0
    avg_loss_D_B = 0
    avg_loss_D_centroidA = 0
    avg_loss_D_centroidB = 0
    print('Epoch {}/{}'.format(epoch + 1, epochs))
    for i, (input_A, input_B, centroid_A, filename_A, filename_B, filename_centroid_A) in enumerate(training_data_loader):
        # print(filename_A, filename_B, filename_centroid_A)
        real_A = Variable(input_synthetic.copy_(input_A)).to(device)
        real_B = Variable(input_neuron.copy_(input_B)).to(device)
        centroid_A = centroid_A.to(device)

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B, same_B_centroid = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)
        loss_identity_B_centroid = criterion_centroid(same_B_centroid, 1-real_B)
        # G_B2A(A) should equal A if real A is fed
        same_A, same_A_centroid = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)
        loss_identity_A_centroid = criterion_centroid(same_A_centroid, centroid_A) + criterion_centroid(same_A_centroid, 1-real_A)

        # GAN loss
        fake_B, fake_B_centroid = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
        loss_GAN_A2B_centroid = criterion_centroid(fake_B_centroid, centroid_A) + \
                                criterion_centroid(fake_B_centroid, 1-real_B) + \
                                criterion_GAN(netD_centroidB(fake_B_centroid), target_fake)

        fake_A, fake_A_centroid = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
        loss_GAN_B2A_centroid = criterion_centroid(fake_A_centroid, 1-real_A) + \
                                criterion_GAN(netD_centroidA(fake_A_centroid), target_fake)

        # Cycle loss
        recovered_A, recovered_A_centroid = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)
        loss_cycle_ABA_centroid = criterion_centroid(recovered_A_centroid, centroid_A) + criterion_centroid(recovered_A_centroid, 1-real_A)

        recovered_B, recovered_B_centroid = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)
        loss_cycle_BAB_centroid = criterion_centroid(recovered_B_centroid, 1-real_B)


        # Centroid Detection loss
        loss_G_centroidA = (loss_identity_A_centroid + loss_GAN_B2A_centroid + loss_cycle_ABA_centroid)
        loss_G_centroidB = (loss_identity_B_centroid + loss_GAN_A2B_centroid + loss_cycle_BAB_centroid)


        # Total loss
        loss_identity = (loss_identity_A + loss_identity_B)*3
        loss_GAN = (loss_GAN_A2B + loss_GAN_B2A)*0.5
        loss_cycle = (loss_cycle_ABA + loss_cycle_BAB)*3
        loss_centroid = (loss_G_centroidA + loss_G_centroidB)*2
        loss_G = loss_identity + loss_GAN + loss_cycle + loss_centroid
        loss_G.backward()

        optimizer_G.step()
        ###################################


        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        ###### Discriminator centroid A ######
        optimizer_D_centroidA.zero_grad()

        # Real loss
        pred_real_centroid = netD_centroidA(centroid_A)
        loss_D_real = criterion_centroid(pred_real_centroid, target_real)

        # Fake loss
        fake_A_centroid = centroid_fakeA_buffer.push_and_pop(fake_A_centroid)
        pred_fake_A_centroid = netD_centroidA(fake_A_centroid.detach())
        loss_D_fake_A_centroid = criterion_GAN(pred_fake_A_centroid, target_fake)

        recovered_A_centroid = centroid_recoveredA_buffer.push_and_pop(recovered_A_centroid)
        pred_recovered_centroid = netD_centroidA(recovered_A_centroid.detach())
        loss_D_recovered = criterion_centroid(pred_recovered_centroid, target_fake)

        same_A_centroid = centroid_same_A_buffer.push_and_pop(same_A_centroid)
        pred_same_centroid = netD_centroidA(same_A_centroid.detach())
        loss_D_same = criterion_centroid(pred_same_centroid, target_fake)

        # Total loss
        loss_D_CentroidA = (loss_D_real + loss_D_fake_A_centroid + loss_D_recovered + loss_D_same)
        loss_D_CentroidA.backward()

        optimizer_D_centroidA.step()
        ###################################

        ###### Discriminator centroid B ######
        optimizer_D_centroidB.zero_grad()

        loss_D_real = 0

        # Fake loss
        fake_B = centroid_fakeB_buffer.push_and_pop(fake_B_centroid)
        pred_fake = netD_centroidB(fake_B.detach())
        loss_D_fake = criterion_centroid(pred_fake, target_fake)

        recovered_B_centroid = centroid_recoveredB_buffer.push_and_pop(recovered_B_centroid)
        pred_recovered_centroid = netD_centroidB(recovered_B_centroid.detach())
        loss_D_recovered = criterion_centroid(pred_recovered_centroid, target_fake)

        same_B_centroid = centroid_same_B_buffer.push_and_pop(same_B_centroid)
        pred_same_centroid = netD_centroidB(same_B_centroid.detach())
        loss_D_same = criterion_centroid(pred_same_centroid, target_fake)

        # Total loss
        loss_D_CentroidB = (loss_D_real + loss_D_fake + loss_D_recovered + loss_D_same)
        loss_D_CentroidB.backward()

        optimizer_D_centroidB.step()
        ###################################


        logger.log({'loss_G': loss_G,
                    'loss_centroid': loss_centroid,
                    'loss_G_identity': loss_identity,
                    'loss_G_GAN': loss_GAN,
                    'loss_G_cycle': loss_cycle,
                    'loss_D_A': loss_D_A,
                    'loss_D_B': loss_D_B})

        avg_loss_G += loss_G
        avg_loss_G_centroid += loss_centroid
        avg_loss_G_identity += loss_identity
        avg_loss_G_GAN += loss_GAN
        avg_loss_G_cycle += loss_cycle
        avg_loss_D_A += loss_D_A
        avg_loss_D_B += loss_D_B
        avg_loss_D_centroidA += loss_D_CentroidA
        avg_loss_D_centroidB += loss_D_CentroidB
        # avg_loss_D_centroid += loss_D_Centroid
    print(f'avg_loss_G: {avg_loss_G / length:4f} '
          f' | avg_loss_G_centroid: {avg_loss_G_centroid/length:4f} '
          f' | avg_loss_G_identity: {avg_loss_G_identity / length:4f}'
          f' | avg_loss_G_GAN: {avg_loss_G_GAN / length:4f}'
          f' | avg_loss_G_cycle: {avg_loss_G_cycle / length:4f}'
          f' | avg_loss_D_A: {avg_loss_D_A / length:4f}'
          f' | avg_loss_D_B: {avg_loss_D_B / length:4f}'
          f' | avg_loss_D_centroidA: {avg_loss_D_centroidA / length:4f}'
          f' | avg_loss_D_centroidB: {avg_loss_D_centroidB / length:4f}')
    with open(join(dest, obj + '.csv'), 'a') as epoch_log:
        epoch_log.write(f'{epoch + 1},'
                        f'{avg_loss_G / length:10f},'
                        f'{avg_loss_G_centroid/length:10f},'
                        f'{avg_loss_G_identity / length:10f},'
                        f'{avg_loss_G_GAN / length:10f},'
                        f'{avg_loss_G_cycle / length:10f},'
                        f'{avg_loss_D_A / length:10f},'
                        f'{avg_loss_D_B / length:10f},'
                        f'{avg_loss_D_centroidA / length:10f},'
                        f'{avg_loss_D_centroidB / length:10f}\n')

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    lr_scheduler_D_CenroidA.step()
    lr_scheduler_D_CenroidB.step()

print('Saving models')
# Create output dirs if they don't exist
os.makedirs(obj, exist_ok=True)
torch.save(netG_A2B.state_dict(), f'{obj}/netG_A2B.pth')
torch.save(netG_B2A.state_dict(), f'{obj}/netG_B2A.pth')
torch.save(netD_A.state_dict(), f'{obj}/netD_A.pth')
torch.save(netD_B.state_dict(), f'{obj}/netD_B.pth')
# torch.save(netG_CentroidB.state_dict(), f'{obj}/netG_Centroid.pth')
end = time.time()
print("\nIt takes {:.2f} h for the training.\n".format((end - start) / 3600))
