import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

        r = 16
        senet_block = [ nn.AvgPool2d(kernel_size=1, stride=1),
                        nn.Conv2d(in_features, in_features//r, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_features//r, in_features, 1),
                        nn.Sigmoid()]
        self.senet_block = nn.Sequential(*senet_block)

    def forward(self, x):
        x1 = self.conv_block(x)
        # x1 = self.senet_block(x1)
        return x + x1

# class Generator(nn.Module):
#     def __init__(self, input_nc, output_nc, n_residual_blocks=9):
#         super(Generator, self).__init__()
#
#         # Initial convolution block
#         model = [   nn.ReflectionPad2d(1),
#                     nn.Conv2d(input_nc, 64, 3),
#                     nn.InstanceNorm2d(64),
#                     nn.ReLU(inplace=True) ]
#
#         # Downsampling
#         in_features = 64
#         out_features = in_features*2
#         for _ in range(4):
#             model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
#                         nn.InstanceNorm2d(out_features),
#                         nn.ReLU(inplace=True) ]
#             in_features = out_features
#             out_features = in_features*2
#
#         # Residual blocks
#         for _ in range(n_residual_blocks):
#             model += [ResidualBlock(in_features)]
#
#         # Upsampling
#         out_features = in_features//2
#         for _ in range(4):
#             model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
#                         nn.InstanceNorm2d(out_features),
#                         nn.ReLU(inplace=True) ]
#             in_features = out_features
#             out_features = in_features//2
#
#         # Output layer
#         model += [  nn.ReflectionPad2d(1),
#                     nn.Conv2d(64, output_nc, 3),
#                     nn.Tanh() ]
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x):
#         return self.model(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(4):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        # print(f'1: {in_features}')
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # model_classifier = model + nn.Linear(in_features, 2)

        # Upsampling
        out_features = in_features//2
        model1 = model
        model2 = model
        # print(f'2: {in_features}')
        for _ in range(4):
            model1 = model1 + [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            model2 = model2 + [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # print(f'3: {in_features}')
        # Output layer
        model1 += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]
        model2 += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, 1, 7),
                    nn.Tanh() ]

        # self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        # self.model_classifier = nn.Sequential(*model_classifier)

    def forward(self, x):
        # print(f'x: {x.shape}')
        # y=self.model(x)
        # print(f'y: {y.shape}')
        return self.model1(x), self.model2(x)#, self.model_classifier

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)