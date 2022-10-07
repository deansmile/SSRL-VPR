import torch
from torch import nn

vgg16_stack = [{"num_layer": 2, "in_channels":3, "out_channels": 64}, 
                {"num_layer": 2, "in_channels":64, "out_channels": 128},
                {"num_layer": 3, "in_channels":128, "out_channels": 256},
                {"num_layer": 3, "in_channels":256, "out_channels": 512},
                {"num_layer": 2, "in_channels":512, "out_channels": 512, "pooling": None},
                {"num_layer": 1, "in_channels":512, "out_channels": 512, "pooling": None, "activation": False}]

class Autoencoder(nn.Module):
    def __init__(self, stack=vgg16_stack):
        super(Autoencoder, self).__init__()
        encoder_conv_layers = []
        for pam in stack:
            encoder_conv_layers += self._make_encoder_layers(**pam)
        encoder_conv_layers = nn.Sequential(*encoder_conv_layers)
        self.encoder = nn.Sequential(encoder_conv_layers)
        
        decoder_conv_layers = []
        for pam in stack[::-1]:
            decoder_conv_layers += self._make_decoder_layers(**pam)
        decoder_conv_layers = nn.Sequential(*decoder_conv_layers)
        self.decoder = nn.Sequential(decoder_conv_layers)

    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def _make_encoder_layers(self, num_layer, in_channels, out_channels, kernel_size=(3, 3), activation=True, pooling="max", pooling_kernel=2, pooling_stride=2, **kwargs):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
        if activation: layers.append(nn.ReLU(inplace=True))
        for i in range(num_layer - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1))
            if activation: layers.append(nn.ReLU(inplace=True))
        if pooling is not None:
            layers.append(nn.MaxPool2d(pooling_kernel, stride=pooling_stride))
        return layers
    
    def _make_decoder_layers(self, num_layer, in_channels, out_channels, kernel_size=(3, 3), activation=True, pooling="max", **kwargs):
        layers = []
        if pooling is not None:
            layers.append(nn.Upsample(scale_factor=(2, 2)))
        for i in range(num_layer - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1))
            if activation:layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, in_channels, kernel_size, padding=1))
        if activation: layers.append(nn.ReLU(inplace=True))
        return layers