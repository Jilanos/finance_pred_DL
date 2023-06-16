import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 80, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(80, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.fc = nn.Linear(10 * 10 * 256, 512)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.batch_norm(self.conv2(x)))
        x = self.relu(self.batch_norm(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(512, 10 * 10 * 256)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(128, 80, kernel_size=5, stride=2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(80, 1, kernel_size=5, stride=2, padding=2)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 10, 10)
        x = self.relu(self.batch_norm(self.deconv1(x)))
        x = self.relu(self.batch_norm(self.deconv2(x)))
        x = self.deconv3(x)
        return x

class VisualAE(nn.Module):
    def __init__(self):
        super(VisualAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Exemple d'utilisation
model = VisualAE()
input_image = torch.randn(1, 1, 80, 80)  # Remplacez cela par votre image d'entrée réelle

output_image = model(input_image)
print(output_image.shape)  