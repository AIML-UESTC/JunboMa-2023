import torch.nn as nn
import torch
# from blitz.modules import BayesianLSTM, BayesianConv2d, BayesianLinear

class Net(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11),                       # input[3, 512, 512]  output[64, 502, 502]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=3),                  # output[64, 166, 166]
            nn.Conv2d(64, 128, kernel_size=11),                     # output[128, 156, 156]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=3),                  # output[128, 51, 51]
            nn.Conv2d(128, 256, kernel_size=11),                    # output[256, 41, 41]
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=3),                  # output[256, 13, 13]
            nn.Conv2d(256, 512, kernel_size=11),                    # output[512, 3, 3]
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),                  # output[512, 1, 1]
        )
        self.classifier = nn.Sequential(
            # nn.LSTM(512, 1024, prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0),
            nn.Linear(512, 1024),
            nn.Linear(1024, 100),
            nn.Linear(100, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
