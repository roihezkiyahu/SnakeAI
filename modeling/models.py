import torch.nn as nn
import torch.nn.functional as F
import torch


class CNNDQNAgent(nn.Module):
    def __init__(self, input_shape, output_size,
                 conv_layers_params=[{'in_channels': 10, 'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 1},
                                    {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
                                    {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}
                                    ], fc_layers=[256]):
        super(CNNDQNAgent, self).__init__()
        self.conv_layers = nn.ModuleList()
        for layer_params in conv_layers_params:
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=layer_params['in_channels'],
                    out_channels=layer_params['out_channels'],
                    kernel_size=layer_params['kernel_size'],
                    stride=layer_params['stride'],
                    padding=layer_params['padding']
                )
            )

        self.fc_layers = nn.ModuleList()
        self.fc_input_dim = self.feature_size(input_shape)
        for units in fc_layers:
            self.fc_layers.append(nn.Linear(self.fc_input_dim, units))
            self.fc_input_dim = units
        self.fc_layers.append(nn.Linear(fc_layers[-1] if fc_layers else self.fc_input_dim, output_size))

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = x.view(x.size(0), -1)  # Flatten
        for i, fc in enumerate(self.fc_layers):
            x = F.relu(fc(x)) if i < len(self.fc_layers) - 1 else fc(x)
        return x

    def feature_size(self, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            for conv in self.conv_layers:
                x = F.relu(conv(x))
            return x.view(1, -1).size(1)



class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations+14, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


