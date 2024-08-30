import torch
import torch.nn as nn

class AE_Encoder(nn.Module):
    def __init__(self, sensor_num, latent_dims):
        super().__init__()

        # 같은 값으로 interpolation 된 애들 줄여줄라고
        self.squeeze_layer = torch.nn.Sequential(
            torch.nn.ReLU(), torch.nn.MaxPool1d(kernel_size=4, stride=4, return_indices=True)
        )
        self.encoder_layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=sensor_num, out_channels=16, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=1, return_indices=True),
        )
        self.encoder_layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=1, return_indices=True),
        )
        self.encoder_layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=1, return_indices=True),
        )
        self.encoder_layer4 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, return_indices=True),
        )
        self.latent_layer = torch.nn.Sequential(
            torch.nn.Linear(384, latent_dims),
        )

    def forward(self, x):

        global_averge_layer = torch.nn.AdaptiveAvgPool1d(1)

        squeezed_x, idx0 = self.squeeze_layer(x)
        x, idx1 = self.encoder_layer1(squeezed_x)
        x1 = global_averge_layer(x)
        x, idx2 = self.encoder_layer2(x)
        x2 = global_averge_layer(x)
        x, idx3 = self.encoder_layer3(x)
        x3 = global_averge_layer(x)
        x, idx4 = self.encoder_layer4(x)
        x4 = global_averge_layer(x)
        x = torch.flatten(x, start_dim=1)
        latent_x = self.latent_layer(x)

        unpooling_idx = [idx0, idx1, idx2, idx3, idx4]  # , idx5
        layer_output = [torch.flatten(x, start_dim=1) for x in [x1, x2, x3, x4]]  # , x5

        return squeezed_x, latent_x, unpooling_idx, layer_output


class AE_Decoder(nn.Module):
    def __init__(self, sensor_num, latent_dims):
        super().__init__()

        self.unsqueeze_layer = torch.nn.MaxUnpool1d(4, stride=4)
        # self.unpool_layer1 = torch.nn.MaxUnpool1d(3, stride=2)
        self.unpool_layer2 = torch.nn.MaxUnpool1d(3, stride=2)
        self.unpool_layer3 = torch.nn.MaxUnpool1d(3, stride=1)
        self.unpool_layer4 = torch.nn.MaxUnpool1d(3, stride=1)
        self.unpool_layer5 = torch.nn.MaxUnpool1d(3, stride=1)

        self.extention_layer = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, 384),
            torch.nn.ReLU(),
        )

        self.decoder_layers2 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            ),
            torch.nn.ReLU(),
        )
        self.decoder_layers3 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1
            ),
            torch.nn.ReLU(),
        )
        self.decoder_layers4 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(
                in_channels=32, out_channels=16, kernel_size=3, stride=1
            ),
            torch.nn.ReLU(),
        )
        self.decoder_layers5 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(
                in_channels=16, out_channels=sensor_num, kernel_size=3, stride=1
            ),
            torch.nn.ReLU(),
        )

    def forward(self, x, unpooling_idx):

        x = self.extention_layer(x)
        x = x.reshape(x.shape[0], 64, -1)
        x = self.unpool_layer2(x, unpooling_idx[-1])
        x = self.decoder_layers2(x)

        x = self.unpool_layer3(x, unpooling_idx[-2])
        x = self.decoder_layers3(x)
        x = self.unpool_layer4(x, unpooling_idx[-3])
        x = self.decoder_layers4(x)
        x = self.unpool_layer5(x, unpooling_idx[-4])
        squeezed_x_hat = self.decoder_layers5(x)
        x_hat = self.unsqueeze_layer(squeezed_x_hat, unpooling_idx[-5])
        return squeezed_x_hat, x_hat


class base_AE(nn.Module):
    def __init__(self, sensor_num, latent_dims):
        super().__init__()
        self.encoder = AE_Encoder(sensor_num, latent_dims)
        self.decoder = AE_Decoder(sensor_num, latent_dims)

    def forward(self, x):

        squeezed_x, latent_x, unpooling_idx, layer_output = self.encoder(x)
        squeezed_x_hat, x_hat = self.decoder(latent_x, unpooling_idx)

        return squeezed_x, squeezed_x_hat, latent_x, layer_output