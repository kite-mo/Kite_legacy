import torch
from tqdm import tqdm
import numpy as np


def get_hypersphere_center(model, pre_train_loader, device, type='vae'):

    latent_z_list = []
    layer1_z_list = []
    layer2_z_list = []
    layer3_z_list = []
    layer4_z_list = []

    model.eval()
    with torch.no_grad():
        for num, data in enumerate(tqdm(pre_train_loader)):
            data, labels = data

            data = data.float().to(device)
            labels = labels.type(torch.LongTensor).to(device)
            if type == 'vae':
                total_outputs = model(data)
                mu, layer_output = total_outputs[2], total_outputs[-1]
                latent_z_list.append(mu.cpu().numpy())
            elif type == 'ae':
                total_outputs = model(data)
                latent_x, layer_output = total_outputs[-2], total_outputs[-1]
                latent_z_list.append(latent_x.cpu().numpy())

            layer1_z_list.append(layer_output[0].cpu().numpy())
            layer2_z_list.append(layer_output[1].cpu().numpy())
            layer3_z_list.append(layer_output[2].cpu().numpy())
            layer4_z_list.append(layer_output[3].cpu().numpy())

        layer_z_list = [
                layer1_z_list,
                layer2_z_list,
                layer3_z_list,
                layer4_z_list,
                latent_z_list,
            ]

        centers = [
            np.squeeze(np.mean(np.mean(np.array(layer_z), axis=0), axis=0))
            for layer_z in layer_z_list
        ]
        center_tensor = [torch.Tensor(center).to(device) for center in centers]

    return center_tensor


def get_anomaly_score(X, y, model_, final_center, type=None):
    if type == 'vae':
        squeezed_x, latent, mu, kl, unpooling_idx, layer_output = model_(X)
        # total_outputs = model_(X)
        # mu, layer_output = total_outputs[-3], total_outputs[-1]
        layer_output.append(mu)
    elif type == 'ae':
        squeezed_x, latent_x, unpooling_idx, layer_output = model_(X)
        # total_outputs = model_(X)
        # latent_x, layer_output = total_outputs[-2], total_outputs[-1]
        layer_output.append(latent_x)

    layer_anomaly = [[], [], [], [], []]
    total_anomaly = 0

    for idx, layer_z in enumerate(layer_output):
        dist = torch.sum((layer_z - final_center[idx]) ** 2, dim=1)
        layer_anomaly[idx].append(dist.cpu().numpy())
        total_anomaly += dist.cpu().numpy()

    return layer_anomaly, total_anomaly


def test_deep_sad(model, test_loader, device, final_center, type):

    layer_score_list = [[], [], [], [], []]
    total_score_list = []
    label_list = []

    model.eval()
    with torch.no_grad():
        for batch, (data, label, torr, id) in tqdm(enumerate(test_loader, 1)):

            data = data.float().to(device)
            label = label.to(device)
            layer_anomaly, total_anomaly = get_anomaly_score(
                data, label, model, final_center, type
            )

            for idx, anomaly in enumerate(layer_anomaly):
                layer_score_list[idx].append(anomaly)
            label_list.append(label.cpu().numpy())
            total_score_list.append(total_anomaly)

    fin_label_list = np.concatenate(label_list, axis=0)
    fin_total_anomaly = np.concatenate(total_score_list, axis=0)

    return layer_score_list, fin_total_anomaly, fin_label_list


def inference_ss_lad(model, test_loader, device, layer_range=[0, 1, 2, 3, 4], type='vae'):
    """Get concatenated layer outputs & label list

    Parameters:
    model (.pt): Trained model
    test_loader (DataLoader) : test set dataloader
    device : cuda device
    layer_range (list) : composed layer number
                         [0, 1, 2, 3, 4] means that use overall layers

    Returns:
    fin_concat_latent (np.array) : concatenated layer outputs
                                   shape -> (test dataset num, concatenated latent dimension)
    fin_label (np.array) : label array
    """
    total_latent = []
    label_list = []

    model.eval()
    with torch.no_grad():
        for batch, (data, label, torr, id) in enumerate(test_loader, 1):

            data = data.float().to(device)
            label = label.to(device)
            label_list.append(label.cpu().numpy())

            squeezed_x, latent, mu, kl, unpooling_idx, layer_output = model(data)

            if type == 'vae':

                layer_output.append(mu)

                if len(layer_range) == 0:
                    raise Exception('Layer range is Empty')

                if min(layer_range) < 0 or max(layer_range) > 4:
                    raise Exception('Layer range should in [0 ~ 4]')

                # outputs part
                concat_latent = np.empty((mu.shape[0], 0), float)
                for layer_num in layer_range:
                    np_layer_ouput = layer_output[layer_num].cpu().numpy()
                    concat_latent = np.hstack((concat_latent, np_layer_ouput))
                total_latent.append(concat_latent)

            elif type == 'ae':
                total_latent.append(mu.cpu().numpy())

    fin_label = np.concatenate(label_list, axis=0)
    fin_concat_latent = np.concatenate(total_latent, axis=0)

    return fin_concat_latent, fin_label
