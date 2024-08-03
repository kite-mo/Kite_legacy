from datetime import datetime
import sys, os, glob
import torch
from tqdm import tqdm
import numpy as np
import copy


def save_model_by_date(
    model, optimizer, model_name, save_path, epoch, n_epochs, avg_valid_losses
):

    global new

    now = datetime.now().strftime('%Y_%m_%d')
    date_save_path = save_path + '/' + now + '/'

    if not os.path.exists(date_save_path):
        os.makedirs(date_save_path)
        new = 1

    if epoch == 1:
        new = 0
        if len(glob.glob(date_save_path + '*.pt')) > 0:
            for ex_model in glob.glob(date_save_path + '*.pt'):
                os.remove(ex_model)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                os.path.join(
                    date_save_path,
                    '{}_epoch_{}_valid_loss_{}.pt'.format(
                        model_name, str(epoch), str(avg_valid_losses[-1])
                    ),
                ),
            )
        else:
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                os.path.join(
                    date_save_path,
                    '{}_epoch_{}_valid_loss_{}.pt'.format(
                        model_name, str(epoch), str(avg_valid_losses[-1])
                    ),
                ),
            )

    elif new == 1:
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            os.path.join(
                date_save_path,
                '{}_epoch_{}_valid_loss_{}.pt'.format(
                    model_name, str(epoch), str(avg_valid_losses[-1])
                ),
            ),
        )
        new = 0

    elif n_epochs == epoch:
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            os.path.join(
                date_save_path,
                '{}_last_epoch_{}_valid_loss_{}.pt'.format(
                    model_name, str(epoch), str(avg_valid_losses[-1])
                ),
            ),
        )

    else:
        if avg_valid_losses[-1] <= np.min(avg_valid_losses[:-1]):
            print('updated model saved!')
            ex_model = glob.glob(date_save_path + '*.pt')[0]
            os.remove(ex_model)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                os.path.join(
                    date_save_path,
                    '{}_epoch_{}_valid_loss_{}.pt'.format(
                        model_name, str(epoch), str(avg_valid_losses[-1])
                    ),
                ),
            )


def loss_vae(model, x, vae_criterion):

    # squeezed_x, squeezed_x_hat, x_hat, latent, mu, kl, layer_output = model(x)
    total_outputs = model(x)
    x_ori, x_hat, kl = total_outputs[0], total_outputs[1], total_outputs[-2]
    vae_loss = vae_criterion(x_ori, x_hat) + kl

    return vae_loss


def loss_mlp_ae(model, x, ae_criterion):

    squeezed_x, squeezed_x_hat, latent_x = model(x)
    ae_loss = ae_criterion(squeezed_x, squeezed_x_hat)

    return ae_loss


def loss_base_ae(model, x, ae_criterion):

    squeezed_x, squeezed_x_hat, latent_x, layer_output = model(x)
    ae_loss = ae_criterion(squeezed_x, squeezed_x_hat)

    return ae_loss


def pre_train_ae_model(
    model,
    model_name,
    save_path,
    n_epochs,
    device,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    criterion_type=None,
):
    model.to(device)
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    avg_valid_losses = []

    print('start-training')
    for epoch in range(1, n_epochs + 1):

        # to track the training loss as the model trains
        train_losses = []
        valid_losses = []

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for batch, (data, label, torr, id) in tqdm(enumerate(train_loader, 1)):
            data = data.float().to(device)
            label = label.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # calculate the loss
            if criterion_type == 'vae':
                loss = loss_vae(model, data, criterion)
            elif criterion_type == 'ae':
                loss = loss_base_ae(model, data, criterion)
            elif criterion_type == 'mlp_ae':
                loss = loss_mlp_ae(model, data, criterion)
            else:
                raise Exception('Not corrected criterion name')

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)

        print('epochs : {} / avg_train_loss : {} '.format(epoch, train_loss))
        avg_train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            for batch, (data, label, torr, id) in tqdm(enumerate(valid_loader, 1)):
                data = data.float().to(device)
                label = label.to(device)
                if criterion_type == 'vae':
                    loss = loss_vae(model, data, criterion)
                elif criterion_type == 'ae':
                    loss = loss_base_ae(model, data, criterion)
                elif criterion_type == 'mlp_ae':
                    loss = loss_mlp_ae(model, data, criterion)
                valid_losses.append(loss.item())
            valid_loss = np.average(valid_losses)

            print('epochs : {} / avg_valid_loss : {} '.format(epoch, valid_loss))
            avg_valid_losses.append(valid_loss)

        if epoch == 1:
            pass
        else:
            if avg_valid_losses[-1] < np.min(avg_valid_losses[:-1]):
                print('updated model saved!')
                model_info = [epoch, model.state_dict()]

        save_model_by_date(
            model, optimizer, model_name, save_path, epoch, n_epochs, avg_valid_losses
        )

    return model, avg_train_losses, avg_valid_losses, model_info


def deep_sad_loss(X, y, model_, final_center, eta, eps, type=None):

    layer_output = []

    if type == 'vae':
        # squeezed_x, latent, mu, kl, unpooling_idx, layer_output = model_(X)
        total_outputs = model_(X)
        mu, layer_output = total_outputs[2], total_outputs[-1]
        layer_outputs = copy.copy(layer_output)
        layer_outputs.append(mu)
    elif type == 'ae':
        # squeezed_x, latent_x, unpooling_idx, layer_output = model_(X)
        total_outputs = model_(X)
        latent_x, layer_output = total_outputs[1], total_outputs[-1]
        layer_outputs = copy.copy(layer_output)
        layer_outputs.append(latent_x)
    else:
        raise Exception('Not corrected model type name')

    fin_loss = 0
    fin_dist = 0
    for idx, layer_z in enumerate(layer_outputs):

        dist = torch.sum((layer_z - final_center[idx]) ** 2, dim=1)
        losses = torch.where(y == 0, dist, eta * ((dist + eps) ** y.float()))
        loss = torch.mean(losses)

        fin_dist += dist
        fin_loss += loss

    return fin_dist, fin_loss


def train_deep_sad_model(
    model,
    model_name,
    save_path,
    n_epochs,
    device,
    train_loader,
    valid_loader,
    optimizer,
    final_center,
    eta,
    eps,
    type,
):
    model.to(device)
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    avg_valid_losses = []

    print('start-training')
    for epoch in range(1, n_epochs + 1):

        # to track the training loss as the model trains
        train_losses = []
        valid_losses = []

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for batch, (data, label, torr, id) in tqdm(enumerate(train_loader, 1)):
            data = data.float().to(device)
            label = label.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # calculate the loss
            dist, loss = deep_sad_loss(data, label, model, final_center, eta, eps, type)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)

        print('epochs : {} / avg_train_loss : {} '.format(epoch, train_loss))
        avg_train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            for batch, (data, label, torr, id) in tqdm(enumerate(valid_loader, 1)):
                data = data.float().to(device)
                label = label.to(device)
                dist, loss = deep_sad_loss(data, label, model, final_center, eta, eps, type)
                valid_losses.append(loss.item())
            valid_loss = np.average(valid_losses)

            print('epochs : {} / avg_valid_loss : {} '.format(epoch, valid_loss))
            avg_valid_losses.append(valid_loss)

        if epoch == 1:
            pass
        else:
            if avg_valid_losses[-1] < np.min(avg_valid_losses[:-1]):
                print('updated model saved!')
                model_info = [epoch, model.state_dict()]

        save_model_by_date(
            model, optimizer, model_name, save_path, epoch, n_epochs, avg_valid_losses
        )

    return model, avg_train_losses, avg_valid_losses, model_info
