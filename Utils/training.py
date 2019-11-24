import torch
import numpy as np
import pandas as pd


def train_epoch(
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        device=torch.device('cuda'),
        verbose=True,
        metric=None,
        accumulate_grad=False,
        accumulate_grad_step=10,
):

    loss_data = dict()

    # Each epoch has a training and validation phase
    for phase in ['train', 'valid']:

        if phase == 'train':
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        running_loss, running_metric, running_counts = 0.0, 0.0, 0
        dataloader = train_loader if phase == 'train' else validation_loader
        n = len(dataloader.dataset)

        for inputs, labels in dataloader:

            # send tensors to specified device
            inputs = inputs.to(device)
            labels = labels.to(device)

            if phase == 'train':

                # zero the parameter gradients
                if not accumulate_grad or (accumulate_grad and running_counts % accumulate_grad_step == 0):
                    optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                metric_ = metric(outputs, labels).item() if metric is not None else 0

                # backward step
                if not accumulate_grad or (accumulate_grad and running_counts % accumulate_grad_step == 0):
                    loss.backward()
                    optimizer.step()

            else:
                with torch.no_grad():

                    # forward
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    metric_ = metric(outputs, labels).item() if metric is not None else 0

            # statistics
            running_loss += loss.item()
            running_metric += metric_
            running_counts += 1

            # print intermediate statistics if verbose==True
            if verbose and phase == 'train':
                if metric is None:
                    print(f'{running_counts * len(inputs)}/{n} loss: {running_loss / running_counts}')
                else:
                    print(f'{running_counts * len(inputs)}/{n} loss: {running_loss / running_counts}; metric: {running_metric / running_counts}')

        # add loss and metric to output
        loss_data[phase] = dict()
        loss_data[phase]['loss'] = running_loss / running_counts
        loss_data[phase]['metric'] = running_metric / running_counts

    return model, loss_data, optimizer.param_groups[0]['lr']


def train_model(
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        device=torch.device('cuda'),
        num_epochs=25,
        early_stopping=True,
        patience=30,
        reduce_lr_on_plateau=None,
        verbose=True,
        metric=None,
        logging_params=None,
        accumulate_grad=False,
        accumulate_grad_step=10,
        frozen_steps=6):

    best_model_wts = model.state_dict()
    best_metric = -np.inf
    best_epoch = 0
    history = []

    counter = 0
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if epoch < frozen_steps:
            model.freeze_to(1)
        else:
            model.unfreeze()

        model, loss_data, curr_lr = train_epoch(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            verbose=verbose,
            metric=metric,
            accumulate_grad=accumulate_grad,
            accumulate_grad_step=accumulate_grad_step
        )

        # unpack train/validation loss and metric
        train_loss, train_metric = loss_data['train']['loss'], loss_data['train']['metric']
        valid_loss, valid_metric = loss_data['valid']['loss'], loss_data['valid']['metric']

        # print epoch performance
        if metric is None:
            print(f'Epoch: {epoch} train loss: {train_loss}')
            print(f'Epoch: {epoch} validation loss: {valid_loss}')
        else:
            print(f'Epoch: {epoch} train loss: {train_loss}; train metric: {train_metric}')
            print(f'Epoch: {epoch} validation loss: {valid_loss}; validation metric: {valid_metric}')

        # update history
        history.append(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_metric': train_metric,
                'validation_loss': valid_loss,
                'validation_metric': valid_metric,
                'lr': curr_lr
            }
        )

        # apply lr reduction if available
        if reduce_lr_on_plateau is not None:
            reduce_lr_on_plateau.step(valid_metric)

        # # apply criterion update
        # if criterion.update_on_training and (epoch + 1) % criterion.update_on_epoch == 0:
        #     criterion.update()

        # deep copy the model
        if valid_metric > best_metric:
            best_metric = valid_metric
            best_model_wts = model.state_dict()
            best_epoch = epoch

            counter = 1  # reset counter of poor epochs if loss was reduced
        else:
            counter += 1  # increment counter of poor epochs if loss increased

        # save state to file if specified
        if logging_params is not None:
            if epoch % logging_params['save_frequency'] == 0:
                if logging_params['replace_with_best']:
                    torch.save(best_model_wts, f'{logging_params["folder"]}/best_state_epoch_{epoch}.pt')
                else:
                    torch.save(model.state_dict(), f'{logging_params["folder"]}/state_epoch_{epoch}.pt')

        # break iterations be early stopping if specified
        if early_stopping and counter == patience:
            print('Break learning by early stopping')
            break

    print(f'Model best loss: {best_metric}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    # store best model if specified
    if logging_params is not None and logging_params['store_best']:
        torch.save(model, f'{logging_params["folder"]}/best_model_epoch_{best_epoch}.pth')

    # store fit trace if specified
    if logging_params is not None and logging_params['store_trace']:
        history = pd.DataFrame(history,
                               columns=['epoch', 'train_loss', 'train_metric',
                                        'validation_loss', 'validation_metric', 'lr'])
        history.to_csv(f'{logging_params["folder"]}/fit_trace.csv', sep=';', header=True, index=False)

    return model
