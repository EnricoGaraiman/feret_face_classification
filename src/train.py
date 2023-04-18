import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm
import time
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

def training_stage(DATASET, dataset_train_loader, dataset_test_loader):
    """
    Training stage

    :param DATASET: dataset info
    :param dataset_train_loader: train loader
    :param dataset_test_loader: test loader
    :return: acc & loss history, best_model
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

    model = torch.hub.load('pytorch/vision:v0.10.0', DATASET['model'])
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=DATASET['learning_rate'])
    # scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=10, factor=0.5)

    best_epoch = 0
    best_model = ''
    count_last_best = 0
    start_time_training = time.time()
    for epoch in range(1, DATASET['epochs'] + 1):
        print(
            f'\n-------------------------------------------------\nEpoch {epoch}\n-------------------------------------------------')

        train_loss, train_acc = train(model, dataset_train_loader, optimizer, loss_fn, device, DATASET)
        test_loss, test_acc = test(model, dataset_test_loader, loss_fn, device, DATASET)
        # scheduler.step(test_loss)

        print(
            f'Training accuracy: {train_acc} % | Training loss: {train_loss} ||| Testing accuracy: {test_acc} % | Testing loss: {test_loss}')

        if len(test_acc_history) > 0 and test_acc > np.max(test_acc_history):
            best_model = 'runs/best_' + str(epoch) + '.pth'
            best_epoch = epoch
            torch.save(model.state_dict(), 'runs/best_' + str(epoch) + '.pth')
            count_last_best = 0
            print(f'Saved best model in epoch {epoch}')

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        if epoch in [10, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
            print(f'Training total time epoch {epoch}: {time.time() - start_time_training} seconds')
            print(f'Best loss in epoch: {best_epoch}')
            print(f'Best loss: {np.min(test_loss_history)}')
            print(f'Best accuracy: {np.max(test_acc_history)} %\n')

        if DATASET['early_stopping'] and DATASET['early_stopping_epochs_no_best'] < count_last_best:
            print('Early Stopping')
            break

        count_last_best += 1

    end_time_training = time.time()
    print(f'Training total time: {end_time_training - start_time_training} seconds')
    print(f'Best loss in epoch: {best_epoch}')
    print(f'Best loss: {np.min(test_loss_history)}')
    print(f'Best accuracy: {np.max(test_acc_history)} %\n')

    return train_loss_history, train_acc_history, test_loss_history, test_acc_history, best_model


def train(model, training_loader, optimizer, loss_function, device, DATASET):
    """
    Train loop

    :param model: network
    :param training_loader: train loader
    :param optimizer: optimizer
    :param loss_function: loss function
    :param device: device (cpu / gpu)
    :param DATASET: info
    :return: acc & loss history
    """
    total_loss = 0
    avg_acc = []

    model.train()

    for images, labels, features in tqdm(training_loader):
        images, labels, features = images.to(device), labels.to(device), features.to(device)

        #  zeroing optimizer gradients
        optimizer.zero_grad()

        #  classifying instances
        if DATASET['feature_extraction'] is not None:
            classifications = model(features)
        else:
            classifications = model(images)

        #  get accuracy
        correct_predictions = sum(torch.argmax(classifications, dim=1) == labels).item()
        avg_acc.append(correct_predictions / len(images) * 100)

        #  computing loss
        loss = loss_function(classifications, labels)
        total_loss += loss.item()

        #  computing gradients
        loss.backward()

        #  optimizing weights
        optimizer.step()

    return np.array(total_loss).mean(), round(np.array(avg_acc).mean(), 2)


def test(model, testing_loader, loss_function, device, DATASET):
    """
    Test loop

    :param model: network
    :param testing_loader: test loader
    :param loss_function: loss function
    :param device: device (cpu / gpu)
    :param DATASET: info
    :return: acc & loss history
    """
    total_loss = 0
    avg_acc = []

    #  defining model state
    model.eval()

    with torch.no_grad():
        for images, labels, features in tqdm(testing_loader):
            images, labels, features = images.to(device), labels.to(device), features.to(device)

            #  making classifications
            if DATASET['feature_extraction'] is not None:
                classifications = model(features)
            else:
                classifications = model(images)

            #  get accuracy
            correct_predictions = sum(torch.argmax(classifications, dim=1) == labels).item()
            avg_acc.append(correct_predictions / len(images) * 100)

            #  computing loss
            loss = loss_function(classifications, labels)
            total_loss += loss.item()

    return np.array(total_loss).mean(), round(np.array(avg_acc).mean(), 2)


def create_mlp_model(DATASET, classes_number):
    """
    Create MLP

    :param DATASET: DATASET info
    :param classes_number: classes number
    :return: model
    """

    model = nn.Sequential(
        nn.Linear(DATASET['mlp_layers_input']['input'], DATASET['mlp_layers_input']['hidden1']),  # input layer
        nn.ReLU(),
        nn.Linear(DATASET['mlp_layers_input']['hidden1'], DATASET['mlp_layers_input']['output']),
        nn.ReLU(),
        # nn.Linear(DATASET['mlp_layers_input']['hidden2'], DATASET['mlp_layers_input']['hidden3']),
        # nn.ReLU(),
        # nn.Linear(DATASET['mlp_layers_input']['hidden3'], DATASET['mlp_layers_input']['hidden4']),
        # nn.ReLU(),
        # nn.Linear(DATASET['mlp_layers_input']['hidden4'], DATASET['mlp_layers_input']['hidden5']),
        # nn.ReLU(),
        # nn.Linear(DATASET['mlp_layers_input']['hidden5'], DATASET['mlp_layers_input']['hidden6']),
        # nn.ReLU(),
        # nn.Linear(DATASET['mlp_layers_input']['hidden6'], DATASET['mlp_layers_input']['output']),
        # nn.ReLU(),
        nn.Linear(DATASET['mlp_layers_input']['output'], classes_number),  # output layer
        nn.Sigmoid()
    )

    return model


def training_stage_mlp(DATASET, dataset_train_loader, dataset_test_loader, classes):
    """
    Training stage mlp

    :param DATASET: DATASET info
    :param dataset_train_loader: dataset_train_loader
    :param dataset_test_loader: dataset_test_loader
    :param classes: classes
    :return: acc & loss history, best_model
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

    model = create_mlp_model(DATASET, len(classes))

    model.to(device)
    print('\n---------------\nMLP summary\n---------------\n')
    summary(model, input_size=(DATASET['mlp_layers_input']['input'],))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=DATASET['learning_rate']) #, weight_decay=0.01)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5, factor=0.5)

    best_epoch = 0
    best_model = ''
    count_last_best = 0
    start_time_training = time.time()
    for epoch in range(1, DATASET['epochs'] + 1):
        print(
            f'\n-------------------------------------------------\nEpoch {epoch}\n-------------------------------------------------')

        train_loss, train_acc = train(model, dataset_train_loader, optimizer, loss_fn, device, DATASET)
        test_loss, test_acc = test(model, dataset_test_loader, loss_fn, device, DATASET)
        # scheduler.step(test_loss)

        print(
            f'Training accuracy: {train_acc} % | Training loss: {train_loss} ||| Testing accuracy: {test_acc} % | Testing loss: {test_loss}')

        if len(test_acc_history) > 0 and test_acc > np.max(test_acc_history):
            best_model = 'runs/best_' + str(epoch) + '.pth'
            best_epoch = epoch
            torch.save(model.state_dict(), 'runs/best_' + str(epoch) + '.pth')
            count_last_best = 0
            print(f'Saved best model in epoch {epoch}')

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        if epoch in [10, 30, 50, 70, 100]:
            print(f'Training total time epoch {epoch}: {time.time() - start_time_training} seconds')
            print(f'Best loss in epoch: {best_epoch}')
            print(f'Best loss: {np.min(test_loss_history)}')
            print(f'Best accuracy: {np.max(test_acc_history)} %\n')

        if DATASET['early_stopping'] and DATASET['early_stopping_epochs_no_best'] < count_last_best:
            print('Early Stopping')
            break

        count_last_best += 1

    end_time_training = time.time()
    print(f'Training total time: {end_time_training - start_time_training} seconds')
    print(f'Best loss in epoch: {best_epoch}')
    print(f'Best loss: {np.min(test_loss_history)}')
    print(f'Best accuracy: {np.max(test_acc_history)} %\n')

    return train_loss_history, train_acc_history, test_loss_history, test_acc_history, best_model
