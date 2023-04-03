import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm


def training_stage(DATASET, dataset_train_loader, dataset_test_loader):
    """
    Training stage

    :param DATASET: dataset info
    :param dataset_train_loader: train loader
    :param dataset_test_loader: test loader
    :return: acc & loss history
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

    model = torch.hub.load('pytorch/vision:v0.10.0', DATASET['model'])
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(1, DATASET['epochs']):
        print(
            f'\n-------------------------------------------------\nEpoch {epoch}\n-------------------------------------------------')

        train_loss, train_acc = train(model, dataset_train_loader, optimizer, loss_fn, device)
        test_loss, test_acc = test(model, dataset_test_loader, loss_fn, device)

        print(
            f'Training accuracy: {train_acc} % | Training loss: {train_loss} ||| Testing accuracy: {test_acc} % | Testing loss: {test_loss}')

        if len(test_acc_history) > 0 and test_acc > np.max(test_acc_history):
            torch.save(model.state_dict(), 'runs/best.pth')
            print(f'Saved best model in epoch {epoch}')

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

    return train_loss_history, train_acc_history, test_loss_history, test_acc_history


def train(model, training_loader, optimizer, loss_function, device):
    """
    Train loop

    :param model: network
    :param training_loader: train loader
    :param optimizer: optimizer
    :param loss_function: loss function
    :param device: device (cpu / gpu)
    :return: acc & loss history
    """
    total_loss = 0
    avg_acc = []

    model.train()

    for images, labels in tqdm(training_loader):
        images, labels = images.to(device), labels.to(device)

        #  zeroing optimizer gradients
        optimizer.zero_grad()

        #  classifying instances
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


def test(model, testing_loader, loss_function, device):
    """
    Test loop

    :param model: network
    :param testing_loader: test loader
    :param loss_function: loss function
    :param device: device (cpu / gpu)
    :return: acc & loss history
    """
    total_loss = 0
    avg_acc = []

    #  defining model state
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(testing_loader):
            images, labels = images.to(device), labels.to(device)

            #  making classifications
            classifications = model(images)

            #  get accuracy
            correct_predictions = sum(torch.argmax(classifications, dim=1) == labels).item()
            avg_acc.append(correct_predictions / len(images) * 100)

            #  computing loss
            loss = loss_function(classifications, labels)
            total_loss += loss.item()

    return np.array(total_loss).mean(), round(np.array(avg_acc).mean(), 2)
