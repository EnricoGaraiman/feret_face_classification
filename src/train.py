import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm


def training_stage(DATASET, dataset_train_loader, dataset_test_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

    model = torch.hub.load('pytorch/vision:v0.10.0', DATASET['model'])
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(1, DATASET['epochs']):
        print(f'\n-------------------------------------------------\nEpoch: {epoch}\n-------------------------------------------------')
        train_loss = train(model, dataset_train_loader, optimizer, loss_fn, device)
        test_loss = test(model, dataset_test_loader, loss_fn, device)

        #  deriving model accuracy on the training set
        train_acc = accuracy(model, dataset_train_loader, device)

        #  deriving model accuracy on the validation set
        test_acc = accuracy(model, dataset_test_loader, device)
        print(f'Training accuracy: {train_acc} % | Training loss: {train_loss} ||| Testing accuracy: {test_acc} % | Testing loss: {test_loss}')

        if len(test_acc_history) > 0 and test_acc > np.max(test_acc_history):
            torch.save(model.state_dict(), 'runs/best.pth')
            print(f'Saved best model in epoch {epoch}')

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

    return train_loss_history, train_acc_history, test_loss_history, test_acc_history


def train(model, training_loader, optimizer, loss_function, device):
    total_loss = 0

    model.train()

    for images, labels in tqdm(training_loader):
        images, labels = images.to(device), labels.to(device)

        #  zeroing optimizer gradients
        optimizer.zero_grad()

        #  classifying instances
        classifications = model(images)

        #  computing loss
        loss = loss_function(classifications, labels)
        total_loss += loss.item()

        #  computing gradients
        loss.backward()

        #  optimizing weights
        optimizer.step()

    return np.array(total_loss).mean()


def test(model, testing_loader, loss_function, device):
    total_loss = 0

    #  defining model state
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(testing_loader):
            images, labels = images.to(device), labels.to(device)

            #  making classifications
            classifications = model(images)

            #  computing loss
            loss = loss_function(classifications, labels)
            total_loss += loss.item()

    return np.array(total_loss).mean()


def accuracy(network, dataloader, device):
    network.eval()

    total_correct = 0
    total_instances = 0

    #  iterating through batches
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            #  making classifications
            classifications = torch.argmax(network(images), dim=1)

            #  comparing
            correct_predictions = sum(classifications == labels).item()

            #  incrementing counters
            total_correct += correct_predictions
            total_instances += len(images)

    return round(total_correct / total_instances * 100, 2)

# def train_loop(dataloader, model, loss_fn, optimizer, device):
#     size = len(dataloader.dataset)
#     train_loss, correct = 0, 0
#
#     for batch, (data, labels) in enumerate(dataloader):
#         data = data.to(device)
#         labels = labels.to(device)
#
#         # Clear the gradients
#         optimizer.zero_grad()
#         # Forward Pass
#         target = model(data)
#         # Find the Loss
#         loss = loss_fn(target, labels)
#         # Calculate gradients
#         loss.backward()
#         # Update Weights
#         optimizer.step()
#         # Calculate Loss
#         train_loss += loss.item()
#
#         #
#         #
#         # # Compute prediction and loss
#         # pred = model(X)
#         # loss = loss_fn(pred, y)
#         #
#         # # get statistics
#         correct += (target.argmax(1) == labe).type(torch.float).sum().item()
#         #
#         # # Backpropagation
#         # optimizer.zero_grad()
#         # loss.backward()
#         # optimizer.step()
#         #
#
#         # if batch % 100 == 0:
#         #     loss, current = loss.item(), (batch + 1) * len(X)
#         #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
#     return train_loss / len(dataloader), 100 * correct / size
#
#
# def test_loop(dataloader, model, loss_fn, device):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss, correct = 0, 0
#
#     with torch.no_grad():
#         for X, y in dataloader:
#             X = X.to(device)
#             y = y.to(device)
#
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#
#     return test_loss, 100 * correct
