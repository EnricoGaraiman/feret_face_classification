from collections import Counter
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm.auto import tqdm
import time
import src.train as train_file

def plot_training_results(DATASET, train_loss_history, train_acc_history, test_loss_history, test_acc_history):
    """
    Plot results from training stage

    :param DATASET: dataset info
    :param train_loss_history: train loss
    :param train_acc_history: train accuracy
    :param test_loss_history: test loss
    :param test_acc_history: test accuracy
    """
    plt.figure()
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train loss', color='brown')
    plt.plot(range(1, len(test_loss_history) + 1), test_loss_history, label='Test loss', color='darkgreen')
    plt.title('Loss results | ' + DATASET['model'] + ' | ' + str(DATASET['epochs']) + ' epochs')
    plt.legend()
    ax = plt.gca()
    ax.set_ylabel('Value')
    ax.set_xlabel('Epoch')
    # plt.hlines(y=np.min(test_loss_history), color='red', zorder=1, xmin=0, xmax=len(test_loss_history) - 1)
    # plt.vlines(x=test_loss_history.index(np.min(test_loss_history)), color='red', zorder=2, ymin=1, ymax=np.max(test_loss_history))
    plt.tight_layout()
    plt.savefig('data/results/training/loss_results_' + DATASET['model'] + '_' + str(DATASET['epochs']) + '.jpg', dpi=300)

    plt.figure()
    plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, label='Train accuracy', color='brown')
    plt.plot(range(1, len(test_acc_history) + 1), test_acc_history, label='Test accuracy', color='darkgreen')
    plt.title('Accuracy results | ' + DATASET['model'] + ' | ' + str(DATASET['epochs']) + ' epochs')
    plt.legend()
    ax = plt.gca()
    ax.set_ylabel('Value [%]')
    ax.set_xlabel('Epoch')
    # plt.hlines(y=np.max(test_acc_history), color='red', zorder=1, xmin=0, xmax=len(test_acc_history) - 1 )
    # plt.vlines(x=test_acc_history.index(np.max(test_acc_history)), color='red', zorder=2, ymin=1, ymax=np.max(test_acc_history))
    plt.tight_layout()
    plt.savefig('data/results/training/accuracy_results_' + DATASET['model'] + '_' + str(DATASET['epochs']) + '.jpg', dpi=300)


def plot_confusion_matrix(DATASET, dataset_test_loader, test_images_name, classes, best_model):
    """
    Plot confusion matrix (full / partial)

    :param DATASET: dataset info
    :param dataset_test_loader: dataset test loader
    :param test_images_name: names of images for test dataset
    :param classes: as string IDs
    :param best_model: best_model
    :return: predictions & real labels
    """
    labels = [classes.index(label.split('_')[0]) for label in test_images_name]

    if DATASET['model'] != 'mlp':
        model = torch.hub.load('pytorch/vision:v0.10.0', DATASET['model'])
    else:
        model = train_file.create_mlp_model(DATASET, len(classes))
    model.load_state_dict(torch.load(best_model))
    model.eval()

    dataset_test_loader.dataset.use_cache = False

    test_accuracy, predictions, real_labels = get_test_predictions(dataset_test_loader, model, DATASET)

    print(f"Test dataset results: \n Accuracy: {test_accuracy :>0.2f}%\n")

    # # plot only a part of CM
    # predictions_copy = predictions.copy()
    # if len(classes) > 50:
    #     labels = [label for label in real_labels if label in range(0, 50)]
    #     predictions = [pred for pred in predictions if pred in range(0, 50)]
    #     classes = classes[0: 50]

    cmx = confusion_matrix(labels, predictions, labels=classes[0: 50] if len(classes) > 50 else classes, normalize=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cmx, display_labels=classes)

    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    fig.suptitle('Confusion Matrix | ' + DATASET['model'] + ' | ' + DATASET['name'], fontsize=20)
    plt.xlabel('True label', fontsize=16)
    plt.ylabel('Predicted label', fontsize=16)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.savefig('data/results/training/confusion-matrix_' + DATASET['model'] + '_' + str(DATASET['epochs']) + '.png', dpi=fig.dpi)

    return predictions, real_labels


def get_test_predictions(dataloader, model, DATASET):
    """
    Get dataset using a custom dataset

    :param dataloader: dataloader
    :param model: network
    :param DATASET: info
    :return: predictions for test dataset
    """
    total_correct, total_instances = 0, 0
    predictions = []
    real_labels = []

    with torch.no_grad():
        start_time_testing = time.time()

        for images, labels, features in tqdm(dataloader):
            if DATASET['feature_extraction'] is not None:
                pred = torch.argmax(model(features), dim=1)
            else:
                pred = torch.argmax(model(images), dim=1)

            correct_predictions = sum(pred == labels).item()

            #  incrementing counters
            total_correct += correct_predictions
            total_instances += len(images)

            predictions.append(pred.numpy()[0])
            real_labels.append(labels.numpy()[0])
        end_time_testing = time.time()

    print(f'Inference time: {(end_time_testing - start_time_testing) / len(dataloader) * 1000} ms / image')

    return round(total_correct / total_instances * 100, 2), predictions, real_labels


def plot_correct_wrong_predictions(DATASET, dataset_test_loader, predictions, real_labels, classes):
    """
    Plot correct & wrong predictions

    :param DATASET: dataset info
    :param dataset_test_loader: dataset test loader
    :param predictions: predicted labels
    :param real_labels: real labels
    :param classes: classes
    """
    # correct
    count = 0
    fig = plt.figure(figsize=(20, 20), dpi=300)
    index = 0
    for img, label, _ in dataset_test_loader:
        if (predictions[index] == real_labels[index]):
            plt.subplot(10, 6, count + 1)
            plt.title('Predict: ' + classes[predictions[index]] + '\n'
                      + ' Real: ' + classes[real_labels[index]])
            plt.imshow(img[0].permute(1, 2, 0), cmap='gray')
            plt.axis('off')
            count = count + 1
        if count == 60:
            break
        index += 1

    plt.tight_layout()
    fig.savefig('data/results/training/test_correct_predictions_visualisation_' + DATASET['model'] + '_' + str(
        DATASET['epochs']) + '.jpg', dpi=fig.dpi)

    # wrong
    count = 0
    fig = plt.figure(figsize=(20, 20), dpi=300)
    index = 0
    for img, label, _ in dataset_test_loader:
        if (predictions[index] != real_labels[index]):
            plt.subplot(10, 6, count + 1)
            plt.title('Predict: ' + classes[predictions[index]] + '\n'
                      + ' Real: ' + classes[real_labels[index]])
            plt.imshow(img[0].permute(1, 2, 0), cmap='gray')
            plt.axis('off')
            count = count + 1
        if count == 60:
            break
        index += 1

    plt.tight_layout()
    fig.savefig(
        'data/results/training/test_wrong_predictions_visualisation_' + DATASET['model'] + '_' + str(
            DATASET['epochs']) + '.jpg', dpi=fig.dpi)
