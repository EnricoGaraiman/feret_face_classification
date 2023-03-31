import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm.auto import tqdm


def plot_training_results(DATASET, train_loss_history, train_acc_history, test_loss_history, test_acc_history):
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
    plt.savefig('data/results/training/loss_results_' + DATASET['model'] + '_' + str(DATASET['epochs']) + '.jpg')

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
    plt.savefig('data/results/training/accuracy_results_' + DATASET['model'] + '_' + str(DATASET['epochs']) + '.jpg')


def plot_confusion_matrix(DATASET, dataset_test_loader, test_images_name, classes):
    labels = [classes.index(label.split('_')[0]) for label in test_images_name]

    model = torch.hub.load('pytorch/vision:v0.10.0', DATASET['model'])
    model.load_state_dict(torch.load('runs/best.pth'))
    model.eval()

    test_accuracy, predictions = get_test_predictions(dataset_test_loader, model)

    print(f"Test dataset results: \n Accuracy: {test_accuracy :>0.1f}%\n")

    cmx = confusion_matrix(labels, predictions, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cmx, display_labels=classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle('Confusion Matrix | ' + DATASET['model'] + ' | ' + DATASET['name'], fontsize=20)
    plt.xlabel('True label', fontsize=16)
    plt.ylabel('Predicted label', fontsize=16)
    disp.plot(ax=ax)
    plt.savefig('data/results/confusion-matrix_' + DATASET['model'] + '_' + str(DATASET['epochs']) + '.png')


def get_test_predictions(dataloader, model):
    total_correct, total_instances = 0, 0
    predictions = []
    real_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            pred = torch.argmax(model(images), dim=1)
            correct_predictions = sum(pred == labels).item()

            #  incrementing counters
            total_correct += correct_predictions
            total_instances += len(images)

            predictions.append(pred.numpy()[0])
            real_labels.append(labels.numpy()[0])

    print(predictions)
    print(real_labels)

    return round(total_correct / total_instances * 100, 2), predictions
