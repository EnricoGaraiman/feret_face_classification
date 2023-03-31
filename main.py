import src.dataset as dataset_file
import src.train as train_file
import src.results as results_file

"""
   PARAMETERS
"""
DATASET = {
    'name': 'COLOR FERET FACE RECOGNITION',
    'images_dir': 'data/colorferet/converted_images/images',
    'ground_truths_dir': 'data/colorferet/converted_images/ground_truths',
    'subjects_info': 'data/colorferet/converted_images/ground_truths/xml/subjects.xml',
    'recordings_info': 'data/colorferet/converted_images/ground_truths/xml/recordings.xml',
    'split_factor': 0.8,
    'subjects_percent': 0.1, #0.015 94.9% pt 100 epoci
    'mtcnn_detect': False,
    'size': [224, 224],
    'data_loader_train': {
        'batch_size': 10,
        'num_workers': 0,
        'shuffle': True
    },
    'data_loader_test': {
        'batch_size': 1,
        'num_workers': 0,
        'shuffle': False
    },
    'model': 'resnet18',
    'epochs': 100,
    'learning_rate': 1e-3,
    'dataset_train_mean': [1.2244, 1.1298, 1.0203],  # calculate dataset_file.dataset_mean_and_std(dataset_train_loader)
    'dataset_train_std': [11.2739, 11.1779, 10.6974],  # calculate dataset_file.dataset_mean_and_std(dataset_train_loader)
}

"""
   MAIN FUNCTION
"""
if __name__ == '__main__':
    # dataset visualisation
    # dataset_file.plot_dataset_visualisation(DATASET)

    # dataset
    train_images_name, test_images_name, classes = dataset_file.get_dataset_images_name(DATASET)
    print('Images: ', len(train_images_name) + len(train_images_name))
    print('Train [%]', len(train_images_name), len(train_images_name) / (len(train_images_name) + len(test_images_name)) * 100)
    print('Test [%]', len(test_images_name), len(test_images_name) / (len(train_images_name) + len(test_images_name)) * 100)
    dataset_train, dataset_test = dataset_file.get_dataset(DATASET, train_images_name, test_images_name, classes)

    # dataset loaders
    dataset_train_loader = dataset_file.get_dataset_loader(dataset_train, DATASET, 0)
    dataset_test_loader = dataset_file.get_dataset_loader(dataset_test, DATASET, 1)

    # dataset train mean & std
    # mean, std = dataset_file.dataset_mean_and_std(dataset_train_loader)
    # print(mean, std)

    # training stage
    # train_loss_history, train_acc_history, test_loss_history, test_acc_history = train_file.training_stage(DATASET, dataset_train_loader, dataset_test_loader)

    # results
    # results_file.plot_training_results(DATASET, train_loss_history, train_acc_history, test_loss_history, test_acc_history)
    results_file.plot_confusion_matrix(DATASET, dataset_test_loader, test_images_name, classes)
