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
    'split_factor': 0.95,  # 0.8 for 70/30, 0.875 for 80/20, 0.95 for 90/10
    'split_factor_sub': 5,  # 7 for 70/30, 7 for 80/20, 5 for 90/10
    'subjects_percent': None,
    'size': [384, 256],  # 768, 512 orig | 384, 256 | 198, 128
    'data_loader_train': {
        'batch_size': 10,
        'num_workers': 0,
        'shuffle': True,
    },
    'data_loader_test': {
        'batch_size': 1,
        'num_workers': 0,
        'shuffle': False,
    },
    'model': 'mlp',  # mlp, resnet18, resnet34, ...
    'mlp_layers_input': {
        'input': 19840, #11040 for 198x128, 19840for 384x256
        'hidden1': 7000,
        'hidden2': 6000,
        'hidden3': 5000,
        'hidden4': 4000,
        'hidden5': 3000,
        'hidden6': 2000,
        'output': 2000,
    },
    'feature_extraction': 'HOG',  # None, HOG, LBP | for MLP
    'epochs': 100,
    'early_stopping': True,
    'early_stopping_epochs_no_best': 100,
    'learning_rate': 1e-4,
    'use_cache': True,
    'augmentation': False,
    'dataset_train_mean': [1.2244, 1.1298, 1.0203],  # calculate dataset_file.dataset_mean_and_std(dataset_train_loader)
    'dataset_train_std': [11.2739, 11.1779, 10.6974],  # calculate dataset_file.dataset_mean_and_std(dataset_train_loader)
}

"""
   MAIN FUNCTION
"""
if __name__ == '__main__':
    try:
        # dataset
        train_images_name, test_images_name, classes = dataset_file.get_dataset_images_name(DATASET)
        print('Images: ', len(train_images_name) + len(test_images_name))
        print('Train [%]', len(train_images_name), len(train_images_name) / (len(train_images_name) + len(test_images_name)) * 100)
        print('Test [%]', len(test_images_name), len(test_images_name) / (len(train_images_name) + len(test_images_name)) * 100)
        dataset_train, dataset_test = dataset_file.get_dataset(DATASET, train_images_name, test_images_name, classes)

        if DATASET['model'] == 'mlp':
            # dataset loaders
            dataset_train_loader = dataset_file.get_dataset_loader_mlp(dataset_train, DATASET, 0)
            dataset_test_loader = dataset_file.get_dataset_loader_mlp(dataset_test, DATASET, 1)

            train_loss_history, train_acc_history, test_loss_history, test_acc_history, best_model = train_file.training_stage_mlp(DATASET, dataset_train_loader, dataset_test_loader, classes)
        else:
            # dataset loaders
            dataset_train_loader = dataset_file.get_dataset_loader(dataset_train, DATASET, 0)
            dataset_test_loader = dataset_file.get_dataset_loader(dataset_test, DATASET, 1)

            # dataset visualisation
            # dataset_file.plot_dataset_visualisation(DATASET, dataset_train_loader, dataset_test_loader, classes)

            # dataset train mean & std
            # mean, std = dataset_file.dataset_mean_and_std(dataset_train_loader)
            # print(mean, std)

            # training stage
            train_loss_history, train_acc_history, test_loss_history, test_acc_history, best_model = train_file.training_stage(DATASET, dataset_train_loader, dataset_test_loader)

        # results
        results_file.plot_training_results(DATASET, train_loss_history, train_acc_history, test_loss_history, test_acc_history)
        predictions, real_labels = results_file.plot_confusion_matrix(DATASET, dataset_test_loader, test_images_name, classes, best_model)
        results_file.plot_correct_wrong_predictions(DATASET, dataset_test_loader, predictions, real_labels, classes)

    except OSError as err:
        print("OS error:", err)
    except ValueError as err:
        print(err)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise
