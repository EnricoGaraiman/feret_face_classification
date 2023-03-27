import src.dataset as dataset_file

"""
   PARAMETERS
"""
DATASET = {
    'name': 'COLOR FERET FACE RECOGNITION',
    'images_dir': 'data/colorferet/converted_images/images',
    'ground_truths_dir': 'data/colorferet/converted_images/ground_truths',
    'subjects_info': 'data/colorferet/converted_images/ground_truths/xml/subjects.xml',
    'recordings_info': 'data/colorferet/converted_images/ground_truths/xml/recordings.xml',
    'test_files': {
        'data/colorferet/converted_images/partitions/dup1.txt',
        'data/colorferet/converted_images/partitions/dup2.txt',
        'data/colorferet/converted_images/partitions/fa.txt',
        'data/colorferet/converted_images/partitions/fb.txt',
    },
    'mtcnn_detect': True,
    'size': [160, 160],
    'data_loader': {
        'batch_size': 4,
        'num_workers': 4,
        'shuffle': True
    }
}

"""
   MAIN FUNCTION
"""
if __name__ == '__main__':
    # dataset visualisation
    dataset_file.plot_dataset_visualisation(DATASET)

    # dataset
    # train_images_name, test_images_name = dataset_file.get_dataset_images_name(DATASET)
    # dataset_train, dataset_test = dataset_file.get_dataset(DATASET, train_images_name, test_images_name)
    #
    # # dataset loaders
    # dataset_train_loader = dataset_file.get_dataset_loader(dataset_train, DATASET)
    # dataset_test_loader = dataset_file.get_dataset_loader(dataset_test, DATASET)
