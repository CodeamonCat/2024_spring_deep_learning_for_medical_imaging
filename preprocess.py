import json
import os
import shutil
from sklearn.model_selection import train_test_split


def create_dataset(type, filenames, labels):
    dataset_dir = './dataset/'
    dataset_dict = dict()

    # copy files
    os.makedirs(os.path.join(dataset_dir, type))
    for filename in filenames:
        source = os.path.join(dataset_dir, filename)
        destination = os.path.join(dataset_dir, type, filename)
        shutil.copy(source, destination)

    # write annotation.json
    dataset_dict['filenames'] = filenames
    dataset_dict['labels'] = labels
    annotation_path = os.path.join(dataset_dir, type, "annotations.json")
    with open(annotation_path, "w") as outfile:
        json.dump(dataset_dict, outfile)

    print("===finished create_dataset {type}===")


if __name__ == '__main__':

    dataset_dir = './dataset/'
    filenames = list()
    labels = list()
    dataset = list()

    for root, dirs, files in os.walk(dataset_dir):
        for filename in files:
            filenames.append(filename)
            if "benign" in filename:
                labels.append(0)
            elif "malignant" in filename:
                labels.append(1)
            elif "normal" in filename:
                labels.append(2)
            else:
                raise NameError('Unknown image classification type')
            dataset.append(os.path.join(root, filename))

    x_train, x_test, y_train, y_test = train_test_split(filenames,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=1216,
                                                        shuffle=True)

    # 0.25 x 0.8 = 0.2
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.25,
                                                      random_state=1216,
                                                      shuffle=True)

    create_dataset("train", x_train, y_train)
    create_dataset("test", x_test, y_test)
    create_dataset("valid", x_val, y_val)
