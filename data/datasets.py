

import scipy
import numpy as np
from skimage.io import imread
from skimage.util import img_as_ubyte
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split

from medmnist import PneumoniaMNIST, DermaMNIST, BloodMNIST

import torchvision.datasets as datasets

def load_mnist_digits():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    train_images = np.asarray([x[0] for x in mnist_trainset])
    train_labels = np.asarray([x[1] for x in mnist_trainset]).reshape((-1, 1))
    test_images = np.asarray([x[0] for x in mnist_testset])
    test_labels = np.asarray([x[1] for x in mnist_testset]).reshape((-1, 1))
    print(train_images.dtype)
    return train_images, train_labels, test_images, test_labels


def build_single_heldout_test_set(images: np.ndarray,
                                  labels: np.ndarray,
                                  test_size: float = 0.15,
                                  random_state: int = 0):
    indices = np.arange(images.shape[0])
    train_dev_indices, test_indices, train_dev_labels, test_labels = train_test_split(
        indices,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels.reshape(-1)
    )
    return (
        images[train_dev_indices],
        train_dev_labels,
        images[test_indices],
        test_labels,
        test_indices,
    )


def load_dataset_with_heldout_test(dataset_name: str, data_path: str):
    if dataset_name == 'tuberculosis':
        images, labels = load_lung_data(data_path)
        return build_single_heldout_test_set(images, labels)
    elif dataset_name == 'raw_tuberculosis':
        images, labels = load_raw_tuberculosis_data(data_path)
        return build_single_heldout_test_set(images, labels)
    elif dataset_name == 'liver_fatty':
        images, labels = load_data(data_path)
        return build_single_heldout_test_set(images, labels)
    elif dataset_name == 'liver_fatty_masked':
        images, labels = load_liver_masked(data_path)
        return build_single_heldout_test_set(images, labels)
    elif dataset_name == 'medmnist_pneumonia':
        images, labels, test_images, test_labels = load_medmnist_pneumonia()
        return images, labels, test_images, test_labels, np.arange(test_images.shape[0])
    elif dataset_name == 'medmnist_derma':
        images, labels, test_images, test_labels = load_medmnist_derma()
        return images, labels, test_images, test_labels, np.arange(test_images.shape[0])
    elif dataset_name == 'brain_mri':
        images, labels, test_images, test_labels = load_brain_mri_data(data_path)
        return images, labels, test_images, test_labels, np.arange(test_images.shape[0])
    elif dataset_name == 'mnist':
        images, labels, test_images, test_labels = load_mnist_digits()
        return images, labels, test_images, test_labels, np.arange(test_images.shape[0])
    raise ValueError(f'Unsupported dataset: {dataset_name}')

def load_data(data_path: str):
    file_name = f'{data_path}/liver.mat'
    mat = scipy.io.loadmat(file_name)
    #mask = nib.load(f'{data_path}/mask.nii').get_fdata()
    images = []
    labels = []
    patient_label = []
    data = mat['data'][0]
    for i in range(55):
        label = data[i][1]
        id = data[i][0]
        for image in data[i][3]:
            patient_label.append(id)
            images.append(img_as_ubyte(resize(image, (256, 256))))
            labels.append(label)
    images = np.asarray(images)
    labels = np.asarray(labels).reshape((-1,1))
    return images, labels
    
def load_liver_masked(data_path: str):
    file_name = f'{data_path}/liver.mat'
    mat = scipy.io.loadmat(file_name)
    mask = img_as_ubyte(imread(f'{data_path}/mask.png'))
    mask[mask > 0] = 1
    images = []
    labels = []
    patient_label = []
    data = mat['data'][0]
    for i in range(55):
        label = data[i][1]
        id = data[i][0]
        for image in data[i][3]:
            patient_label.append(id)
            images.append(img_as_ubyte(resize(image, (256, 256))) * mask)
            labels.append(label)
    images = np.asarray(images)
    labels = np.asarray(labels).reshape((-1,1))
    return images, labels


def load_medmnist_pneumonia():
    train_dataset = PneumoniaMNIST(split="train", size=224, download=True)
    val_dataset = PneumoniaMNIST(split="val", size=224, download=True)
    test_dataset = PneumoniaMNIST(split="test", size=224, download=True)
    images = np.concatenate([train_dataset.imgs, val_dataset.imgs], axis=0).reshape((-1, 224, 224))
    labels = np.concatenate([train_dataset.labels, val_dataset.labels], axis=0)
    test_images = test_dataset.imgs.reshape((-1, 224, 224))
    test_labels = test_dataset.labels
    return images, labels, test_images, test_labels

def load_medmnist_derma():
    train_dataset = BloodMNIST(split="train", size=64, download=True)
    val_dataset = BloodMNIST(split="val", size=64, download=True)
    test_dataset = BloodMNIST(split="test", size=64, download=True)
    images = np.concatenate([train_dataset.imgs, val_dataset.imgs], axis=0)
    labels = np.concatenate([train_dataset.labels, val_dataset.labels], axis=0)
    images_result = np.zeros(shape=images.shape[:-1])
    test_images_result = np.zeros(shape=test_dataset.imgs.shape[:-1])
    print(images.shape)
    for i in range(len(images)):
        images_result[i] = img_as_ubyte(rgb2gray(images[i]))
    for i in range(len(test_dataset.imgs)):
        test_images_result[i] = img_as_ubyte(rgb2gray(test_dataset.imgs[i]))
    return images_result, labels, test_images_result, test_dataset.labels


def load_lung_data(data_path: str):
    images = np.load(f'{data_path}/images_only_roi.npy')
    labels = np.load(f'{data_path}/labels.npy').astype(np.int32).reshape((-1, 1))
    return images, labels

def load_raw_tuberculosis_data(data_path: str):
    images = np.load(f'{data_path}/images.npy')
    labels = np.load(f'{data_path}/labels.npy').astype(np.int32).reshape((-1, 1))
    return images, labels

def load_brain_mri_data(data_path: str):
    images_train = np.load(f'{data_path}/brain_mri_train.npy')
    images_test = np.load(f'{data_path}/brain_mri_test.npy')
    labels_train = np.load(f'{data_path}/brain_mri_train_labels.npy').astype(np.int32)
    labels_test = np.load(f'{data_path}/brain_mri_test_labels.npy').astype(np.int32)
    return images_train, labels_train.reshape((-1, 1)), images_test, labels_test.reshape((-1, 1))