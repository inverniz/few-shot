from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import h5py

from config import DATA_PATH


class OmniglotDataset(Dataset):
    def __init__(self, subset):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        instance = io.imread(self.datasetid_to_filepath[item])
        # Reindex to channels first format as supported by pytorch
        instance = instance[np.newaxis, :, :]

        # Normalise to 0-1
        instance = (instance - instance.min()) / (instance.max() - instance.min())

        label = self.datasetid_to_class_id[item]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Omniglot dataset dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            alphabet = root.split('/')[-2]
            class_name = '{}.{}'.format(alphabet, root.split('/')[-1])

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'alphabet': alphabet,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class MiniImageNet(Dataset):
    def __init__(self, subset):
        """Dataset class representing miniImageNet dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images

class ImageNetKamonDataset(Dataset):
    def __init__(self, subset):
        """Dataset class representing miniImageNet dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        elif subset == 'background':
            self.subset = subset
            
            self.df = pd.DataFrame(self.index_subset(self.subset))
            
            # Index of dataframe has direct correspondence to item in dataset
            self.df = self.df.assign(id=self.df.index.values)

            # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
            self.unique_characters = sorted(self.df['class_name'].unique())
            num_classes = len(self.df['class_name'].unique())
            self.class_name_to_id = {self.unique_characters[i]: i for i in range(num_classes)}
            self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))
            
            # Setup transforms
            self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        elif subset == 'evaluation':
            self.subset = subset
            
            csv_filepath = '/data/input/configs/input_files/eval_input_charge_classification_v1.csv'
            self.df = pd.read_csv(csv_filepath, names = ['filepath', 'class_id'])
            
            # Index of dataframe has direct correspondence to item in dataset
            self.df = self.df.assign(id=self.df.index.values)
            
            # Setup transforms
            self.transform = transforms.Compose([
            transforms.Resize((86, 86)),
            transforms.Grayscale(3),
            transforms.ToTensor()
            ])
            
        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_id'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images
    
class DummyDataset(Dataset):
    def __init__(self, samples_per_class=10, n_classes=10, n_features=1):
        """Dummy dataset for debugging/testing purposes

        A sample from the DummyDataset has (n_features + 1) features. The first feature is the index of the sample
        in the data and the remaining features are the class index.

        # Arguments
            samples_per_class: Number of samples per class in the dataset
            n_classes: Number of distinct classes in the dataset
            n_features: Number of extra features each sample should have.
        """
        self.samples_per_class = samples_per_class
        self.n_classes = n_classes
        self.n_features = n_features

        # Create a dataframe to be consistent with other Datasets
        self.df = pd.DataFrame({
            'class_id': [i % self.n_classes for i in range(len(self))]
        })
        self.df = self.df.assign(id=self.df.index.values)

    def __len__(self):
        return self.samples_per_class * self.n_classes

    def __getitem__(self, item):
        class_id = item % self.n_classes
        return np.array([item] + [class_id]*self.n_features, dtype=np.float), float(class_id)

class KamonDataset(Dataset):
    def __init__(self, subset):
        """TBD
            """
        csv_filepath = None
        if subset == 'background':
            csv_filepath = '/data/input/configs/input_files/training_input_charge_classification_v1.csv'
        elif subset == 'evaluation':
            csv_filepath = '/data/input/configs/input_files/eval_input_charge_classification_v1.csv'
        else:
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset
        # Create a dataframe to be consistent with other Datasets
        self.df = pd.read_csv(csv_filepath, names = ['filepath', 'class_id'])
        self.df = self.df.assign(id=self.df.index.values)
    
        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((86, 86)),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label
    
class OldKamonDataset(Dataset):
    def __init__(self, subset):
        """TBD
            """
        csv_filepath = None
        if subset == 'background':
            csv_filepath = '/data/input/configs/input_files/training_input_classification_extended_v6.csv'
        elif subset == 'evaluation':
            csv_filepath = '/data/input/configs/input_files/eval_input_classification_extended_v6.csv'
        else:
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset
        # Create a dataframe to be consistent with other Datasets
        self.df = pd.read_csv(csv_filepath, names = ['filepath', 'class_id'])
        self.df = self.df.assign(id=self.df.index.values)
    
        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((86, 86)),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label
    
class LogoDataset(Dataset):
    def __init__(self, subset):
        """TBD
            """
        self.hdf5_filepath = '/data/input/logo/LLD-logo.hdf5'
        hdf5_file = h5py.File(self.hdf5_filepath, 'r')
        
        starting_index = -1
        ending_index = -1
        if subset == 'background':
            starting_index = 0
            ending_index = 20000
        elif subset == 'evaluation':
            starting_index = 20000
            ending_index = 30000
        else:
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        # Create a dataframe to be consistent with other Datasets
        index = pd.Index(np.arange(0, ending_index-starting_index))
        self.df = pd.DataFrame(index = index, columns = ['image_id', 'class_id'])
        
        # simple way to load the complete dataset
        for i in index:
            image_id = i + starting_index
            class_id = hdf5_file['labels/resnet/rc_32'][image_id]
            self.df.loc[i, 'image_id'] = image_id
            self.df.loc[i, 'class_id'] = class_id

        self.df = self.df.assign(id=self.df.index.values)
    
        # Create dicts
        self.datasetid_to_image_id = self.df.to_dict()['image_id']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((86, 86)),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        hdf5_file = h5py.File(self.hdf5_filepath, 'r')
        image_id = self.datasetid_to_image_id[item]
        image_shape = hdf5_file['shapes'][image_id]
        image = hdf5_file['data'][image_id, :, :image_shape[1], :image_shape[2]].astype(np.uint8)
        instance = Image.fromarray(image.T)
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label
