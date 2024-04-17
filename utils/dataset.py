import json
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from utils.utils import get_args


def get_dataloader(dataset_dir: str, batch_size: int = 1, split: str = 'test'):
    args = get_args()
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((args.size, args.size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20, expand=False),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    dataset = BUSI(dataset_dir, split=split, transform=transform)
    if dataset[0] is None:
        raise NotImplementedError('No data found!')

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=(split == 'train'),
                            num_workers=0,
                            pin_memory=True,
                            drop_last=(split == 'train'))
    return dataloader


class BUSI(Dataset):

    def __init__(self,
                 dataset_dir: str,
                 split: str = 'test',
                 transform=None) -> None:
        super(BUSI, self).__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        with open(os.path.join(self.dataset_dir, 'annotations.json'),
                  'r') as f:
            json_data = json.load(f)

        self.image_names = json_data['filenames']
        if self.split != 'test':
            self.labels = json_data['labels']

        print(f'Number of {self.split} images is {len(self.image_names)}')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = Image.open(
            os.path.join(self.dataset_dir,
                         self.image_names[index])).convert('RGB')
        image_transformed = self.transform(image)

        if self.split == 'test':
            return {'images': image_transformed}
        else:
            return {
                'images': image_transformed,
                'labels': int(self.labels[index])
            }
