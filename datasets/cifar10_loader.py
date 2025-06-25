import torch
from torchvision import datasets, transforms

class CIFAR10OneClass(torch.utils.data.Dataset):
    def __init__(self, root, normal_class=0, train=True):
        self.normal_class = normal_class
        self.train = train
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # or (64, 64)
            transforms.ToTensor()
        ])
        
        full_dataset = datasets.CIFAR10(
            root=root,
            train=self.train,
            download=True,
            transform=self.transform
        )

        if self.train:
            self.indices = [i for i, (_, label) in enumerate(full_dataset) if label == self.normal_class]
        else:
            self.indices = list(range(len(full_dataset)))

        self.dataset = full_dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]
        if self.train:
            return img, 0  # All normal
        else:
            return img, 0 if label == self.normal_class else 1
