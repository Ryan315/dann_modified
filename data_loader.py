import torch.utils.data as data
from PIL import Image
import os


class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        # self.img_paths = []
        # self.img_labels = []
        #
        # with open(data_list, 'r') as f:
        #     while True:
        #         line = f.readline()
        #         if not line:
        #             break
        #             pass
        #         temp_line = line.strip().split(" ")
        #         self.img_paths.append(temp_line[0])
        #         self.img_labels.append(temp_line[1])
        #         pass
        #     pass
        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)
        #
        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-4])
            self.img_labels.append(data[-3:])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data
