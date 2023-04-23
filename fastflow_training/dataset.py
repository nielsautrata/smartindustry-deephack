import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2 as cv

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=False, is_inference=False):
        # print(input_size)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((input_size,input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        def get_img_paths(list_paths):
            return [name for name in list_paths if name.split(".")[-1].lower() in ["jpg", "jpeg", "png"]]

        if is_train:
            self.image_files = get_img_paths(glob(
                os.path.join(root, category, "train", "good", "*.png")
            ))
        else:
            self.image_files = get_img_paths(glob(os.path.join(root, category, "test", "*", "*")))
            

        self.is_train = is_train
        self.is_inference = is_inference
        assert self.__len__() > 0, "Num data = 0, cant find images"
        print("Num_data:", self.__len__())

    def __getitem__(self, index):
        image_file = self.image_files[index]
        # raw = cv.imread(image_file)
        # raw = cv.cvtColor(raw, cv.COLOR_BGR2RGB)
        raw = Image.open(image_file).convert("RGB")
        # print(np.array(raw).shape)
        image = self.image_transform(raw)

        if self.is_train:
            return image
        
        target_path = image_file.replace("/test/", "/ground_truth/").replace(".png", "_mask.png")
        if os.path.isfile(target_path):
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(target_path)
                target = self.target_transform(target)
        else:
            target = torch.ones([1, image.shape[-2], image.shape[-1]])*255
        
        if self.is_inference:
            return image, target, np.array(raw)
        else:
            return image, target

    def __len__(self):
        return len(self.image_files)
