import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
import cv2
import imgaug.augmenters as iaa
from utils.perlin import rand_perlin_2d_np
import yaml
import imageio.v3 as iio


# DATASETS_PATH = "/data1/chenruitao/eyecandies/Eyecandies"

def eyecandies_classes():
    return [
        "CandyCane" ,
        "ChocolateCookie",
        "ChocolatePraline",
        "Confetto",
        "GummyBear" ,
        "HazelnutTruffle",
        "LicoriceSandwich",
        "Lollipop",
        "Marshmallow",
        "PeppermintCandy"
    ]


class Eyecandies(Dataset):

    def __init__(self, split, class_name, img_size,dataset_path):
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split,"data")
    def load_and_convert_depth(self,depth_img,info_depth):
        with open(info_depth) as f:
            data = yaml.safe_load(f)
        mind, maxd = data["normalization"]["min"], data["normalization"]["max"]

        dimg = iio.imread(depth_img)
        dimg = dimg.astype(np.float32)
        dimg = dimg / 65535.0 * (maxd - mind) + mind
        return dimg


class EyecandiesTrain(Eyecandies):
    def __init__(self, class_name, img_size,dataset_path,anomaly_source_path):
        super().__init__(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.dataset_len = 1000
        self.img_paths, self.path_depth, self.path_info_depth = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.resize_shape = img_size
 
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]
        # There is a chance of rotation between -90 and 90 degrees
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        # Path of noise dataset
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))


    def load_dataset(self):
        path_img = []
        path_depth = []
        path_info_depth = []
        for num in range(self.dataset_len):
            # for i in range(6):
            i = np.random.randint(6)
            path_img.append(os.path.join(self.img_path,str(num).zfill(3)+"_image_"+str(i)+".png"))
            path_info_depth.append(os.path.join(self.img_path,str(num).zfill(3)+"_info_depth"+".yaml"))
            path_depth.append(os.path.join(self.img_path,str(num).zfill(3)+"_depth"+".png"))
        return path_img,path_depth,path_info_depth
    


    def __len__(self):
        return self.dataset_len
    
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, depth, anomaly_source_path):
        # Random rotation with three variations
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        # Generate even numbers from 0 to 12
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)

        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        perlin_noise = np.expand_dims(perlin_noise, axis=2)


        # load noise image
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        # Randomly perform three aug
        anomaly_img_augmented = aug(image=anomaly_source_img)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        
        beta = torch.rand(1).numpy()[0] * 0.8


        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
        augmented_zzz = depth * (1 - perlin_thr) + perlin_thr * perlin_noise

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, depth, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = perlin_thr
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, augmented_zzz.astype(np.float32), msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, depth_path, depth_info_path, anomaly_source_path):
        # Generate numpy format functions for raw and noisy images

        rgb_img = iio.imread(image_path)
        rgb_img = rgb_img.astype(np.float32)
        rgb_img = rgb_img / 255.0
        rgb_img = cv2.resize(rgb_img, dsize=(self.resize_shape[1], self.resize_shape[0]))


        depth_img = self.load_and_convert_depth(depth_path,depth_info_path)
        depth_img = np.expand_dims(depth_img,axis=2)
        depth_img = np.repeat(depth_img,3,axis=2)
        depth_img = cv2.resize(depth_img, dsize=(self.resize_shape[1], self.resize_shape[0]))


        augmented_image, augmented_depth, mask, has_anomaly = self.augment_image(rgb_img,depth_img,anomaly_source_path)
        # print("augmented_image",augmented_image.shape)

        # Adjusting dimensions
        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        depth_img = np.transpose(depth_img, (2, 0, 1))

        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        augmented_depth = np.transpose(augmented_depth, (2, 0, 1))

        mask = np.transpose(mask, (2, 0, 1))
        
        return (rgb_img, augmented_image, depth_img, augmented_depth, has_anomaly,mask)

    def __getitem__(self, idx):
        # idx = torch.randint(0, len(self.image_paths), (1,)).item()
        # choose a random picture to generate noise
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        result = self.transform_image(self.img_paths[idx], self.path_depth[idx], self.path_info_depth[idx],self.anomaly_source_paths[anomaly_source_idx])

        sample = {'image': result[0], 'augmented_image': result[1],
                    'zzz': result[2], 'augmented_zzz': result[3],
                  "has_anomaly":result[4],"mask": result[5]}
        return sample


class EyecandiesTest(Eyecandies):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(split="test_public", class_name=class_name, img_size=img_size,dataset_path=dataset_path)
        self.dataset_len = 50*6
        self.img_paths, self.depth_paths, self.gt_paths, self.info_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.resize_shape = img_size

    def load_dataset(self):
        path_img = []
        path_depth = []
        path_mask = []
        path_info_depth = []
        for num in range(self.dataset_len):
            for i in range(6):
                path_img.append(os.path.join(self.img_path,str(num).zfill(2)+"_image_"+str(i)+".png"))
                path_info_depth.append(os.path.join(self.img_path,str(num).zfill(2)+"_info_depth"+".yaml"))
                path_depth.append(os.path.join(self.img_path,str(num).zfill(2)+"_depth"+".png"))
                path_mask.append(os.path.join(self.img_path,str(num).zfill(2)+"_mask"+".png"))
        return path_img, path_depth, path_mask, path_info_depth

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):

        image_path, depth_path, mask_path, depth_info_path = self.img_paths[idx], self.depth_paths[idx], self.gt_paths[idx],self.info_paths[idx]

        rgb_img = iio.imread(image_path)
        rgb_img = rgb_img.astype(np.float32)
        rgb_img = rgb_img / 255.0
        rgb_img = cv2.resize(rgb_img, dsize=(self.resize_shape[1], self.resize_shape[0]))


        depth_img = self.load_and_convert_depth(depth_path,depth_info_path)
        depth_img = np.expand_dims(depth_img,axis=2)
        depth_img = np.repeat(depth_img,3,axis=2)
        depth_img = cv2.resize(depth_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        mask_img = iio.imread(mask_path)
        mask_img = mask_img.astype(np.float32)
        mask_img = mask_img / 255.0
        mask_img = cv2.resize(mask_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        mask_img = np.expand_dims(mask_img, axis=0)

        # load mask
        if np.sum(mask_img) == 0:
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            has_anomaly = np.array([1], dtype=np.float32)

        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        depth_img = np.transpose(depth_img, (2, 0, 1))


        sample = {'image': rgb_img,'mask': mask_img ,'has_anomaly': has_anomaly,'zzz': depth_img}

        return sample


def get_data_loader(args,split, class_name, img_size,batch_size=1, num_workers=1,shuffle=False):
    if split in ['train']:
        dataset = EyecandiesTrain(class_name=class_name, img_size=img_size,dataset_path=args.dataset_path_Eyecandies,anomaly_source_path=args.anomaly_source_path)
    elif split in ['test']:
        dataset = EyecandiesTest(class_name=class_name, img_size=img_size,dataset_path=args.dataset_path_Eyecandies)
    datas_len = len(dataset)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers, 
                             drop_last=False)
    return data_loader,datas_len
