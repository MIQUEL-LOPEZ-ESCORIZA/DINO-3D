from monai import transforms
from monai.data import MetaTensor
import numpy as np
import torch
import nibabel as nib
from math import floor
class DataAugmentationDINO3D(object):
    def __init__(self, final_size, global_crops_size, local_crops_size, local_crops_number):
        # Define transforms for flipping and random affine transformations
        # flip_and_noise = transforms.Compose([
        #     transforms.RandFlip(prob=0.5, spatial_axis=[0, 1, 2]),  # Random flip across different axes
        #     transforms.RandBiasField(prob=0.8),
        # ])
        flip_and_noise = transforms.Compose([transforms.RandFlip(prob=0.2, spatial_axis=0), 
                                    transforms.RandFlip(prob=0.2, spatial_axis=1), 
                                    transforms.RandFlip(prob=0.2, spatial_axis=2), 
                                    transforms.RandShiftIntensity(offsets=0.2, prob=0.5)
                                    ])

        # Normalization transform - adjust mean and std as per your dataset
        # normalize = transforms.Compose([transforms.ToTensor(), 
        #                                 transforms.NormalizeIntensity()])
        normalize = transforms.ToTensor()
        max_global_crops_size_roi = floor(global_crops_size[0]*1.2)
        max_global_crops_size = (max_global_crops_size_roi, max_global_crops_size_roi, max_global_crops_size_roi)


        # Global crop transforms
        self.global_transfo1 = transforms.Compose([
            transforms.CastToType(dtype=np.float32),
            transforms.ResizeWithPadOrCrop(spatial_size=(234,234,234)), #(250,250,250)
            transforms.RandSpatialCrop(global_crops_size, random_center=True, random_size=False),
            transforms.Resize(spatial_size=final_size),
            flip_and_noise, 
            transforms.RandGaussianSmooth(sigma_x=(0.5,1.0), sigma_y=(0.5,1.0), sigma_z=(0.5,1.0), prob=0.2), 
            normalize, 
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.CastToType(dtype=np.float32),
            transforms.ResizeWithPadOrCrop(spatial_size=(234,234,234)),
            transforms.RandSpatialCrop(global_crops_size, random_center=True, random_size=False),
            transforms.Resize(spatial_size=final_size),
            flip_and_noise, 
            transforms.RandAdjustContrast(gamma=(0.2,1.),prob=0.2),
            normalize,
        ])

        # Local crop transform
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.CastToType(dtype=np.float32),
            transforms.ResizeWithPadOrCrop(spatial_size=(234, 234, 234)),
            transforms.CenterSpatialCrop((192, 192, 192)), 
            transforms.RandSpatialCrop(local_crops_size, max_roi_size=global_crops_size, random_center=True, random_size=True),
            transforms.Resize(spatial_size=final_size),
            # transforms.RandScaleCrop(roi_scale=local_crops_scale[0], max_roi_scale=local_crops_scale[1], random_size=True),
            # transforms.Resize(spatial_size=local_crops_size),
            # flip_and_noise,
            # transforms.RandGaussianSmooth(sigma_x=(0.05,0.1), sigma_y=(0.05,0.1), sigma_z=(0.05,0.1), prob=0.2), 
            normalize
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class SimpleAugmenation3D(object):
    def __init__(self, final_size, global_crops_size, local_crops_size, local_crops_number):
        # Define transforms for flipping and random affine transformations
        flip_and_noise = transforms.Compose([transforms.RandFlip(prob=0.2, spatial_axis=0), 
                                    transforms.RandFlip(prob=0.2, spatial_axis=1), 
                                    transforms.RandFlip(prob=0.2, spatial_axis=2), 
                                    transforms.RandShiftIntensity(offsets=0.2, prob=0.5)
                                    ])
        normalize = transforms.ToTensor()

        # Global crop transforms
        self.global_transfo1 = transforms.Compose([
            transforms.CastToType(dtype=np.float32),
            transforms.ResizeWithPadOrCrop(spatial_size=(224, 224, 224)),
            transforms.RandSpatialCrop(global_crops_size, random_center=True, random_size=True),
            transforms.Resize(spatial_size=final_size),
            # flip_and_noise, 
            # transforms.RandGaussianSmooth(sigma_x=(0.5,1.0), sigma_y=(0.5,1.0), sigma_z=(0.5,1.0), prob=0.2), 
            normalize, 
        ])
        # Local crop transform
        self.local_crops_number = local_crops_number
    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        return crops
    


class MultipleWindowScaleStack(transforms.MapTransform):
    def __init__(
        self,
        keys,
        window_sizes,
    ) -> None:
        transforms.MapTransform.__init__(self, keys)
        self.keys = keys
        self.window_sizes = window_sizes
        self.scale_transforms = [
            transforms.ScaleIntensityRange(
                a_min=l - w//2,
                a_max=l + w//2,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ) for l, w in window_sizes]
        
    def __call__(self, data):
        d = dict(data)
        image = torch.cat([t(d["image"]) for t in self.scale_transforms], dim=0)
        d['image'] = np.array(image)
        return d

class LoadMGZInMemory:
    def __init__(self, allow_missing_keys=True):
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data):
       
        print(data['image'])


        return data



def get_loading_transform():
    window_sizes = [(40, 80), (80, 200), (600, 2800)]
    print('mal')
    trans = transforms.Compose(
        [
            LoadMGZInMemory(),
            transforms.EnsureChannelFirstd(
                keys=["image"],
                allow_missing_keys=True,
            ),
            transforms.Orientationd(
                keys=["image"],
                axcodes="RAS",
                allow_missing_keys=True,
            ),
            transforms.Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode=3,
                allow_missing_keys=True
            ),
            transforms.CropForegroundd(
                keys=["image"],
                source_key="image",
                allow_smaller=False,
                allow_missing_keys=True,
            ),
            MultipleWindowScaleStack(
                keys=["image"], 
                window_sizes=window_sizes,
            ),
             transforms.CastToTyped(
                keys=["image"], 
                dtype=np.float16,
            )
        ])
    return trans


def get_loading_transform_f16():
    window_sizes = [(40, 80), (80, 200), (600, 2800)]
    trans = transforms.Compose(
        [
            transforms.LoadImaged(
                keys=["image"],
                reader="NibabelReader", 
                image_only=True,
                allow_missing_keys=True,
            ),
            transforms.EnsureChannelFirstd(
                keys=["image"],
                allow_missing_keys=True,
            ),
            transforms.Orientationd(
                keys=["image"],
                axcodes="RAS",
                allow_missing_keys=True,
            ),
            transforms.Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode=3,
                allow_missing_keys=True
            ),
            transforms.CropForegroundd(
                keys=["image"],
                source_key="image",
                allow_smaller=False,
                allow_missing_keys=True,
            ),
            # MultipleWindowScaleStack(
            #     keys=["image"], 
            #     window_sizes=window_sizes,
            # ),
            transforms.CastToTyped(
                keys=["image"], 
                dtype=np.float16,
            )
        ])
    return trans

# Jack's augmentations

# # Global crop transforms
#         self.global_transfo1 = transforms.Compose([
#             transforms.CastToType(dtype=np.float32),
#             transforms.ResizeWithPadOrCrop(spatial_size=(224, 224, 224)),
#             transforms.RandSpatialCrop(global_crops_size, random_center=True, random_size=True),
#             transforms.Resize(spatial_size=final_size),
#             flip_and_noise, 
#             transforms.RandGaussianSmooth(sigma_x=(0.5,1.0), sigma_y=(0.5,1.0), sigma_z=(0.5,1.0), prob=0.2), 
#             normalize, 
#         ])
#         self.global_transfo2 = transforms.Compose([
#             transforms.CastToType(dtype=np.float32),
#             transforms.ResizeWithPadOrCrop(spatial_size=(224, 224, 224)),
#             transforms.RandSpatialCrop(global_crops_size, random_center=True, random_size=True),
#             transforms.Resize(spatial_size=final_size),
#             flip_and_noise, 
#             transforms.RandAdjustContrast(gamma=(0.2,1.),prob=0.2),
#             normalize,
#         ])

#         # Local crop transform
#         self.local_crops_number = local_crops_number
#         self.local_transfo = transforms.Compose([
#             transforms.CastToType(dtype=np.float32),
#             transforms.ResizeWithPadOrCrop(spatial_size=(224, 224, 224)),
#             transforms.CenterSpatialCrop((192, 192, 192)), 
#             transforms.RandSpatialCrop(local_crops_size, max_roi_size=global_crops_size, random_center=True, random_size=True),
#             transforms.Resize(spatial_size=final_size),
#             # transforms.RandScaleCrop(roi_scale=local_crops_scale[0], max_roi_scale=local_crops_scale[1], random_size=True),
#             # transforms.Resize(spatial_size=local_crops_size),
#             # flip_and_noise,
#             # transforms.RandGaussianSmooth(sigma_x=(0.05,0.1), sigma_y=(0.05,0.1), sigma_z=(0.05,0.1), prob=0.2), 
#             normalize
#         ])

class MY_AUGMENTATION(object):
    def __init__(self, final_size, global_crops_size, local_crops_size, local_crops_number):
        # Define transforms for flipping and random affine transformations
        flip_and_noise = transforms.Compose([transforms.RandFlip(prob=0.2, spatial_axis=0), 
                                    transforms.RandFlip(prob=0.2, spatial_axis=1), 
                                    transforms.RandFlip(prob=0.2, spatial_axis=2), 
                                    transforms.RandShiftIntensity(offsets=0.2, prob=0.5)
                                    ])

        # Normalization transform - adjust mean and std as per your dataset
        normalize = transforms.Compose([transforms.ToTensor(), 
                                         transforms.NormalizeIntensity()])
        def threshold(x):
            # threshold at 1
            return x > 100

        



        # Global crop transforms
        self.global_transfo1 = transforms.Compose([
            transforms.CastToType(dtype=np.float32),
            transforms.CropForeground(select_fn=threshold, margin=0), #(250,250,250)
            transforms.RandSpatialCrop(global_crops_size, random_center=True, random_size=False),
            transforms.Resize(spatial_size=final_size),
            flip_and_noise, 
            transforms.RandGaussianSmooth(sigma_x=(0.5,1.0), sigma_y=(0.5,1.0), sigma_z=(0.5,1.0), prob=0.2), 
            normalize, 
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.CastToType(dtype=np.float32),
            transforms.CropForeground(select_fn=threshold, margin=0),
            transforms.RandSpatialCrop(global_crops_size, random_center=True, random_size=False),
            transforms.Resize(spatial_size=final_size),
            flip_and_noise, 
            transforms.RandAdjustContrast(gamma=(0.2,1.),prob=0.2),
            normalize,
        ])

        # Local crop transform
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.CastToType(dtype=np.float32),
            transforms.CropForeground(select_fn=threshold, margin=0),
            transforms.RandSpatialCrop(local_crops_size, random_center=True, random_size=False),
            transforms.Resize(spatial_size=final_size),
            flip_and_noise,
            transforms.RandGaussianSmooth(sigma_x=(0.05,0.1), sigma_y=(0.05,0.1), sigma_z=(0.05,0.1), prob=0.2), 
            normalize
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops