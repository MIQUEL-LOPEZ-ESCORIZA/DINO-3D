import pandas as pd
import sys
import os
import nibabel as nib
import torch 
import numpy as np
from monai import transforms

from torch.utils.data import DataLoader, Dataset
import pandas as pd
from monai.data import PersistentDataset

sys.path.append(os.path.abspath("/scratch/ml9715/DINO-3D/base_modules"))
from src.models.vit import ViT


class ImageDataset(Dataset):
    def __init__(self, cache_dir, dataset_df, features_to_validate, transform_function):
        self.cache_dir = cache_dir
        self.dataset_df = dataset_df
        self.features_to_validate = features_to_validate
        self.load = transform_function

        # Prepare the data list, including both the image paths and metadata
        self.data_list = [
            {
                "image": row["img_path_T1_mni152"], 
                **{col: row[col] for col in features_to_validate}  # Include metadata in each dictionary
            }
            for _, row in self.dataset_df.iterrows()
        ]

        # Create PersistentDataset to cache images and metadata
        self.cache_dataset = PersistentDataset(
            data=self.data_list,
            transform=self.load, 
            cache_dir=self.cache_dir,
        )

    def __len__(self):
        return len(self.cache_dataset)

    def __getitem__(self, idx):
        # Load image and metadata from cache
        data = self.cache_dataset[idx]
        
        # Extract the image tensor
        img_tensor = data["image"].float()  # Ensure it's in float32
        
        # Collect metadata
        metadata = {col: data[col] for col in self.features_to_validate}
        
        return img_tensor, metadata

def assert_path(TRAIN_CSV_PATH, CHECKPOINT, OUTPUT_DIR):
    # Check if TRAIN_CSV_PATH exists and is a valid file
    assert os.path.isfile(TRAIN_CSV_PATH), f"TRAIN_CSV_PATH '{TRAIN_CSV_PATH}' does not exist or is not a file."
    assert TRAIN_CSV_PATH.endswith('.csv'), f"TRAIN_CSV_PATH '{TRAIN_CSV_PATH}' must be a CSV file."

    # Check if CHECKPOINT exists and is a valid file
    assert os.path.isfile(CHECKPOINT), f"CHECKPOINT '{CHECKPOINT}' does not exist or is not a file."
    assert CHECKPOINT.endswith('.pth') or CHECKPOINT.endswith('.pt'), \
        f"CHECKPOINT '{CHECKPOINT}' must be a PyTorch checkpoint file (.pth or .pt)."

    # Check if OUTPUT_DIR exists and is a valid directory; if not, create it
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create the directory if it doesn't exist
    assert os.path.isdir(OUTPUT_DIR), f"OUTPUT_DIR '{OUTPUT_DIR}' is not a valid directory."

def assert_vit_params(VIT_PARAMS):
    required_params = {
        "in_chans": int,
        "img_size": tuple,
        "patch_size": tuple,
        "patch_embed": str,
        "pos_embed": str,
        "use_flash_attn": bool
    }
    
    # Check that VIT_PARAMS contains all required keys and types
    for key, expected_type in required_params.items():
        assert key in VIT_PARAMS, f"Missing required parameter '{key}' in VIT_PARAMS."
        assert isinstance(VIT_PARAMS[key], expected_type), \
            f"Parameter '{key}' should be of type {expected_type.__name__}, but got {type(VIT_PARAMS[key]).__name__}."
    
    # Additional checks for specific parameter values (e.g., dimensions of img_size and patch_size)
    assert len(VIT_PARAMS["img_size"]) == 3, "Parameter 'img_size' should be a tuple with three elements (e.g., (96, 96, 96))."
    assert len(VIT_PARAMS["patch_size"]) == 3, "Parameter 'patch_size' should be a tuple with three elements (e.g., (12, 12, 12))."

    # Instantiate the ViT model using VIT_PARAMS

def load_vit_model(CHECKPOINT, VIT_PARAMS):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    vit_model = ViT(in_chans=1, img_size=(128, 128, 128), patch_size=(16, 16, 16), 
                    patch_embed='conv', pos_embed='sincos', use_flash_attn=True)
    vit_state_dict = {k.replace("model.", ""): v for k, v in checkpoint['student'].items() if "model." in k}
    vit_model.load_state_dict(vit_state_dict, strict=False)
    vit_model.eval()

    return vit_model


def preprocess_dataframe(CSV_PATH):
    df = pd.read_csv(CSV_PATH)

    df = df[(df != -99).all(axis=1)]

    return df


def loading_and_central_cropping(img_size):
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
                
                transforms.CenterSpatialCropd(
                    keys=["image"],
                    roi_size=img_size,
                    allow_missing_keys=True
                ),
                transforms.CastToTyped(
                    keys=["image"], 
                    dtype=np.float16,
                )
            ])
        return trans

def create_dataframe(vit_model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
    vit_model = vit_model.to(device)

    # Prepare a list to collect data for the new DataFrame
    data_list = []

    # Iterate through the DataLoader in batches
    for img_tensors, metadata_batch in dataloader:
        # Move image tensors to the appropriate device (GPU)
        img_tensors = img_tensors.to(device).float()  # Ensure float32

        # Obtain features from the ViT model in batch
        with torch.no_grad():
            features_batch = vit_model(img_tensors)  # Get features for the batch

        # Move features back to CPU for further processing
        features_batch = features_batch.cpu()

        # Process each feature and corresponding metadata
        for i in range(len(features_batch)):
            features = features_batch[i].squeeze().numpy()
            
            # Create a dictionary for each image's metadata by extracting each key's ith element
            metadata = {key: metadata_batch[key][i].item() if torch.is_tensor(metadata_batch[key][i]) else metadata_batch[key][i]
                        for key in metadata_batch}

            # Collect the features and selected columns
            data_entry = {'features': features}
            data_entry.update(metadata)
            
            data_list.append(data_entry)

    # Create a new DataFrame with features and selected columns
    return pd.DataFrame(data_list)


def compute_features(TRAIN_CSV_PATH, CHECKPOINT, VIT_PARAMS, EPOCH, OUTPUT_DIR, CACHE_DIRECTORY, BATCH_SIZE, FEATURES_TO_VALIDATE):

    assert_vit_params(VIT_PARAMS)
    assert_path(TRAIN_CSV_PATH, CHECKPOINT, OUTPUT_DIR)

    vit_model = load_vit_model(CHECKPOINT, VIT_PARAMS)

    train_df = preprocess_dataframe(TRAIN_CSV_PATH).head(10000)

    # Instantiate the dataset and dataloader
    dataset = ImageDataset(
        cache_dir=CACHE_DIRECTORY,
        dataset_df=train_df,
        features_to_validate=FEATURES_TO_VALIDATE,
        transform_function=loading_and_central_cropping(VIT_PARAMS["img_size"])
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    features = create_dataframe(vit_model, dataloader)

     # Construct the filename
    filename = f"features_{EPOCH}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Save the DataFrame as a CSV
    features.to_csv(filepath, index=False)
    print(f"DataFrame saved to: {filepath}")




  

        
        

