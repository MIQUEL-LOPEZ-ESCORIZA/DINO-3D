# %% [markdown]
# ## DATA DEFINITION
# 

# %%
import pandas as pd
import sys
import os
import nibabel as nib
import torch 
import numpy as np
from monai import transforms

sys.path.append(os.path.abspath("/scratch/ml9715/DINO-3D/base_modules"))
from src.models.vit import ViT

TRAIN_DATASET_CSV = '/scratch/by2026/BrainATLAS/mae/mae/data_csv/mae_UKB_ADNI_HCP_CamCAN_IXI_train_with_stat.csv'
VALIDATION_DATASET_CSV = '/scratch/by2026/BrainATLAS/mae/mae/data_csv/mae_UKB_ADNI_HCP_CamCAN_IXI_val_with_stat.csv'
MODEL_CHECKPOINT_PATH = '/scratch/ml9715/DINO-3D/checkpoints/RUN_1/checkpoint0050.pth'
FEATURES_TO_VALIDATE = ['age', 'gender']
CACHE_DIRECTORY = '/scratch/ml9715/DINO-3D/temp_cache_large_3_channel/'
BATCH_SIZE = 128






# %% [markdown]
# ## PREPROCESSING

# %%
validation_dataset_df = pd.read_csv(VALIDATION_DATASET_CSV)

validation_dataset_df = validation_dataset_df[(validation_dataset_df != -99).all(axis=1)]

len(validation_dataset_df)

# %% [markdown]
# ## LOADING ViT WITH PRE-TRAINED WEIGHTS

# %%
# Load the CSV and model checkpoint

checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu")

# Define the ViT model and load the checkpoint
vit_model = ViT(in_chans=1, img_size=(96, 96, 96), patch_size=(12, 12, 12), 
                patch_embed='conv', pos_embed='sincos', use_flash_attn=True)
vit_state_dict = {k.replace("model.", ""): v for k, v in checkpoint['student'].items() if "model." in k}
vit_model.load_state_dict(vit_state_dict, strict=False)
vit_model.eval()

# %% [markdown]
# ## EXTRACTING EMBEDDINGS AND CREATING NEW DATAFRAME
# 

# %% [markdown]
# ### DEFINING THE TRANSFORM

# %%
def loading_and_central_cropping():
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
                roi_size=(96, 96, 96),
                allow_missing_keys=True
            ),
            transforms.CastToTyped(
                keys=["image"], 
                dtype=np.float16,
            )
        ])
    return trans

# %% [markdown]
# ### CREATING DATALOADER

# %%
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from monai.data import PersistentDataset

class ImageDataset(Dataset):
    def __init__(self, cache_dir, validation_dataset_df, features_to_validate, transform_function):
        self.cache_dir = cache_dir
        self.validation_dataset_df = validation_dataset_df
        self.features_to_validate = features_to_validate
        self.load = transform_function

        # Prepare the data list, including both the image paths and metadata
        self.data_list = [
            {
                "image": row["img_path_T1_mni152"], 
                **{col: row[col] for col in features_to_validate}  # Include metadata in each dictionary
            }
            for _, row in self.validation_dataset_df.iterrows()
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

# Instantiate the dataset and dataloader
dataset = ImageDataset(
    cache_dir=CACHE_DIRECTORY,
    validation_dataset_df=validation_dataset_df,
    features_to_validate=FEATURES_TO_VALIDATE,
    transform_function=loading_and_central_cropping()
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)





# %% [markdown]
# ### CREATING FINAL DATAFRAME

# %%

# Specify the device (use GPU if available)
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
features_df = pd.DataFrame(data_list)

save_path = "/scratch/ml9715/DINO-3D/LINEAR_PROBING/DATAFRAMES/RUN_1"
os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

# Save the DataFrame as a .pth file
file_name = "features0100.pth"
torch.save(features_df, os.path.join(save_path, file_name))

print(f"DataFrame saved to {os.path.join(save_path, file_name)}")



# %% [markdown]
# ## TRAINING AND TESTING

# %%
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

# Extract features, age, and gender
features_df = features_df.dropna(subset=['age'])
X = np.array(features_df['features'].tolist())  # Convert list of features to numpy array
y_age = features_df['age'].values
y_gender = features_df['gender'].values

# Split the data into training and testing sets
X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
    X, y_age, y_gender, test_size=0.2, random_state=42
)

# Pipeline for age prediction (Linear Regression)
age_pipeline = Pipeline([
    ('scaler', StandardScaler()),      # Standardize the input features
    ('regressor', LinearRegression())  # Linear regression for age prediction
])

# Train the pipeline on the standardized features and age target
age_pipeline.fit(X_train, y_age_train)

# Predict age and evaluate with R^2
y_age_pred = age_pipeline.predict(X_test)
age_r2 = r2_score(y_age_test, y_age_pred)
print("R^2 Score for Age Prediction:", age_r2)

# Pipeline for gender prediction (Logistic Regression)
gender_pipeline = Pipeline([
    ('scaler', StandardScaler()),          # Standardize the input features
    ('classifier', LogisticRegression(max_iter=1000))  # Logistic regression for gender classification
])

# Train the pipeline on the standardized features and gender target
gender_pipeline.fit(X_train, y_gender_train)

# Predict gender and evaluate accuracy
y_gender_pred = gender_pipeline.predict(X_test)
gender_accuracy = accuracy_score(y_gender_test, y_gender_pred)
print("Accuracy for Gender Prediction:", gender_accuracy)


# %%



