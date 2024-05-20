import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PretrainedConfig, CLIPTokenizer, CLIPImageProcessor
from PIL import Image
import json
import os
import numpy as np

# CompoDiff classes (as provided)
class CompoDiffConfig(PretrainedConfig):
    model_type = "CompoDiff"

    def __init__(self, embed_dim: int = 768, model_depth: int = 12, model_dim: int = 64, model_heads: int = 16, timesteps: int = 1000, **kwargs):
        self.embed_dim = embed_dim
        self.model_depth = model_depth
        self.model_dim = model_dim
        self.model_heads = model_heads
        self.timesteps = timesteps
        super().__init__(**kwargs)

class CompoDiffModel(PreTrainedModel):
    config_class = CompoDiffConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = build_compodiff(config.embed_dim, config.model_depth, config.model_dim, config.model_heads, config.timesteps)

    def _init_weights(self, module):
        pass

    def sample(self, image_cond, text_cond, negative_text_cond, input_mask, num_samples_per_batch=4, cond_scale=1., timesteps=None, random_seed=None):
        return self.model.sample(image_cond, text_cond, negative_text_cond, input_mask, num_samples_per_batch, cond_scale, timesteps, random_seed)

# Data Loading and Preprocessing
# class CompoDiffDataset(Dataset):
    # def __init__(self, metadata_path, prompt_path, image_dir):
    #     self.image_dir = image_dir
    #     with open(metadata_path, 'r') as f:
    #         self.metadata = [json.loads(line) for line in f]
    #     with open(prompt_path, 'r') as f:
    #         self.prompts = {json.loads(line) for line in f}
        
    # def __len__(self):
    #     return len(self.metadata)
    
    # def __getitem__(self, idx):
    #     data = self.metadata[idx]
    #     prompt = self.prompts[idx]
        
    #     seed = data['seed']
    #     input_image_filename = f"{seed}_1.jpg"
    #     reference_image_filename = f"{seed}_0.jpg"
        
    #     input_image_path = os.path.join(self.image_dir, input_image_filename)
    #     reference_image_path = os.path.join(self.image_dir, reference_image_filename)
        
    #     input_image = Image.open(input_image_path).convert('RGB')
    #     reference_image = Image.open(reference_image_path).convert('RGB')
        
    #     text_condition = prompt['output']  # Use the 'edit' field for the text condition
    #     mask_condition = None  # Modify as necessary if mask condition is available
        
    #     return input_image, reference_image, text_condition, mask_condition

# Data Loading and Preprocessing
# class CompoDiffDataset(Dataset):
#     def __init__(self, seeds_path, base_dir):
#         self.base_dir = base_dir
#         with open(seeds_path, 'r') as f:
#             self.seeds = json.load(f)
        
#         self.data = []
#         for folder, files in self.seeds.items():
#             prompt_path = os.path.join(base_dir, folder, 'prompt.json')
            
#             with open(prompt_path, 'r') as f:
#                 prompt_data = json.load(f)
#                 for file_pair in files:
#                     self.data.append({
#                         'folder': folder,
#                         'input_image': file_pair + "_1.jpg",
#                         'reference_image': file_pair + "_0.jpg",
#                         'text_condition': prompt_data['output']
#                     })
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         item = self.data[idx]
        
#         input_image_path = os.path.join(self.base_dir, item['folder'], item['input_image'])
#         reference_image_path = os.path.join(self.base_dir, item['folder'], item['reference_image'])
        
#         input_image = Image.open(input_image_path).convert('RGB')
#         reference_image = Image.open(reference_image_path).convert('RGB')
        
#         text_condition = item['text_condition']
#         mask_condition = None
        
#         return input_image, reference_image, text_condition, mask_condition

# Data Loading and Preprocessing
class CompoDiffDataset(Dataset):
    def __init__(self, seeds_path, base_dir):
        self.base_dir = base_dir
        with open(seeds_path, 'r') as f:
            self.seeds = json.load(f)
        
        self.data = []
        for entry in self.seeds:
            folder = entry[0]
            files = entry[1]
            # Construct the path to the prompt.json file within the current folder
            prompt_path = os.path.join(base_dir, folder, 'prompt.json')
            
            with open(prompt_path, 'r') as f:
                prompt_data = json.load(f)
                # For each file pair (seed) in the current folder, add the relevant data to the dataset
                for file_pair in files:
                    self.data.append({
                        'folder': folder,  # The current folder name
                        'input_image': file_pair + "_1.jpg",  # The input image file name
                        'reference_image': file_pair + "_0.jpg",  # The reference image file name
                        'text_condition': prompt_data['output']  # The text condition from prompt.json
                    })
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Construct the full paths to the input and reference images
        input_image_path = os.path.join(self.base_dir, item['folder'], item['input_image'])
        reference_image_path = os.path.join(self.base_dir, item['folder'], item['reference_image'])
        
        # Load the images
        input_image = Image.open(input_image_path).convert('RGB')
        reference_image = Image.open(reference_image_path).convert('RGB')
        
        # Retrieve the text condition
        text_condition = item['text_condition']
        mask_condition = None  # No mask condition in this implementation
        
        return input_image, reference_image, text_condition, mask_condition

# Define build_compodiff and build_clip stubs (implement these based on your actual model definitions)
def build_compodiff(embed_dim, model_depth, model_dim, model_heads, timesteps):
    # Return a dummy model for now
    return nn.Module()

def build_clip():
    # Return a dummy model for now
    return nn.Module()

# Train Stage 1
def train_stage1(model, data_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for input_image, reference_image, text_condition, _ in data_loader:
            optimizer.zero_grad()
            output = model(input_image, text_condition)
            loss = criterion(output, reference_image)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Train Stage 2
# def train_stage2(model, data_loader, optimizer, criterion, num_epochs=10):
#     model.train()
#     for epoch in range(num_epochs):
#         for input_image, reference_image, text_condition, mask_condition in data_loader:
#             optimizer.zero_grad()
#             # Choose task randomly
#             task = np.random.choice(['conversion', 'mask_conversion', 'triplet'], p=[0.3, 0.3, 0.4])
            
#             if task == 'conversion':
#                 output = model(input_image, text_condition)
#                 loss = criterion(output, reference_image)
            
#             elif task == 'mask_conversion' and mask_condition is not None:
#                 output = model(input_image, text_condition, mask_condition=mask_condition)
#                 loss = criterion(output, reference_image)
            
#             elif task == 'triplet':
#                 output = model(input_image, text_condition, reference_image=reference_image, mask_condition=mask_condition)
#                 loss = criterion(output, reference_image)
            
#             loss.backward()
#             optimizer.step()
#             print(f"Epoch [{epoch+1}/{num_epochs}], Task: {task}, Loss: {loss.item()}")


# Train Stage 2 without Mask Condition
def train_stage2(model, data_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for input_image, reference_image, text_condition, mask_condition in data_loader:
            optimizer.zero_grad()
            # Choose task randomly
            task = np.random.choice(['conversion', 'triplet'], p=[0.5, 0.5])
            
            if task == 'conversion':
                output = model(input_image, text_condition)
                loss = criterion(output, reference_image)
            
            elif task == 'triplet':
                output = model(input_image, text_condition, reference_image=reference_image)
                loss = criterion(output, reference_image)
            
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Task: {task}, Loss: {loss.item()}")

# Function to save the model
def save_model(model, config, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    model.save_pretrained(save_directory)
    config.save_pretrained(save_directory)
    print(f"Model and configuration saved to {save_directory}")


# Main execution
if __name__ == '__main__':
    compodiff_config = CompoDiffConfig()
    compodiff = CompoDiffModel(compodiff_config)
    
    # Assume the model checkpoint is already saved in '/data/data_zoo/logs/stage2_arch.depth12-heads16_lr1e-4_text-bigG_add-art-datasets/checkpoints/model_000710000.pt'
    # compodiff.model.load_state_dict(torch.load('/data/data_zoo/logs/stage2_arch.depth12-heads16_lr1e-4_text-bigG_add-art-datasets/checkpoints/model_000710000.pt')['ema_model'])
    
    # # Save the pre-trained model
    # compodiff_config.save_pretrained('/data/CompoDiff_HF')
    # compodiff.save_pretrained('/data/CompoDiff_HF')
    
    # Dataset and DataLoader
    # dataset = CompoDiffDataset(metadata_path='D:/instruct-pix2pix/data/dataset/metadata.jsonl', prompt_path='D:/instruct-pix2pix/data/dataset/prompt.json', image_dir='D:/instruct-pix2pix/data/dataset')
    dataset = CompoDiffDataset(seeds_path='D:\instruct-pix2pix\data\dataset\seeds.json', base_dir='D:\instruct-pix2pix\data\dataset')
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    
    # Optimizer and Criterion
    optimizer = optim.Adam(compodiff.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Train Stage 1
    train_stage1(compodiff, data_loader, optimizer, criterion, num_epochs=10)

    # Save the model after Stage 1
    save_model(compodiff, compodiff_config, '/data/CompoDiff_HF_stage1')
    
    # Train Stage 2
    train_stage2(compodiff, data_loader, optimizer, criterion, num_epochs=10)

    # Save the model after Stage 2
    save_model(compodiff, compodiff_config, '/data/CompoDiff_HF_stage2')
