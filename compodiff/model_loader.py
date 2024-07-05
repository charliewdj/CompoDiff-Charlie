import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PretrainedConfig, CLIPTokenizer, CLIPImageProcessor
from PIL import Image
from torchvision import transforms
import json
import os
import numpy as np

# Import the model building functions
try:
    from .models import build_compodiff, build_clip
except:
    from models import build_compodiff, build_clip

# CompoDiff classes (as provided)
class CompoDiffConfig(PretrainedConfig):
    model_type = "CompoDiff"

    def __init__(self, embed_dim: int = 32, model_depth: int = 12, model_dim: int = 64, model_heads: int = 16, timesteps: int = 1000, **kwargs):
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
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    def _init_weights(self, module):
        pass

    def sample(self, image_cond, text_cond, negative_text_cond, input_mask, num_samples_per_batch=4, cond_scale=1., timesteps=None, random_seed=None):
        return self.model.sample(image_cond, text_cond, negative_text_cond, input_mask, num_samples_per_batch, cond_scale, timesteps, random_seed)

    def forward(self, input_image_embed, target_image_embed, image_cond, text_cond, input_mask, text_cond_uc=None):
        print("Compodiff Model Forward start")
        return self.model(input_image_embed, target_image_embed, image_cond, text_cond, input_mask, text_cond_uc)

# Data Loading and Preprocessing
class CompoDiffDataset(Dataset):
    def __init__(self, seeds_path, base_dir):
        self.base_dir = base_dir
        with open(seeds_path, 'r') as f:
            self.seeds = json.load(f)
        
        self.data = []
        prompt_path = os.path.join(base_dir, '0028217', 'prompt.json')
        with open(prompt_path, 'r') as f:
            prompt_data = json.load(f) 
            self.data.append({'text_condition' : prompt_data['output']})

        # Define the transform to convert images to tensors
        self.transform = transforms.ToTensor()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_image_path = os.path.join(self.base_dir, '0028217', '1805523698_1.jpg')
        reference_image_path = os.path.join(self.base_dir, '0028217', '1805523698_0.jpg')
        
        input_image = Image.open(input_image_path).convert('RGB')
        reference_image = Image.open(reference_image_path).convert('RGB')

        # Apply the transform to convert images to tensors
        input_image = self.transform(input_image)
        reference_image = self.transform(reference_image)

        text_condition = item['text_condition']
        
        
        
        return input_image, reference_image, text_condition

# Train Stage 1
def train_stage1(model, data_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for input_image, reference_image, text_condition in data_loader:
            optimizer.zero_grad()
            # Prepare the inputs for the forward pass
            image_cond = None  # Replace with actual image condition if available
            input_mask = None  # Replace with actual input mask if available
            # Encode the text condition
            text_condition_encoded = model.tokenizer(text_condition, return_tensors="pt", padding=True, truncation=True).input_ids.to(input_image.device)
            print(f"Encoded text condition shape: {text_condition_encoded.shape}")  # Add this line to print shape
            output = model(input_image, reference_image, image_cond, text_condition_encoded, input_mask)
            loss = criterion(output, reference_image)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Train Stage 2 without Mask Condition
def train_stage2(model, data_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for input_image, reference_image, text_condition in data_loader:
            optimizer.zero_grad()
            # Prepare the inputs for the forward pass
            image_cond = None  # Replace with actual image condition if available
            input_mask = None  # Replace with actual input mask if available
            # Encode the text condition
            text_condition_encoded = model.tokenizer(text_condition, return_tensors="pt", padding=True, truncation=True).input_ids.to(input_image.device)
            # Choose task randomly
            task = np.random.choice(['conversion', 'triplet'], p=[0.5, 0.5])
            
            if task == 'conversion':
                output = model(input_image, reference_image, image_cond, text_condition_encoded, input_mask)
                loss = criterion(output, reference_image)
            
            elif task == 'triplet':
                output = model(input_image, reference_image, image_cond, text_condition_encoded, input_mask, text_cond_uc=text_condition_encoded)
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
    
    # Dataset and DataLoader
    dataset = CompoDiffDataset(seeds_path='/mnt/hdd2/datasets/clip-filtered-dataset/data/clip-filtered-dataset/seeds.json', base_dir='/mnt/hdd2/datasets/clip-filtered-dataset/data/clip-filtered-dataset')
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
