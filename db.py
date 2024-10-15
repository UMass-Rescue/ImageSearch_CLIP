import torch
import numpy as np

def convert_image_embeddings_to_numpy(image_embeddings):
    return image_embeddings.cpu().numpy() if torch.is_tensor(image_embeddings) else image_embeddings
