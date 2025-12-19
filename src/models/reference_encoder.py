"""
Reference image encoder using DINOv2 or CLIP for feature extraction
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union
from PIL import Image
import torchvision.transforms as transforms


class ReferenceEncoder:
    """
    Encode reference images using pretrained vision models
    """
    
    def __init__(
        self,
        model_name: str = "dinov2",
        device: Optional[str] = None,
        aggregation_method: str = "mean"
    ):
        """
        Initialize reference encoder
        
        Args:
            model_name: Model to use ('dinov2', 'clip')
            device: Device to use (cuda/cpu)
            aggregation_method: How to aggregate multiple reference embeddings ('mean', 'max', 'concat')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.aggregation_method = aggregation_method
        
        self.model = None
        self.processor = None
        self.feature_dim = None
        
        self._load_model()
    
    def _load_model(self):
        """Load pretrained model"""
        if self.model_name == "dinov2":
            self._load_dinov2()
        elif self.model_name == "clip":
            self._load_clip()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _load_dinov2(self):
        """Load DINOv2 model"""
        try:
            # Try loading from torch hub
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.feature_dim = 1024  # DINOv2 ViT-L/14 output dimension
        except:
            # Fallback to transformers library
            from transformers import AutoModel, AutoImageProcessor
            
            self.model = AutoModel.from_pretrained('facebook/dinov2-large')
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
            self.feature_dim = 1024
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_clip(self):
        """Load CLIP model"""
        from transformers import CLIPModel, CLIPProcessor
        
        self.model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
        self.feature_dim = 768  # CLIP ViT-L/14 vision output dimension
        
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def encode_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Encode single image
        
        Args:
            image: Input image (RGB numpy array or PIL Image)
            
        Returns:
            Feature vector (numpy array)
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        if self.model_name == "dinov2":
            if self.processor is not None:
                # Use processor
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]  # CLS token
            else:
                # Use manual preprocessing
                img_tensor = self.transform(image).unsqueeze(0).to(self.device)
                features = self.model(img_tensor)
                
        elif self.model_name == "clip":
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_image_features(**inputs)
        
        # Convert to numpy
        features = features.cpu().numpy().squeeze()
        
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    @torch.no_grad()
    def encode_images(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        Encode multiple images
        
        Args:
            images: List of images
            
        Returns:
            Feature vectors (numpy array of shape [N, feature_dim])
        """
        features = []
        
        for image in images:
            feat = self.encode_image(image)
            features.append(feat)
        
        return np.array(features)
    
    def aggregate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Aggregate multiple embeddings into single representation
        
        Args:
            embeddings: Array of shape [N, feature_dim]
            
        Returns:
            Aggregated embedding of shape [feature_dim]
        """
        if self.aggregation_method == "mean":
            aggregated = np.mean(embeddings, axis=0)
        elif self.aggregation_method == "max":
            aggregated = np.max(embeddings, axis=0)
        elif self.aggregation_method == "concat":
            aggregated = embeddings.flatten()
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
        
        # Normalize
        aggregated = aggregated / (np.linalg.norm(aggregated) + 1e-8)
        
        return aggregated
    
    def encode_reference_images(self, reference_images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        Encode and aggregate reference images
        
        Args:
            reference_images: List of reference images (typically 3)
            
        Returns:
            Aggregated reference embedding
        """
        embeddings = self.encode_images(reference_images)
        aggregated = self.aggregate_embeddings(embeddings)
        
        return aggregated
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def encode_and_crop(
        self,
        image: Union[np.ndarray, Image.Image],
        bboxes: List[np.ndarray]
    ) -> np.ndarray:
        """
        Extract and encode cropped regions from image
        
        Args:
            image: Input image
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            
        Returns:
            Array of embeddings for each crop
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        embeddings = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            # Crop region
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0:
                embeddings.append(np.zeros(self.feature_dim))
                continue
            
            # Encode crop
            crop_embedding = self.encode_image(crop)
            embeddings.append(crop_embedding)
        
        return np.array(embeddings)


class SiameseReferenceEncoder(nn.Module):
    """
    Siamese network for reference matching (trainable version)
    """
    
    def __init__(
        self,
        backbone: str = "dinov2",
        feature_dim: int = 1024,
        projection_dim: int = 256
    ):
        """
        Initialize Siamese encoder
        
        Args:
            backbone: Backbone model
            feature_dim: Input feature dimension
            projection_dim: Output projection dimension
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        
        # Load backbone
        self.backbone = self._load_backbone()
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def _load_backbone(self):
        """Load and freeze backbone"""
        if self.backbone_name == "dinov2":
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        else:
            from transformers import AutoModel
            model = AutoModel.from_pretrained('facebook/dinov2-large')
        
        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def forward_once(self, x):
        """Forward pass for single image"""
        features = self.backbone(x)
        projected = self.projection(features)
        
        # L2 normalize
        projected = nn.functional.normalize(projected, p=2, dim=1)
        
        return projected
    
    def forward(self, x1, x2):
        """
        Forward pass for image pair
        
        Args:
            x1: Reference image
            x2: Query image
            
        Returns:
            Tuple of (embedding1, embedding2)
        """
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        
        return out1, out2

