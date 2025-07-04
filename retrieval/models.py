"""
Model definitions for retrieval system.
Contains SharedModels singleton and MLP model architecture.
"""
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from llava.model.multimodal_encoder.builder import extractor
from llava.utils import disable_torch_init


class SharedModels:
    """Singleton class to hold shared models across datasets."""
    _instance = None
    _initialized = False
    
    def __new__(cls, args=None):
        if cls._instance is None:
            cls._instance = super(SharedModels, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, args=None):
        if not SharedModels._initialized and args is not None:
            self._setup_models(args)
            SharedModels._initialized = True
    
    def _setup_models(self, args):
        """Setup video/image and text models."""
        print("Initializing shared models...")
        
        # Load video/image model
        disable_torch_init()
        self.video_model = extractor()
        self.video_model.eval()  # Set to evaluation mode
        self.video_model.to(args.device)
        self.video_processor = self.video_model.video_processor
        
        # Load text model
        self.sentence_model = SentenceTransformer(
            args.text_model, device=args.device
        )
        print("Shared models initialized successfully!")
    
    def get_models(self):
        """Get the shared models."""
        if not SharedModels._initialized:
            raise RuntimeError("SharedModels not initialized. Call with args first.")
        return self.video_model, self.video_processor, self.sentence_model


class MLP(nn.Module):
    """Multi-layer perceptron for multimodal embedding fusion."""
    
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        
        # Define dimensions
        self.signal_dim = 2000  # Signal embedding dimension  
        self.visual_dim = 1024  # Video/image embedding dimension
        self.predicate_dim = 1024  # Predicate embedding dimension
        self.hidden_dim = 1024  # Output dimension
        
        # Signal projection layers
        self.signal_proj = nn.Sequential(
            nn.Linear(self.signal_dim, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(512, self.hidden_dim)
        )
        
        # Visual projection layers  
        self.visual_proj = nn.Sequential(
            nn.Linear(self.visual_dim, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(512, self.hidden_dim)
        )
        
        # Predicate projection layers
        self.predicate_proj = nn.Sequential(
            nn.Linear(self.predicate_dim, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(512, self.hidden_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, visual_embedding: torch.Tensor, signal_embedding: torch.Tensor, 
                predicate_embedding: torch.Tensor):
        """
        Forward pass through the MLP.
        
        Args:
            visual_embedding: Visual features [batch_size, visual_dim]
            signal_embedding: Signal features [batch_size, signal_dim] 
            predicate_embedding: Predicate features [batch_size, predicate_dim]
            
        Returns:
            tuple: (signal_projected, visual_projected, fused_embedding)
        """
        # Project each modality
        signal_projected = self.signal_proj(signal_embedding)
        visual_projected = self.visual_proj(visual_embedding)
        predicate_projected = self.predicate_proj(predicate_embedding)
        
        # Weighted fusion
        fused = (
            self.args.signal_weight * signal_projected +
            self.args.video_weight * visual_projected +
            self.args.predicate_weight * predicate_projected
        )
        
        # Final fusion layer
        fused_embedding = self.fusion(fused)
        
        return signal_projected, visual_projected, fused_embedding 