"""
Model definitions for retrieval system.
Contains SharedModels singleton and MLP model architecture.
"""
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from llava.model.multimodal_encoder.builder import extractor, build_image_tower
from llava.utils import disable_torch_init


### for loading the image tower
def image_extractor(**kwargs):
    """Create image tower using the same pattern as video extractor."""
    class ImageTowerConfig:
        def __init__(self):
            self.mm_image_tower = "./cache_dir/LanguageBind_Image"
            self.mm_vision_select_feature = "patch"
            self.mm_vision_select_layer = -2
            self.model_type = "llava"
            self.num_attention_heads = 32
            self.num_hidden_layers = 32
            self.num_key_value_heads = 32
            self.pad_token_id = 0
            self.pretraining_tp = 1
            self.rms_norm_eps = 1e-05
            self.vocab_size = 32000

    image_tower_cfg = ImageTowerConfig()
    return build_image_tower(image_tower_cfg)

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
        
        # Load video model using existing extractor
        disable_torch_init()
        self.video_model = extractor()
        self.video_model.eval()
        self.video_model.to(args.device)
        self.video_processor = self.video_model.video_processor
        
        # Load image model using elegant image_extractor
        self.image_model = image_extractor()
        self.image_model.eval()
        self.image_model.to(args.device)
        self.image_processor = self.image_model.image_processor
        
        # Load text model
        self.sentence_model = SentenceTransformer(
            args.text_model, device=args.device
        )
        print("Shared models initialized successfully!")
    
    def get_models(self):
        """Get the shared models."""
        if not SharedModels._initialized:
            raise RuntimeError("SharedModels not initialized. Call with args first.")
        return self.video_model, self.video_processor, self.image_model, self.image_processor, self.sentence_model


class FusionModel(nn.Module):
    """Multi-layer perceptron for multimodal embedding fusion."""
    

    def __init__(self, signal_dim, visual_dim, predicate_dim, args):
        super(FusionModel, self).__init__()
        self.signal_dim = signal_dim
        self.visual_dim = visual_dim
        self.predicate_dim = predicate_dim
        self.fusion_dim = args.fusion_dim  # Final projection dimension for all modalities
        self.args = args
        dropout_rate = args.dropout_rate
        
        def block(in_dim, out_dim, dropout=True):
            layers = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout_rate))
            return layers

        # Signal MLP
        self.signal_layer = nn.Sequential(
            *block(signal_dim, 64),
            *block(64, 128),
            *block(128, 256),
            *block(256, self.fusion_dim, dropout=False)
        )

        # Visual MLP
        self.visual_layer = nn.Sequential(
            *block(visual_dim, 1024),
            *block(1024, 512),
            *block(512, 256),
            *block(256, self.fusion_dim, dropout=False)
        )

        # Predicate MLP
        self.predicate_layer = nn.Sequential(
            *block(predicate_dim, 64),
            *block(64, 128),
            *block(128, 256),
            *block(256, self.fusion_dim, dropout=False)
        )

        # Main fusion MLP
        self.fusion = nn.Sequential(
            *block(self.fusion_dim, 256),
            *block(256, 256),
            *block(256, 256),
            *block(256, self.fusion_dim, dropout=False)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize weights using He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def project_and_normalize(self, embedding, projector):
        projected = projector(embedding)
        return projected / (projected.norm(dim=-1, keepdim=True) + 1e-16)

    def forward(self, visual_embedding: torch.Tensor, signal_embedding: torch.Tensor, 
                predicate_embedding: torch.Tensor):
        """
        Forward pass through the MLP.
        
        Args:
            visual_embedding: Visual features [batch_size, visual_dim]
            signal_embedding: Signal features [batch_size, signal_dim] 
            predicate_embedding: Predicate features [batch_size, predicate_dim]
            
        Returns:
            fused_embedding: Fused multimodal embedding
        """
        # Ensure all embeddings have the same dtype as the model parameters
        # Get the dtype of the first parameter (linear layer weight)
        model_dtype = next(self.parameters()).dtype
        
        visual_embedding = visual_embedding.to(dtype=model_dtype)
        signal_embedding = signal_embedding.to(dtype=model_dtype)
        predicate_embedding = predicate_embedding.to(dtype=model_dtype)

        signal_projected = self.project_and_normalize(signal_embedding, self.signal_layer)
        visual_projected = self.project_and_normalize(visual_embedding, self.visual_layer)
        predicate_projected = self.project_and_normalize(predicate_embedding, self.predicate_layer)
        
        # Weighted fusion
        fused = (
            self.args.signal_weight * signal_projected +
            self.args.video_weight * visual_projected +
            self.args.predicate_weight * predicate_projected
        )
        
        # Final fusion layer
        fused_embedding = self.fusion(fused)
        return fused_embedding / (fused_embedding.norm(dim=-1, keepdim=True)+1e-16)