"""
Clean and modular retrieval training script.
Trains multimodal embeddings for RAG-based retrieval using video, text, signal, and predicate embeddings.
"""
import setGPU
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

# Import our modular components
from models import SharedModels, FusionModel
from datasets import EmbeddingsDataset
from training import train_model, generate_retrieval_index
from utils import parse_args, setup_output_dirs, get_checkpoint_path, get_retrieval_index_path


def main():
    """Main training function with clean modular structure."""
    # Parse arguments
    args = parse_args()
    args.conversation_dir = f'data/conversation/{args.dataset}'
    # Setup output directories
    setup_output_dirs(args)
    
    # Initialize shared models once (singleton pattern)
    print("=" * 60)
    print(f"Starting {args.dataset} retrieval training")
    print("=" * 60)
    
    shared_models = SharedModels(args)
    
    # Load datasets (no redundant data loading!)
    print(f"\nLoading {args.dataset} datasets...")
    train_dataset = EmbeddingsDataset('train', args, shared_models)
    signal_dim, visual_dim, predicate_dim = train_dataset.get_feature_dims()
    print(f"✓ Training dataset loaded: {len(train_dataset)} samples")
        
    eval_dataset = EmbeddingsDataset('eval', args, shared_models)  
    print(f"✓ Evaluation dataset loaded: {len(eval_dataset)} samples")
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    # Initialize model
    print(f"\nInitializing MLP model...")
    model = FusionModel(signal_dim, visual_dim, predicate_dim, args).to(args.device)
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"\nLoading checkpoint from {args.load_checkpoint}")
        model.load_state_dict(torch.load(args.load_checkpoint))
        print("✓ Checkpoint loaded successfully")
    else:
        # Train the model
        print(f"\nStarting training for {args.num_epochs} epochs...")
        print("-" * 40)
        
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        train_model(model, train_dataloader, optimizer, args)
        
        print("✓ Training completed!")
    
    # Save model checkpoint
    if args.save_checkpoint:
        checkpoint_path = get_checkpoint_path(args)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✓ Model saved to {checkpoint_path}")
    
    # Generate retrieval index (using existing dataset objects - no redundant loading!)
    print(f"\nGenerating retrieval index...")
    print("-" * 40)
    
    rag_dict = generate_retrieval_index(model, train_dataset, eval_dataset, args)
    
    # Save retrieval index
    rag_path = get_retrieval_index_path(args)
    os.makedirs(os.path.dirname(rag_path), exist_ok=True)
    with open(rag_path, 'w') as f:
        json.dump(rag_dict, f, indent=2)
    
    print(f"✓ Retrieval index saved to {rag_path}")
    print(f"✓ Index contains {len(rag_dict)} entries with top-{args.top_k} similarities")
    
    print("\n" + "=" * 60)
    print("Training and indexing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main() 