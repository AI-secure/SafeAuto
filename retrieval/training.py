"""
Training and evaluation utilities for retrieval model.
Handles training loops, loss computation, and retrieval index generation.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List


def kl_divergence(logits, targets, dim, temperature=1.0):
    """Compute KL divergence loss with temperature scaling."""
    log_pred = F.log_softmax(logits / temperature, dim=dim)
    targets = F.softmax(targets / temperature, dim=dim)
    return F.kl_div(log_pred, targets, reduction='batchmean')


def train_model(model, dataloader: DataLoader, optimizer, args):
    """Train the retrieval model."""
    model.train()
    
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        
        for batch_idx, (text_emb, visual_emb, signal_emb, predicate_emb) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            
            # Move to device
            text_emb = text_emb.to(args.device)
            visual_emb = visual_emb.to(args.device)
            signal_emb = signal_emb.to(args.device)
            predicate_emb = predicate_emb.to(args.device)
            
            # Forward pass
            signal_embedding, visual_embedding, hidden_embedding = model(
                visual_emb, signal_emb, predicate_emb
            )
            
            # Compute losses
            ground_logits = text_emb @ text_emb.T
            logits = hidden_embedding @ hidden_embedding.T
            
            loss_i = kl_divergence(logits, ground_logits, dim=0, temperature=args.temperature)
            loss_t = kl_divergence(logits, ground_logits, dim=1, temperature=args.temperature)
            loss = (loss_i + loss_t) / 2
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Log progress
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{args.num_epochs}, Average Loss: {avg_loss:.6f}')


def generate_retrieval_index(model, train_dataset, eval_dataset, args) -> Dict[str, List[str]]:
    """
    Generate retrieval index for evaluation using existing dataset objects.
    
    Args:
        model: Trained retrieval model
        train_dataset: Training dataset object (already loaded)
        eval_dataset: Evaluation dataset object (already loaded) 
        args: Configuration arguments
        
    Returns:
        Dictionary mapping visual paths to similar visual paths
    """
    model.eval()
    
    # Get data from existing dataset objects (no redundant loading!)
    train_visual_paths = train_dataset.visual_paths
    train_texts = train_dataset.texts
    eval_visual_paths = eval_dataset.visual_paths
    eval_texts = eval_dataset.texts
    
    print(f"Loaded {len(train_visual_paths)} training samples and {len(eval_visual_paths)} eval samples")
    
    # Process training embeddings
    print("Processing training embeddings...")
    visual_embeddings = []
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, drop_last=False)
    
    with torch.no_grad():
        for batch_idx, (text_emb, visual_emb, signal_emb, predicate_emb) in enumerate(tqdm(train_dataloader, desc="Processing training embeddings")):
            visual_emb = visual_emb.to(args.device)
            signal_emb = signal_emb.to(args.device)
            predicate_emb = predicate_emb.to(args.device)
            
            # Compute projections
            _, _, hidden_emb = model(visual_emb, signal_emb, predicate_emb)
            visual_embeddings.append(hidden_emb.cpu())
    
    train_embeddings = torch.cat(visual_embeddings)
    
    # Create retrieval index for training data (self-similarity)
    print("Computing training similarities...")
    similarity_matrix = train_embeddings @ train_embeddings.T
    # Set diagonal to -inf to exclude self-similarity
    similarity_matrix.fill_diagonal_(-float('inf'))
    # Get top-k indices
    topk_values, topk_indices = torch.topk(similarity_matrix, k=args.top_k, dim=1)
    
    rag_dict = {}
    
    # Add training similarities to dictionary
    for i, path in enumerate(train_visual_paths):
        current_index = topk_indices[i]
        if len(current_index):
            if args.dataset == 'bddx':
                path_key = path.split('/')[-1]
                similar_paths = [train_visual_paths[j].split('/')[-1] for j in current_index]
            else:
                # For drivelm, path is a list of image paths
                path_key = str(path)  # Convert list to string for key
                similar_paths = [str(train_visual_paths[j]) for j in current_index]
            
            rag_dict[path_key] = similar_paths
        else:
            print(f"No similar samples found for training sample {i}")
    
    # Process evaluation embeddings
    print("Processing evaluation embeddings...")
    eval_visual_embeddings = []
    eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False, drop_last=False)
    
    with torch.no_grad():
        for batch_idx, (text_emb, visual_emb, signal_emb, predicate_emb) in enumerate(tqdm(eval_dataloader, desc="Processing eval embeddings")):
            visual_emb = visual_emb.to(args.device)
            signal_emb = signal_emb.to(args.device)
            predicate_emb = predicate_emb.to(args.device)
            
            # Compute projections
            _, _, hidden_emb = model(visual_emb, signal_emb, predicate_emb)
            eval_visual_embeddings.append(hidden_emb.cpu())
    
    eval_embeddings = torch.cat(eval_visual_embeddings)
    
    # Compute similarity between eval and train embeddings
    print("Computing eval-train similarities...")
    similarity_matrix = eval_embeddings @ train_embeddings.T
    # Get top-k indices
    eval_topk_values, eval_topk_indices = torch.topk(similarity_matrix, k=args.top_k, dim=1)
    
    # Add evaluation similarities to dictionary
    for i, path in enumerate(eval_visual_paths):
        current_index = eval_topk_indices[i]
        if len(current_index):
            if args.dataset == 'bddx':
                path_key = path.split('/')[-1]
                similar_paths = [train_visual_paths[j].split('/')[-1] for j in current_index]
            else:
                path_key = str(path)
                similar_paths = [str(train_visual_paths[j]) for j in current_index]
            
            rag_dict[path_key] = similar_paths
        else:
            print(f"No similar samples found for eval sample {i}")
    
    print(f"Generated retrieval index with {len(rag_dict)} entries")
    return rag_dict 