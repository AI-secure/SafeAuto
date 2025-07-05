"""
Training and evaluation utilities for retrieval model.
Handles training loops, loss computation, and retrieval index generation.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List


def kl_divergence(p_logits, q_logits, dim, temperature=0.5):
    # Apply temperature scaling to q_logits
    q_logits = q_logits / temperature
    # Compute q distribution after temperature scaling
    q = F.softmax(q_logits, dim=dim)
    # Compute KL divergence with softmax for p_logits
    kl_div = F.kl_div(F.log_softmax(p_logits, dim=dim), q, reduction='batchmean')
    return kl_div


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
            fused_embedding = model(
                visual_emb, signal_emb, predicate_emb
            )
            
            # Compute losses
            ground_logits = text_emb @ text_emb.T
            logits = fused_embedding @ fused_embedding.T
            
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
        Dictionary mapping keys to similar keys:
        - BDD-X: path -> [similar_paths]
        - DriveLM: id -> [similar_ids]
    """
    model.eval()
    
    # Get data from existing dataset objects
    train_visual_paths = train_dataset.visual_paths
    train_ids = train_dataset.ids
    eval_visual_paths = eval_dataset.visual_paths
    eval_ids = eval_dataset.ids

    print(f"Loaded {len(train_visual_paths)} training samples and {len(eval_visual_paths)} eval samples")
    
    # Process training embeddings
    print("Processing training embeddings...")
    train_embeddings_list = []
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, drop_last=False)
    
    with torch.no_grad():
        for batch_idx, (text_emb, visual_emb, signal_emb, predicate_emb) in enumerate(tqdm(train_dataloader, desc="Processing training embeddings")):
            visual_emb = visual_emb.to(args.device)
            signal_emb = signal_emb.to(args.device)
            predicate_emb = predicate_emb.to(args.device)
            
            # Compute fused embeddings
            fused_emb = model(visual_emb, signal_emb, predicate_emb)
            train_embeddings_list.append(fused_emb.cpu())
    
    train_embeddings = torch.cat(train_embeddings_list)
    
    # Create retrieval index for training data (self-similarity)
    print("Computing training similarities...")
    similarity_matrix = train_embeddings @ train_embeddings.T
    # Set diagonal to -inf to exclude self-similarity
    similarity_matrix.fill_diagonal_(-float('inf'))
    # Get top-k indices
    topk_values, topk_indices = torch.topk(similarity_matrix, k=args.top_k, dim=1)
    
    rag_dict = {}
    
    # Add training similarities to dictionary
    for i in range(len(train_visual_paths)):
        current_index = topk_indices[i]
        if len(current_index) > 0:
            if args.dataset == 'bddx':
                # For BDD-X, use path as key
                path_key = train_visual_paths[i].split('/')[-1]  # Extract filename
                similar_paths = [train_visual_paths[j].split('/')[-1] for j in current_index]
                rag_dict[path_key] = similar_paths
            else:
                # For DriveLM, use id as key
                id_key = str(train_ids[i])
                similar_ids = [str(train_ids[j]) for j in current_index]
                rag_dict[id_key] = similar_ids
        else:
            print(f"No similar samples found for training sample {i}")
                
    # Process evaluation embeddings
    print("Processing evaluation embeddings...")
    eval_embeddings_list = []
    eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False, drop_last=False)
    
    with torch.no_grad():
        for batch_idx, (text_emb, visual_emb, signal_emb, predicate_emb) in enumerate(tqdm(eval_dataloader, desc="Processing eval embeddings")):
            visual_emb = visual_emb.to(args.device)
            signal_emb = signal_emb.to(args.device)
            predicate_emb = predicate_emb.to(args.device)
            
            # Compute fused embeddings
            fused_emb = model(visual_emb, signal_emb, predicate_emb)
            eval_embeddings_list.append(fused_emb.cpu())
    
    eval_embeddings = torch.cat(eval_embeddings_list)
    
    # Compute similarity between eval and train embeddings
    print("Computing eval-train similarities...")
    similarity_matrix = eval_embeddings @ train_embeddings.T
    # Get top-k indices
    eval_topk_values, eval_topk_indices = torch.topk(similarity_matrix, k=args.top_k, dim=1)
    
    # Add evaluation similarities to dictionary
    for i in range(len(eval_visual_paths)):
        current_index = eval_topk_indices[i]
        if len(current_index) > 0:
            if args.dataset == 'bddx':
                # For BDD-X, use path as key, retrieve similar paths from training
                eval_path_key = eval_visual_paths[i].split('/')[-1]  # Extract filename
                similar_paths = [train_visual_paths[j].split('/')[-1] for j in current_index]
                rag_dict[eval_path_key] = similar_paths
            else:
                # For DriveLM, use id as key, retrieve similar ids from training
                eval_id_key = str(eval_ids[i])
                similar_ids = [str(train_ids[j]) for j in current_index]
                rag_dict[eval_id_key] = similar_ids
        else:
            print(f"No similar samples found for eval sample {i}")
    
    print(f"Generated retrieval index with {len(rag_dict)} entries")
    return rag_dict 