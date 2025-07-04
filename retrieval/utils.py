"""
Utility functions and configuration for retrieval system.
Contains argument parsing and data path management.
"""
import argparse
import os
from typing import Tuple


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train retrieval model for multimodal RAG')
    
    # Model and data paths
    parser.add_argument('--model_path', type=str, default='/home/jiaweizhang/Nuro/RAGDriver/checkpoints/Video-LLaVA-7B_RAGDRIVER',
                        help='Path to pretrained model')
    parser.add_argument('--data_dir', type=str, default='data/conversation/bddx',
                        help='Directory containing training data')
    parser.add_argument('--video_dir', type=str, default='data',
                        help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, default='./retrieval/ckpts',
                        help='Directory to save outputs')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, choices=['bddx', 'drivelm'], default='bddx',
                        help='Dataset to use for training (bddx or drivelm)')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help='Evaluation batch size')
    parser.add_argument('--processing_batch_size', type=int, default=100,
                        help='Batch size for processing videos/texts')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate for MLP layers')
    
    # Model architecture
    parser.add_argument('--text_model', type=str, default='sentence-transformers/sentence-t5-xl',
                        help='Sentence transformer model for text embeddings')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for KL divergence loss')
    parser.add_argument('--top_k', type=int, default=2,
                        help='Number of top similar samples to retrieve')
    
    # Embedding weights
    parser.add_argument('--signal_weight', type=float, default=0.4,
                        help='Weight for signal embeddings in fusion')
    parser.add_argument('--video_weight', type=float, default=0.4,
                        help='Weight for video embeddings in fusion')
    parser.add_argument('--predicate_weight', type=float, default=0.2,
                        help='Weight for predicate embeddings in fusion')
    
    # Other options
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation')
    parser.add_argument('--save_checkpoint', action='store_true',
                        help='Save model checkpoint after training')
    
    return parser.parse_args()


def get_data_paths(args) -> Tuple[str, str]:
    """
    Get training and evaluation data paths based on dataset.
    
    Args:
        args: Parsed arguments containing dataset and data_dir
        
    Returns:
        Tuple of (train_data_path, eval_data_path)
    """
    train_data_path = os.path.join(args.data_dir, f'conversation_{args.dataset}_train.json')
    eval_data_path = os.path.join(args.data_dir, f'conversation_{args.dataset}_eval.json')
    
    return train_data_path, eval_data_path


def setup_output_dirs(args):
    """Create necessary output directories."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create retrieval subdirectory
    retrieval_dir = os.path.join(args.output_dir, 'retrieval/ckpts')
    os.makedirs(retrieval_dir, exist_ok=True)
    
    return retrieval_dir


def get_checkpoint_path(args) -> str:
    """Get the checkpoint save path."""
    return os.path.join(args.output_dir, f'retrieval/ckpts/rag_projector_{args.dataset}.pth')


def get_retrieval_index_path(args) -> str:
    """Get the retrieval index save path."""
    return os.path.join(args.output_dir, f'retrieval/rag_json/{args.dataset}_rag_top{args.top_k}.json') 