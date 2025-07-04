"""
Dataset classes and data processing utilities for multimodal retrieval.
Handles BDDX and DriveLM datasets with signal extraction and embedding processing.
"""
import os
import json
import pickle
import re
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class EmbeddingsDataset(Dataset):
    """Dataset for loading and processing multimodal embeddings."""
    
    def __init__(self, split: str, args, shared_models):
        self.args = args
        self.split = split
        self.shared_models = shared_models
        self.dataset = args.dataset
        
        # Load conversation data based on dataset
        self.data_path = os.path.join(args.data_dir, f'conversation_{self.dataset}_{split}.json')
        predicate_path = f'pgm/predicates/{self.dataset}/{split}_vectors.pkl'
        
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        
        # Load predicate vectors
        self.predicates = self._load_pickle_data(predicate_path)
        
        # Get shared models
        self.video_model, self.video_processor, self.sentence_model = shared_models.get_models()
        
        # Process all embeddings
        self._process_embeddings()
    
    @property
    def conversation_data(self):
        """Get the raw conversation data."""
        return self.data
    
    @property 
    def visual_paths(self):
        """Get list of visual paths (videos/images) from the dataset."""
        paths = []
        for item in self.data:
            if self.dataset == 'bddx':
                paths.append(item["video"][0])
            else:  # drivelm
                paths.append(item["image"])
        return paths
    
    @property
    def texts(self):
        """Get list of text content from the dataset."""
        texts = []
        for item in self.data:
            if self.dataset == 'bddx':
                text = item["conversations"][1]["value"] + ' ' + item["conversations"][3]["value"].lower()
            else:  # drivelm
                text = self._extract_before_ids(item['conversations'][1]['value'])
            texts.append(text)
        return texts
    
    def _load_pickle_data(self, file_path: str) -> torch.Tensor:
        """Load pickle data and convert to tensor."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return torch.FloatTensor(data)
    
    def _extract_vehicle_signals_bddx(self, input_string: str) -> torch.Tensor:
        """Extract vehicle signals from BDDX conversation string."""
        patterns = {
            'speed': r"Speed: \[([0-9., -]+)\]",
            'curvature': r"Curvature: \[([0-9., -]+)\]",
            'acceleration': r"Acceleration: \[([0-9., -]+)\]",
            'course': r"Course: \[([0-9., -]+)\]"
        }
        
        def extract_data(pattern):
            match = re.search(pattern, input_string)
            return [float(x) for x in match.group(1).split(',')] if match else []
        
        all_signals = []
        for pattern in patterns.values():
            all_signals.extend(extract_data(pattern))
        
        return torch.tensor(all_signals, dtype=torch.float32)
    
    def _extract_vehicle_signals_drivelm(self, input_string: str) -> torch.Tensor:
        """Extract vehicle signals from drivelm conversation string."""
        # Updated patterns for matching position, speed, and orientation data
        position_pattern = r"Position: \[\((.*?)\)\]"
        speed_pattern = r"Speed: \[\((.*?)\)\]"
        orientation_pattern = r"Orientation: \[\((.*?)\)\]"
        
        # Function to extract and convert data into a list of float tuples
        def extract_data(pattern):
            match = re.search(pattern, input_string)
            if match:
                data = match.group(1).split("), (")  # Split the matched data into individual tuples
                return [tuple(map(float, x.split(', '))) for x in data]  # Convert each tuple to a float tuple
            return []
        
        # Extract data
        position = extract_data(position_pattern)
        speed = extract_data(speed_pattern)
        orientation = extract_data(orientation_pattern)
        
        # Concatenate all signals and convert them to a tensor
        all_signals = position + speed + orientation
        tensor_signals = torch.tensor(all_signals, dtype=torch.float32).transpose(0,1).reshape(1, -1)
        
        return tensor_signals.flatten()
    
    def _extract_before_ids(self, text: str) -> str:
        """Extract text content before 'IDs:' marker."""
        pattern = r'(.*?)(?=IDs:|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text
    
    def _extract_sentence(self, text: str, choice: str) -> str:
        """Extract sentence based on choice marker."""
        # Find the choice and extract the corresponding sentence
        choice_pattern = rf"{re.escape(choice)}:\s*(.*?)(?=\n[A-E]:|$)"
        match = re.search(choice_pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        return text
    
    def _process_batch_bddx(self, texts: List[str], video_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process batch of BDDX data."""
        # Process texts
        text_embeddings = self.sentence_model.encode(texts, convert_to_tensor=True, device=self.args.device)
        
        # Process videos
        full_paths = [os.path.join(self.args.video_dir, path) for path in video_paths]
        video_tensors = self.video_processor(full_paths, return_tensors='pt')['pixel_values']
        video_embeddings = self.video_model(video_tensors.half()).view(-1, 2056, 1024).mean(1)
        
        return text_embeddings.cpu(), video_embeddings.cpu()
    
    def _process_batch_drivelm(self, texts: List[str], image_paths: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process batch of drivelm data."""
        # Process texts
        text_embeddings = self.sentence_model.encode(texts, convert_to_tensor=True, device=self.args.device)
        
        # Process images - convert lists to image paths
        video_embeddings = []
        for paths in image_paths:
            full_paths = [os.path.join(self.args.video_dir, path) for path in paths]
            video_tensors = self.video_processor(full_paths, return_tensors='pt')['pixel_values']
            video_emb = self.video_model(video_tensors.half()).view(-1, 2056, 1024).mean(1).mean(0)
            video_embeddings.append(video_emb)
        
        video_embeddings = torch.stack(video_embeddings)
        
        return text_embeddings.cpu(), video_embeddings.cpu()
    
    def _process_embeddings(self):
        """Process all embeddings in batches."""
        print(f"Processing {len(self.data)} samples in batches...")
        
        all_text_embeddings = []
        all_visual_embeddings = []
        all_signal_embeddings = []
        
        # Process in batches for memory efficiency
        batch_size = self.args.processing_batch_size
        
        for i in tqdm(range(0, len(self.data), batch_size), desc=f"Processing {self.split} embeddings"):
            batch = self.data[i:i+batch_size]
            
            batch_texts = []
            batch_visual_paths = []
            batch_signals = []
            
            for item in batch:
                if self.dataset == 'bddx':
                    # Extract text and video path
                    text = item["conversations"][1]["value"] + ' ' + item["conversations"][3]["value"].lower()
                    video_path = item["video"][0]
                    signal = self._extract_vehicle_signals_bddx(item["conversations"][0]["value"])
                    
                    batch_texts.append(text)
                    batch_visual_paths.append(video_path)
                    batch_signals.append(signal)
                    
                else:  # drivelm
                    # Extract text and image paths
                    text = self._extract_before_ids(item['conversations'][1]['value'])
                    image_paths = item["image"]
                    signal = self._extract_vehicle_signals_drivelm(item["conversations"][0]["value"])
                    
                    batch_texts.append(text)
                    batch_visual_paths.append(image_paths)
                    batch_signals.append(signal)
            
            # Process batch based on dataset type
            if self.dataset == 'bddx':
                text_emb, visual_emb = self._process_batch_bddx(batch_texts, batch_visual_paths)
            else:
                text_emb, visual_emb = self._process_batch_drivelm(batch_texts, batch_visual_paths)
            
            # Process signals
            max_len = max(len(signal) for signal in batch_signals)
            padded_signals = []
            for signal in batch_signals:
                if len(signal) < max_len:
                    padding = torch.zeros(max_len - len(signal))
                    signal = torch.cat([signal, padding])
                padded_signals.append(signal)
            signal_emb = torch.stack(padded_signals)
            
            all_text_embeddings.append(text_emb)
            all_visual_embeddings.append(visual_emb)
            all_signal_embeddings.append(signal_emb)
        
        # Concatenate all batches
        self.text_embeddings = torch.cat(all_text_embeddings)
        self.visual_embeddings = torch.cat(all_visual_embeddings)
        
        # Pad signal embeddings to consistent size
        max_signal_len = max(emb.size(1) for emb in all_signal_embeddings)
        padded_signal_embeddings = []
        for emb in all_signal_embeddings:
            if emb.size(1) < max_signal_len:
                padding = torch.zeros(emb.size(0), max_signal_len - emb.size(1))
                emb = torch.cat([emb, padding], dim=1)
            padded_signal_embeddings.append(emb)
        
        self.signal_embeddings = torch.cat(padded_signal_embeddings)
        
        # Ensure all have same number of samples
        min_samples = min(len(self.text_embeddings), len(self.visual_embeddings), 
                         len(self.signal_embeddings), len(self.predicates))
        
        self.text_embeddings = self.text_embeddings[:min_samples]
        self.visual_embeddings = self.visual_embeddings[:min_samples]
        self.signal_embeddings = self.signal_embeddings[:min_samples]
        self.predicates = self.predicates[:min_samples]
        
        print(f"Processed {min_samples} samples successfully!")
        print(f"Text embeddings shape: {self.text_embeddings.shape}")
        print(f"Visual embeddings shape: {self.visual_embeddings.shape}")
        print(f"Signal embeddings shape: {self.signal_embeddings.shape}")
        print(f"Predicate embeddings shape: {self.predicates.shape}")
    
    def __len__(self) -> int:
        return len(self.text_embeddings)
    
    def __getitem__(self, idx: int):
        return (
            self.text_embeddings[idx],
            self.visual_embeddings[idx],
            self.signal_embeddings[idx],
            self.predicates[idx]
        ) 