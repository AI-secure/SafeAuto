#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import time
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from scipy.stats import norm

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.global_step = 0
        self.max_step = None  # Will be set dynamically
        self.update_step = None # Will be set dynamically
        self.init_sigma = None  # Will be set by initialize_sigma_schedule
        self.final_sigma = None  # Will be set by initialize_sigma_schedule
        self.digit_token_list = None
        self.token2digit = None
        # Initialize weights and apply final processing
        self.post_init()
        
    def setup_training_configuration(self, 
                                   # Sigma parameters
                                   init_sigma: float, final_sigma: float, tokenizer,
                                   # Training schedule parameters  
                                   num_epochs: int, dataset_size: int, batch_size: int,
                                   num_gpus: int = 1, gradient_accumulation_steps: int = 1):
        """
        Unified method to configure all training parameters including sigma schedule and training schedule.
        This combines the functionality of initialize_sigma_schedule and setup_training_schedule.
        
        Args:
            # Sigma parameters
            init_sigma: Initial sigma value for KL divergence loss scheduling
            final_sigma: Final sigma value for annealing
            tokenizer: Tokenizer for collecting digit tokens
            
            # Training schedule parameters
            num_epochs: Number of training epochs
            dataset_size: Total number of samples in the dataset
            batch_size: Batch size per GPU
            num_gpus: Number of GPUs (default: 1)
            gradient_accumulation_steps: Gradient accumulation steps (default: 1)
        """
        # === Sigma Schedule Configuration ===
        self.init_sigma = init_sigma
        self.final_sigma = final_sigma
        
        # Encode full string "0123456789." once for efficiency
        full_chars = "0123456789."
        token_ids = tokenizer.encode(full_chars, add_special_tokens=False)[-11:]

        # digit_ids: exclude the dot (only digits)
        self.digit_token_list = token_ids[:-1]  # first 10 tokens = digits

        # digit_map: include dot for parsing
        self.token2digit = {tid: ch for tid, ch in zip(token_ids, full_chars)}

        # === Training Schedule Configuration ===
        # Set update step (frequency of global_step increments)
        if gradient_accumulation_steps and gradient_accumulation_steps > 0:
            self.update_step = 1.0 / gradient_accumulation_steps
        else:
            self.update_step = 1.0
        
        # Calculate and set max training steps
        steps_per_epoch = dataset_size // (batch_size * num_gpus * gradient_accumulation_steps)
        self.max_step = num_epochs * steps_per_epoch
        
        # === Unified Logging ===
        print(f"Training configuration setup complete:")
        print(f"  Sigma schedule: init_sigma={self.init_sigma}, final_sigma={self.final_sigma}")
        print(f"  Update frequency: {self.update_step:.2f}")
        print(f"  Max steps: {self.max_step} (epochs: {num_epochs}, steps_per_epoch: {steps_per_epoch})")
        print(f"  Dataset size: {dataset_size}, effective batch size: {batch_size * num_gpus * gradient_accumulation_steps}")
        print(f"  Digit tokens: {self.digit_token_list}")

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            
            if self.init_sigma != 0:
                ## Position-Dependent Cross-Entropy (PDCE) Loss
                # PDCE addresses the limitation of standard CE loss for numerical predictions
                # by incorporating two key principles:
                # 1. Digit-Level Proximity: Digits closer to target incur lower loss
                # 2. Place-Level Importance: More significant positions have greater influence
                
                # Shift tokens for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten the tokens and filter out padding tokens
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                    
                shift_logits = shift_logits[shift_labels != -100]
                shift_labels = shift_labels[shift_labels != -100]
                
                # Identify numerical tokens and compute PDCE components
                float_mask, target_distribution, weights = self.get_kl_weight(shift_labels)
                digit_index = self.digit_token_list  # Token IDs for digits 0-9
                
                # KL divergence loss for numerical tokens (PDCE)
                # Uses soft target distributions instead of one-hot labels
                kl_loss = - target_distribution * F.log_softmax(shift_logits[float_mask], dim=-1)[:, digit_index]
                kl_loss = (kl_loss.sum(-1) * weights).sum() / len(shift_labels)
                
                # Standard Cross-Entropy loss for non-numerical tokens
                loss_fct = CrossEntropyLoss(reduction='none')
                ce_loss = loss_fct(shift_logits[~float_mask], shift_labels[~float_mask]).sum() / len(shift_labels)
                
                # Combined loss: PDCE for numbers + CE for text
                loss = kl_loss + ce_loss
                
                # Sigma annealing: exponentially increase sigma during training
                sigma = self.init_sigma * ((self.final_sigma/self.init_sigma) ** (int(self.global_step)/self.max_step))
    
                print(f"CE loss: {ce_loss.item():.2f}, PDCE loss: {kl_loss.item():.2f}, sigma: {sigma:.4f}")
                self.global_step += self.update_step
                
            else:
                ## Standard Cross-Entropy Loss (original behavior)
                # Shift tokens for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def get_kl_weight(self, input_sequence):
        """
        Identify numerical tokens and compute PDCE loss components.
        
        This method implements the core logic for Position-Dependent Cross-Entropy (PDCE) loss:
        1. Detects numerical sequences (e.g., "12.46") in the token sequence
        2. Creates soft target distributions for each digit using Gaussian distributions
        3. Computes place-level weights using cumulative probabilities
        
        Args:
            input_sequence: Token sequence containing potential numerical values
            
        Returns:
            float_mask: Boolean mask indicating which tokens are numerical
            target_distribution: Soft target probability distributions for each digit
            weights: Place-level weights for each digit position
        """
        float_mask = torch.zeros_like(input_sequence).long()
        target_distribution = []
        weights = []
        
        # Safety check: ensure token2digit mapping is initialized
        if self.token2digit is None:
            raise RuntimeError("Token mapping not initialized. Call setup_training_configuration() first.")
            
        digit_map = self.token2digit
        sequence = [digit_map.get(token_id.item(), ' ') for token_id in input_sequence]

        # State tracking for numerical sequence detection
        current_index = 0
        start_new_number = True
        index = []  # Track token indices for current number
        current_char = ''  # Build current number string

        # Scan through token sequence to identify numerical patterns
        for i, token_id in enumerate(input_sequence):
            char = digit_map.get(token_id.item(), None)

            if char is not None:  # Found a digit or decimal point
                if start_new_number and char == '.':
                    # Skip leading decimal points
                    pass
                else:
                    current_char += char
                    if start_new_number:
                        index = [i]
                        start_new_number = False
                        num_start_index = i
                    elif char != '.':
                        index.append(i)
            else:
                # Non-numerical token encountered, process accumulated number
                # Here, we hard-code the float format to be either xx.xx or xx.xxx to avoid assigning pdce loss to irrelevant numbers that are not control signals.
                if len(index) > 0 and '.' in current_char and len(current_char.split('.')[0]) == 2 and (len(current_char.split('.')[1]) == 3 or len(current_char.split('.')[1]) == 2):
                    # Valid numerical format detected (e.g., "12.46" or "08.100")
                    float_mask[index] = 1
                    
                    # Generate soft target distributions for each digit (PDCE principle 1)
                    current_target_distribution = self.calculate_probabilities(current_char)
                    target_distribution.extend(current_target_distribution)
                    
                    # Calculate place-level weights using cumulative probabilities (PDCE principle 2)
                    prob_sum = sum([sum(dist) for dist in current_target_distribution])
                    cur_weights = [len(current_target_distribution)/prob_sum] * len(current_target_distribution)
                    weights.extend(cur_weights)
                        
                # Reset for next potential number
                start_new_number = True
                index = []
                current_char = ''

        # Handle case where numerical sequence is at the end
        if len(index) > 0:
            float_mask[index] = 1
            current_target_distribution = self.calculate_probabilities(current_char)
            target_distribution.extend(current_target_distribution)
            prob_sum = sum([sum(dist) for dist in current_target_distribution])
            weights.extend([len(current_target_distribution)/prob_sum] * len(current_target_distribution))

        return float_mask.bool(), torch.FloatTensor(target_distribution).to(input_sequence.device), torch.FloatTensor(weights).to(input_sequence.device)
  
    def calculate_probabilities(self, str_num):
        """
        Generate soft target distributions for PDCE loss using Gaussian distributions.
        
        This method implements the core mathematical foundation of PDCE loss:
        - Creates Gaussian-centered probability distributions for each digit position
        - Implements place-level weighting using cumulative probabilities
        - Ensures digits closer to target have higher probabilities (digit-level proximity)
        
        For example, for "12.46":
        - Position 1: Gaussian centered at 1, assigns higher prob to 0,1,2 than 8,9
        - Position 2: Gaussian centered at 2, weighted by prob of position 1
        - And so on...
        
        Args:
            str_num: String representation of the number (e.g., "12.46")
            
        Returns:
            probabilities: List of probability distributions for each digit position
        """
        # Convert number to string to easily access digits
        mu = float(str_num)  # Mean for Gaussian distribution

        def prob_range(num, precision, sigma):
            """
            Generate discrete probability distribution for digits 0-9 using Gaussian CDF.
            
            This implements the "Digit-Level Proximity" principle where digits closer 
            to the target digit receive higher probabilities.
            """
            # Centering the integration around the mid-point of the range
            lis = []
            num = int(num)
            for i in range(10):
                # Use Gaussian CDF to compute probability for each digit
                lis.append(norm(num, sigma).cdf(i+0.5) - norm(num, sigma).cdf(i-0.5))
            lis = np.array(lis)
            return (lis / sum(lis))  # Normalize to sum to 1

        # Parse integer and decimal parts
        parts = str_num.split('.')
        integer_part = parts[0]
        decimal_part = parts[1] if len(parts) > 1 else ''
      
        # Get current sigma value (with annealing if configured)
        if (self.max_step is not None and self.max_step > 0 and 
            self.init_sigma is not None and self.final_sigma is not None):
            sigma = self.init_sigma * ((self.final_sigma/self.init_sigma) ** (int(self.global_step)/self.max_step))
        else:
            # Fallback to initial sigma if parameters not set
            sigma = self.init_sigma if self.init_sigma is not None else 1.0

        # Generate probability distributions for each digit position
        # Implements "Place-Level Importance" using cumulative probabilities
        probabilities = []
        scale = 10 ** len(integer_part)  # Scale factor for digit positions
        current_number = 0
        next_weights = 1.0  # Cumulative weight for place-level importance
        new_part = integer_part + decimal_part  # Combined digit sequence
        
        for i, digit in enumerate(new_part):
            # Create Gaussian distribution centered around current digit
            # The precision parameter accounts for remaining digits
            digit_probs = prob_range(float(digit + '.' + new_part[i+1:]), (10 ** (-len(new_part[i+1:]))) /2, sigma)
            
            # Apply place-level weighting (cumulative probability approach)
            # Earlier positions have higher influence than later positions
            digit_probs = (np.array(digit_probs) * next_weights).tolist()
            
            # Update weight for next position using current digit's probability
            # This ensures weights decrease monotonically with position
            next_weights *= digit_probs[int(digit)]
            
            probabilities.append(digit_probs)
            
        return probabilities

    
AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)