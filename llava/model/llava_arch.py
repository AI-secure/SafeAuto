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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_image_tower, build_video_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, X_TOKEN_INDEX, DEFAULT_X_PATCH_TOKEN, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_image_tower"):
            self.image_tower = build_image_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        if hasattr(config, "mm_video_tower"):
            self.video_tower = build_video_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_image_tower(self):
        image_tower = getattr(self, 'image_tower', None)
        if type(image_tower) is list:
            image_tower = image_tower[0]
        return image_tower

    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower

    def initialize_image_modules(self, model_args, fsdp=None):
        image_tower = model_args.image_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_image_tower = image_tower

        image_tower = build_image_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.image_tower = [image_tower]
        else:
            self.image_tower = image_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = image_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_video_modules(self, model_args, fsdp=None):
        video_tower = model_args.video_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_video_tower = video_tower

        video_tower = build_video_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.video_tower = [video_tower]
        else:
            self.video_tower = video_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = video_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_image_tower(self):
        return self.get_model().get_image_tower()

    def get_video_tower(self):
        return self.get_model().get_video_tower()

    def get_all_tower(self, keys):
        tower = {key: getattr(self, f'get_{key}_tower') for key in keys}
        return tower

    def encode_images(self, images):
        image_features = self.get_model().get_image_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_videos(self, videos):
        video_features = self.get_model().get_video_tower()(videos)

        video_features = self.get_model().mm_projector(video_features)

        # import pdb; pdb.set_trace()
        return video_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, X_modalities
    ):
        '''
        X_modalities [
        [img_feature, img_feature, video_feature, audio_feature],
        ['image', 'image', 'video', 'audio']
        ]
        '''
        Xs, keys = X_modalities
        all_tower = self.get_all_tower(set(keys)) if len(keys) > 0 else None
        
        if all_tower is None or X_modalities[0][0] is None or input_ids.shape[1] == 1:
            if past_key_values is not None and all_tower is not None and Xs is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels


        X_features = []
        # Xs = [Xs]
        
        for X, key in zip(Xs, keys):
            # stackX = torch.stack(X, dim=0).view(3,3,8,224,224)
            if key == 'image':
                stackX = torch.stack(X, dim=0).view(-1,3,224,224)
            else:
                stackX = torch.stack(X, dim=0).view(-1,3,8,224,224)
            # print(stackX.shape)
            
            encoded_feature = getattr(self, f'encode_{key}s')(stackX)
#             print("Feature Dim", encoded_feature.shape)
            X_features.append(encoded_feature)
            
        flattened_X_features = []
        for x in X_features:
            if key == 'video':
                flattened_feature = x.flatten(0, 1)
            else:
                flattened_feature = x
            flattened_X_features.append(flattened_feature)

        X_features = flattened_X_features


        new_input_embeds = []
        new_labels = [] if labels is not None else None
        
        # print("BATCH size", len(X_features), input_ids.shape)
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_X_idx = 0
            if (torch.any(torch.stack([cur_input_ids == X_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_X_features = X_features[cur_X_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                
                if len(cur_X_features.shape) == 3:
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_X_features[0], cur_input_embeds_2], dim=0)
                else:
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_X_features[0:0], cur_input_embeds_2], dim=0)
                    
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_X_idx += 1 
                continue
            X_token_indices = torch.where(torch.any(torch.stack([cur_input_ids == X_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0] 
            # print(batch_idx, cur_input_ids.size(),X_token_indices)
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            # print(4444444444)
            while X_token_indices.numel() > 0:
                cur_X_features = X_features[batch_idx][cur_X_idx]
                X_token_start = X_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_x_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:X_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[X_token_start-1:X_token_start]))
                    cur_new_input_embeds.append(cur_X_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[X_token_start+1:X_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:X_token_start])
                        cur_new_labels.append(torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[X_token_start:X_token_start+1])
                        cur_labels = cur_labels[X_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:X_token_start])) 
                    cur_new_input_embeds.append(cur_X_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:X_token_start])
                        cur_new_labels.append(torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[X_token_start+1:]
                cur_X_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_x_start_end', False):
                    cur_input_ids = cur_input_ids[X_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[X_token_start+1:] 
                X_token_indices = torch.where(torch.any(torch.stack([cur_input_ids == X_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0]
            # print(55555555555555555)
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_x_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
                    
            for i in range(len(cur_new_input_embeds)):
                cur_new_input_embeds[i] = cur_new_input_embeds[i].view(-1, 4096)
                
#             import pdb; pdb.set_trace()
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds] 
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            
            
            
            new_input_embeds.append(cur_new_input_embeds)
            
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
            
            print("New emb", new_input_embeds.shape)
            # import pdb;pdb.set_trace()

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_X_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_x_patch_token:
            for x in model_args.X:
                tokenizer.add_tokens([DEFAULT_X_PATCH_TOKEN[x.upper()]], special_tokens=True)
            # tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_x_start_end:
            num_new_tokens = 0
            for x in model_args.X:
                num_new_tokens += tokenizer.add_tokens([DEFAULT_X_START_TOKEN[x.upper()], DEFAULT_X_END_TOKEN[x.upper()]], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_x_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False