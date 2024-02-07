import torch
import torch.nn as nn
import torch.nn.init as init
import timm

import math
import inspect 

'''
Xavier_uniform
'''
def xavier_uniform_(tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * torch.sqrt(torch.tensor(6.0 / (fan_in + fan_out)))
    a = -std
    b = std

    with torch.no_grad():
        return tensor.uniform_(a, b)

def _calculate_fan_in_and_fan_out(tensor):
    # 주어진 tensor로부터 fan_in, fan_out 계산
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    if dimensions > 2:
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    else:
        fan_in = num_input_fmaps
        fan_out = num_output_fmaps
    
    return fan_in, fan_out


'''
Xavier Uniform 함수 사용 Prompt Embedding
'''
'''class PromptInput(nn.Module):
    def __init__(self, num_prompts, embed_dim, num_layers):
        super().__init__()
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim

        # Initialize prompt embeddings
        self.prompts = nn.Parameter(torch.zeros(num_layers, num_prompts, embed_dim)) 
        
        # Xavier Uniform
        xavier_uniform_(self.prompts, gain=1.0)

    # prepend the prompt to ebeded image patches
    def prepend_prompt(self, x, layer_idx):
        # x : [batch_size, sequence's length, embed_dim] 형태의 tensor
        batch_size = x.shape[0]
        
        prompt_tokens = self.prompts[layer_idx,:,:].expand(batch_size, -1, -1) # [batch_size, num_prompt, embed_dim]

        if layer_idx == 0:  # 첫 번째 layer
            x = torch.cat((x[:, :1, :], prompt_tokens, x[:, 1:, :]), dim=1)
                            # cls_token,     prompt,    image_patch_embedding  => x : [batch_size, cls_token + prompt_tokens + num_patches, hidden_dim]
                            # cls_token : 모든 입력에서의 첫 번째 위치에 있어야 함.
        else:
            x = torch.cat((x[:, :1, :], prompt_tokens, x[:, (1+self.num_prompts):, :]), dim=1)
                            # cls_token,  promt_tokens, remaining_patch_embedding
                            # remaining_patch_embedding : cls_token과 이전 레이어에서 추가된 프롬프트를 제외한 나머지 패치 임베딩. (나머지 패치 임베딩 부분을 시작하는 idx)

        return x'''
    

'''
Kaiming 함수 사용 Prompt Embedding
'''
class PromptInput(nn.Module):
    def __init__(self, num_prompts, embed_dim=768, num_layers=12):
        super().__init__()
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim

        # Initialize prompt embeddings
        self.prompts = nn.Parameter(torch.zeros(num_layers, num_prompts, embed_dim)) 
        
        # He(Kaiming_uniform)으로 초기화
        init.kaiming_uniform_(self.prompts)

    # prepend the prompt to ebeded image patches
    def prepend_prompt(self, x, layer_idx):
        # x : [batch_size, sequence's length, embed_dim] 형태의 tensor
        
        batch_size = x.shape[0]
        
        prompt_tokens = self.prompts[layer_idx,:,:].expand(batch_size, -1, -1) # [batch_size, num_prompt, embed_dim]

        if layer_idx == 0:  # 첫 번째 layer
            x = torch.cat((x[:, :1, :], prompt_tokens, x[:, 1:, :]), dim=1)
                            # cls_token,     prompt,    image_patch_embedding  => x : [batch_size, cls_token + prompt_tokens + num_patches, hidden_dim]
                            # cls_token : 모든 입력에서의 첫 번째 위치에 있어야 함.
        else:
            x = torch.cat((x[:, :1, :], prompt_tokens, x[:, (1+self.num_prompts):, :]), dim=1)
                            # cls_token,  promt_tokens, remaining_patch_embedding
                            # remaining_patch_embedding : cls_token과 이전 레이어에서 추가된 프롬프트를 제외한 나머지 패치 임베딩. (나머지 패치 임베딩 부분을 시작하는 idx)

        return x
    

'''
Prompt를 추가한 ViT
'''
class CustomViT(nn.Module):
    def __init__(self, pretrained_model='vit_base_patch16_224', img_size=32, patch_size=4, num_classes=10, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.prompt_embedding = PromptInput(num_prompts=100, embed_dim=768, num_layers=depth) # 사용자 정의 프롬프트 임베딩
        
        # timm 라이브러리를 사용하여 사전 학습된 ViT 모델 로드
        self.model = timm.create_model(pretrained_model, pretrained=True, img_size=img_size, patch_size=patch_size, num_classes=num_classes)

    def forward(self, x):
        # 이미지 패치 임베딩 및 위치 임베딩 처리
        x = self.model.patch_embed(x)  # 패치 임베딩
        cls_tokens = self.model.cls_token.expand(x.shape[0], -1, -1)  # 클래스 토큰 추가
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.model.pos_embed  # 위치 임베딩 추가
        x = self.model.pos_drop(x)

        # 각 Transformer 블록에 대해 프롬프트 추가
        for idx, block in enumerate(self.model.blocks):
            x = self.prompt_embedding.prepend_prompt(x, idx)
            x = block(x)

        x = self.model.norm(x)  # 최종 레이어 정규화
        x = self.model.forward_head(x)
        return x  # 분류 헤드를 통한 출력
