'''
함수와 모델에 대한 정의가 제대로 이루어졌는지 확인하기 위한 파일입니다.
'''

import inspect
from model.custom_vit import CustomViT 
import timm
import torch


model = CustomViT()
model = model.load_state_dict(torch.load('/root/workspace/minjae/VPT/best/vpt_he_cifar.pt'))
#model = timm.create_model('vit_base_patch16_224', num_classes=47, pretrained=True)

# 클래스나 함수의 소스 코드 확인
#class_or_func_source = inspect.getsource(model.forward)
#print(class_or_func_source)
#print("---------------------------------------")

'''
bbb = inspect.getsource(CustomViT().__init__)
print(bbb)
'''
