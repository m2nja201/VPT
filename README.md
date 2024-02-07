# 🦾 VPT
experimental code of **``Visual Prompt Tuning``**

## ⚙️ Setting
- **Dataset** : CIFAR-10
- **Backbone Model** : ViT

## 📕 How to
1. Select the train file you want to run.
  - ``train_full.py``
  - ``train_vpt.py``
2. If you want to use **VPT**, Select the initialization method for the prompt embedding.
  - Xavier uniform
  - Kaiming uniform 
3. Uncomment the techniques you want to experiment with.
4. Execute the following command:
   
```
python {train file name} --name="{result file name}" > {result file name}.txt
```

## 🎣 Experiment
- **VPT**
  - initialization method
  - prompt length
- **VPT** vs **FFT**
  - learning rate

## 🔍 Result and Conclusion
> I used a part of the PowerPoint presentation I had made as the results section.

![image](https://github.com/m2nja201/VPT/assets/80443295/29819eaa-bf51-45ae-b6e0-d69a31cc2e54)
![image](https://github.com/m2nja201/VPT/assets/80443295/11974dc9-1836-4c2e-9deb-57686d98af0b)
![image](https://github.com/m2nja201/VPT/assets/80443295/cb1ec964-c1ae-4f34-81fd-017a80d91be2)
![image](https://github.com/m2nja201/VPT/assets/80443295/7d7000d5-f588-4335-9da4-c39d84268c20)
![image](https://github.com/m2nja201/VPT/assets/80443295/99953dd5-c669-4d7b-b305-fb17a0eddef6)

## 🖇️ Reference
1. **Visual Prompt Tuning** : https://arxiv.org/abs/2203.12119
2. **Vision Transformer** : https://arxiv.org/abs/2010.11929


