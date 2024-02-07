
import torch
import torch.nn as nn
import torchvision.models as models
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts # pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
import copy
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from model.custom_vit import CustomViT
from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
import os
import torchvision
import matplotlib
matplotlib.use('Agg')
from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip, InterpolationMode, ToPILImage


'''
Validation
'''
def validate(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0

    with torch.no_grad():  # No need to track gradients
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total_predictions += labels.size(0)
            val_correct_predictions += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = val_correct_predictions / val_total_predictions * 100
    return val_loss, val_accuracy


'''
Train
'''
def train(model, train_loader, test_loader, criterion, optimizer, total_epochs, device, lr):
    print("\nTraining Starts...")
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=100, cycle_mult=1.0, max_lr=lr, min_lr=0.0001, warmup_steps=1, gamma=0.5)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    model.to(device)
    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Add accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions * 100

        val_loss, val_acc = validate(model, test_loader, criterion, device)

        last_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # tensorBoard
        writer.add_scalar('loss/train', epoch_loss, epoch)
        writer.add_scalar('accuracy/train', epoch_acc, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('accuracy/val', val_acc, epoch)

        # update
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"/root/workspace/minjae/VPT/best/{args.name}.pt")
            print(f"{epoch} is the Best accuracy!")

        print(f'Epoch {epoch+1}/{total_epochs}, LR: {last_lr}\nLoss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        print(f'Val_Loss: {val_loss:.2f}, Val_Acc: {val_acc:.2f}%\n')

    print(f'Best Validation Accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f"/root/workspace/minjae/VPT/best/{args.name}.pt")
    print("Traning Ends...")
    writer.close()
    return model


'''
Final Accuracy
'''
# Final performance
def final_acc(model, test_loader):
    all_labels = []
    all_preds = []

    model.eval()
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        pred = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    return all_labels, all_preds


'''
Main
'''
if __name__== '__main__':
    # 실행 title 설정
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--name', type=str, help='An input name')

    args = parser.parse_args()
    if args.name:
        print(f"I will try <<{args.name}>>! Training Starts..")
    else:
        args.name = 'none_name'


    # TensorBoard 정의
    layout = {
        "Visual Prompt Tuning" : {
            "loss" : ["Multiline", ["loss/train", "loss/val"]],
            "accuracy" : ["Multiline", ["accuracy/train", "accuracy/val"]],
        },
    }
    writer = SummaryWriter(f"./logs/{args.name}") 
    writer.add_custom_scalars(layout=layout)


    # Device 연결
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("========================")
    print(device)
    print("========================")


    # model (pretrained)
    model = CustomViT(pretrained_model='vit_base_patch16_224', img_size=32, patch_size=4, num_classes=10)  # cifar dataset을 사용할 때 
    model = model.to(device=device)


    # untrainable
    for name, param in model.named_parameters():
        if 'blocks' in name:
                param.requires_grad = False

    
    # Freeze가 되었는지 확인하는 코드
    def print_layer_trainable_status(model):
        for name, param in model.named_parameters():
            print(f"{name} is {'trainable' if param.requires_grad else 'frozen'}")

    print_layer_trainable_status(model)


    # Parameter 출력
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: FULL[{params}]")
    print(f"Trainable Parameters: FULL[{trainable}]")


    # Transforms
    train_mean = [0.4913997645378113, 0.48215836706161499, 0.44653093814849854]
    train_std = [0.24703224098682404, 0.24348513793945312, 0.2615878584384918]

    test_mean = [0.4942141773700714, 0.4851310842037201, 0.45040971088409424]
    test_std = [0.2466523642539978, 0.24289205610752106, 0.2615927150249481]

    cifar_train_transform = transforms.Compose([
        transforms.RandomCrop((32, 32), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])
    cifar_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(test_mean, test_std)
    ])


    # Parameter 설정
    batch_size = 128
    learning_rate = 0.001
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=5e-5)
    criterion = nn.CrossEntropyLoss()

    
    # Dataset2 - 'CIFAR-10'
    trainset = datasets.CIFAR10(root='../data', train=True, transform=cifar_train_transform, download=True)
    testset = datasets.CIFAR10(root='../data', train=False, transform=cifar_test_transform, download=True)


    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

    # Train
    model = train(model=model, train_loader=trainloader, test_loader=testloader, criterion=criterion, optimizer=optimizer, total_epochs=10, device=device, lr=learning_rate)


    # Test
    test_labels, test_preds = final_acc(model, testloader)
    print(f"Accuracy : {(accuracy_score(test_labels, test_preds) * 100):.2f}%")
    print(f"Precision : {(precision_score(test_labels, test_preds, average='macro')):.4f}")
    print(f"Recall : {(recall_score(test_labels, test_preds, average='macro')):.4f}")
    print(f"F1 Score : {(f1_score(test_labels, test_preds, average='macro')):.4f}")
    
