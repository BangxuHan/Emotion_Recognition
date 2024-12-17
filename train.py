# train.py
# 定义训练轮
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from face_dataset import FaceDataset
from model import FaceCNN
from mobile_net import MobileNet_v2
log = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, weight_decay):
    # 加载数据集并分割batch
    train_loader = data.DataLoader(train_dataset, batch_size)
    # 构建模型
    # model = FaceCNN().to(device)
    model = MobileNet_v2().to(device)

    save_path = 'models/mobilenet_v2.pth'
    # 判断是否有之前训练过的网络参数，有的话就加载参数接着训练
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path), False)

    # 损失函数和优化器
    compute_loss = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(models.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 学习率衰减
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    for epoch in range(epochs):
        loss = 0
        sum_loss = 0.
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.cpu().detach().item()

        # 保存损失值
        log.add_scalars("train_loss", {"train_loss": sum_loss / len(train_loader)}, epoch)
        # 打印损失值
        print('epoch {} - train_loss:'.format(epoch), loss.item())

        # 评估模型准确率
        if epoch % 20 == 19:
            model.eval()
            acc_train = validate(model, train_dataset, batch_size)
            acc_val = validate(model, val_dataset, batch_size)
            print('train_acc: %.1f %%' % (acc_train * 100))
            print('val_acc: %.1f %%' % (acc_val * 100))

            log.add_scalar("train_acc", acc_train, epoch)
            log.add_scalar("val_acc", acc_val, epoch)

            torch.save(model.state_dict(), 'models/mobilenet_v2_t{}.pth'.format(epoch))

    return model


# 验证模型在验证集上的正确率
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    result, total = 0.0, 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        pred = model.forward(images)
        pred = np.argmax(pred.cpu().data.numpy(), axis=1)
        labels = labels.cpu().data.numpy()
        result += np.sum((pred == labels))
        total += len(images)
    acc = result / total
    return acc


if __name__ == '__main__':
    train_dataset = FaceDataset(root=r'/home/kls/data/fer2013/cnn_train')
    val_dataset = FaceDataset(root=r'/home/kls/data/fer2013/cnn_val')
    model = train(train_dataset, val_dataset, batch_size=128, epochs=200, learning_rate=0.001, weight_decay=0)
    # torch.save(models, 'cnn_model.pkl')  # 保存模型
