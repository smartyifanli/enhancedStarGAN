import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from scipy import stats

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = "/home/li46460/abaAbstract/experiment/"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['original', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
               for x in ['original', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['original', 'test']}
class_names = image_datasets['original'].classes

model_ft = models.googlenet(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['original']:
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['original']
        epoch_acc = running_corrects.double() / dataset_sizes['original']

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

model_ft = train_model(model_ft, criterion, optimizer, num_epochs=25)

def evaluate_model(model, dataloader, class_names):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred

y_true, y_pred = evaluate_model(model_ft, dataloaders['test'], class_names)

print(classification_report(y_true, y_pred, target_names=class_names))

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)

    # Calculate metrics for each class
    sensitivity = np.zeros(len(class_names))
    specificity = np.zeros(len(class_names))
    precision = np.zeros(len(class_names))
    f1_score = np.zeros(len(class_names))

    for i in range(len(class_names)):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        
        sensitivity[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity[i] = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1_score[i] = 2 * (precision[i] * sensitivity[i]) / (precision[i] + sensitivity[i]) if (precision[i] + sensitivity[i]) > 0 else 0

    # Average across all classes
    sensitivity_avg = np.mean(sensitivity)
    specificity_avg = np.mean(specificity)
    precision_avg = np.mean(precision)
    f1_score_avg = np.mean(f1_score)

    # Confidence Interval for Accuracy
    conf_interval = stats.norm.interval(0.95, loc=accuracy, scale=stats.sem(y_pred))

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision_avg:.4f}')
    print(f'F1 Score: {f1_score_avg:.4f}')
    print(f'Sensitivity: {sensitivity_avg:.4f}')
    print(f'Specificity: {specificity_avg:.4f}')
    print(f'Confidence Interval: {conf_interval}')


calculate_metrics(y_true, y_pred)
