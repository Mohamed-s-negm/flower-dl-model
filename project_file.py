import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Class for the CNN structure
class ScratchCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    #Forward pass method
    def forward(self, x):
        return self.classifier(self.features(x))

# The main model training class
class Custom_Model:
    def __init__(self, model_type='scratch', device=None, num_classes=None, model_path=None,
                 num_epochs=150, batch_size=32, lr=0.001, early_stop_patience=20):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.num_classes = num_classes
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), f"model_{model_type}.pth")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.early_stop_patience = early_stop_patience
        self.model = None
        self.dataloaders = {}
        self.criterion = None
        self.optimizer = None
        self.test_loader = None
        self.class_names = []

    # We prepare the dataset into train, validate, and test datasets
    def prepare(self, data_dir, input_size=(224, 224)):
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4363, 0.4328, 0.329], [0.2129, 0.2075, 0.2038])
        ])
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
        train_len = int(0.8 * len(dataset))
        val_len = int(0.1 * len(dataset))
        test_len = len(dataset) - train_len - val_len

        train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len])
        self.dataloaders['train'] = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.dataloaders['val'] = DataLoader(val_data, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        self.class_names = dataset.classes

    # We initialize the model depending on the model_type
    def set_model(self):
        if self.model_type == 'scratch':
            self.model = ScratchCNN(self.num_classes).to(self.device)
        else:
            vgg = models.vgg16(pretrained=True)

            if self.model_type == 'feature_extraction':
                for param in vgg.parameters():
                    param.requires_grad = False

            elif self.model_type == 'fine_tuning':
                for param in vgg.features.parameters():
                  param.requires_grad = False

                for param in vgg.classifier.parameters():
                  param.requires_grad = True

                for name, param in vgg.features.named_parameters():
                  if int(name.split('.')[0]) >= 10:  
                    param.requires_grad = True
                print(f"Fine-tuning VGG16: Total trainable layers: {sum(p.requires_grad for p in vgg.parameters())}")

            vgg.classifier[6] = nn.Linear(4096, self.num_classes)
            self.model = vgg.to(self.device)

    # We train the model that was initiated
    def train(self):
        if self.model_type == 'scratch':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.model_type == 'feature_extraction':
            self.optimizer = optim.Adam(self.model.classifier[6].parameters(), lr=self.lr)
        else:  # fine_tuning
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = optim.Adam(trainable_params, lr=self.lr)

        self.criterion = nn.CrossEntropyLoss()
        best_acc = 0
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for xb, yb in self.dataloaders['train']:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_acc = self.evaluate()
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}")

            # We save the model with the highest validation accuracy and make sure that the model doesn't get into the state of underfitting.
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stop_patience:
                print("Early stopping triggered.")
                break

    #We evaaluate the model
    def evaluate(self):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in self.dataloaders['val']:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total if total > 0 else 0.0

    #We get the test accuracy of the model.
    def test(self):
        """Evaluate the model on the held-out test set."""
        if self.model is None or self.test_loader is None:
            print("Model or test loader not initialized.")
            return 0.0

        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        test_acc = correct / total if total > 0 else 0.0
        print(f"Test Accuracy: {test_acc:.4f}")
        return test_acc

    # Save the model 
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    # Load an existing trained model
    def load_model(self):
        if not os.path.exists(self.model_path):
            print("No saved model to load.")
            return
        self.set_model()
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {self.model_path}")

    # We use the model to classify the images
    def predict(self, image_path):
        if self.model is None:
            print("Model is not initialized. Cannot perform prediction.")
            return None

        if not hasattr(self, 'class_names') or not self.class_names:
            print("Class names are not defined. Cannot interpret prediction.")
            return None

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4363, 0.4328, 0.329], [0.2129, 0.2075, 0.2038])
        ])

        image_tensor = transform(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        
        index = predicted.item()
        return self.class_names[index]

    # Plot a confusion matrix for the trained model
    def plot_confusion_matrix(self):
        if self.model is None or self.test_loader is None:
            return

        y_true, y_pred = [], []
        self.model.eval()
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                outputs = self.model(xb)
                preds = outputs.argmax(1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(f"Confusion Matrix - {self.model_type}")
        plt.show()

    

