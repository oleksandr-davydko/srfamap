import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, classification_report
from tqdm import tqdm

from radiomics_saliency.feature_extraction import generate_features
from radiomics_saliency.models import get_model


class ModelTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 texture_statistics_matrices,
                 labels,
                 train_indices,
                 val_indices,
                 test_texture_statistics_matrices,
                 test_labels,
                 storage_path: str,
                 batch_size: int):
        self.model = model
        self.device = device
        self.data = texture_statistics_matrices
        self.labels = np.asarray(labels).reshape(-1)
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_data = test_texture_statistics_matrices
        self.test_labels = np.asarray(test_labels).reshape(-1)
        self.current_epoch = 1
        self.best_metric = 0
        self.patience_value = 40
        self.storage_path = storage_path
        self.model_metrics = {}
        self.batch_size = batch_size

    def eval_model(self, eval_data, eval_labels):
        predicted_labels = []
        self.model.eval()
        with torch.no_grad():
            for example in tqdm(eval_data):
                texture_matrices = example.reshape(1, -1).to(self.device)
                logits = F.softmax(self.model.forward(texture_matrices), dim=1)
                predicted_labels += torch.argmax(logits, dim=1).cpu().detach().numpy().tolist()
        eval_labels = np.asarray(eval_labels).reshape(-1)
        mcc = matthews_corrcoef(predicted_labels, eval_labels)
        f1_scores = f1_score(predicted_labels, eval_labels, average=None)
        f1_avg = f1_score(predicted_labels, eval_labels, average='weighted')
        epoch_accuracy = accuracy_score(predicted_labels, eval_labels)
        metrics_dict = {
            "mcc": mcc,
            "f1": {k: v for (k, v) in zip(range(1, len(f1_scores) + 1), f1_scores)},
            "f1_avg": f1_avg,
            "accuracy": epoch_accuracy,
        }
        return predicted_labels, metrics_dict, classification_report(predicted_labels, eval_labels)

    def eval_model_by_indices(self, eval_indices: np.ndarray):
        return self.eval_model(self.data[eval_indices], self.labels[eval_indices])

    def eval_test_model(self):
        return self.eval_model(self.test_data, self.test_labels)

    def train_step(self, optimizer, criterion):
        self.model.train()
        print(f'Train epoch {self.current_epoch + 1}')
        epoch_losses = []
        train_preds = []
        with tqdm(total=len(self.train_indices)) as progress:
            batches = np.array_split(self.train_indices, len(self.train_indices) // self.batch_size)
            for i in batches:
                data = self.data[i]
                texture_matrices = data.to(self.device).reshape(-1, data.shape[1])
                label = torch.as_tensor(self.labels[i], dtype=torch.long, device=self.device).reshape(-1)
                optimizer.zero_grad()
                logits = F.softmax(self.model.forward(texture_matrices), dim=1)
                prediction = torch.argmax(logits, dim=1)
                train_preds += prediction.type(torch.LongTensor).cpu().detach().numpy().tolist()
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                progress.set_postfix(loss=np.mean(epoch_losses), epoch=self.current_epoch, refresh=False)
                progress.update(len(i))

    def train_model(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        epochs_without_improvement = 0
        while True:
            self.train_step(optimizer, criterion)
            print(f'Development set evaluation epoch {self.current_epoch}')
            _, val_metrics, val_report = self.eval_model_by_indices(self.val_indices)
            print(val_report)
            epoch_matthews = val_metrics['mcc']
            print(f'{epoch_matthews}')
            if epoch_matthews > self.best_metric:
                print(f'Epoch {self.current_epoch} has MCC improvement.')
                torch.save(self.model.state_dict(), f'{self.storage_path}/weights.pth')
                self.best_metric = epoch_matthews
                self.model_metrics['dev'] = val_metrics
                with open(f'{self.storage_path}/validation.json', 'w') as f:
                    f.write(json.dumps(val_metrics))
                epochs_without_improvement = 0
            else:
                epochs_without_improvement = epochs_without_improvement + 1
            if epochs_without_improvement >= self.patience_value:
                break
            self.current_epoch = self.current_epoch + 1
        self.model.load_state_dict(torch.load(f'{self.storage_path}/weights.pth'))
        _, test_metrics, test_report = self.eval_test_model()
        self.model_metrics['test'] = test_metrics
        with open(f'{self.storage_path}/test.json', 'w') as f:
            f.write(json.dumps(test_metrics))
        print(f'Test set evaluation')
        print(test_report)


def train_and_evaluate_model(images, labels, indices_train, indices_dev, test_images,
                              test_labels, extraction_parameters, device, model_path,
                              feature_count, unique_labels, skip_training,
                              model_type_name, masks=None, test_masks=None):
    """
    Train a model on the given images and return the model, trainer, features, and statistics.
    """
    # Generate or load features
    if (os.path.exists(f'{model_path}/features_cache.pt') and
            os.path.exists(f'{model_path}/test_features_cache.pt') and
            os.path.exists(f'{model_path}/mean_cache.pt') and
            os.path.exists(f'{model_path}/std_cache.pt')):
        features = torch.load(f'{model_path}/features_cache.pt')
        test_features = torch.load(f'{model_path}/test_features_cache.pt')
        m = torch.load(f'{model_path}/mean_cache.pt')
        s = torch.load(f'{model_path}/std_cache.pt')
    else:
        with torch.no_grad():
            features = generate_features(images, extraction_parameters, True, masks=masks)
            test_features = generate_features(test_images, extraction_parameters, True, masks=test_masks)
            m = features[indices_train].mean(0, keepdim=True)
            s = features[indices_train].std(0, unbiased=False, keepdim=True)
            features -= m
            features /= s
            test_features -= m
            test_features /= s
            torch.save(features, f'{model_path}/features_cache.pt')
            torch.save(test_features, f'{model_path}/test_features_cache.pt')
            torch.save(m, f'{model_path}/mean_cache.pt')
            torch.save(s, f'{model_path}/std_cache.pt')
    feature_count = features.shape[1]
    # Create and train model
    model = get_model(model_type_name, device, feature_count, len(unique_labels))
    model.to(device).to(torch.float)
    model_trainer = ModelTrainer(model, device, features, labels, indices_train, indices_dev,
                                 test_features, test_labels, model_path, 60)

    if not skip_training:
        model_trainer.train_model()
    else:
        model_trainer.model.load_state_dict(torch.load(f'{model_path}/weights.pth'))

    return model, model_trainer, features, test_features, m, s
