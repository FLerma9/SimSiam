# https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch import nn


class KNNValidation(object):
    def __init__(self, batch_size, train_path, val_path, model, K=1):
        self.model = model
        self.device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
        self.K = K
        self.batch_size =batch_size
        base_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.ImageFolder(train_path, transform=base_transform)
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=True,
                                           drop_last=True)

        val_dataset = datasets.ImageFolder(val_path, transform=base_transform)
        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         num_workers=4,
                                         pin_memory=True,
                                         drop_last=True)

    def _topk_retrieval(self):
        """Extract features from validation split and search on train split features."""
        n_data = len(self.train_dataloader.dataset)
        feat_dim = 2048

        self.model.eval()
        if str(self.device) == 'cuda':
            torch.cuda.empty_cache()

        train_features = torch.zeros([feat_dim, n_data], device=self.device)
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_dataloader):
                inputs = inputs.to(self.device)
                batch_size = inputs.size(0)

                # forward
                if self.model.arch == 'squeezenet1_1':
                    features = self.model.encoder.features(inputs)
                    features = features.view(features.size(0), -1)
                    features = self.model.encoder.classifier(features)
                else:
                    features = self.model.encoder(inputs)
                features = nn.functional.normalize(features)
                train_features[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = features.data.t()

            train_labels = torch.LongTensor(self.train_dataloader.dataset.targets).cuda()

        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_dataloader):
                targets = targets.cuda(non_blocking=True)
                batch_size = inputs.size(0)
                
                if self.model.arch == 'squeezenet1_1':
                    features = self.model.encoder.features(inputs.to(self.device))
                    features = features.view(features.size(0), -1)
                    features = self.model.encoder.classifier(features)
                else:
                    features = self.model.encoder(inputs.to(self.device))

                dist = torch.mm(features, train_features)
                yd, yi = dist.topk(self.K, dim=1, largest=True, sorted=True)
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)

                total += targets.size(0)
                correct += retrieval.eq(targets.data).sum().item()
        top1 = correct / total

        return top1

    def eval(self):
        return self._topk_retrieval()


