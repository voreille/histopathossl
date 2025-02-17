import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import transforms
from pytorch_lightning import LightningModule
from torchvision.models import (ResNet18_Weights, ResNet34_Weights,
                                ResNet50_Weights, ResNet101_Weights,
                                ResNet152_Weights)

from histopathossl.training.augmentations import GaussianBlur


def get_resnet_weights(base_encoder):
    resnet_weights_map = {
        "resnet18": ResNet18_Weights.DEFAULT,
        "resnet34": ResNet34_Weights.DEFAULT,
        "resnet50": ResNet50_Weights.DEFAULT,
        "resnet101": ResNet101_Weights.DEFAULT,
        "resnet152": ResNet152_Weights.DEFAULT,
    }
    base_encoder = base_encoder.lower()
    if base_encoder in resnet_weights_map:
        return resnet_weights_map[base_encoder]
    else:
        raise ValueError(
            f"Unsupported base_encoder: {base_encoder}. Supported values are: {list(resnet_weights_map.keys())}"
        )


class MoCoV2Lightning(LightningModule):

    def __init__(self,
                 base_encoder="resnet50",
                 output_dim=128,
                 queue_size=65536,
                 momentum=0.999,
                 temperature=0.07,
                 lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_q = self._load_resnet(base_encoder, output_dim)
        self.encoder_k = self._load_resnet(base_encoder, output_dim)

        self.temperature = temperature
        self.momentum = momentum
        self.lr = lr

        self.register_buffer("queue", torch.randn(queue_size, output_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self._initialize_momentum_encoder()

    def _load_resnet(self, base_encoder, output_dim, pretrained=False):
        weights = get_resnet_weights(base_encoder) if pretrained else None
        encoder = getattr(models, base_encoder)(weights=weights)
        if base_encoder == "resnet50":
            hidden_dim = encoder.fc.in_features
            encoder.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, output_dim))

        elif base_encoder == "convnext_large":
            hidden_dim = encoder.classifier[-1].in_features
            encoder.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )

        return encoder

    def _initialize_momentum_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data,
                                                  alpha=1 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        end_ptr = ptr + batch_size
        if end_ptr <= self.queue.size(0):
            self.queue[ptr:end_ptr, :] = keys
        else:
            first_part = self.queue.size(0) - ptr
            self.queue[ptr:, :] = keys[:first_part, :]
            self.queue[:end_ptr % self.queue.size(0), :] = keys[first_part:, :]

        self.queue_ptr[0] = (ptr + batch_size) % self.queue.size(0)

    def forward(self, x_q, x_k):
        q = self.encoder_q(x_q)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(x_k)
            k = F.normalize(k, dim=1)

        return q, k

    def training_step(self, batch, batch_idx):
        x_q, x_k = batch
        q, k = self(x_q, x_k)

        # Contrastive loss

        # pos_logits = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # neg_logits = torch.einsum("nc,kc->nk",
        #                           [q, self.queue.clone().detach()])

        pos_logits = (q * k).sum(dim=1, keepdim=True)  # Faster than einsum
        neg_logits = q @ self.queue.clone().detach(
        ).T  # Matrix multiplication (faster)

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        logits /= self.temperature

        labels = torch.zeros(logits.size(0),
                             dtype=torch.long,
                             device=self.device)
        loss = F.cross_entropy(logits, labels)

        # Update queue
        self._dequeue_and_enqueue(k)

        self.log("train_loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
