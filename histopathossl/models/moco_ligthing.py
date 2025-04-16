import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)
from torchvision.transforms import transforms

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
    def __init__(
        self,
        base_encoder="resnet50",
        output_dim=128,
        queue_size=65536,
        lr=1e-3,
        momentum_key_encoder=0.999,
        momentum_sgd=0.9,
        weight_decay=1e-4,
        temperature=0.07,
        warmup_epochs=10,
        epoch_max=200,
        pretrained=False,
    ):
        super().__init__()

        self.encoder_q = self._load_resnet(
            base_encoder, output_dim, pretrained=pretrained
        )
        self.encoder_k = self._load_resnet(base_encoder, output_dim)

        self.temperature = temperature
        self.momentum_key_encoder = momentum_key_encoder
        self.momentum_sgd = momentum_sgd
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.epoch_max = epoch_max

        self.register_buffer("queue", torch.randn(queue_size, output_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self._initialize_momentum_encoder()
        self.save_hyperparameters()

    def _load_resnet(self, base_encoder, output_dim, pretrained=False):
        weights = get_resnet_weights(base_encoder) if pretrained else None
        encoder = getattr(models, base_encoder)(weights=weights)
        if base_encoder == "resnet50":
            hidden_dim = encoder.fc.in_features
            encoder.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

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
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.mul_(self.momentum_key_encoder).add_(
                param_q.data, alpha=1 - self.momentum_key_encoder
            )

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
            self.queue[: end_ptr % self.queue.size(0), :] = keys[first_part:, :]

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
        x_q = x_q.cuda(non_blocking=True)
        x_k = x_k.cuda(non_blocking=True)
        q, k = self(x_q, x_k)

        # Contrastive loss

        # pos_logits = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # neg_logits = torch.einsum("nc,kc->nk",
        #                           [q, self.queue.clone().detach()])

        pos_logits = (q * k).sum(dim=1, keepdim=True)  # Faster than einsum
        neg_logits = q @ self.queue.clone().detach().T  # Matrix multiplication (faster)

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        logits /= self.temperature

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels)

        # Update queue
        self._dequeue_and_enqueue(k)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.lr,
            momentum=self.momentum_sgd,
            weight_decay=self.weight_decay,
        )

        # Create a linear warmup scheduler followed by cosine annealing
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: self._get_lr_scale(epoch, self.warmup_epochs),
            ),
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _get_lr_scale(self, epoch, warmup_epochs):
        """
        Returns a learning rate multiplier:
        - Linear warmup for the first `warmup_epochs`
        - Cosine annealing afterwards
        """
        if epoch < warmup_epochs:
            # Linear warmup
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # Cosine annealing
            return 0.5 * (
                1.0
                + math.cos(
                    math.pi * (epoch - warmup_epochs) / (self.epoch_max - warmup_epochs)
                )
            )


class MoCoV2LightningPus(MoCoV2Lightning):
    def _load_resnet(self, base_encoder, output_dim, pretrained=False):
        weights = get_resnet_weights(base_encoder) if pretrained else None
        encoder = getattr(models, base_encoder)(weights=weights)
        if base_encoder == "resnet50":
            hidden_dim = encoder.fc.in_features
            encoder.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),  # first layer
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),  # second layer
                self.encoder.fc,
                nn.BatchNorm1d(output_dim, affine=False),
            )  # output layer

        else:
            raise ValueError(
                f"Unsupported base_encoder: {base_encoder}. Supported values are: resnet50, convnext_large"
            )

        return encoder

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for current_layer in range(num_layers):
            dim1 = input_dim if current_layer == 0 else mlp_dim
            dim2 = output_dim if current_layer == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if current_layer < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc  # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)

    def compute_loss(self, q, k):
        pos_logits = (q * k).sum(dim=1, keepdim=True)  # Faster than einsum
        neg_logits = q @ self.queue.clone().detach().T  # Matrix multiplication (faster)

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        logits /= self.temperature

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, labels)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        # Einstein sum is more intuitive
        logits = torch.einsum("nc,mc->nm", [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (
            torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()
        ).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def training_step(self, batch, batch_idx):
        x_1, x_2 = batch
        x_1 = x_1.cuda(non_blocking=True)
        x_2 = x_2.cuda(non_blocking=True)
        q1, k1 = self(x_1, x_2)
        q2, k2 = self(x_2, x_1)

        loss = 0.5 * (self.compute_loss(q1, k1) + self.compute_loss(q2, k2))
        # Update queue
        self._dequeue_and_enqueue(k1)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
