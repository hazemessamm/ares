from ares.eval import pooling
import torch
import torch.nn as nn
from typing import Union, Callable
from ares.eval import common as common_module


def freezed_fwd(
    model: nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    _checked: bool = False,
) -> torch.Tensor:
    model.eval()
    if not _checked:
        if input_ids.device != next(model.parameters()).device:
            model.to(device=input_ids.device)
            _checked = True
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    return outputs, _checked


class OutputHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        intermediate_dim: Union[int, None] = None,
        mlp: bool = False,
        seed: int = None,
    ):
        super().__init__()
        self.mlp = mlp
        self.embed_dim = embed_dim
        self.intermediate_dim = intermediate_dim
        self.seed = seed
        if intermediate_dim is None and mlp:
            raise ValueError("intermediate_dim must be provided if mlp is True")

        if mlp:
            self.output_proj = nn.Sequential(
                nn.Linear(embed_dim, intermediate_dim),
                nn.GELU(),
                nn.LayerNorm(intermediate_dim),
                nn.Linear(intermediate_dim, num_classes),
            )
            print("Embed dim: ", embed_dim)
            print("Intermediate dim: ", intermediate_dim)
            print("Num classes: ", num_classes)
        else:
            self.output_proj = nn.Linear(embed_dim, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        range = 0.1
        if self.mlp:
            if self.seed is not None:
                print("Resetting MLP parameters with seeds: ", self.seed, self.seed + 1)
                intermediate_generator = torch.Generator()
                intermediate_generator.manual_seed(self.seed)
                output_generator = torch.Generator()
                output_generator.manual_seed(self.seed + 1)
            else:
                intermediate_generator = None
                output_generator = None

            nn.init.normal_(self.output_proj[0].weight, mean=0, std=self.intermediate_dim**-0.5, generator=intermediate_generator)
            nn.init.uniform_(self.output_proj[-1].weight, a=-range, b=range, generator=output_generator)
            nn.init.constant_(self.output_proj[0].bias, 0.0)
            nn.init.constant_(self.output_proj[-1].bias, 0.0)
        else:
            if self.seed is not None:
                print("Resetting Single output parameters with seed: ", self.seed)
                output_generator = torch.Generator()
                output_generator.manual_seed(self.seed)
            else:
                output_generator = None
            nn.init.uniform_(self.output_proj.weight, a=-range, b=range, generator=output_generator)
            nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(x)


class RegressionHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        intermediate_dim: Union[int, None] = 768,
        pooler: str = "avg",
        mlp: bool = True,
        freeze: bool = True,
        embedding_fn: Callable = common_module.ares_embedding_fn,
        seed: int = None,
    ):
        super().__init__()
        common_module.freeze_backbone(model, freeze)
        self.embedding_fn = embedding_fn
        if freeze:
            # To avoid saving the model to the checkpoint
            object.__setattr__(self, "model", model)
            self._device_checked = False
        else:
            self.model = model
        self.freeze = freeze
        if hasattr(model.config, "embed_dim"):
            self.embed_dim = model.config.embed_dim
        else:
            self.embed_dim = model.config.hidden_size
        self.pooler = pooling.get(pooler, embed_dim=self.embed_dim) if pooler is not None else None

        self.head = OutputHead(
            embed_dim=self.embed_dim,
            num_classes=1,
            intermediate_dim=intermediate_dim,
            mlp=mlp,
            seed=seed,
        )

    def fwd_no_grad(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        outputs, _checked = freezed_fwd(
            self.model, input_ids, attention_mask, self._device_checked
        )
        self._device_checked = _checked
        return self.embedding_fn(outputs)

    def fwd_with_grad(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask)
        return self.embedding_fn(outputs)

    def fwd_backbone(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        if self.freeze:
            return self.fwd_no_grad(input_ids, attention_mask)
        else:
            return self.fwd_with_grad(input_ids, attention_mask)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        special_tokens_mask: torch.LongTensor = None,
        labels: torch.FloatTensor = None,
    ) -> torch.Tensor:
        embeddings = self.fwd_backbone(input_ids, attention_mask)
        if self.pooler is not None:
            embeddings = self.pooler(embeddings, attention_mask, special_tokens_mask)
        logits = self.head(embeddings)
        outputs = {"logits": logits}

        if labels is not None:
            loss = nn.functional.mse_loss(logits, labels)
            outputs["loss"] = loss
        return outputs


class RegressionEmbeddingsHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        intermediate_dim: Union[int, None] = None,
        mlp: bool = True,
        seed: int = None,
    ):
        super().__init__()
        self.head = OutputHead(
            embed_dim=embed_dim,
            num_classes=1,
            intermediate_dim=intermediate_dim,
            mlp=mlp,
            seed=seed,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.FloatTensor = None,
    ) -> torch.Tensor:
        logits = self.head(embeddings)
        outputs = {"logits": logits}

        if labels is not None:
            loss = nn.functional.mse_loss(logits, labels)
            outputs["loss"] = loss
        return outputs


class MultiClassEmbeddingsHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        intermediate_dim: Union[int, None] = None,
        mlp: bool = True,
        seed: int = None,
    ):
        super().__init__()
        self.head = OutputHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            intermediate_dim=intermediate_dim,
            mlp=mlp,
            seed=seed,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.LongTensor = None,
    ) -> torch.Tensor:
        logits = self.head(embeddings)
        outputs = {"logits": logits}

        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            outputs["loss"] = loss
        return outputs


class BinaryClassificationHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        pooler: str,
        intermediate_dim: Union[int, None] = 768,
        mlp: bool = False,
        freeze: bool = False,
        embedding_fn: Callable = common_module.ares_embedding_fn,
        seed: int = None,
    ):
        super().__init__()
        common_module.freeze_backbone(model, freeze)
        if freeze:
            # To avoid saving the model to the checkpoint
            object.__setattr__(self, "model", model)
            self._device_checked = False
        else:
            self.model = model
        self.embedding_fn = embedding_fn
        self.freeze = freeze
        embed_dim = common_module.get_embed_dim(model)
        self.pooler = pooling.get(pooler, embed_dim=embed_dim) if pooler is not None else None
        self.head = OutputHead(
            embed_dim=embed_dim,
            num_classes=1,
            intermediate_dim=intermediate_dim,
            mlp=mlp,
            seed=seed,
        )

    def fwd_no_grad(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        outputs, _checked = freezed_fwd(
            self.model, input_ids, attention_mask, self._device_checked
        )
        self._device_checked = _checked
        return self.embedding_fn(outputs)

    def fwd_with_grad(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask)
        return self.embedding_fn(outputs)

    def fwd_backbone(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        if self.freeze:
            return self.fwd_no_grad(input_ids, attention_mask)
        else:
            return self.fwd_with_grad(input_ids, attention_mask)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        special_tokens_mask: torch.LongTensor = None,
        labels: torch.LongTensor = None,
    ) -> torch.Tensor:
        embeddings = self.fwd_backbone(input_ids, attention_mask)
        if self.pooler is not None:
            embeddings = self.pooler(embeddings, attention_mask, special_tokens_mask)
        logits = self.head(embeddings)
        outputs = {"logits": logits}

        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels
            )
            outputs["loss"] = loss
        return outputs


class MultiClassClassificationHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        intermediate_dim: Union[int, None] = 768,
        pooler: str = "avg",
        mlp: bool = True,
        freeze: bool = True,
        embedding_fn: Callable = common_module.ares_embedding_fn,
        seed: int = None,
    ):
        super().__init__()
        common_module.freeze_backbone(model, freeze)
        if freeze:
            object.__setattr__(self, "model", model)
            self._device_checked = False
        else:
            self.model = model
        self.embedding_fn = embedding_fn
        self.freeze = freeze
        embed_dim = common_module.get_embed_dim(model)
        self.pooler = pooling.get(pooler, embed_dim=embed_dim) if pooler is not None else None
        self.head = OutputHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            intermediate_dim=intermediate_dim,
            mlp=mlp,
            seed=seed,
        )

    def fwd_no_grad(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        outputs, _checked = freezed_fwd(
            self.model, input_ids, attention_mask, self._device_checked
        )
        self._device_checked = _checked
        return self.embedding_fn(outputs)

    def fwd_with_grad(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask)
        return self.embedding_fn(outputs)

    def fwd_backbone(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        if self.freeze:
            return self.fwd_no_grad(input_ids, attention_mask)
        else:
            return self.fwd_with_grad(input_ids, attention_mask)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        special_tokens_mask: torch.LongTensor = None,
        labels: torch.LongTensor = None,
    ) -> torch.Tensor:
        embeddings = self.fwd_backbone(input_ids, attention_mask)
        if self.pooler is not None:
            embeddings = self.pooler(embeddings, attention_mask, special_tokens_mask)
        logits = self.head(embeddings)
        outputs = {"logits": logits}

        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            outputs["loss"] = loss
        return outputs


class TokenClassificationHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        intermediate_dim: Union[int, None] = 768,
        mlp: bool = False,
        freeze: bool = False,
        embedding_fn: Callable = common_module.ares_embedding_fn,
        seed: int = None,
    ):
        super().__init__()
        common_module.freeze_backbone(model, freeze)
        if freeze:
            # To avoid saving the model to the checkpoint
            object.__setattr__(self, "model", model)
            self._device_checked = False
        else:
            self.model = model
        self.embedding_fn = embedding_fn
        self.freeze = freeze
        self.head = OutputHead(
            embed_dim=common_module.get_embed_dim(model),
            num_classes=num_classes,
            intermediate_dim=intermediate_dim,
            mlp=mlp,
            seed=seed,
        )

    def fwd_no_grad(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        outputs, _checked = freezed_fwd(
            self.model, input_ids, attention_mask, self._device_checked
        )
        self._device_checked = _checked
        return self.embedding_fn(outputs)

    def fwd_with_grad(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask)
        return self.embedding_fn(outputs)

    def fwd_backbone(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        if self.freeze:
            return self.fwd_no_grad(input_ids, attention_mask)
        else:
            return self.fwd_with_grad(input_ids, attention_mask)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        special_tokens_mask: torch.LongTensor = None,
        labels: torch.LongTensor = None,
    ) -> torch.Tensor:
        embeddings = self.fwd_backbone(input_ids, attention_mask)
        logits = self.head(embeddings)
        outputs = {"logits": logits}
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
            )
            outputs["loss"] = loss
        return outputs


class PPIBinaryClassificationHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        pooler: str,
        mlp: bool = False,
        freeze: bool = False,
        embedding_fn: Callable = common_module.ares_embedding_fn,
        seed: int = None,
    ):
        super().__init__()
        common_module.freeze_backbone(model, freeze)
        if freeze:
            object.__setattr__(self, "model", model)
            self._device_checked = False
        else:
            self.model = model
        self.embedding_fn = embedding_fn
        self.freeze = freeze
        embed_dim = common_module.get_embed_dim(model)
        self.pooler = pooling.get(pooler, embed_dim=embed_dim) if pooler is not None else None
        self.head = OutputHead(
            embed_dim=embed_dim * 2,
            num_classes=1,
            mlp=mlp,
            seed=seed,
        )

    def fwd_no_grad(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        outputs, _checked = freezed_fwd(
            self.model, input_ids, attention_mask, self._device_checked
        )
        self._device_checked = _checked
        return self.embedding_fn(outputs)

    def fwd_with_grad(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask)
        return self.embedding_fn(outputs)

    def fwd_backbone(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        if self.freeze:
            return self.fwd_no_grad(input_ids, attention_mask)
        else:
            return self.fwd_with_grad(input_ids, attention_mask)

    def forward(
        self,
        input_ids_1: torch.LongTensor,
        input_ids_2: torch.LongTensor,
        attention_mask_1: torch.LongTensor,
        attention_mask_2: torch.LongTensor,
        labels: torch.LongTensor = None,
    ) -> torch.Tensor:
        embeddings_1 = self.fwd_backbone(input_ids_1, attention_mask_1)
        embeddings_2 = self.fwd_backbone(input_ids_2, attention_mask_2)
        if self.pooler is not None:
            embeddings_1 = self.pooler(embeddings_1, attention_mask_1)
            embeddings_2 = self.pooler(embeddings_2, attention_mask_2)
        pair_embeddings = torch.cat([embeddings_1, embeddings_2], dim=-1)
        logits = self.head(pair_embeddings)
        outputs = {"logits": logits}
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels
            )
            outputs["loss"] = loss
        return outputs


class PPIRegressionHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        pooler: str,
        mlp: bool = False,
        freeze: bool = False,
        embedding_fn: Callable = common_module.ares_embedding_fn,
        seed: int = None,
    ):
        super().__init__()
        common_module.freeze_backbone(model, freeze)
        if freeze:
            object.__setattr__(self, "model", model)
            self._device_checked = False
        else:
            self.model = model
        self.embedding_fn = embedding_fn
        self.freeze = freeze
        embed_dim = common_module.get_embed_dim(model)
        self.pooler = pooling.get(pooler, embed_dim=embed_dim) if pooler is not None else None
        self.head = OutputHead(
            embed_dim=embed_dim * 2,
            num_classes=1,
            mlp=mlp,
            seed=seed,
        )

    def fwd_no_grad(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        outputs, _checked = freezed_fwd(
            self.model, input_ids, attention_mask, self._device_checked
        )
        self._device_checked = _checked
        return self.embedding_fn(outputs)

    def fwd_with_grad(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask)
        return self.embedding_fn(outputs)

    def fwd_backbone(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.Tensor:
        if self.freeze:
            return self.fwd_no_grad(input_ids, attention_mask)
        else:
            return self.fwd_with_grad(input_ids, attention_mask)

    def forward(
        self,
        input_ids_1: torch.LongTensor,
        input_ids_2: torch.LongTensor,
        attention_mask_1: torch.LongTensor,
        attention_mask_2: torch.LongTensor,
        labels: torch.LongTensor = None,
    ) -> torch.Tensor:
        embeddings_1 = self.fwd_backbone(input_ids_1, attention_mask_1)
        embeddings_2 = self.fwd_backbone(input_ids_2, attention_mask_2)
        if self.pooler is not None:
            embeddings_1 = self.pooler(embeddings_1, attention_mask_1)
            embeddings_2 = self.pooler(embeddings_2, attention_mask_2)
        pair_embeddings = torch.cat([embeddings_1, embeddings_2], dim=-1)
        logits = self.head(pair_embeddings)
        outputs = {"logits": logits}
        if labels is not None:
            loss = nn.functional.mse_loss(
                logits, labels
            )
            outputs["loss"] = loss
        return outputs
