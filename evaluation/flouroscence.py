from omegaconf import DictConfig
from transformers import TrainingArguments, Trainer
from ares.eval.metrics import RegressionMetrics
from ares.eval import models as models_module
from ares.eval import utils as utils_module
from ares.eval import dataset as dataset_module
import hydra
import wandb
import os
from ares.eval import common as common_module


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Running fluorescence evaluation on model: ", cfg.model.name, "with checkpoint: ", getattr(cfg, cfg.model.name).checkpoint, "and seed: ", cfg.training.seed)
    utils_module.set_seed(cfg.training.seed)
    model_config = getattr(cfg, cfg.model.name)
    wandb_run_name = utils_module.create_wandb_run_name(cfg)
    config_path = os.path.join("configs", "config.yaml")

    if not os.path.exists(config_path):
        print("Config file not found, saving config to wandb run directory. Directory: ", config_path)

    output_dir = os.path.join(model_config.output_dir, "flouroscence", f"seed_{cfg.training.seed}")
    tokenizer, backbone = common_module.load_model(cfg.model.family, model_config.checkpoint)

    extraction_fn = common_module.get_extraction_function(cfg.model.family)

    shared_config = cfg.shared_modeling_config
    model = models_module.RegressionHead(
        model=backbone,
        intermediate_dim=shared_config.intermediate_dim,
        pooler=shared_config.pooler,
        mlp=shared_config.mlp,
        freeze=shared_config.freeze,
        embedding_fn=extraction_fn,
        seed=cfg.training.seed,
    )

    train_dataset = dataset_module.get(cfg.task.name, "train")
    validation_dataset = dataset_module.get(cfg.task.name, "validation")
    test_dataset = dataset_module.get(cfg.task.name, "test")

    collator = dataset_module.SingleTargetCollator(
        tokenizer=tokenizer,
        sequences_column=train_dataset.sequences_column,
        labels_column=train_dataset.labels_column,
        family=cfg.model.family,
    )
    train_args = TrainingArguments(**cfg.training, run_name=wandb_run_name, output_dir=output_dir, data_seed=cfg.training.seed)

    regression_metrics = RegressionMetrics()
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset={"validation": validation_dataset, "test": test_dataset},
        data_collator=collator,
        compute_metrics=regression_metrics,
    )
    trainer.train()
    wandb.save(config_path)


if __name__ == "__main__":
    main()