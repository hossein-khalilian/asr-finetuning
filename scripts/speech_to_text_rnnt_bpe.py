# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Preparing the Tokenizer for the dataset
Use the `process_asr_text_tokenizer.py` script under <NEMO_ROOT>/scripts/tokenizers/ in order to prepare the tokenizer.

```sh
python <NEMO_ROOT>/scripts/tokenizers/process_asr_text_tokenizer.py \
        --manifest=<path to train manifest files, seperated by commas>
        OR
        --data_file=<path to text data, seperated by commas> \
        --data_root="<output directory>" \
        --vocab_size=<number of tokens in vocabulary> \
        --tokenizer=<"spe" or "wpe"> \
        --no_lower_case \
        --spe_type=<"unigram", "bpe", "char" or "word"> \
        --spe_character_coverage=1.0 \
        --log
```

# Training the model
```sh
python speech_to_text_rnnt_bpe.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    model.tokenizer.dir=<path to directory of tokenizer (not full path to the vocab file!)> \
    model.tokenizer.type=<either bpe or wpe> \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp" \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

"""

import lightning.pytorch as pl
import torch.nn as nn
from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from omegaconf import OmegaConf


def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if "SqueezeExcite" in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)


def check_vocabulary(asr_model, cfg):
    """
    Checks if the decoder and vocabulary of the model needs to be updated.
    If either of them needs to be updated, it updates them and returns the updated model.
    else vocabulary will be reused from the pre-trained model.
    Args:
        asr_model: ASRModel instance
        cfg: config
    Returns:
        asr_model: ASRModel instance with updated decoder and vocabulary
    """
    if (
        hasattr(cfg.model.tokenizer, "update_tokenizer")
        and cfg.model.tokenizer.update_tokenizer
    ):
        if (
            hasattr(cfg.model.char_labels, "update_labels")
            and cfg.model.char_labels.update_labels
        ):
            raise ValueError(
                "Both `model.tokenizer.update_tokenizer` and `model.char_labels.update_labels` cannot be passed together"
            )
        else:
            asr_model = update_tokenizer(
                asr_model, cfg.model.tokenizer.dir, cfg.model.tokenizer.type
            )
    elif hasattr(cfg.model, "char_labels") and cfg.model.char_labels.update_labels:
        asr_model.change_vocabulary(new_vocabulary=cfg.model.char_labels.labels)
        logging.warning(
            "The vocabulary of the model has been updated with provided char labels."
        )
    else:
        logging.info("Reusing the vocabulary from the pre-trained model.")

    return asr_model


def update_tokenizer(asr_model, tokenizer_dir, tokenizer_type):
    """
    Updates the tokenizer of the model and also reinitializes the decoder if the vocabulary size
    of the new tokenizer differs from that of the loaded model.
    Args:
        asr_model: ASRModel instance
        tokenizer_dir: tokenizer directory
        tokenizer_type: tokenizer type
    Returns:
        asr_model: ASRModel instance with updated tokenizer and decoder
    """
    vocab_size = asr_model.tokenizer.vocab_size
    decoder = asr_model.decoder.state_dict()
    if hasattr(asr_model, "joint"):
        joint_state = asr_model.joint.state_dict()
    else:
        joint_state = None

    if tokenizer_dir is None:
        raise ValueError("dir must be specified if update_tokenizer is True")
    logging.info("Using the tokenizer provided through config")
    asr_model.change_vocabulary(
        new_tokenizer_dir=tokenizer_dir, new_tokenizer_type=tokenizer_type
    )
    if asr_model.tokenizer.vocab_size != vocab_size:
        logging.warning(
            "The vocabulary size of the new tokenizer differs from that of the loaded model. As a result, finetuning will proceed with the new vocabulary, and the decoder will be reinitialized."
        )
    else:
        asr_model.decoder.load_state_dict(decoder)
        if joint_state is not None:
            asr_model.joint.load_state_dict(joint_state)

    return asr_model


def setup_dataloaders(asr_model, cfg):
    """
    Sets up the training, validation and test dataloaders for the model.
    Args:
        asr_model: ASRModel instance
        cfg: config
    Returns:
        asr_model: ASRModel instance with updated dataloaders
    """
    cfg = model_utils.convert_model_config_to_dict_config(cfg)
    asr_model.setup_training_data(cfg.model.train_ds)
    asr_model.setup_multiple_validation_data(cfg.model.validation_ds)
    if (
        hasattr(cfg.model, "test_ds")
        and cfg.model.test_ds.manifest_filepath is not None
    ):
        asr_model.setup_multiple_test_data(cfg.model.test_ds)

    return asr_model


@hydra_runner(config_path="experimental/contextnet_rnnt", config_name="config_rnnt_bpe")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # asr_model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)
    # asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    print(cfg.get("init_from_pretrained_model"))
    asr_model = ASRModel.from_pretrained(
        model_name=cfg.get("init_from_pretrained_model")
    )

    asr_model = update_tokenizer(
        asr_model, cfg.model.tokenizer.dir, cfg.model.tokenizer.type
    )

    if cfg.model.freeze_encoder:
        asr_model.encoder.freeze()
        asr_model.encoder.apply(enable_bn_se)
        logging.info("Model encoder has been frozen")
    else:
        asr_model.encoder.unfreeze()
        logging.info("Model encoder has been un-frozen")

    # asr_model = check_vocabulary(asr_model, cfg)
    # asr_model = setup_dataloaders(asr_model, cfg)
    asr_model.setup_training_data(cfg.model.train_ds)
    asr_model.setup_validation_data(cfg.model.validation_ds)

    # asr_model.setup_optimization(cfg.model.optim)
    # if hasattr(cfg.model, "spec_augment") and cfg.model.spec_augment is not None:
    #     asr_model.spec_augment = ASRModel.from_config_dict(cfg.model.spec_augment)

    # asr_model.setup_training_data(cfg.model.train_ds)
    # asr_model.setup_validation_data(cfg.model.validation_ds)
    # asr_model.tokenizer_dir = cfg.model.tokenizer.dir
    # asr_model.change_vocabulary(
    #     new_tokenizer_dir=cfg.model.tokenizer.dir,
    #     new_tokenizer_type=cfg.model.tokenizer.type,
    # )
    # asr_model.set_trainer(trainer)

    trainer.fit(asr_model)

    if (
        hasattr(cfg.model, "test_ds")
        and cfg.model.test_ds.manifest_filepath is not None
    ):
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
