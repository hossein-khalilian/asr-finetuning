"""
This script can used to fine-tune a speech-to-text model of any instance type when users want to 
fine-tune an existing model without changing its core architecture but may change the tokenizer. 
One can mention the pretrained model in two ways: 
1) `init_from_nemo_model` or 
2) `init_from_pretrained_model` in the configuration.

****************************************************************************************
This script is mainly intended for changing the dataset, optim, spec_augment, vocabulary/tokenizer of the model.
To update the model architecture in conjunction with other modifications, 
it is advisable to use the primary 'speech_to_text_rnnt/ctc_*.py' script.
****************************************************************************************

Note: To create a single script for all model types, we currently only support two types of 
initializations:
1) `init_from_nemo_model`, and
2) `init_from_pretrained_model`,
but not `init_from_ptl_ckpt`.

To train with prior base model tokenizer keep `model.tokenizer.update_tokenizer` as false else
make it true and provide tokenizer dir along with tokenizer type.

To fine-tune the model, use the following commands:

For initialization from a NEMO model:
```sh
python <NEMO_ROOT>/examples/asr/speech_to_text_finetune.py \
    init_from_nemo_model=<path_to_nemo_model>
```

For initialization from a pretrained model:
```sh
python <NEMO_ROOT>/examples/asr/speech_to_text_finetune.py \
    init_from_pretrained_model=<pretrained_model_name>
```
"""

import lightning.pytorch as pl
import torch.nn as nn
from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
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


def get_base_model(trainer, cfg):
    """
    Returns the base model to be fine-tuned.
    Currently supports two types of initializations:
    1) `init_from_nemo_model`, and
    2) `init_from_pretrained_model`.
    Args:
        trainer: PyTorch Lightning Trainer
        cfg: config
    Returns:
        asr_model: ASRModel instance
    """
    asr_model = None
    nemo_model_path = cfg.get("init_from_nemo_model", None)
    pretrained_name = cfg.get("init_from_pretrained_model", None)
    if nemo_model_path is not None and pretrained_name is not None:
        raise ValueError(
            "Only pass `init_from_nemo_model` or `init_from_pretrained_model` but not both"
        )
    elif nemo_model_path is None and pretrained_name is None:
        raise ValueError(
            "Both `init_from_nemo_model` and `init_from_pretrained_model cannot be None, should pass atleast one of them"
        )

    if nemo_model_path is not None:
        asr_model = ASRModel.restore_from(restore_path=nemo_model_path)
    elif pretrained_name is not None:
        asr_model = ASRModel.from_pretrained(model_name=pretrained_name)

    if cfg.model.freeze_encoder:
        asr_model.encoder.freeze()
        asr_model.encoder.apply(enable_bn_se)
        logging.info("Model encoder has been frozen")
    else:
        asr_model.encoder.unfreeze()
        logging.info("Model encoder has been un-frozen")

    asr_model.set_trainer(trainer)

    return asr_model


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


@hydra_runner(config_path="conf/asr_finetune", config_name="speech_to_text_finetune")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    if hasattr(cfg, "init_from_ptl_ckpt") and cfg.init_from_ptl_ckpt is not None:
        raise NotImplementedError(
            "Currently for simplicity of single script for all model types, we only support `init_from_nemo_model` and `init_from_pretrained_model`"
        )

    asr_model = get_base_model(trainer, cfg)

    # Check vocabulary type and update if needed
    asr_model = check_vocabulary(asr_model, cfg)

    # Setup Data
    asr_model = setup_dataloaders(asr_model, cfg)

    # Setup Optimizer
    asr_model.setup_optimization(cfg.model.optim)

    # Setup SpecAug
    if hasattr(cfg.model, "spec_augment") and cfg.model.spec_augment is not None:
        asr_model.spec_augment = ASRModel.from_config_dict(cfg.model.spec_augment)

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
