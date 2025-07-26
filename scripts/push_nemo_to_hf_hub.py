from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

asr_model = EncDecHybridRNNTCTCBPEModel.restore_from(
    "/home/user/.cache/nemo_experiments/Speech_To_Text_Finetuning/2025-07-26_08-55-36/checkpoints/Speech_To_Text_Finetuning.nemo"
)

asr_model.push_to_hf_hub(
    "hsekhalilian/Speech_To_Text_Finetuning_03_no_punc_with_encoder", private=True
)
