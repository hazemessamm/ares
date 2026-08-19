from tokenizers.models import BPE
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing
from ares.tokenization import constants


class AresProteinTokenizer(PreTrainedTokenizerFast):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        cls_token="<cls>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
        **kwargs,
    ):
        tokenizer = Tokenizer(
            BPE(vocab=constants.AA_VOCAB, merges=[], unk_token="<unk>"),
        )
        special_tokens = [
            cls_token,
            pad_token,
            mask_token,
            eos_token,
            unk_token,
        ]
        additional_special_tokens = []

        tokenizer.add_special_tokens(special_tokens)

        tokenizer.post_processor = TemplateProcessing(
            single="<cls> $A <eos>",
            special_tokens=[
                ("<cls>", tokenizer.token_to_id("<cls>")),
                ("<eos>", tokenizer.token_to_id("<eos>")),
            ],
        )

        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
