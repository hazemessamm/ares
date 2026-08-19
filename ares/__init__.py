from ares import preprocessing
from ares import tokenization
from ares import models
from ares.models import Ares, AresConfig
from ares.tokenization import AresProteinTokenizer

__all__ = [
    "Ares",
    "AresConfig",
    "AresProteinTokenizer",
    "models",
    "preprocessing",
    "tokenization",
]


def _register_auto_classes() -> None:
    """Make Ares resolvable through HuggingFace ``Auto*`` classes.

    Importing ``ares`` is enough for ``AutoModelForMaskedLM.from_pretrained(...)``
    to load an Ares checkpoint, without ``trust_remote_code``.
    """
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForMaskedLM,
        AutoTokenizer,
    )

    # Re-registration raises ValueError, which is harmless on repeat imports.
    try:
        AutoConfig.register(AresConfig.model_type, AresConfig)
    except ValueError:
        pass

    for auto_class in (AutoModel, AutoModelForMaskedLM):
        try:
            auto_class.register(AresConfig, Ares)
        except ValueError:
            pass

    try:
        AutoTokenizer.register(AresConfig, fast_tokenizer_class=AresProteinTokenizer)
    except ValueError:
        pass

    # Emit the auto_map / code files on save_pretrained too, so a checkpoint
    # stays loadable via trust_remote_code when ares is not installed.
    Ares.register_for_auto_class("AutoModelForMaskedLM")
    AresConfig.register_for_auto_class("AutoConfig")
    AresProteinTokenizer.register_for_auto_class("AutoTokenizer")


_register_auto_classes()
