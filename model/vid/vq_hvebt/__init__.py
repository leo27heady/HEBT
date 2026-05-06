"""
VQ-HVEBT: Vector-Quantized Hierarchical Video Energy-Based Transformer.

Public API
----------
from model.vid.vq_hvebt import (
    VQHVEBTConfig, VQStageConfig, VQCodebookConfig,
    VQHVEBTModel, VQHVEBTOutput, StageForwardResult,
    VectorQuantizer, QuantizerOutput,
    VQHVEBTStage,
    VQClipBackbone,
)
"""
from model.vid.vq_hvebt.config import (
    VQCodebookConfig,
    VQHVEBTConfig,
    VQStageConfig,
    _default_stages,
)
from model.vid.vq_hvebt.hierarchy import (
    StageForwardResult,
    VQHVEBTModel,
    VQHVEBTOutput,
)
from model.vid.vq_hvebt.quantizer import QuantizerOutput, VectorQuantizer
from model.vid.vq_hvebt.stage_predictor import VQHVEBTStage
from model.vid.vq_hvebt.clip_backbone import VQClipBackbone
from model.vid.vq_hvebt import losses

__all__ = [
    # config
    "VQCodebookConfig",
    "VQStageConfig",
    "VQHVEBTConfig",
    "_default_stages",
    # model
    "VQHVEBTModel",
    "VQHVEBTOutput",
    "StageForwardResult",
    # quantizer
    "VectorQuantizer",
    "QuantizerOutput",
    # stage predictor
    "VQHVEBTStage",
    # backbone
    "VQClipBackbone",
    # losses module
    "losses",
]
