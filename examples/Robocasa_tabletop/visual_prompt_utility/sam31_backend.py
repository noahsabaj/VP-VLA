# SAM 3.1 backend for sam3_server.py
#
# SAM 3.1 (facebook/sam3.1, sam3.1_multiplex.pt) has no Hugging Face
# `transformers` port, so it loads through Meta's own code
# (github.com/facebookresearch/sam3). Meta only ships it as a video predictor;
# this module builds the predictor's detector half (a Sam3Image with the
# three-head SAM 3.1 neck) on its own and runs it on single images with Meta's
# Sam3Processor, the same way the SAM 3 image model is used.
#
# The `sam3` package needs Python >= 3.10, numpy < 2 and setuptools < 81, which
# clash with the transformers-5 `sam3` env, so it runs from its own env
# (see build_env_sam31.sh).

import logging
from typing import Optional

import numpy as np
import torch

SAM31_HF_REPO = "facebook/sam3.1"
SAM31_CKPT_NAME = "sam3.1_multiplex.pt"


def state_to_result(state: dict, mask_threshold: float) -> dict:
    """Convert a Sam3Processor inference state into the server's reply format.

    Sam3Processor already filters instances by confidence and returns boxes in
    pixel (x1, y1, x2, y2) format. Its own `masks` are fixed at 0.5, so they are
    re-thresholded here from `masks_logits` (post-sigmoid probabilities).
    """
    probs = state["masks_logits"]  # (N, 1, H, W), values in [0, 1]
    masks = (probs > mask_threshold).squeeze(1)
    boxes = state["boxes"]
    scores = state["scores"]
    return {
        "masks": masks.cpu().numpy().astype(np.uint8),  # (N, H, W)
        "boxes": boxes.float().cpu().numpy().astype(np.float32),  # (N, 4) x1, y1, x2, y2
        "scores": scores.float().cpu().numpy().astype(np.float32),  # (N,)
        "num_masks": int(masks.shape[0]),
    }


def _build_sam31_detector(use_fa3: bool = False):
    """Build the SAM 3.1 detector exactly as build_sam3_multiplex_video_predictor does."""
    from importlib.resources import files

    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.vl_combiner import SAM3VLBackboneTri
    from sam3.model_builder import (
        _create_dot_product_scoring,
        _create_geometry_encoder,
        _create_multiplex_tri_backbone,
        _create_sam3_transformer,
        _create_segmentation_head,
        _create_text_encoder,
    )

    bpe_path = str(files("sam3") / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    tri_neck = _create_multiplex_tri_backbone(compile_mode=None, use_fa3=use_fa3, use_rope_real=False)
    backbone = SAM3VLBackboneTri(scalp=0, visual=tri_neck, text=_create_text_encoder(bpe_path))
    return Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=backbone,
        transformer=_create_sam3_transformer(use_fa3=use_fa3),
        segmentation_head=_create_segmentation_head(use_fa3=use_fa3),
        semantic_segmentation_head=None,
        input_geometry_encoder=_create_geometry_encoder(),
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=_create_dot_product_scoring(),
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )


def _load_detector_weights(detector: torch.nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    # Hugging Face checkpoints use "detector." / "tracker."; older local ones "sam3_model." / "sam2_predictor.".
    prefixes = ("detector.", "sam3_model.")
    det_ckpt = {k.split(".", 1)[1]: v for k, v in ckpt.items() if k.startswith(prefixes)}
    if not det_ckpt:
        raise RuntimeError(f"No detector weights found in {checkpoint_path}")
    missing, unexpected = detector.load_state_dict(det_ckpt, strict=False)
    logging.info(f"Loaded {len(det_ckpt)} detector tensors from {checkpoint_path}")
    if missing:
        logging.warning(f"SAM3.1 detector: {len(missing)} missing keys, e.g. {missing[:5]}")
    if unexpected:
        logging.warning(f"SAM3.1 detector: {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")


class SAM31Model:
    """SAM 3.1 detector with the same segment() interface as sam3_server.SAM3Model."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        from sam3.model.sam3_image_processor import Sam3Processor

        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download

            checkpoint_path = hf_hub_download(repo_id=SAM31_HF_REPO, filename=SAM31_CKPT_NAME)

        self.device = device

        logging.info(f"Loading SAM3.1 detector from {checkpoint_path}")
        detector = _build_sam31_detector()
        _load_detector_weights(detector, checkpoint_path)
        detector = detector.to(device).eval()

        # Only the detection head's features are needed; skip the tracker's two neck heads.
        forward_image = detector.backbone.forward_image
        detector.backbone.forward_image = lambda samples: forward_image(
            samples, need_interactive_out=False, need_propagation_out=False
        )

        self.model = detector
        self.processor = Sam3Processor(detector, device=device)
        logging.info("SAM3.1 model loaded successfully")

    def segment(
        self,
        image: np.ndarray,
        text_prompt: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> dict:
        """Same contract as SAM3Model.segment: (N, H, W) uint8 masks, (N, 4) xyxy boxes, (N,) scores."""
        from PIL import Image

        pil_image = Image.fromarray(image).convert("RGB")
        self.processor.confidence_threshold = threshold
        # Meta's SAM 3.1 code only runs under bf16 autocast (its own predictors wrap every call in it);
        # in plain fp32 some layers get bf16 inputs and fail with a dtype mismatch.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(pil_image)
            state = self.processor.set_text_prompt(prompt=text_prompt, state=state)
        return state_to_result(state, mask_threshold)
