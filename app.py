#!/usr/bin/env python3
"""
HuggingFace Space entry point for OmniVoice demo.

"""

import logging
import os
import tempfile
from typing import Any, Dict

try:
    import spaces
    _USING_ZERO_GPU = True
except ImportError:
    _USING_ZERO_GPU = False

import torch
import torchaudio

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.cli.demo import build_demo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
CHECKPOINT = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")

model = None
if not _USING_ZERO_GPU:
    # Non-ZeroGPU: load model at startup on the best available device
    logger.info(f"Loading model from {CHECKPOINT} on {DEVICE} ...")
    model = OmniVoice.from_pretrained(
        CHECKPOINT,
        device_map=DEVICE,
        dtype=torch.float16,
        load_asr=True,
    )
    logger.info("Model loaded on %s.", DEVICE)
else:
    logger.info("ZeroGPU mode: model will be loaded inside @spaces.GPU() function.")

sampling_rate = 16000  # fallback; will be overwritten after model loads


# ---------------------------------------------------------------------------
# Generation logic (outside build_demo so we can wrap with spaces.GPU)
# ---------------------------------------------------------------------------


def _gen_core(
    text,
    language,
    ref_audio,
    instruct,
    num_step,
    guidance_scale,
    denoise,
    speed,
    duration,
    preprocess_prompt,
    postprocess_output,
    mode,
    ref_text=None,
):
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None

    kw: Dict[str, Any] = dict(
        text=text.strip(), language=lang, generation_config=gen_config
    )

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if not ref_audio:
            return None, "Please upload a reference audio."
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

    if mode == "design":
        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

    try:
        out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        audio = model.generate(**kw)
        torchaudio.save(out_path, audio[0], sampling_rate)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    return out_path, "Done."


# ---------------------------------------------------------------------------
# ZeroGPU wrapper
# ---------------------------------------------------------------------------
generate_fn = None
if _USING_ZERO_GPU:
    @spaces.GPU()
    def generate_fn(*args, **kwargs):
        # Lazy-load model on first call (inside GPU context)
        global model, sampling_rate
        if model is None:
            logger.info(f"Loading model from {CHECKPOINT} on cuda (ZeroGPU) ...")
            model = OmniVoice.from_pretrained(
                CHECKPOINT,
                device_map="cuda",
                dtype=torch.float16,
                load_asr=True,
            )
            sampling_rate = model.sampling_rate
            logger.info("Model loaded on cuda (ZeroGPU).")
        return _gen_core(*args, **kwargs)

    logger.info("Using spaces.GPU() wrapper.")

# ---------------------------------------------------------------------------
# Build and launch demo — reuses the full UI from omnivoice.cli.demo
# ---------------------------------------------------------------------------
demo = build_demo(model, CHECKPOINT, generate_fn=generate_fn)

if __name__ == "__main__":
    demo.queue().launch()
