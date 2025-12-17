#!/usr/bin/env python3
# extract_latents_wav2vec2.py
"""poetry run python ssl_latent_features_extraction.py \
  --base-folder /Users/.../ \
  --output-dir /Users/.../ \
  --dtype float16
  
  """
import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel


DEFAULT_MODELS = [
    "facebook/wav2vec2-large",
    "facebook/mms-300m",
    "facebook/wav2vec2-large-xlsr-53",
    "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "Finnish-NLP/wav2vec2-xlsr-300m-finnish-1m",
]


def sanitize_model_id(model_id: str) -> str:
    # safe folder/file name
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)


def load_audio_mono_16k(wav_path: Path, target_sr: int = 16000) -> np.ndarray:
    wav, sr = torchaudio.load(str(wav_path))  # (channels, time)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    wav = wav.squeeze(0)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    # Return float32 numpy in [-1, 1]
    return wav.numpy().astype(np.float32)


@torch.no_grad()
def extract_hidden_states(
    audio_16k: np.ndarray,
    feature_extractor,
    model,
    device: torch.device,
) -> list[np.ndarray]:
    inputs = feature_extractor(
        audio_16k,
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_values = inputs["input_values"].to(device)

    outputs = model(
        input_values,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states  # tuple: (layer0, layer1, ...)

    # Convert each layer: (T, D) numpy
    layers_np = []
    for layer in hidden_states:
        emb = layer.squeeze(0).cpu().numpy()  # (time, dim)
        layers_np.append(emb)
    return layers_np


def save_npz(
    save_path: Path,
    layers: list[np.ndarray],
    metadata: dict,
    dtype: str = "float32",
    compressed: bool = True,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if dtype == "float16":
        layers = [x.astype(np.float16) for x in layers]
    else:
        layers = [x.astype(np.float32) for x in layers]

    arrays = {f"layer_{i:02d}": arr for i, arr in enumerate(layers)}
    arrays["metadata_json"] = np.array([json.dumps(metadata)], dtype=object)

    if compressed:
        np.savez_compressed(str(save_path), **arrays)
    else:
        np.savez(str(save_path), **arrays)


def iter_wavs(base_folder: Path):
    # expects: base_folder/<speaker>/wav/*.wav (como tu estructura)
    for speaker_dir in sorted(base_folder.iterdir()):
        wav_dir = speaker_dir / "wav"
        if not wav_dir.is_dir():
            continue
        for wav_path in sorted(wav_dir.glob("*.wav")):
            yield speaker_dir.name, wav_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-folder", type=str, required=True,
                        help="Folder that contains speaker subfolders with a 'wav/' directory.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output root dir. Default: <base-folder>/features_latents")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                        help="List of HuggingFace model IDs.")
    parser.add_argument("--revision", type=str, default=None,
                        help="Optional HF revision/commit hash for reproducibility.")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / mps / cpu. Default: auto.")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16"], default="float16",
                        help="float16 saves space; float32 is safer.")
    parser.add_argument("--no-compress", action="store_true",
                        help="Disable npz compression (faster, bigger).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Root directory where latent features will be saved. Default: <base-folder>/features_latents")
    args = parser.parse_args()

    base_folder = Path(args.base_folder).expanduser().resolve()
    if args.output_dir is not None:
        out_root = Path(args.output_dir).expanduser().resolve()
    else:
        out_root = base_folder / "features_latents"

    out_root.mkdir(parents=True, exist_ok=True)

    # device auto
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Global metadata for reproducibility
    run_meta = {
        "torch_version": torch.__version__,
        "device": str(device),
    }

    for model_id in args.models:
        safe_id = sanitize_model_id(model_id)
        print(f"\n=== Model: {model_id} (folder: {safe_id}) ===")

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=args.revision)
        model = AutoModel.from_pretrained(model_id, revision=args.revision)
        model.to(device)
        model.eval()

        # Save model/config metadata
        model_meta = {
            **run_meta,
            "model_id": model_id,
            "revision": args.revision,
            "transformers_config": model.config.to_dict(),
        }
        meta_path = out_root / safe_id / "model_metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

        for speaker, wav_path in iter_wavs(base_folder):
            utt_id = wav_path.stem
            rel_save = Path(safe_id) / speaker / f"{utt_id}.npz"
            save_path = out_root / rel_save

            if save_path.exists():
                continue

            audio = load_audio_mono_16k(wav_path)
            layers = extract_hidden_states(audio, feature_extractor, model, device)

            per_file_meta = {
                "speaker": speaker,
                "utt_id": utt_id,
                "wav_path": str(wav_path),
                "sr": 16000,
                "n_layers": len(layers),
                "layer_shapes": [list(x.shape) for x in layers],
                "dtype": args.dtype,
            }

            save_npz(
                save_path=save_path,
                layers=layers,
                metadata=per_file_meta,
                dtype=args.dtype,
                compressed=not args.no_compress,
            )
            print(f"saved: {save_path}  (layers={len(layers)})")


if __name__ == "__main__":
    main()
