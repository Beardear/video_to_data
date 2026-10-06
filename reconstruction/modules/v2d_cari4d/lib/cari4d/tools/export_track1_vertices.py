"""Internal decoder subprocess for the Track 1 adapter; no image inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lib_mhr import MHRLayer, MHR_PARAM_DIMS
from lib_mhr.track1 import track1_validate_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    # Only load our own CARI4D pipeline artifacts, which contain NumPy/pickle.
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    motion = track1_validate_bundle(bundle, args.expected_frames)
    output = Path(args.output)
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=Path(args.weights) / "sam3d_body",
                                   device=args.device)
    decoder_identity = layer.decoder_identity()
    vertices = np.lib.format.open_memmap(output / "human_vertices.npy", mode="w+",
                                         dtype=np.float32, shape=(args.expected_frames, 18439, 3))
    with torch.inference_mode():
        for start in range(0, args.expected_frames, args.batch_size):
            stop = min(start + args.batch_size, args.expected_frames)
            params = {key: torch.as_tensor(bundle["pr"][key][start:stop],
                                          device=args.device, dtype=torch.float32)
                      for key in MHR_PARAM_DIMS}
            decoded = layer.mhr_forward_vertices(params).detach().cpu().numpy()
            if decoded.shape != (stop - start, 18439, 3) or not np.isfinite(decoded).all():
                raise ValueError(f"Invalid lod1 MHR vertices at frames {start}:{stop}")
            # Already in the same metre/axis convention as the official fitter:
            # diag(1,-1,-1) @ MHR(...)/100 + mhr_trans. No second flip or scaling.
            vertices[start:stop] = decoded
    vertices.flush()
    del vertices
    np.savez(output / "object_motion.npz", rotation=motion.rotation, translation=motion.translation)
    if layer.decoder_identity() != decoder_identity:
        raise ValueError("MHR decoder assets changed while decoding")
    (output / "decoder.json").write_text(json.dumps({
        **decoder_identity, "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "device": args.device,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
