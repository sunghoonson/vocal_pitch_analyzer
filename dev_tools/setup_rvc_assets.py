from __future__ import annotations

from pathlib import Path
import sys

from huggingface_hub import (
    hf_hub_download,
    snapshot_download,
)


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: setup_rvc_assets.py <RVC_ROOT>"
        )
        return 2

    root = Path(
        sys.argv[1]
    ).resolve()

    assets = root / "assets"
    assets.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[INFO] Downloading HuBERT..."
    )
    snapshot_download(
        repo_id="lj1995/VoiceConversionWebUI",
        revision="main",
        allow_patterns=[
            "hubert_base/*",
        ],
        local_dir=str(assets),
    )

    rmvpe_dir = (
        assets / "rmvpe"
    )
    rmvpe_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[INFO] Downloading RMVPE..."
    )
    hf_hub_download(
        repo_id="lj1995/VoiceConversionWebUI",
        filename="rmvpe.pt",
        revision="main",
        local_dir=str(
            rmvpe_dir
        ),
    )

    hubert = (
        assets
        / "hubert_base"
        / "pytorch_model.bin"
    )
    rmvpe = (
        assets
        / "rmvpe"
        / "rmvpe.pt"
    )

    if not hubert.is_file():
        raise FileNotFoundError(
            hubert
        )

    if not rmvpe.is_file():
        raise FileNotFoundError(
            rmvpe
        )

    print(
        "[OK] HuBERT:",
        hubert,
    )
    print(
        "[OK] RMVPE:",
        rmvpe,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
