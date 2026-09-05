from __future__ import annotations

from pathlib import Path
import shutil
import sys
import zipfile

from huggingface_hub import (
    hf_hub_download,
    snapshot_download,
)


REPO_ID = "lj1995/VoiceConversionWebUI"


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: setup_rvc_training_assets.py <RVC_ROOT>"
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
        "[1/2] Downloading RVC v2 40k pretrained models..."
    )

    snapshot_download(
        repo_id=REPO_ID,
        revision="main",
        allow_patterns=[
            "pretrained_v2/f0G40k.pth",
            "pretrained_v2/f0D40k.pth",
        ],
        local_dir=str(
            assets
        ),
    )

    print(
        "[2/2] Downloading mute training dataset..."
    )

    download_dir = (
        root / ".model-downloads"
    )
    download_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mute_zip = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename="mute.zip",
            revision="main",
            local_dir=str(
                download_dir
            ),
        )
    )

    logs_dir = root / "logs"
    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        mute_zip,
        "r",
    ) as zf:
        zf.extractall(
            logs_dir
        )

    required = [
        assets
        / "pretrained_v2"
        / "f0G40k.pth",
        assets
        / "pretrained_v2"
        / "f0D40k.pth",
        logs_dir
        / "mute"
        / "0_gt_wavs"
        / "mute40k.wav",
        logs_dir
        / "mute"
        / "3_feature768"
        / "mute.npy",
        logs_dir
        / "mute"
        / "2a_f0"
        / "mute.wav.npy",
        logs_dir
        / "mute"
        / "2b-f0nsf"
        / "mute.wav.npy",
    ]

    missing = [
        path
        for path in required
        if not path.is_file()
    ]

    if missing:
        print(
            "[ERROR] Required training files are missing:"
        )
        for path in missing:
            print(
                " -",
                path,
            )
        return 1

    print(
        "[OK] RVC training assets are ready."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
