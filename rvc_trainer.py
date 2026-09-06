from __future__ import annotations

# RVC_TRAINING_MODULE_LAUNCH_HOTFIX

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json
import os
import re
import shutil
import subprocess
from datetime import datetime


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, str], None]


class RVCTrainingError(RuntimeError):
    pass


@dataclass(slots=True)
class RVCTrainingResult:
    model_path: Path
    index_path: Path | None
    experiment_dir: Path


@dataclass(slots=True)
class RVCExperimentInfo:
    name: str
    experiment_dir: Path
    dataset_dir: Path | None
    dataset_file_count: int | None
    dataset_path_exists: bool
    generator_checkpoints: int
    discriminator_checkpoints: int
    estimated_epoch: int | None
    final_model_path: Path | None
    index_path: Path | None
    modified_timestamp: float

    @property
    def has_checkpoint_pair(self) -> bool:
        return (
            self.generator_checkpoints > 0
            and self.discriminator_checkpoints > 0
        )


# V29_FINETUNE_LEAD_LINEAGE_HOTFIX
# V23_RVC_EXPERIMENT_BROWSER_PATCH
# V23_RVC_FINETUNE_INDEX_REFRESH_HOTFIX
# V22_RVC_FINETUNE_PATCH
TRAINING_MODE_NEW = "new"
TRAINING_MODE_RESUME = "resume"
TRAINING_MODE_FINETUNE_ADD = "finetune_add"
TRAINING_MODES = {
    TRAINING_MODE_NEW,
    TRAINING_MODE_RESUME,
    TRAINING_MODE_FINETUNE_ADD,
}

DERIVED_TRAINING_DIRS = (
    "0_gt_wavs",
    "1_16k_wavs",
    "2a_f0",
    "2b-f0nsf",
    "3_feature768",
)

DATASET_MANIFEST_NAME = "vocal_pitch_dataset_manifest.json"
LEAD_CLEAN_MANIFEST_NAME = ".rvc_lead_clean_manifest.json"


SUPPORTED_DATASET_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".mp4",
    ".mkv",
    ".webm",
}


def project_root() -> Path:
    return Path(__file__).resolve().parent


def rvc_root() -> Path:
    return project_root() / "tools" / "rvc"


def rvc_python() -> Path:
    return project_root() / ".venv_rvc" / "Scripts" / "python.exe"


def training_log_path() -> Path:
    return project_root() / "logs" / "rvc_training_last.log"


def trained_models_root() -> Path:
    return project_root() / "rvc_models"


def finetune_backups_root() -> Path:
    return project_root() / "rvc_finetune_backups"


def experiment_dir(
    experiment_name: str,
) -> Path:
    return (
        rvc_root()
        / "logs"
        / validate_experiment_name(
            experiment_name
        )
    )


def experiment_checkpoint_files(
    experiment_name: str,
) -> tuple[list[Path], list[Path]]:
    exp_dir = experiment_dir(
        experiment_name
    )

    generators = sorted(
        exp_dir.glob("G_*.pth"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    discriminators = sorted(
        exp_dir.glob("D_*.pth"),
        key=lambda path: path.stat().st_mtime_ns,
    )

    return (
        generators,
        discriminators,
    )


def experiment_status_text(
    experiment_name: str,
) -> str:
    try:
        exp_name = validate_experiment_name(
            experiment_name
        )
    except RVCTrainingError:
        return (
            "모델 이름을 입력하면 기존 학습 상태를 확인합니다."
        )

    exp_dir = (
        rvc_root()
        / "logs"
        / exp_name
    )

    generators = list(
        exp_dir.glob("G_*.pth")
    )
    discriminators = list(
        exp_dir.glob("D_*.pth")
    )

    final_model = (
        trained_models_root()
        / exp_name
        / f"{exp_name}.pth"
    )

    if generators and discriminators:
        final_text = (
            " / 최종 모델 있음"
            if final_model.is_file()
            else ""
        )

        return (
            f"기존 학습 체크포인트 발견 "
            f"(G {len(generators)} / D {len(discriminators)})"
            f"{final_text} - 이어학습/파인튜닝 가능"
        )

    if exp_dir.exists():
        return (
            "동일 이름의 실험 폴더는 있지만 완전한 G/D 체크포인트가 없습니다. "
            "새 학습으로 재구축하거나 다른 이름을 권장합니다."
        )

    return (
        "기존 체크포인트 없음 - 새 모델 학습용 이름입니다."
    )


def _safe_path_exists(
    path: Path | None,
) -> bool:
    if path is None:
        return False

    try:
        return path.is_dir()
    except OSError:
        return False


def _read_experiment_manifest(
    exp_dir: Path,
) -> tuple[Path | None, int | None]:
    path = (
        exp_dir
        / DATASET_MANIFEST_NAME
    )

    if not path.is_file():
        return (
            None,
            None,
        )

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        raw_dataset = str(
            data.get(
                "dataset_dir",
                "",
            )
        ).strip()

        dataset = (
            Path(
                raw_dataset
            ).expanduser()
            if raw_dataset
            else None
        )

        files = data.get(
            "files",
            [],
        )

        count = (
            len(files)
            if isinstance(
                files,
                list,
            )
            else None
        )

        return (
            dataset,
            count,
        )

    except Exception:
        return (
            None,
            None,
        )


def _infer_pre_v22_dataset_count(
    exp_dir: Path,
) -> int | None:
    gt_dir = (
        exp_dir
        / "0_gt_wavs"
    )

    if not gt_dir.is_dir():
        return None

    prefixes: set[str] = set()

    try:
        for path in gt_dir.glob(
            "*.wav"
        ):
            match = re.match(
                r"^(\d+)_\d+\.wav$",
                path.name,
            )

            if match:
                prefixes.add(
                    match.group(1)
                )
    except OSError:
        return None

    return (
        len(prefixes)
        if prefixes
        else None
    )


def _estimated_experiment_epoch(
    experiment_name: str,
) -> int | None:
    weights_dir = (
        rvc_root()
        / "assets"
        / "weights"
    )

    if not weights_dir.is_dir():
        return None

    values: list[int] = []

    try:
        for path in weights_dir.glob(
            f"{experiment_name}*.pth"
        ):
            match = re.search(
                r"(?:^|_)e(\d+)(?:_|\.|$)",
                path.name,
                flags=re.IGNORECASE,
            )

            if match:
                values.append(
                    int(
                        match.group(1)
                    )
                )
    except OSError:
        return None

    return (
        max(values)
        if values
        else None
    )


def _find_experiment_final_model(
    experiment_name: str,
) -> Path | None:
    folder = (
        trained_models_root()
        / experiment_name
    )

    preferred = (
        folder
        / f"{experiment_name}.pth"
    )

    if preferred.is_file():
        return preferred

    if not folder.is_dir():
        return None

    try:
        candidates = sorted(
            folder.glob(
                "*.pth"
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return None

    return (
        candidates[0]
        if candidates
        else None
    )


def _find_experiment_index(
    experiment_name: str,
) -> Path | None:
    permanent = (
        trained_models_root()
        / experiment_name
    )

    if permanent.is_dir():
        try:
            candidates = sorted(
                permanent.glob(
                    "*.index"
                ),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        except OSError:
            candidates = []

        if candidates:
            return candidates[0]

    shared = (
        rvc_root()
        / "assets"
        / "indices"
    )

    if not shared.is_dir():
        return None

    try:
        candidates = sorted(
            shared.glob(
                f"*{experiment_name}*.index"
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return None

    return (
        candidates[0]
        if candidates
        else None
    )


def _experiment_modified_timestamp(
    paths: list[Path | None],
) -> float:
    values: list[float] = []

    for path in paths:
        if path is None:
            continue

        try:
            values.append(
                path.stat().st_mtime
            )
        except OSError:
            continue

    return (
        max(values)
        if values
        else 0.0
    )


def scan_training_experiments() -> list[RVCExperimentInfo]:
    logs_root = (
        rvc_root()
        / "logs"
    )

    if not logs_root.is_dir():
        return []

    results: list[RVCExperimentInfo] = []

    try:
        children = sorted(
            (
                path
                for path in logs_root.iterdir()
                if path.is_dir()
            ),
            key=lambda path: path.name.lower(),
        )
    except OSError:
        return []

    for exp_dir in children:
        name = exp_dir.name

        if (
            name.lower() == "mute"
            or name.startswith(".")
        ):
            continue

        try:
            generators = list(
                exp_dir.glob(
                    "G_*.pth"
                )
            )
            discriminators = list(
                exp_dir.glob(
                    "D_*.pth"
                )
            )
        except OSError:
            generators = []
            discriminators = []

        dataset_dir, dataset_count = (
            _read_experiment_manifest(
                exp_dir
            )
        )

        if dataset_count is None:
            dataset_count = (
                _infer_pre_v22_dataset_count(
                    exp_dir
                )
            )

        final_model = (
            _find_experiment_final_model(
                name
            )
        )
        index_path = (
            _find_experiment_index(
                name
            )
        )

        # Ignore unrelated/empty RVC log folders.
        has_relevant_data = bool(
            generators
            or discriminators
            or dataset_dir is not None
            or dataset_count is not None
            or final_model is not None
            or index_path is not None
            or (
                exp_dir
                / "config.json"
            ).is_file()
            or (
                exp_dir
                / "filelist.txt"
            ).is_file()
        )

        if not has_relevant_data:
            continue

        results.append(
            RVCExperimentInfo(
                name=name,
                experiment_dir=exp_dir,
                dataset_dir=dataset_dir,
                dataset_file_count=dataset_count,
                dataset_path_exists=(
                    _safe_path_exists(
                        dataset_dir
                    )
                ),
                generator_checkpoints=len(
                    generators
                ),
                discriminator_checkpoints=len(
                    discriminators
                ),
                estimated_epoch=(
                    _estimated_experiment_epoch(
                        name
                    )
                ),
                final_model_path=final_model,
                index_path=index_path,
                modified_timestamp=(
                    _experiment_modified_timestamp(
                        [
                            exp_dir,
                            final_model,
                            index_path,
                            *generators,
                            *discriminators,
                        ]
                    )
                ),
            )
        )

    results.sort(
        key=lambda item: (
            -item.modified_timestamp,
            item.name.lower(),
        )
    )

    return results


def get_training_experiment(
    experiment_name: str,
) -> RVCExperimentInfo | None:
    name = str(
        experiment_name
    ).strip()

    if not name:
        return None

    for info in scan_training_experiments():
        if info.name == name:
            return info

    return None


def experiment_list_label(
    info: RVCExperimentInfo,
) -> str:
    epoch_text = (
        f"{info.estimated_epoch}ep"
        if info.estimated_epoch is not None
        else "?ep"
    )

    data_text = (
        str(
            info.dataset_file_count
        )
        if info.dataset_file_count is not None
        else "?"
    )

    checkpoint_text = (
        "CKPT✓"
        if info.has_checkpoint_pair
        else "CKPT-"
    )
    model_text = (
        "MODEL✓"
        if info.final_model_path is not None
        else "MODEL-"
    )
    index_text = (
        "INDEX✓"
        if info.index_path is not None
        else "INDEX-"
    )

    return (
        f"{info.name} | {epoch_text} | data {data_text} | "
        f"{checkpoint_text} {model_text} {index_text}"
    )


def experiment_detail_text(
    info: RVCExperimentInfo,
) -> str:
    checkpoint_text = (
        f"G {info.generator_checkpoints} / D {info.discriminator_checkpoints}"
        if (
            info.generator_checkpoints
            or info.discriminator_checkpoints
        )
        else "없음"
    )

    epoch_text = (
        str(
            info.estimated_epoch
        )
        if info.estimated_epoch is not None
        else "중간 weight 이름에서 확인 불가"
    )

    if info.dataset_dir is None:
        dataset_text = (
            "경로 기록 없음"
        )
    else:
        exists_text = (
            "경로 정상"
            if info.dataset_path_exists
            else "현재 경로를 찾을 수 없음"
        )
        dataset_text = (
            f"{info.dataset_dir} "
            f"({exists_text})"
        )

    count_text = (
        str(
            info.dataset_file_count
        )
        if info.dataset_file_count is not None
        else "알 수 없음"
    )

    model_text = (
        str(
            info.final_model_path
        )
        if info.final_model_path is not None
        else "없음"
    )

    index_text = (
        str(
            info.index_path
        )
        if info.index_path is not None
        else "없음"
    )

    return (
        f"실험: {info.name}\n"
        f"체크포인트: {checkpoint_text} / 추정 저장 epoch: {epoch_text}\n"
        f"학습 데이터: {dataset_text} / 기록 파일 수: {count_text}\n"
        f"변환 모델: {model_text}\n"
        f"Feature Index: {index_text}"
    )


def training_assets_status() -> tuple[bool, str]:
    checks = [
        (
            rvc_python(),
            ".venv_rvc",
        ),
        (
            rvc_root() / "train" / "preprocess.py",
            "RVC training code",
        ),
        (
            rvc_root() / "assets" / "hubert_base" / "pytorch_model.bin",
            "HuBERT",
        ),
        (
            rvc_root() / "assets" / "rmvpe" / "rmvpe.pt",
            "RMVPE",
        ),
        (
            rvc_root() / "assets" / "pretrained_v2" / "f0G40k.pth",
            "RVC pretrained G",
        ),
        (
            rvc_root() / "assets" / "pretrained_v2" / "f0D40k.pth",
            "RVC pretrained D",
        ),
        (
            rvc_root() / "logs" / "mute" / "0_gt_wavs" / "mute40k.wav",
            "RVC mute dataset",
        ),
        (
            rvc_root() / "logs" / "mute" / "3_feature768" / "mute.npy",
            "RVC mute HuBERT feature",
        ),
    ]

    missing = [
        label
        for path, label in checks
        if not path.is_file()
    ]

    if missing:
        return (
            False,
            "RVC 학습 준비 미완료: "
            + ", ".join(missing)
            + "\nSETUP_RVC_TRAINING_ASSETS.bat을 실행하세요.",
        )

    return (
        True,
        "RVC 학습 준비 완료 - CUDA/RMVPE/HuBERT/pretrained/mute",
    )


def validate_experiment_name(name: str) -> str:
    value = str(name).strip()

    if not value:
        raise RVCTrainingError(
            "실험/모델 이름을 입력하세요."
        )

    if not re.fullmatch(
        r"[A-Za-z0-9_-]+",
        value,
    ):
        raise RVCTrainingError(
            "실험/모델 이름은 영문, 숫자, _ , - 만 사용할 수 있습니다."
        )

    return value


def dataset_files(
    folder: str | Path,
) -> list[Path]:
    root = Path(folder).expanduser().resolve()

    if not root.is_dir():
        return []

    return sorted(
        [
            p
            for p in root.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in SUPPORTED_DATASET_EXTENSIONS
            )
        ],
        key=lambda p: p.name.lower(),
    )


def dataset_status_text(
    folder: str | Path,
) -> str:
    root = Path(folder).expanduser()

    if not root.is_dir():
        return "데이터셋 폴더를 선택하세요."

    files = dataset_files(root)

    if not files:
        return (
            "지원되는 음성/음원 파일이 없습니다. "
            "선택한 폴더 바로 아래에 파일을 넣어주세요."
        )

    return (
        f"데이터 파일 {len(files)}개 감지 "
        "(하위 폴더는 스캔하지 않음)"
    )


def _escape_training_path(path: Path) -> str:
    return str(path).replace(
        "\\",
        "\\\\",
    )


class RVCTrainingPipeline:
    def __init__(
        self,
        *,
        log_callback: LogCallback | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        self.log_callback = log_callback
        self.progress_callback = (
            progress_callback
        )
        self.current_process: subprocess.Popen | None = None
        self.cancelled = False

        log_file = training_log_path()
        log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        log_file.write_text(
            "",
            encoding="utf-8",
        )

    def _log(
        self,
        message: str,
    ) -> None:
        clean = str(message).rstrip()

        if not clean:
            return

        try:
            with training_log_path().open(
                "a",
                encoding="utf-8",
            ) as fp:
                fp.write(
                    clean + "\n"
                )
        except OSError:
            pass

        if self.log_callback:
            self.log_callback(
                clean
            )

    def _progress(
        self,
        value: int,
        message: str,
    ) -> None:
        if self.progress_callback:
            self.progress_callback(
                max(
                    0,
                    min(
                        100,
                        int(value),
                    ),
                ),
                str(message),
            )

    def cancel(self) -> None:
        self.cancelled = True
        process = self.current_process

        if (
            process is None
            or process.poll() is not None
        ):
            return

        self._log(
            "[CANCEL] 학습 작업 종료 요청"
        )

        try:
            if os.name == "nt":
                subprocess.run(
                    [
                        "taskkill",
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                process.terminate()
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _check_cancelled(
        self,
    ) -> None:
        if self.cancelled:
            raise RVCTrainingError(
                "사용자가 RVC 학습을 중지했습니다."
            )

    def _run(
        self,
        command: list[str],
        *,
        cwd: Path,
        stage: str,
        env: dict[str, str] | None = None,
    ) -> None:
        self._check_cancelled()

        self._log("")
        self._log(
            "=" * 72
        )
        self._log(
            f"[STAGE] {stage}"
        )
        self._log(
            " ".join(
                f'"{item}"'
                if " " in str(item)
                else str(item)
                for item in command
            )
        )
        self._log(
            "=" * 72
        )

        creationflags = getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0,
        )

        # RVC_TRAINING_PYTHONPATH_HOTFIX
        process_env = os.environ.copy()

        # RVC scripts import sibling top-level packages such as
        # infer/configs/tools/train.  When launched as
        # `python train/preprocess.py`, sys.path[0] is tools/rvc/train,
        # so explicitly expose the RVC project root to every child.
        pythonpath_entries = [
            str(cwd),
        ]

        existing_pythonpath = process_env.get(
            "PYTHONPATH",
            "",
        )

        if existing_pythonpath:
            pythonpath_entries.append(
                existing_pythonpath
            )

        process_env["PYTHONPATH"] = os.pathsep.join(
            pythonpath_entries
        )

        if env:
            process_env.update(
                env
            )

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=(
                creationflags
                if os.name == "nt"
                else 0
            ),
        )

        self.current_process = process

        try:
            assert process.stdout is not None

            for raw_line in process.stdout:
                self._check_cancelled()

                for part in raw_line.replace(
                    "\r",
                    "\n",
                ).splitlines():
                    if part.strip():
                        self._log(
                            part
                        )

            return_code = process.wait()

        finally:
            self.current_process = None

        self._check_cancelled()

        if return_code != 0:
            raise RVCTrainingError(
                f"{stage} 단계가 실패했습니다. "
                f"반환 코드={return_code}\n\n"
                f"로그: {training_log_path()}"
            )

    def _checkpoint_pair(
        self,
        exp_dir: Path,
    ) -> tuple[Path, Path]:
        generators = sorted(
            exp_dir.glob("G_*.pth"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        discriminators = sorted(
            exp_dir.glob("D_*.pth"),
            key=lambda path: path.stat().st_mtime_ns,
        )

        if not generators or not discriminators:
            raise RVCTrainingError(
                "기존 학습을 이어갈 G/D 체크포인트를 찾지 못했습니다.\n"
                f"실험 폴더: {exp_dir}\n\n"
                "최종 inference용 .pth만으로는 이 기능에서 이어학습할 수 없습니다."
            )

        return (
            generators[-1],
            discriminators[-1],
        )

    def _checkpoint_epoch(
        self,
        checkpoint: Path,
    ) -> int | None:
        code = (
            "import sys, torch; "
            "d=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
            "print(int(d.get('iteration',0)))"
        )

        try:
            completed = subprocess.run(
                [
                    str(
                        rvc_python()
                    ),
                    "-c",
                    code,
                    str(checkpoint),
                ],
                cwd=str(
                    rvc_root()
                ),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                check=False,
            )

            if completed.returncode != 0:
                self._log(
                    "[WARN] 현재 checkpoint epoch를 읽지 못했습니다: "
                    + completed.stderr.strip()
                )
                return None

            value = int(
                completed.stdout.strip().splitlines()[-1]
            )

            return max(
                0,
                value,
            )

        except Exception as exc:
            self._log(
                "[WARN] checkpoint epoch 확인 실패: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    def _dataset_manifest_path(
        self,
        exp_dir: Path,
    ) -> Path:
        return (
            exp_dir
            / DATASET_MANIFEST_NAME
        )

    def _dataset_manifest(
        self,
        dataset: Path,
        files: list[Path],
    ) -> dict:
        entries = []

        for path in files:
            stat = path.stat()
            entries.append(
                {
                    "name": path.name,
                    "path": str(
                        path.resolve()
                    ),
                    "size": int(
                        stat.st_size
                    ),
                    "mtime_ns": int(
                        stat.st_mtime_ns
                    ),
                }
            )

        return {
            "version": 1,
            "dataset_dir": str(
                dataset.resolve()
            ),
            "files": entries,
        }

    def _load_dataset_manifest(
        self,
        exp_dir: Path,
    ) -> dict | None:
        path = self._dataset_manifest_path(
            exp_dir
        )

        if not path.is_file():
            return None

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            if not isinstance(
                data,
                dict,
            ):
                return None

            if not isinstance(
                data.get("files"),
                list,
            ):
                return None

            return data

        except Exception:
            return None

    def _save_dataset_manifest(
        self,
        exp_dir: Path,
        dataset: Path,
        files: list[Path],
    ) -> None:
        path = self._dataset_manifest_path(
            exp_dir
        )

        path.write_text(
            json.dumps(
                self._dataset_manifest(
                    dataset,
                    files,
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        self._log(
            f"[OK] 데이터셋 기록 저장: {path.name} / {len(files)} files"
        )

    def _infer_old_source_count(
        self,
        exp_dir: Path,
    ) -> int:
        manifest = self._load_dataset_manifest(
            exp_dir
        )

        if manifest is not None:
            return len(
                manifest.get(
                    "files",
                    [],
                )
            )

        gt_dir = (
            exp_dir
            / "0_gt_wavs"
        )

        if not gt_dir.is_dir():
            return 0

        prefixes: set[str] = set()

        for path in gt_dir.glob(
            "*.wav"
        ):
            match = re.match(
                r"^(\d+)_\d+\.wav$",
                path.name,
            )

            if match:
                prefixes.add(
                    match.group(1)
                )

        return len(
            prefixes
        )

    def _validate_resume_dataset(
        self,
        exp_dir: Path,
        files: list[Path],
    ) -> None:
        previous = self._load_dataset_manifest(
            exp_dir
        )

        if previous is not None:
            old_names = {
                str(item.get("name", ""))
                for item in previous.get(
                    "files",
                    [],
                )
                if item.get("name")
            }
            current_names = {
                path.name
                for path in files
            }

            if current_names != old_names:
                raise RVCTrainingError(
                    "'기존 학습 이어하기'는 데이터셋이 바뀌지 않았을 때만 사용합니다.\n\n"
                    f"이전 파일 수: {len(old_names)}\n"
                    f"현재 파일 수: {len(current_names)}\n\n"
                    "파일을 추가/삭제/이름 변경했다면 "
                    "'데이터 추가 후 파인튜닝'을 선택하세요."
                )

            return

        old_count = self._infer_old_source_count(
            exp_dir
        )

        if (
            old_count > 0
            and len(files) != old_count
        ):
            raise RVCTrainingError(
                "'기존 학습 이어하기'에서 데이터 파일 수 변경을 감지했습니다.\n\n"
                f"기존 추정 원본 수: {old_count}\n"
                f"현재 데이터 수: {len(files)}\n\n"
                "데이터를 추가했다면 '데이터 추가 후 파인튜닝'을 사용하세요."
            )

    def _lead_clean_lineage_names_from_dir(
        self,
        dataset_dir: Path,
    ) -> set[str]:
        """
        Return original source names recorded by the v2.8
        Training Lead Dataset Cleaner.

        A cleaned file such as:
            song_vocals_lead.wav

        can therefore be matched to the original:
            song_vocals.wav
        """
        try:
            dataset = Path(
                dataset_dir
            ).expanduser().resolve()
        except Exception:
            return set()

        manifest_path = (
            dataset
            / LEAD_CLEAN_MANIFEST_NAME
        )

        if not manifest_path.is_file():
            return set()

        try:
            data = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as exc:
            self._log(
                "[WARN][LEAD LINEAGE] Lead Cleaner manifest 읽기 실패: "
                f"{type(exc).__name__}: {exc}"
            )
            return set()

        records = data.get(
            "files",
            [],
        )

        if not isinstance(
            records,
            list,
        ):
            return set()

        names: set[str] = set()

        for record in records:
            if not isinstance(
                record,
                dict,
            ):
                continue

            raw_name = str(
                record.get(
                    "source_name",
                    "",
                )
            ).strip()

            if not raw_name:
                raw_path = str(
                    record.get(
                        "source_path",
                        "",
                    )
                ).strip()

                if raw_path:
                    try:
                        raw_name = Path(
                            raw_path
                        ).name
                    except Exception:
                        raw_name = ""

            if raw_name:
                names.add(
                    raw_name
                )

        if names:
            self._log(
                "[LEAD LINEAGE] Cleaner manifest 원본 계보 "
                f"{len(names)}개 확인"
            )

        return names

    def _previous_finetune_lineage_names(
        self,
        previous: dict,
    ) -> set[str]:
        raw_dataset = str(
            previous.get(
                "dataset_dir",
                "",
            )
        ).strip()

        if raw_dataset:
            lineage = (
                self._lead_clean_lineage_names_from_dir(
                    Path(
                        raw_dataset
                    )
                )
            )

            if lineage:
                return lineage

        return {
            str(
                item.get(
                    "name",
                    "",
                )
            )
            for item in previous.get(
                "files",
                [],
            )
            if (
                isinstance(
                    item,
                    dict,
                )
                and item.get(
                    "name"
                )
            )
        }

    def _current_finetune_lineage_names(
        self,
        files: list[Path],
    ) -> set[str]:
        if not files:
            return set()

        lineage = (
            self._lead_clean_lineage_names_from_dir(
                files[
                    0
                ].parent
            )
        )

        if lineage:
            return lineage

        return {
            path.name
            for path in files
        }

    def _validate_finetune_dataset(
        self,
        exp_dir: Path,
        files: list[Path],
    ) -> None:
        previous = self._load_dataset_manifest(
            exp_dir
        )

        if previous is not None:
            previous_names = (
                self._previous_finetune_lineage_names(
                    previous
                )
            )
            current_names = (
                self._current_finetune_lineage_names(
                    files
                )
            )

            missing = sorted(
                previous_names
                - current_names
            )

            if missing:
                preview = ", ".join(
                    missing[
                        :5
                    ]
                )

                if len(
                    missing
                ) > 5:
                    preview += (
                        f" 외 {len(missing) - 5}개"
                    )

                raise RVCTrainingError(
                    "데이터 추가 파인튜닝에서는 기존 데이터 + 신규 데이터가 "
                    "모두 같은 데이터셋 계보에 있어야 합니다.\n\n"
                    "Lead Cleaner 정제본은 _lead.wav 파일명 자체가 아니라 "
                    "Cleaner manifest의 source_name/source_path를 기준으로 비교합니다.\n\n"
                    f"이전 데이터 중 현재 계보에서 찾지 못한 파일: {preview}"
                )

            added = (
                current_names
                - previous_names
            )

            current_file_names = {
                path.name
                for path in files
            }

            if (
                current_names
                != current_file_names
            ):
                self._log(
                    "[FINETUNE][LEAD LINEAGE] Lead Cleaner 정제본 감지 - "
                    "정제 파일명이 아니라 원본 source lineage로 검증했습니다."
                )

            self._log(
                "[FINETUNE] 이전 데이터 계보 "
                f"{len(previous_names)}개 / "
                f"현재 계보 {len(current_names)}개 / "
                f"추가 감지 {len(added)}개"
            )

            if added:
                added_preview = ", ".join(
                    sorted(
                        added
                    )[
                        :5
                    ]
                )

                if len(
                    added
                ) > 5:
                    added_preview += (
                        f" 외 {len(added) - 5}개"
                    )

                self._log(
                    "[FINETUNE] 신규 데이터 계보: "
                    + added_preview
                )

            return

        old_count = self._infer_old_source_count(
            exp_dir
        )

        if (
            old_count > 0
            and len(files) < old_count
        ):
            raise RVCTrainingError(
                "기존 학습 데이터보다 현재 선택한 데이터 파일 수가 적습니다.\n\n"
                f"기존 추정 원본 수: {old_count}\n"
                f"현재 데이터 수: {len(files)}\n\n"
                "데이터 추가 파인튜닝은 기존 데이터 + 신규 데이터를 모두 넣은 "
                "폴더를 선택하세요."
            )

        if old_count > 0:
            self._log(
                "[FINETUNE] 이전 manifest가 없어 전처리 파일에서 "
                f"기존 원본 수를 약 {old_count}개로 추정했습니다."
            )

    def _backup_training_state(
        self,
        exp_name: str,
        exp_dir: Path,
    ) -> Path:
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        backup_dir = (
            finetune_backups_root()
            / exp_name
            / stamp
        )
        backup_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        g_path, d_path = (
            self._checkpoint_pair(
                exp_dir
            )
        )

        for source in (
            g_path,
            d_path,
        ):
            shutil.copy2(
                source,
                backup_dir
                / source.name,
            )

        for name in (
            "config.json",
            "filelist.txt",
            DATASET_MANIFEST_NAME,
        ):
            source = (
                exp_dir
                / name
            )

            if source.is_file():
                shutil.copy2(
                    source,
                    backup_dir
                    / source.name,
                )

        permanent = (
            trained_models_root()
            / exp_name
        )

        if permanent.is_dir():
            target = (
                backup_dir
                / "rvc_models"
            )
            shutil.copytree(
                permanent,
                target,
                dirs_exist_ok=True,
            )

        weights_dir = (
            rvc_root()
            / "assets"
            / "weights"
        )
        final_weight = (
            weights_dir
            / f"{exp_name}.pth"
        )

        if final_weight.is_file():
            shutil.copy2(
                final_weight,
                backup_dir
                / final_weight.name,
            )

        self._log(
            f"[BACKUP] 파인튜닝 전 학습 상태 백업: {backup_dir}"
        )

        return backup_dir

    def _feature_index_candidates(
        self,
        exp_name: str,
        exp_dir: Path,
        *,
        include_permanent: bool = True,
    ) -> list[Path]:
        candidates: list[Path] = []

        try:
            candidates.extend(
                exp_dir.glob(
                    "*.index"
                )
            )
        except OSError:
            pass

        shared_dir = (
            rvc_root()
            / "assets"
            / "indices"
        )

        if shared_dir.is_dir():
            try:
                for path in shared_dir.glob(
                    "*.index"
                ):
                    if (
                        exp_name.lower()
                        in path.name.lower()
                    ):
                        candidates.append(
                            path
                        )
            except OSError:
                pass

        if include_permanent:
            permanent_dir = (
                trained_models_root()
                / exp_name
            )

            if permanent_dir.is_dir():
                try:
                    candidates.extend(
                        permanent_dir.glob(
                            "*.index"
                        )
                    )
                except OSError:
                    pass

        unique: dict[str, Path] = {}

        for path in candidates:
            try:
                key = str(
                    path.resolve()
                ).lower()
            except OSError:
                key = str(
                    path
                ).lower()

            unique[key] = path

        return list(
            unique.values()
        )

    def _backup_and_remove_feature_indexes(
        self,
        exp_name: str,
        exp_dir: Path,
        backup_dir: Path,
    ) -> None:
        candidates = (
            self._feature_index_candidates(
                exp_name,
                exp_dir,
                include_permanent=True,
            )
        )

        if not candidates:
            self._log(
                "[INDEX] 제거할 기존 Feature Index가 없습니다."
            )
            return

        index_backup_dir = (
            backup_dir
            / "feature_indexes"
        )
        index_backup_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        shared_dir = (
            rvc_root()
            / "assets"
            / "indices"
        )
        permanent_dir = (
            trained_models_root()
            / exp_name
        )

        removed = 0

        for source in candidates:
            try:
                if source.parent == exp_dir:
                    prefix = "experiment__"
                elif source.parent == shared_dir:
                    prefix = "assets_indices__"
                elif source.parent == permanent_dir:
                    prefix = "rvc_models__"
                else:
                    prefix = "other__"

                target = (
                    index_backup_dir
                    / (
                        prefix
                        + source.name
                    )
                )

                counter = 1

                while target.exists():
                    target = (
                        index_backup_dir
                        / (
                            prefix
                            + str(counter)
                            + "__"
                            + source.name
                        )
                    )
                    counter += 1

                shutil.copy2(
                    source,
                    target,
                )
                source.unlink()
                removed += 1

                self._log(
                    f"[INDEX] 기존 Index 백업/제거: {source}"
                )

            except Exception as exc:
                raise RVCTrainingError(
                    "기존 Feature Index를 백업/제거하지 못했습니다.\n"
                    f"파일: {source}\n"
                    f"오류: {type(exc).__name__}: {exc}"
                ) from exc

        self._log(
            f"[INDEX] 기존 Feature Index {removed}개 제거 완료. "
            "이번 파인튜닝의 새 HuBERT 특징으로 다시 생성합니다."
        )

    def _validate_generated_feature_index(
        self,
        exp_name: str,
        exp_dir: Path,
    ) -> Path:
        candidates = (
            self._feature_index_candidates(
                exp_name,
                exp_dir,
                include_permanent=False,
            )
        )

        added = [
            path
            for path in candidates
            if "added" in path.name.lower()
        ]

        if added:
            candidates = added

        if not candidates:
            raise RVCTrainingError(
                "Feature Index 생성 단계는 종료됐지만 새 .index 파일을 찾지 못했습니다.\n"
                "기존 Index는 파인튜닝 전에 제거했으므로 이전 Index를 계속 "
                "사용하지 않도록 완료 처리를 중단합니다."
            )

        newest = max(
            candidates,
            key=lambda path: path.stat().st_mtime_ns,
        )

        self._log(
            f"[INDEX] 새 Feature Index 확인: {newest}"
        )

        return newest

    def _clear_derived_training_data(
        self,
        exp_dir: Path,
    ) -> None:
        self._log(
            "[FINETUNE] 기존 전처리/F0/HuBERT 결과를 정리합니다. "
            "G/D 체크포인트는 보존합니다."
        )

        for name in DERIVED_TRAINING_DIRS:
            path = (
                exp_dir
                / name
            )

            if path.exists():
                shutil.rmtree(
                    path
                )

            path.mkdir(
                parents=True,
                exist_ok=True,
            )

        for name in (
            "filelist.txt",
            "config.json",
        ):
            path = (
                exp_dir
                / name
            )

            if path.is_file():
                path.unlink()

    def _validate_preprocess(
        self,
        exp_dir: Path,
    ) -> None:
        gt = list(
            (exp_dir / "0_gt_wavs").glob(
                "*.wav"
            )
        )
        wav16 = list(
            (exp_dir / "1_16k_wavs").glob(
                "*.wav"
            )
        )

        if not gt or not wav16:
            raise RVCTrainingError(
                "데이터 전처리 결과가 비어 있습니다."
            )

    def _validate_features(
        self,
        exp_dir: Path,
    ) -> None:
        names = self._training_names(
            exp_dir
        )

        if not names:
            raise RVCTrainingError(
                "F0/HuBERT 특징이 완전하게 생성된 학습 샘플이 없습니다."
            )

    def _training_names(
        self,
        exp_dir: Path,
    ) -> list[str]:
        def stems(
            folder: Path,
            suffix: str,
        ) -> set[str]:
            if not folder.is_dir():
                return set()

            result: set[str] = set()

            for path in folder.iterdir():
                if (
                    path.is_file()
                    and path.name.endswith(
                        suffix
                    )
                ):
                    name = path.name[
                        : -len(suffix)
                    ]
                    result.add(
                        name
                    )

            return result

        gt = stems(
            exp_dir / "0_gt_wavs",
            ".wav",
        )
        feature = stems(
            exp_dir / "3_feature768",
            ".npy",
        )

        # F0 paths produced by RVC are e.g. "0_1.wav.npy";
        # match them back to the base wav stem.
        f0_raw = stems(
            exp_dir / "2a_f0",
            ".npy",
        )
        f0nsf_raw = stems(
            exp_dir / "2b-f0nsf",
            ".npy",
        )

        def normalize(
            values: set[str],
        ) -> set[str]:
            return {
                value[:-4]
                if value.lower().endswith(
                    ".wav"
                )
                else value
                for value in values
            }

        return sorted(
            gt
            & feature
            & normalize(f0_raw)
            & normalize(f0nsf_raw)
        )

    def _prepare_config_and_filelist(
        self,
        exp_name: str,
        exp_dir: Path,
    ) -> None:
        names = self._training_names(
            exp_dir
        )

        if not names:
            raise RVCTrainingError(
                "filelist를 만들 학습 샘플이 없습니다."
            )

        gt_dir = (
            exp_dir / "0_gt_wavs"
        )
        feature_dir = (
            exp_dir / "3_feature768"
        )
        f0_dir = (
            exp_dir / "2a_f0"
        )
        f0nsf_dir = (
            exp_dir / "2b-f0nsf"
        )

        lines: list[str] = []

        for name in names:
            line = (
                f"{_escape_training_path(gt_dir)}/{name}.wav|"
                f"{_escape_training_path(feature_dir)}/{name}.npy|"
                f"{_escape_training_path(f0_dir)}/{name}.wav.npy|"
                f"{_escape_training_path(f0nsf_dir)}/{name}.wav.npy|"
                "0"
            )
            lines.append(
                line
            )

        mute_root = (
            rvc_root()
            / "logs"
            / "mute"
        )

        mute_line = (
            f"{_escape_training_path(mute_root / '0_gt_wavs')}/mute40k.wav|"
            f"{_escape_training_path(mute_root / '3_feature768')}/mute.npy|"
            f"{_escape_training_path(mute_root / '2a_f0')}/mute.wav.npy|"
            f"{_escape_training_path(mute_root / '2b-f0nsf')}/mute.wav.npy|"
            "0"
        )

        # Match the official WebUI: add two mute records.
        lines.extend(
            [
                mute_line,
                mute_line,
            ]
        )

        (exp_dir / "filelist.txt").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        # Current upstream WebUI uses configs/v1/40k.json
        # for 40k training even when model version is v2.
        template = (
            rvc_root()
            / "configs"
            / "v1"
            / "40k.json"
        )

        if not template.is_file():
            raise RVCTrainingError(
                f"RVC 40k config를 찾을 수 없습니다: {template}"
            )

        config = json.loads(
            template.read_text(
                encoding="utf-8",
            )
        )

        config.pop(
            "speaker_info",
            None,
        )

        (exp_dir / "config.json").write_text(
            json.dumps(
                config,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        self._log(
            f"[OK] filelist.txt: {len(lines)} records"
        )
        self._log(
            f"[OK] config.json: {template.name} / RVC v2 40k"
        )

    def _copy_final_outputs(
        self,
        exp_name: str,
        exp_dir: Path,
    ) -> RVCTrainingResult:
        source_model = (
            rvc_root()
            / "assets"
            / "weights"
            / f"{exp_name}.pth"
        )

        if not source_model.is_file():
            raise RVCTrainingError(
                "학습은 종료됐지만 최종 inference 모델을 찾지 못했습니다.\n"
                f"예상 경로: {source_model}"
            )

        output_dir = (
            trained_models_root()
            / exp_name
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        final_model = (
            output_dir
            / f"{exp_name}.pth"
        )
        shutil.copy2(
            source_model,
            final_model,
        )

        index_candidates = list(
            (
                rvc_root()
                / "assets"
                / "indices"
            ).glob(
                f"{exp_name}_*added*.index"
            )
        )

        if not index_candidates:
            index_candidates = list(
                exp_dir.glob(
                    "added_*.index"
                )
            )

        final_index: Path | None = None

        if index_candidates:
            source_index = max(
                index_candidates,
                key=lambda p: p.stat().st_mtime_ns,
            )
            final_index = (
                output_dir
                / source_index.name
            )
            shutil.copy2(
                source_index,
                final_index,
            )

        self._log("")
        self._log(
            f"[RESULT] Model: {final_model}"
        )

        if final_index is not None:
            self._log(
                f"[RESULT] Index: {final_index}"
            )
        else:
            self._log(
                "[RESULT] Index: 생성 결과를 찾지 못함"
            )

        return RVCTrainingResult(
            model_path=final_model,
            index_path=final_index,
            experiment_dir=exp_dir,
        )

    def run(
        self,
        *,
        dataset_dir: str | Path,
        experiment_name: str,
        epochs: int = 200,
        batch_size: int = 8,
        save_every: int = 10,
        workers: int = 8,
        gpu_id: int = 0,
        cache_gpu: bool = False,
        training_mode: str = TRAINING_MODE_NEW,
    ) -> RVCTrainingResult:
        ready, status = (
            training_assets_status()
        )

        if not ready:
            raise RVCTrainingError(
                status
            )

        if training_mode not in TRAINING_MODES:
            raise RVCTrainingError(
                f"알 수 없는 학습 방식입니다: {training_mode}"
            )

        dataset = Path(
            dataset_dir
        ).expanduser().resolve()

        files = dataset_files(
            dataset
        )

        if not files:
            raise RVCTrainingError(
                "학습 데이터셋 폴더에 사용할 수 있는 파일이 없습니다.\n"
                "파일은 선택 폴더 바로 아래에 두세요."
            )

        exp_name = (
            validate_experiment_name(
                experiment_name
            )
        )

        epochs = max(
            10,
            min(
                1200,
                int(epochs),
            ),
        )
        batch_size = max(
            1,
            min(
                64,
                int(batch_size),
            ),
        )
        save_every = max(
            1,
            min(
                epochs,
                int(save_every),
            ),
        )
        workers = max(
            1,
            min(
                32,
                int(workers),
            ),
        )
        gpu_id = max(
            0,
            int(gpu_id),
        )

        root = rvc_root()
        py = rvc_python()
        exp_dir = (
            root
            / "logs"
            / exp_name
        )
        exp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            root
            / "assets"
            / "weights"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )
        (
            root
            / "assets"
            / "indices"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        mode_labels = {
            TRAINING_MODE_NEW: "새 모델 학습",
            TRAINING_MODE_RESUME: "기존 학습 이어하기",
            TRAINING_MODE_FINETUNE_ADD: "데이터 추가 후 파인튜닝",
        }

        self._log(
            "RVC one-click single-speaker training"
        )
        self._log(
            f"Mode: {mode_labels[training_mode]}"
        )
        self._log(
            f"Dataset: {dataset}"
        )
        self._log(
            f"Files: {len(files)}"
        )
        self._log(
            f"Experiment: {exp_name}"
        )
        self._log(
            f"Target total epochs: {epochs}"
        )
        self._log(
            f"Batch: {batch_size}"
        )
        self._log(
            f"GPU: {gpu_id}"
        )

        if training_mode == TRAINING_MODE_NEW:
            generators = list(
                exp_dir.glob(
                    "G_*.pth"
                )
            )
            discriminators = list(
                exp_dir.glob(
                    "D_*.pth"
                )
            )

            if generators or discriminators:
                raise RVCTrainingError(
                    "같은 모델 이름에 기존 학습 체크포인트가 있습니다.\n\n"
                    f"모델: {exp_name}\n"
                    "새 모델을 만들려면 다른 이름을 사용하거나, "
                    "'기존 학습 이어하기' 또는 '데이터 추가 후 파인튜닝'을 선택하세요."
                )

            self._clear_derived_training_data(
                exp_dir
            )

        else:
            checkpoint_g, checkpoint_d = (
                self._checkpoint_pair(
                    exp_dir
                )
            )
            current_epoch = self._checkpoint_epoch(
                checkpoint_g
            )

            self._log(
                f"[RESUME] Generator checkpoint: {checkpoint_g.name}"
            )
            self._log(
                f"[RESUME] Discriminator checkpoint: {checkpoint_d.name}"
            )

            if current_epoch is not None:
                self._log(
                    f"[RESUME] 현재 체크포인트 epoch: {current_epoch}"
                )

                if epochs <= current_epoch:
                    raise RVCTrainingError(
                        "목표 Epochs는 현재 체크포인트보다 커야 합니다.\n\n"
                        f"현재 epoch: {current_epoch}\n"
                        f"입력한 목표 epoch: {epochs}\n\n"
                        "예: 현재 200 epoch에서 100 epoch를 더 학습하려면 "
                        "목표 Epochs를 300으로 설정하세요."
                    )

        if training_mode == TRAINING_MODE_RESUME:
            self._validate_resume_dataset(
                exp_dir,
                files,
            )

        rebuild_features = (
            training_mode
            in {
                TRAINING_MODE_NEW,
                TRAINING_MODE_FINETUNE_ADD,
            }
        )

        if (
            training_mode
            == TRAINING_MODE_FINETUNE_ADD
        ):
            self._validate_finetune_dataset(
                exp_dir,
                files,
            )
            finetune_backup_dir = (
                self._backup_training_state(
                    exp_name,
                    exp_dir,
                )
            )
            self._backup_and_remove_feature_indexes(
                exp_name,
                exp_dir,
                finetune_backup_dir,
            )
            self._clear_derived_training_data(
                exp_dir
            )

        if rebuild_features:
            self._progress(
                3,
                "1/6 데이터 전처리 중...",
            )
            self._run(
                [
                    str(py),
                    "-m",
                    "train.preprocess",
                    str(dataset),
                    "40000",
                    str(workers),
                    str(exp_dir),
                    "False",
                    "3.7",
                ],
                cwd=root,
                stage="1/6 데이터 전처리",
            )
            self._validate_preprocess(
                exp_dir
            )

            self._progress(
                18,
                "2/6 RMVPE F0 추출 중...",
            )
            self._run(
                [
                    str(py),
                    "-m",
                    "train.dataset.extract_f0",
                    "cuda",
                    "1",
                    "0",
                    str(gpu_id),
                    str(exp_dir),
                    "true",
                ],
                cwd=root,
                stage="2/6 RMVPE F0 추출",
            )

            self._progress(
                32,
                "3/6 HuBERT 특징 추출 중...",
            )
            self._run(
                [
                    str(py),
                    "-m",
                    "train.dataset.extract_hubert_feature",
                    f"cuda:{gpu_id}",
                    "1",
                    "0",
                    str(gpu_id),
                    str(exp_dir),
                    "v2",
                    "true",
                ],
                cwd=root,
                stage="3/6 HuBERT 특징 추출",
            )
            self._validate_features(
                exp_dir
            )

            self._progress(
                45,
                "4/6 학습 설정 생성 중...",
            )
            self._prepare_config_and_filelist(
                exp_name,
                exp_dir,
            )
            self._save_dataset_manifest(
                exp_dir,
                dataset,
                files,
            )

        else:
            self._progress(
                45,
                "기존 전처리/F0/HuBERT 데이터 검증 중...",
            )
            self._validate_preprocess(
                exp_dir
            )
            self._validate_features(
                exp_dir
            )

            if (
                not (
                    exp_dir
                    / "filelist.txt"
                ).is_file()
                or not (
                    exp_dir
                    / "config.json"
                ).is_file()
            ):
                self._prepare_config_and_filelist(
                    exp_name,
                    exp_dir,
                )

            self._log(
                "[RESUME] 전처리/RMVPE/HuBERT를 재실행하지 않고 "
                "기존 특징 데이터를 사용합니다."
            )

        pretrained_g = (
            root
            / "assets"
            / "pretrained_v2"
            / "f0G40k.pth"
        )
        pretrained_d = (
            root
            / "assets"
            / "pretrained_v2"
            / "f0D40k.pth"
        )

        self._progress(
            50,
            (
                "5/6 RVC 모델 학습 중... "
                "이 단계가 가장 오래 걸립니다."
            ),
        )

        env = os.environ.copy()
        env["RVC_CUDA_GRAPH"] = "0"
        env.setdefault(
            "PYTHONUTF8",
            "1",
        )
        env.setdefault(
            "PYTHONIOENCODING",
            "utf-8",
        )

        if training_mode != TRAINING_MODE_NEW:
            self._log(
                "[RESUME] RVC train.py가 같은 실험 폴더의 최신 G/D "
                "체크포인트를 자동 로드하여 이어서 학습합니다."
            )

        self._run(
            [
                str(py),
                "-m",
                "train.train",
                "-e",
                exp_name,
                "-sr",
                "40k",
                "-f0",
                "1",
                "-bs",
                str(batch_size),
                "-g",
                str(gpu_id),
                "-te",
                str(epochs),
                "-se",
                str(save_every),
                "-pg",
                str(pretrained_g),
                "-pd",
                str(pretrained_d),
                "-l",
                "1",
                "-c",
                "1"
                if cache_gpu
                else "0",
                "-sw",
                "1",
                "-v",
                "v2",
            ],
            cwd=root,
            stage="5/6 RVC 모델 학습",
            env=env,
        )

        self._progress(
            94,
            "6/6 Feature Index 생성 중...",
        )
        self._run(
            [
                str(py),
                "-m",
                "train.train_index",
                exp_name,
                "v2",
                str(
                    root
                    / "assets"
                    / "indices"
                ),
                str(workers),
                "single",
            ],
            cwd=root,
            stage="6/6 Feature Index 생성",
        )

        if (
            training_mode
            == TRAINING_MODE_FINETUNE_ADD
        ):
            self._validate_generated_feature_index(
                exp_name,
                exp_dir,
            )

        result = (
            self._copy_final_outputs(
                exp_name,
                exp_dir,
            )
        )

        self._progress(
            100,
            "RVC 모델 학습 완료",
        )

        return result

