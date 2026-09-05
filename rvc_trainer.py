from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json
import os
import re
import shutil
import subprocess


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, str], None]


class RVCTrainingError(RuntimeError):
    pass


@dataclass(slots=True)
class RVCTrainingResult:
    model_path: Path
    index_path: Path | None
    experiment_dir: Path


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

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
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
    ) -> RVCTrainingResult:
        ready, status = (
            training_assets_status()
        )

        if not ready:
            raise RVCTrainingError(
                status
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

        self._log(
            "RVC one-click single-speaker training"
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
            f"Epochs: {epochs}"
        )
        self._log(
            f"Batch: {batch_size}"
        )
        self._log(
            f"GPU: {gpu_id}"
        )

        # 1. preprocess
        self._progress(
            3,
            "1/6 데이터 전처리 중...",
        )
        self._run(
            [
                str(py),
                "train/preprocess.py",
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

        # 2. RMVPE F0
        self._progress(
            18,
            "2/6 RMVPE F0 추출 중...",
        )
        self._run(
            [
                str(py),
                "train/dataset/extract_f0.py",
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

        # 3. HuBERT
        self._progress(
            32,
            "3/6 HuBERT 특징 추출 중...",
        )
        self._run(
            [
                str(py),
                "train/dataset/extract_hubert_feature.py",
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

        # 4. config / filelist
        self._progress(
            45,
            "4/6 학습 설정 생성 중...",
        )
        self._prepare_config_and_filelist(
            exp_name,
            exp_dir,
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

        # 5. train
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

        self._run(
            [
                str(py),
                "train/train.py",
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

        # 6. index
        self._progress(
            94,
            "6/6 Feature Index 생성 중...",
        )
        self._run(
            [
                str(py),
                "train/train_index.py",
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
