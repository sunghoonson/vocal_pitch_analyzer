from __future__ import annotations

import csv
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSettings, QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from media_input import ffmpeg_status_text
from pitch_analyzer import (
    AnalysisResult,
    analyze_audio,
    midi_to_korean_name,
    midi_to_note_name,
)
from vocal_separator import (
    DEFAULT_MODEL,
    SeparatedVocal,
    separate_vocals,
    separator_status_text,
)
from subtitle_generator import (
    build_subtitle_groups,
    write_ass,
    write_srt,
)
from audio_transposer import (
    AudioTransposeError,
    rubberband_filter_available,
    semitone_to_ratio,
    transpose_audio,
)
from seed_vc_svc import (
    SeedVCSVCError,
    convert_full_mix_seed_vc,
    convert_vocal_seed_vc,
    seed_vc_available,
    seed_vc_status_text,
)
from rvc_rmvpe import (
    RVCRMVPEError,
    convert_full_mix_rvc,
    convert_vocal_rvc,
    rvc_available,
    rvc_status_text,
)
from rvc_trainer import (
    RVCTrainingError,
    RVCTrainingPipeline,
    dataset_status_text,
    training_assets_status,
)


APP_TITLE = "Vocal Pitch Analyzer - Prototype v2.0 / RVC Training"

# V20_RVC_TRAINING_PATCH
# V19_RVC_RMVPE_PATCH
# V18_SEED_VC_SVC_PATCH
# V17_KEY_TRANSPOSE_PATCH

# V16_TABBED_UI_PATCH
SUPPORTED_MEDIA_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".mp4",
    ".aac",
    ".webm",
    ".mkv",
    ".mov",
    ".wma",
    ".opus",
    ".m4v",
}



@dataclass(slots=True)
class PipelineResult:
    pitch: AnalysisResult
    original_path: str
    analysis_source_path: str
    used_vocal_separation: bool
    vocal_resource: SeparatedVocal | None


class NoteAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        labels = []
        for value in values:
            midi = int(round(value))
            labels.append(
                midi_to_note_name(midi)
                if 0 <= midi <= 127
                else ""
            )
        return labels


class AnalysisThread(QThread):
    progress_changed = Signal(int, str)
    separator_log = Signal(str)
    analysis_done = Signal(object)
    analysis_failed = Signal(str)

    def __init__(
        self,
        *,
        path: str,
        use_vocal_separation: bool,
        separator_model: str,
        separator_cache: bool,
        separator_autocast: bool,
        fmin: float,
        fmax: float,
        threshold: float,
        max_dropout_ms: float,
        smoothing_window: int,
        hysteresis_cents: float,
        min_note_ms: float,
        use_energy_gate: bool,
        energy_margin_db: float,
        energy_floor_dbfs: float,
        energy_hysteresis_db: float,
        min_activity_ms: float,
        max_activity_gap_ms: float,
        range_min_note_ms: float,
        range_min_confidence: float,
        parent=None,
    ):
        super().__init__(parent)

        self.path = path
        self.use_vocal_separation = use_vocal_separation
        self.separator_model = separator_model
        self.separator_cache = separator_cache
        self.separator_autocast = separator_autocast

        self.fmin = fmin
        self.fmax = fmax
        self.threshold = threshold
        self.max_dropout_ms = max_dropout_ms
        self.smoothing_window = smoothing_window
        self.hysteresis_cents = hysteresis_cents
        self.min_note_ms = min_note_ms

        self.use_energy_gate = use_energy_gate
        self.energy_margin_db = energy_margin_db
        self.energy_floor_dbfs = energy_floor_dbfs
        self.energy_hysteresis_db = energy_hysteresis_db
        self.min_activity_ms = min_activity_ms
        self.max_activity_gap_ms = max_activity_gap_ms
        self.range_min_note_ms = range_min_note_ms
        self.range_min_confidence = range_min_confidence

    def run(self):
        vocal_resource: SeparatedVocal | None = None

        try:
            analysis_path = self.path

            if self.use_vocal_separation:
                self.progress_changed.emit(
                    3,
                    "AI 보컬 분리를 준비하는 중...",
                )

                vocal_resource = separate_vocals(
                    self.path,
                    model_filename=self.separator_model,
                    use_cache=self.separator_cache,
                    use_autocast=self.separator_autocast,
                    log_callback=lambda text: self.separator_log.emit(text),
                )

                analysis_path = str(
                    vocal_resource.vocal_path
                )

                self.progress_changed.emit(
                    28,
                    (
                        "보컬 캐시를 불러왔습니다. Pitch 분석을 시작합니다."
                        if vocal_resource.cache_hit
                        else "보컬 분리 완료. Pitch 분석을 시작합니다."
                    ),
                )

                def pitch_progress(
                    percent: int,
                    text: str,
                ) -> None:
                    mapped = 28 + int(
                        percent * 0.72
                    )
                    self.progress_changed.emit(
                        min(mapped, 100),
                        text,
                    )
            else:
                def pitch_progress(
                    percent: int,
                    text: str,
                ) -> None:
                    self.progress_changed.emit(
                        percent,
                        text,
                    )

            pitch = analyze_audio(
                analysis_path,
                fmin_hz=self.fmin,
                fmax_hz=self.fmax,
                voiced_threshold=self.threshold,
                max_dropout_ms=self.max_dropout_ms,
                smoothing_window=self.smoothing_window,
                hysteresis_cents=self.hysteresis_cents,
                min_note_ms=self.min_note_ms,

                use_energy_gate=self.use_energy_gate,
                energy_margin_db=self.energy_margin_db,
                energy_floor_dbfs=self.energy_floor_dbfs,
                energy_hysteresis_db=self.energy_hysteresis_db,
                min_activity_ms=self.min_activity_ms,
                max_activity_gap_ms=self.max_activity_gap_ms,
                range_min_note_ms=self.range_min_note_ms,
                range_min_confidence=self.range_min_confidence,

                progress=pitch_progress,
            )

            self.analysis_done.emit(
                PipelineResult(
                    pitch=pitch,
                    original_path=self.path,
                    analysis_source_path=analysis_path,
                    used_vocal_separation=self.use_vocal_separation,
                    vocal_resource=vocal_resource,
                )
            )

        except Exception:
            if vocal_resource is not None:
                vocal_resource.cleanup()

            self.analysis_failed.emit(
                traceback.format_exc()
            )


class TransposeThread(QThread):
    progress_changed = Signal(
        int,
        str,
    )
    transpose_done = Signal(str)
    transpose_failed = Signal(str)

    def __init__(
        self,
        *,
        input_path: str,
        output_path: str,
        semitones: int,
        preserve_formant: bool,
        quality: str,
        parent=None,
    ):
        super().__init__(parent)

        self.input_path = input_path
        self.output_path = output_path
        self.semitones = semitones
        self.preserve_formant = (
            preserve_formant
        )
        self.quality = quality

    def run(self):
        try:
            result = transpose_audio(
                self.input_path,
                self.output_path,
                semitones=self.semitones,
                preserve_formant=self.preserve_formant,
                quality=self.quality,
                progress=lambda p, t: (
                    self.progress_changed.emit(
                        p,
                        t,
                    )
                ),
            )

            self.transpose_done.emit(
                str(result)
            )

        except Exception:
            self.transpose_failed.emit(
                traceback.format_exc()
            )



class SeedVCTransposeThread(QThread):
    progress_changed = Signal(
        int,
        str,
    )
    transpose_done = Signal(str)
    transpose_failed = Signal(str)
    log_line = Signal(str)

    def __init__(
        self,
        *,
        input_path: str,
        output_path: str,
        source_mode: str,
        semitones: int,
        auto_reference: bool,
        reference_path: str | None,
        diffusion_steps: int,
        cfg_rate: float,
        fp16: bool,
        separator_model: str,
        separator_cache: bool,
        separator_autocast: bool,
        parent=None,
    ):
        super().__init__(parent)

        self.input_path = input_path
        self.output_path = output_path
        self.source_mode = source_mode
        self.semitones = semitones
        self.auto_reference = auto_reference
        self.reference_path = reference_path
        self.diffusion_steps = diffusion_steps
        self.cfg_rate = cfg_rate
        self.fp16 = fp16
        self.separator_model = separator_model
        self.separator_cache = separator_cache
        self.separator_autocast = (
            separator_autocast
        )

    def run(self):
        try:
            common = dict(
                semitones=self.semitones,
                reference_path=self.reference_path,
                auto_reference=self.auto_reference,
                diffusion_steps=self.diffusion_steps,
                cfg_rate=self.cfg_rate,
                fp16=self.fp16,
                progress=lambda p, t: (
                    self.progress_changed.emit(
                        p,
                        t,
                    )
                ),
                log_callback=lambda t: (
                    self.log_line.emit(t)
                ),
            )

            if self.source_mode == "vocals":
                result = convert_vocal_seed_vc(
                    self.input_path,
                    self.output_path,
                    **common,
                )
            else:
                result = convert_full_mix_seed_vc(
                    self.input_path,
                    self.output_path,
                    separator_model=self.separator_model,
                    separator_cache=self.separator_cache,
                    separator_autocast=self.separator_autocast,
                    **common,
                )

            self.transpose_done.emit(
                str(result)
            )

        except Exception:
            self.transpose_failed.emit(
                traceback.format_exc()
            )



class RVCTransposeThread(QThread):
    progress_changed = Signal(
        int,
        str,
    )
    transpose_done = Signal(str)
    transpose_failed = Signal(str)
    log_line = Signal(str)

    def __init__(
        self,
        *,
        input_path: str,
        output_path: str,
        source_mode: str,
        semitones: int,
        model_path: str,
        index_path: str | None,
        index_rate: float,
        protect: float,
        rms_mix_rate: float,
        speaker_id: int,
        separator_model: str,
        separator_cache: bool,
        separator_autocast: bool,
        parent=None,
    ):
        super().__init__(parent)

        self.input_path = input_path
        self.output_path = output_path
        self.source_mode = source_mode
        self.semitones = semitones
        self.model_path = model_path
        self.index_path = index_path
        self.index_rate = index_rate
        self.protect = protect
        self.rms_mix_rate = rms_mix_rate
        self.speaker_id = speaker_id
        self.separator_model = (
            separator_model
        )
        self.separator_cache = (
            separator_cache
        )
        self.separator_autocast = (
            separator_autocast
        )

    def run(self):
        try:
            common = dict(
                model_path=self.model_path,
                index_path=self.index_path,
                semitones=self.semitones,
                index_rate=self.index_rate,
                protect=self.protect,
                rms_mix_rate=self.rms_mix_rate,
                speaker_id=self.speaker_id,
                progress=lambda p, t: (
                    self.progress_changed.emit(
                        p,
                        t,
                    )
                ),
                log_callback=lambda t: (
                    self.log_line.emit(t)
                ),
            )

            if self.source_mode == "vocals":
                result = convert_vocal_rvc(
                    self.input_path,
                    self.output_path,
                    **common,
                )
            else:
                result = convert_full_mix_rvc(
                    self.input_path,
                    self.output_path,
                    separator_model=self.separator_model,
                    separator_cache=self.separator_cache,
                    separator_autocast=self.separator_autocast,
                    **common,
                )

            self.transpose_done.emit(
                str(result)
            )

        except Exception:
            self.transpose_failed.emit(
                traceback.format_exc()
            )



class RVCTrainingThread(QThread):
    progress_changed = Signal(
        int,
        str,
    )
    log_line = Signal(str)
    training_done = Signal(
        str,
        str,
    )
    training_failed = Signal(str)

    def __init__(
        self,
        *,
        dataset_dir: str,
        experiment_name: str,
        epochs: int,
        batch_size: int,
        save_every: int,
        workers: int,
        gpu_id: int,
        cache_gpu: bool,
        parent=None,
    ):
        super().__init__(parent)

        self.dataset_dir = dataset_dir
        self.experiment_name = (
            experiment_name
        )
        self.epochs = epochs
        self.batch_size = (
            batch_size
        )
        self.save_every = (
            save_every
        )
        self.workers = workers
        self.gpu_id = gpu_id
        self.cache_gpu = (
            cache_gpu
        )

        self.pipeline = RVCTrainingPipeline(
            log_callback=lambda text: (
                self.log_line.emit(
                    text
                )
            ),
            progress_callback=lambda p, text: (
                self.progress_changed.emit(
                    p,
                    text,
                )
            ),
        )

    def cancel(self):
        self.pipeline.cancel()

    def run(self):
        try:
            result = self.pipeline.run(
                dataset_dir=self.dataset_dir,
                experiment_name=self.experiment_name,
                epochs=self.epochs,
                batch_size=self.batch_size,
                save_every=self.save_every,
                workers=self.workers,
                gpu_id=self.gpu_id,
                cache_gpu=self.cache_gpu,
            )

            self.training_done.emit(
                str(
                    result.model_path
                ),
                (
                    str(
                        result.index_path
                    )
                    if result.index_path
                    else ""
                ),
            )

        except Exception:
            self.training_failed.emit(
                traceback.format_exc()
            )



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.audio_path: str | None = None
        self.result: AnalysisResult | None = None
        self.pipeline_result: PipelineResult | None = None
        self.worker: AnalysisThread | None = None
        self.transpose_worker: QThread | None = None
        self.rvc_training_worker: RVCTrainingThread | None = None
        self.current_vocal_resource: SeparatedVocal | None = None
        self.seedvc_reference_path = ""
        self.active_transpose_semitones = 0
        self.active_transpose_engine = "rubberband"

        self.settings = QSettings(
            "VocalPitchAnalyzer",
            "VocalPitchAnalyzer",
        )

        self.rvc_model_path = self.settings.value(
            "rvc_model_path",
            "",
            type=str,
        )
        self.rvc_index_path = self.settings.value(
            "rvc_index_path",
            "",
            type=str,
        )

        self.setWindowTitle(APP_TITLE)
        self.resize(1620, 1000)
        self.setAcceptDrops(True)

        self._build_ui()
        self._build_menu()

        self.statusBar().showMessage(
            ffmpeg_status_text()
            + " | "
            + separator_status_text(),
            20000,
        )

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)

        self.main_tabs = QTabWidget()

        self.analysis_tab = QWidget()
        analysis_root = QVBoxLayout(
            self.analysis_tab
        )

        self.result_tab = QWidget()
        result_root = QVBoxLayout(
            self.result_tab
        )

        self.transpose_tab = QWidget()
        transpose_root = QVBoxLayout(
            self.transpose_tab
        )

        self.rvc_training_tab = QWidget()
        rvc_training_root = QVBoxLayout(
            self.rvc_training_tab
        )

        self.main_tabs.addTab(
            self.analysis_tab,
            "분석 / 설정",
        )
        self.main_tabs.addTab(
            self.result_tab,
            "결과 / 자막",
        )
        self.main_tabs.addTab(
            self.transpose_tab,
            "키 변환 / 음원 추출",
        )
        self.main_tabs.addTab(
            self.rvc_training_tab,
            "RVC 모델 학습",
        )

        training_data_group = QGroupBox(
            "1. 남성/타깃 음성 데이터셋"
        )
        training_data_layout = QFormLayout(
            training_data_group
        )

        dataset_row = QWidget()
        dataset_row_layout = QHBoxLayout(
            dataset_row
        )
        dataset_row_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self.rvc_training_dataset_label = QLabel(
            "데이터셋 폴더를 선택하세요."
        )
        self.rvc_training_dataset_label.setWordWrap(
            True
        )

        self.rvc_training_dataset_button = QPushButton(
            "폴더 선택"
        )
        self.rvc_training_dataset_button.clicked.connect(
            self.choose_rvc_training_dataset
        )

        dataset_row_layout.addWidget(
            self.rvc_training_dataset_label,
            1,
        )
        dataset_row_layout.addWidget(
            self.rvc_training_dataset_button,
        )

        self.rvc_training_dataset_status = QLabel(
            "한 사람의 깨끗한 남성 음성/보컬만 넣는 것을 권장합니다. "
            "반주나 다른 화자가 섞이면 모델 품질이 크게 떨어집니다."
        )
        self.rvc_training_dataset_status.setWordWrap(
            True
        )

        training_data_layout.addRow(
            "데이터셋",
            dataset_row,
        )
        training_data_layout.addRow(
            "",
            self.rvc_training_dataset_status,
        )

        rvc_training_root.addWidget(
            training_data_group
        )

        training_option_group = QGroupBox(
            "2. 학습 설정 - Single Speaker / RVC v2 / 40k / RMVPE"
        )
        training_option_layout = QFormLayout(
            training_option_group
        )

        self.rvc_training_name_edit = QLineEdit()
        self.rvc_training_name_edit.setText(
            "male_voice_01"
        )
        self.rvc_training_name_edit.setToolTip(
            "영문/숫자/_/- 만 사용"
        )

        self.rvc_training_epochs_spin = QSpinBox()
        self.rvc_training_epochs_spin.setRange(
            10,
            1200,
        )
        self.rvc_training_epochs_spin.setValue(
            200
        )

        self.rvc_training_batch_spin = QSpinBox()
        self.rvc_training_batch_spin.setRange(
            1,
            64,
        )
        self.rvc_training_batch_spin.setValue(
            8
        )

        self.rvc_training_save_spin = QSpinBox()
        self.rvc_training_save_spin.setRange(
            1,
            200,
        )
        self.rvc_training_save_spin.setValue(
            10
        )
        self.rvc_training_save_spin.setSuffix(
            " epoch"
        )

        self.rvc_training_workers_spin = QSpinBox()
        self.rvc_training_workers_spin.setRange(
            1,
            32,
        )
        self.rvc_training_workers_spin.setValue(
            8
        )

        self.rvc_training_gpu_spin = QSpinBox()
        self.rvc_training_gpu_spin.setRange(
            0,
            15,
        )
        self.rvc_training_gpu_spin.setValue(
            0
        )

        self.rvc_training_cache_gpu_check = QCheckBox(
            "학습 데이터를 GPU 메모리에 캐시"
        )
        self.rvc_training_cache_gpu_check.setChecked(
            False
        )
        self.rvc_training_cache_gpu_check.setToolTip(
            "VRAM 사용량이 크게 늘 수 있으므로 기본 OFF입니다."
        )

        self.rvc_training_runtime_label = QLabel()
        self.rvc_training_runtime_label.setWordWrap(
            True
        )

        training_option_layout.addRow(
            "모델 이름",
            self.rvc_training_name_edit,
        )
        training_option_layout.addRow(
            "Epochs",
            self.rvc_training_epochs_spin,
        )
        training_option_layout.addRow(
            "Batch Size",
            self.rvc_training_batch_spin,
        )
        training_option_layout.addRow(
            "중간 저장",
            self.rvc_training_save_spin,
        )
        training_option_layout.addRow(
            "CPU Workers",
            self.rvc_training_workers_spin,
        )
        training_option_layout.addRow(
            "GPU ID",
            self.rvc_training_gpu_spin,
        )
        training_option_layout.addRow(
            "",
            self.rvc_training_cache_gpu_check,
        )
        training_option_layout.addRow(
            "환경",
            self.rvc_training_runtime_label,
        )

        rvc_training_root.addWidget(
            training_option_group
        )

        training_action_group = QGroupBox(
            "3. 원클릭 학습"
        )
        training_action_layout = QVBoxLayout(
            training_action_group
        )

        training_button_row = QHBoxLayout()

        self.rvc_training_start_button = QPushButton(
            "전처리 → RMVPE → HuBERT → 학습 → Index 생성"
        )
        self.rvc_training_start_button.clicked.connect(
            self.start_rvc_training
        )

        self.rvc_training_stop_button = QPushButton(
            "학습 중지"
        )
        self.rvc_training_stop_button.setEnabled(
            False
        )
        self.rvc_training_stop_button.clicked.connect(
            self.stop_rvc_training
        )

        training_button_row.addWidget(
            self.rvc_training_start_button,
            1,
        )
        training_button_row.addWidget(
            self.rvc_training_stop_button,
        )

        self.rvc_training_progress = QProgressBar()
        self.rvc_training_progress.setRange(
            0,
            100,
        )
        self.rvc_training_progress.setValue(
            0
        )

        self.rvc_training_progress_label = QLabel(
            "준비"
        )
        self.rvc_training_progress_label.setWordWrap(
            True
        )

        self.rvc_training_log = QPlainTextEdit()
        self.rvc_training_log.setReadOnly(
            True
        )
        self.rvc_training_log.setMaximumBlockCount(
            4000
        )

        training_action_layout.addLayout(
            training_button_row
        )
        training_action_layout.addWidget(
            self.rvc_training_progress
        )
        training_action_layout.addWidget(
            self.rvc_training_progress_label
        )
        training_action_layout.addWidget(
            self.rvc_training_log,
            1,
        )

        rvc_training_root.addWidget(
            training_action_group,
            1,
        )

        self.refresh_rvc_training_status()

        engine_group = QGroupBox(
            "키 변환 엔진"
        )
        engine_layout = QFormLayout(
            engine_group
        )

        self.transpose_engine_combo = QComboBox()
        self.transpose_engine_combo.addItem(
            "빠른 DSP - FFmpeg RubberBand",
            "rubberband",
        )
        self.transpose_engine_combo.addItem(
            "AI 고음질 - Seed-VC SVC",
            "seed_vc",
        )
        self.transpose_engine_combo.addItem(
            "AI 음색 변환 - RVC + RMVPE",
            "rvc",
        )
        self.transpose_engine_combo.currentIndexChanged.connect(
            self.update_transpose_engine_ui
        )

        self.transpose_engine_status_label = QLabel(
            "엔진 상태 확인 전"
        )
        self.transpose_engine_status_label.setWordWrap(
            True
        )

        engine_layout.addRow(
            "변환 방식",
            self.transpose_engine_combo,
        )
        engine_layout.addRow(
            "상태",
            self.transpose_engine_status_label,
        )

        transpose_root.addWidget(
            engine_group
        )

        source_group = QGroupBox(
            "변환 소스"
        )
        source_layout = QFormLayout(
            source_group
        )

        self.transpose_source_combo = QComboBox()
        self.transpose_source_combo.addItem(
            "현재 선택한 원본 전체 음원",
            "original",
        )
        self.transpose_source_combo.addItem(
            "분리된 보컬 stem",
            "vocals",
        )
        self.transpose_source_combo.currentIndexChanged.connect(
            self.update_transpose_preview
        )

        self.transpose_source_label = QLabel(
            "선택된 파일 없음"
        )
        self.transpose_source_label.setWordWrap(
            True
        )

        self.rubberband_status_label = QLabel(
            "RubberBand 확인 전"
        )
        self.rubberband_status_label.setWordWrap(
            True
        )

        source_layout.addRow(
            "입력",
            self.transpose_source_combo,
        )
        source_layout.addRow(
            "현재 소스",
            self.transpose_source_label,
        )
        source_layout.addRow(
            "Pitch Shift 엔진",
            self.rubberband_status_label,
        )

        transpose_root.addWidget(
            source_group
        )

        key_group = QGroupBox(
            "키 변경"
        )
        key_layout = QFormLayout(
            key_group
        )

        self.key_semitone_spin = QSpinBox()
        self.key_semitone_spin.setRange(
            -12,
            12,
        )
        self.key_semitone_spin.setValue(
            0
        )
        self.key_semitone_spin.setPrefix(
            "키 "
        )
        self.key_semitone_spin.valueChanged.connect(
            self.update_transpose_preview
        )

        self.transpose_formant_check = QCheckBox(
            "Formant 보존 (원래 보컬 음색 유지)"
        )
        self.transpose_formant_check.setChecked(
            True
        )
        self.transpose_formant_check.setToolTip(
            "체크하면 키를 바꿔도 목소리의 성도/음색 특성을 "
            "가능한 유지합니다. 더 극적인 음색 변화를 원하면 끄세요."
        )

        self.transpose_quality_combo = QComboBox()
        self.transpose_quality_combo.addItem(
            "음질 우선",
            "quality",
        )
        self.transpose_quality_combo.addItem(
            "속도 우선",
            "speed",
        )
        self.transpose_quality_combo.addItem(
            "일관성 우선",
            "consistency",
        )

        self.transpose_preview_label = QLabel(
            "키 변경: 0 semitone"
        )
        self.transpose_preview_label.setWordWrap(
            True
        )

        key_layout.addRow(
            "변경량",
            self.key_semitone_spin,
        )
        key_layout.addRow(
            "",
            self.transpose_formant_check,
        )
        key_layout.addRow(
            "처리 품질",
            self.transpose_quality_combo,
        )
        key_layout.addRow(
            "예상 결과",
            self.transpose_preview_label,
        )

        transpose_root.addWidget(
            key_group
        )

        self.seedvc_group = QGroupBox(
            "Seed-VC SVC 옵션"
        )
        seedvc_layout = QFormLayout(
            self.seedvc_group
        )

        self.seedvc_reference_combo = QComboBox()
        self.seedvc_reference_combo.addItem(
            "같은 가수 음색 유지 - 보컬 stem에서 자동 참조",
            "auto",
        )
        self.seedvc_reference_combo.addItem(
            "별도 참조 음성 파일 사용",
            "custom",
        )
        self.seedvc_reference_combo.currentIndexChanged.connect(
            self.update_seedvc_reference_ui
        )

        reference_row = QWidget()
        reference_row_layout = QHBoxLayout(
            reference_row
        )
        reference_row_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self.seedvc_reference_label = QLabel(
            "자동 참조: 보컬이 잘 들리는 약 12초 구간을 자동 선택"
        )
        self.seedvc_reference_label.setWordWrap(
            True
        )

        self.seedvc_reference_button = QPushButton(
            "참조 파일 선택"
        )
        self.seedvc_reference_button.clicked.connect(
            self.choose_seedvc_reference
        )

        reference_row_layout.addWidget(
            self.seedvc_reference_label,
            1,
        )
        reference_row_layout.addWidget(
            self.seedvc_reference_button,
        )

        self.seedvc_steps_spin = QSpinBox()
        self.seedvc_steps_spin.setRange(
            4,
            50,
        )
        self.seedvc_steps_spin.setValue(
            30
        )
        self.seedvc_steps_spin.setSuffix(
            " steps"
        )
        self.seedvc_steps_spin.setToolTip(
            "노래 변환은 30~50 step이 고음질 권장 범위입니다. "
            "값이 높을수록 느려집니다."
        )

        self.seedvc_cfg_spin = QDoubleSpinBox()
        self.seedvc_cfg_spin.setRange(
            0.0,
            1.2,
        )
        self.seedvc_cfg_spin.setDecimals(
            2
        )
        self.seedvc_cfg_spin.setSingleStep(
            0.05
        )
        self.seedvc_cfg_spin.setValue(
            0.70
        )

        self.seedvc_fp16_check = QCheckBox(
            "FP16 사용 (CUDA 권장)"
        )
        self.seedvc_fp16_check.setChecked(
            True
        )

        self.seedvc_help_label = QLabel(
            "원곡 전체 모드: BS-RoFormer 2-stem → "
            "보컬 Seed-VC SVC → 반주 동일 키 이동 → 재합성\n"
            "첫 실행은 Seed-VC/Whisper/RMVPE/BigVGAN 모델 다운로드 때문에 오래 걸릴 수 있습니다."
        )
        self.seedvc_help_label.setWordWrap(
            True
        )

        seedvc_layout.addRow(
            "참조 음색",
            self.seedvc_reference_combo,
        )
        seedvc_layout.addRow(
            "참조",
            reference_row,
        )
        seedvc_layout.addRow(
            "Diffusion",
            self.seedvc_steps_spin,
        )
        seedvc_layout.addRow(
            "CFG",
            self.seedvc_cfg_spin,
        )
        seedvc_layout.addRow(
            "",
            self.seedvc_fp16_check,
        )
        seedvc_layout.addRow(
            "",
            self.seedvc_help_label,
        )

        transpose_root.addWidget(
            self.seedvc_group
        )

        self.seedvc_group.setVisible(
            False
        )
        self.seedvc_reference_button.setEnabled(
            False
        )

        self.rvc_group = QGroupBox(
            "RVC + RMVPE 옵션"
        )
        rvc_layout = QFormLayout(
            self.rvc_group
        )

        model_row = QWidget()
        model_row_layout = QHBoxLayout(
            model_row
        )
        model_row_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self.rvc_model_label = QLabel(
            self.rvc_model_path
            or "RVC .pth 모델을 선택하세요."
        )
        self.rvc_model_label.setWordWrap(
            True
        )

        self.rvc_model_button = QPushButton(
            "모델 선택"
        )
        self.rvc_model_button.clicked.connect(
            self.choose_rvc_model
        )

        model_row_layout.addWidget(
            self.rvc_model_label,
            1,
        )
        model_row_layout.addWidget(
            self.rvc_model_button,
        )

        index_row = QWidget()
        index_row_layout = QHBoxLayout(
            index_row
        )
        index_row_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self.rvc_index_label = QLabel(
            self.rvc_index_path
            or "Index 미선택 - 선택하지 않으면 index rate 0으로 처리"
        )
        self.rvc_index_label.setWordWrap(
            True
        )

        self.rvc_index_button = QPushButton(
            "Index 선택"
        )
        self.rvc_index_button.clicked.connect(
            self.choose_rvc_index
        )

        self.rvc_index_clear_button = QPushButton(
            "해제"
        )
        self.rvc_index_clear_button.clicked.connect(
            self.clear_rvc_index
        )

        index_row_layout.addWidget(
            self.rvc_index_label,
            1,
        )
        index_row_layout.addWidget(
            self.rvc_index_button,
        )
        index_row_layout.addWidget(
            self.rvc_index_clear_button,
        )

        self.rvc_f0_label = QLabel(
            "RMVPE (CUDA 자동 사용)"
        )

        self.rvc_index_rate_spin = QDoubleSpinBox()
        self.rvc_index_rate_spin.setRange(
            0.0,
            1.0,
        )
        self.rvc_index_rate_spin.setDecimals(
            2
        )
        self.rvc_index_rate_spin.setSingleStep(
            0.05
        )
        self.rvc_index_rate_spin.setValue(
            0.75
        )

        self.rvc_protect_spin = QDoubleSpinBox()
        self.rvc_protect_spin.setRange(
            0.0,
            0.50,
        )
        self.rvc_protect_spin.setDecimals(
            2
        )
        self.rvc_protect_spin.setSingleStep(
            0.01
        )
        self.rvc_protect_spin.setValue(
            0.33
        )
        self.rvc_protect_spin.setToolTip(
            "무성음/자음 보호. 기본 0.33. "
            "낮출수록 변환 음색이 강해지고, 높일수록 원본 발음 보호가 강해집니다."
        )

        self.rvc_rms_mix_spin = QDoubleSpinBox()
        self.rvc_rms_mix_spin.setRange(
            0.0,
            1.0,
        )
        self.rvc_rms_mix_spin.setDecimals(
            2
        )
        self.rvc_rms_mix_spin.setSingleStep(
            0.05
        )
        self.rvc_rms_mix_spin.setValue(
            1.00
        )

        self.rvc_speaker_spin = QSpinBox()
        self.rvc_speaker_spin.setRange(
            0,
            109,
        )
        self.rvc_speaker_spin.setValue(
            0
        )

        self.rvc_help_label = QLabel(
            "남성 음색으로 학습된 RVC .pth 모델을 선택하면 "
            "RMVPE가 원곡 보컬의 F0를 추적하고, 현재 키 변경값을 적용한 뒤 "
            "선택한 모델 음색으로 보컬을 재합성합니다.\n"
            ".index 파일은 권장하지만 필수는 아닙니다."
        )
        self.rvc_help_label.setWordWrap(
            True
        )

        rvc_layout.addRow(
            "RVC 모델",
            model_row,
        )
        rvc_layout.addRow(
            "Feature Index",
            index_row,
        )
        rvc_layout.addRow(
            "F0",
            self.rvc_f0_label,
        )
        rvc_layout.addRow(
            "Index Rate",
            self.rvc_index_rate_spin,
        )
        rvc_layout.addRow(
            "Protect",
            self.rvc_protect_spin,
        )
        rvc_layout.addRow(
            "RMS Mix",
            self.rvc_rms_mix_spin,
        )
        rvc_layout.addRow(
            "Speaker ID",
            self.rvc_speaker_spin,
        )
        rvc_layout.addRow(
            "",
            self.rvc_help_label,
        )

        transpose_root.addWidget(
            self.rvc_group
        )
        self.rvc_group.setVisible(
            False
        )

        output_group = QGroupBox(
            "출력"
        )
        output_layout = QFormLayout(
            output_group
        )

        self.transpose_format_combo = QComboBox()
        self.transpose_format_combo.addItem(
            "WAV (24-bit, 권장)",
            "wav",
        )
        self.transpose_format_combo.addItem(
            "FLAC",
            "flac",
        )
        self.transpose_format_combo.addItem(
            "MP3",
            "mp3",
        )
        self.transpose_format_combo.addItem(
            "M4A / AAC",
            "m4a",
        )

        self.transpose_subtitle_check = QCheckBox(
            "현재 분석 결과가 있으면 변환된 음계 자막도 함께 생성"
        )
        self.transpose_subtitle_check.setChecked(
            True
        )

        self.transpose_button = QPushButton(
            "키 변경 음원 생성"
        )
        self.transpose_button.setEnabled(
            False
        )
        self.transpose_button.clicked.connect(
            self.start_transpose
        )

        output_layout.addRow(
            "파일 형식",
            self.transpose_format_combo,
        )
        output_layout.addRow(
            "",
            self.transpose_subtitle_check,
        )
        output_layout.addRow(
            "",
            self.transpose_button,
        )

        transpose_root.addWidget(
            output_group
        )

        self.transpose_progress = QProgressBar()
        self.transpose_progress.setRange(
            0,
            100,
        )
        self.transpose_progress.setValue(
            0
        )

        self.transpose_progress_text = QLabel(
            "준비"
        )
        self.transpose_progress_text.setWordWrap(
            True
        )

        transpose_root.addWidget(
            self.transpose_progress
        )
        transpose_root.addWidget(
            self.transpose_progress_text
        )

        self.update_transpose_engine_ui()
        self.update_seedvc_reference_ui()

        transpose_root.addStretch(1)

        root.addWidget(
            self.main_tabs,
            1,
        )

        file_row = QHBoxLayout()

        self.path_label = QLabel(
            "음원/영상 파일을 선택하세요."
        )

        self.open_button = QPushButton("파일 선택")
        self.open_button.clicked.connect(
            self.choose_audio
        )

        self.analyze_button = QPushButton("분석 시작")
        self.analyze_button.clicked.connect(
            self.start_analysis
        )
        self.analyze_button.setEnabled(False)

        self.export_button = QPushButton(
            "음표 CSV 저장"
        )
        self.export_button.clicked.connect(
            self.export_note_csv
        )
        self.export_button.setEnabled(False)

        self.export_raw_button = QPushButton(
            "Raw CSV 저장"
        )
        self.export_raw_button.clicked.connect(
            self.export_raw_csv
        )
        self.export_raw_button.setEnabled(False)

        self.save_vocal_button = QPushButton(
            "분리 보컬 WAV 저장"
        )
        self.save_vocal_button.clicked.connect(
            self.save_separated_vocal
        )
        self.save_vocal_button.setEnabled(False)

        file_row.addWidget(self.path_label, 1)
        file_row.addWidget(self.open_button)
        file_row.addWidget(self.analyze_button)
        file_row.addWidget(self.export_button)
        file_row.addWidget(self.export_raw_button)
        file_row.addWidget(self.save_vocal_button)

        analysis_root.addLayout(file_row)

        self.ffmpeg_label = QLabel(
            ffmpeg_status_text()
        )
        self.ffmpeg_label.setWordWrap(True)
        analysis_root.addWidget(self.ffmpeg_label)

        self.separator_label = QLabel(
            separator_status_text()
        )
        self.separator_label.setWordWrap(True)
        analysis_root.addWidget(self.separator_label)

        separator_group = QGroupBox(
            "AI 보컬 분리"
        )
        separator_layout = QFormLayout(
            separator_group
        )

        self.analysis_mode_combo = QComboBox()
        self.analysis_mode_combo.addItem(
            "보컬 분리 후 분석 (권장)",
            "vocal",
        )
        self.analysis_mode_combo.addItem(
            "원본 전체 믹스 분석",
            "original",
        )

        self.separator_model_combo = QComboBox()
        self.separator_model_combo.addItem(
            "BS-RoFormer Vocal",
            DEFAULT_MODEL,
        )

        self.separator_cache_check = QCheckBox(
            "분리된 vocals.wav 캐시 사용"
        )
        self.separator_cache_check.setChecked(
            True
        )

        self.separator_autocast_check = QCheckBox(
            "GPU autocast 사용"
        )
        self.separator_autocast_check.setChecked(
            True
        )

        self.separator_log_label = QLabel(
            "분리 로그: -"
        )
        self.separator_log_label.setWordWrap(
            True
        )

        separator_layout.addRow(
            "분석 대상",
            self.analysis_mode_combo,
        )
        separator_layout.addRow(
            "분리 모델",
            self.separator_model_combo,
        )
        separator_layout.addRow(
            "",
            self.separator_cache_check,
        )
        separator_layout.addRow(
            "",
            self.separator_autocast_check,
        )
        separator_layout.addRow(
            "상태",
            self.separator_log_label,
        )

        analysis_root.addWidget(separator_group)

        gate_group = QGroupBox(
            "보컬 활동 게이트 - Engine v3"
        )
        gate_layout = QFormLayout(
            gate_group
        )

        self.energy_gate_check = QCheckBox(
            "RMS 기반 보컬 활동 게이트 사용"
        )
        self.energy_gate_check.setChecked(True)

        self.energy_margin_spin = QDoubleSpinBox()
        self.energy_margin_spin.setRange(
            10.0,
            60.0,
        )
        self.energy_margin_spin.setValue(32.0)
        self.energy_margin_spin.setSuffix(" dB")
        self.energy_margin_spin.setToolTip(
            "곡의 강한 보컬 레벨(90 percentile)보다 "
            "몇 dB 아래까지 활동으로 인정할지 결정합니다.\n"
            "값이 작을수록 더 엄격하게 제거합니다."
        )

        self.energy_floor_spin = QDoubleSpinBox()
        self.energy_floor_spin.setRange(
            -100.0,
            -20.0,
        )
        self.energy_floor_spin.setValue(-55.0)
        self.energy_floor_spin.setSuffix(" dBFS")
        self.energy_floor_spin.setToolTip(
            "적응형 threshold가 지나치게 낮아지는 것을 막는 "
            "절대 하한입니다."
        )

        self.energy_hyst_spin = QDoubleSpinBox()
        self.energy_hyst_spin.setRange(
            0.0,
            15.0,
        )
        self.energy_hyst_spin.setValue(4.0)
        self.energy_hyst_spin.setSuffix(" dB")

        self.min_activity_spin = QDoubleSpinBox()
        self.min_activity_spin.setRange(
            0.0,
            500.0,
        )
        self.min_activity_spin.setValue(80.0)
        self.min_activity_spin.setSuffix(" ms")

        self.activity_gap_spin = QDoubleSpinBox()
        self.activity_gap_spin.setRange(
            0.0,
            500.0,
        )
        self.activity_gap_spin.setValue(100.0)
        self.activity_gap_spin.setSuffix(" ms")

        self.range_min_note_spin = QDoubleSpinBox()
        self.range_min_note_spin.setRange(
            0.0,
            1000.0,
        )
        self.range_min_note_spin.setValue(100.0)
        self.range_min_note_spin.setSuffix(" ms")

        self.range_conf_spin = QDoubleSpinBox()
        self.range_conf_spin.setRange(
            0.0,
            1.0,
        )
        self.range_conf_spin.setDecimals(2)
        self.range_conf_spin.setSingleStep(0.05)
        self.range_conf_spin.setValue(0.35)

        gate_layout.addRow(
            "",
            self.energy_gate_check,
        )
        gate_layout.addRow(
            "활동 margin",
            self.energy_margin_spin,
        )
        gate_layout.addRow(
            "절대 floor",
            self.energy_floor_spin,
        )
        gate_layout.addRow(
            "게이트 hysteresis",
            self.energy_hyst_spin,
        )
        gate_layout.addRow(
            "최소 활동 구간",
            self.min_activity_spin,
        )
        gate_layout.addRow(
            "활동 gap 연결",
            self.activity_gap_spin,
        )
        gate_layout.addRow(
            "최고/최저음 최소 지속",
            self.range_min_note_spin,
        )
        gate_layout.addRow(
            "최고/최저음 최소 신뢰도",
            self.range_conf_spin,
        )

        analysis_root.addWidget(gate_group)

        options_group = QGroupBox(
            "Pitch Engine 옵션"
        )
        options_layout = QFormLayout(
            options_group
        )

        self.fmin_spin = QDoubleSpinBox()
        self.fmin_spin.setRange(40.0, 500.0)
        self.fmin_spin.setValue(65.4)
        self.fmin_spin.setSuffix(" Hz")

        self.fmax_spin = QDoubleSpinBox()
        self.fmax_spin.setRange(
            300.0,
            2500.0,
        )
        self.fmax_spin.setValue(1396.9)
        self.fmax_spin.setSuffix(" Hz")

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(
            0.05,
            0.99,
        )
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.25)

        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(
            0.0,
            300.0,
        )
        self.dropout_spin.setValue(80.0)
        self.dropout_spin.setSuffix(" ms")

        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(1, 21)
        self.smoothing_spin.setSingleStep(2)
        self.smoothing_spin.setValue(5)

        self.hysteresis_spin = QDoubleSpinBox()
        self.hysteresis_spin.setRange(
            0.0,
            49.0,
        )
        self.hysteresis_spin.setValue(20.0)
        self.hysteresis_spin.setSuffix(" cent")

        self.min_note_spin = QDoubleSpinBox()
        self.min_note_spin.setRange(
            0.0,
            300.0,
        )
        self.min_note_spin.setValue(35.0)
        self.min_note_spin.setSuffix(" ms")

        self.show_raw_check = QCheckBox(
            "Raw pitch 그래프 같이 표시"
        )
        self.show_raw_check.setChecked(True)
        self.show_raw_check.stateChanged.connect(
            self.redraw_plot
        )

        options_layout.addRow(
            "최저 검출 주파수",
            self.fmin_spin,
        )
        options_layout.addRow(
            "최고 검출 주파수",
            self.fmax_spin,
        )
        options_layout.addRow(
            "유성음 신뢰도 기준",
            self.threshold_spin,
        )
        options_layout.addRow(
            "짧은 dropout 연결",
            self.dropout_spin,
        )
        options_layout.addRow(
            "Pitch smoothing",
            self.smoothing_spin,
        )
        options_layout.addRow(
            "음표 전환 hysteresis",
            self.hysteresis_spin,
        )
        options_layout.addRow(
            "짧은 note 병합",
            self.min_note_spin,
        )
        options_layout.addRow(
            "",
            self.show_raw_check,
        )

        analysis_root.addWidget(options_group)


        subtitle_group = QGroupBox(
            "음계 자막 생성 - v1.5"
        )
        subtitle_layout = QHBoxLayout(
            subtitle_group
        )

        subtitle_layout.addWidget(
            QLabel("포맷")
        )

        self.subtitle_format_combo = QComboBox()
        self.subtitle_format_combo.addItem(
            "ASS (권장)",
            "ass",
        )
        self.subtitle_format_combo.addItem(
            "SRT",
            "srt",
        )
        self.subtitle_format_combo.addItem(
            "ASS + SRT",
            "both",
        )
        subtitle_layout.addWidget(
            self.subtitle_format_combo
        )

        subtitle_layout.addWidget(
            QLabel("표시")
        )

        self.subtitle_note_format_combo = QComboBox()
        self.subtitle_note_format_combo.addItem(
            "2옥타브 레 (한글, 권장)",
            "korean",
        )
        self.subtitle_note_format_combo.addItem(
            "D4",
            "note",
        )
        self.subtitle_note_format_combo.addItem(
            "D4 · 2옥타브 레",
            "both",
        )
        subtitle_layout.addWidget(
            self.subtitle_note_format_combo
        )

        subtitle_layout.addWidget(
            QLabel("최대 시간")
        )

        self.subtitle_seconds_spin = QDoubleSpinBox()
        self.subtitle_seconds_spin.setRange(
            2.0,
            20.0,
        )
        self.subtitle_seconds_spin.setSingleStep(
            0.5
        )
        self.subtitle_seconds_spin.setValue(8.0)
        self.subtitle_seconds_spin.setSuffix(
            " 초"
        )
        subtitle_layout.addWidget(
            self.subtitle_seconds_spin
        )

        subtitle_layout.addWidget(
            QLabel("최대 음표")
        )

        self.subtitle_max_notes_spin = QSpinBox()
        self.subtitle_max_notes_spin.setRange(
            4,
            40,
        )
        self.subtitle_max_notes_spin.setValue(
            18
        )
        subtitle_layout.addWidget(
            self.subtitle_max_notes_spin
        )

        subtitle_layout.addWidget(
            QLabel("긴 공백 분리")
        )

        self.subtitle_silence_spin = QDoubleSpinBox()
        self.subtitle_silence_spin.setRange(
            100.0,
            3000.0,
        )
        self.subtitle_silence_spin.setSingleStep(
            50.0
        )
        self.subtitle_silence_spin.setDecimals(0)
        self.subtitle_silence_spin.setValue(
            400.0
        )
        self.subtitle_silence_spin.setSuffix(
            " ms"
        )
        subtitle_layout.addWidget(
            self.subtitle_silence_spin
        )

        subtitle_layout.addWidget(
            QLabel("한 줄")
        )

        self.subtitle_line_notes_spin = QSpinBox()
        self.subtitle_line_notes_spin.setRange(
            3,
            20,
        )
        self.subtitle_line_notes_spin.setValue(
            6
        )
        self.subtitle_line_notes_spin.setSuffix(
            " 개"
        )
        subtitle_layout.addWidget(
            self.subtitle_line_notes_spin
        )

        self.subtitle_highlight_check = QCheckBox(
            "현재 음 【괄호 + 색상】 강조"
        )
        self.subtitle_highlight_check.setChecked(
            True
        )
        self.subtitle_highlight_check.setToolTip(
            "ASS에서 현재 재생 중인 음표를 【괄호】로 감싸고 색상/굵기로 강조합니다. "
            "SRT에서는 적용되지 않습니다."
        )
        subtitle_layout.addWidget(
            self.subtitle_highlight_check
        )

        self.subtitle_generate_button = QPushButton(
            "음계 자막 생성"
        )
        self.subtitle_generate_button.setEnabled(
            False
        )
        self.subtitle_generate_button.clicked.connect(
            self.generate_note_subtitles
        )
        subtitle_layout.addWidget(
            self.subtitle_generate_button
        )

        subtitle_layout.addStretch(1)
        result_root.addWidget(subtitle_group)

        stats_row = QHBoxLayout()

        self.source_label = QLabel("소스: -")
        self.duration_label = QLabel("길이: -")
        self.gate_label = QLabel("활동 게이트: -")
        self.raw_label = QLabel("Raw 유성: -")
        self.accepted_label = QLabel(
            "임계값 통과: -"
        )
        self.processed_label = QLabel(
            "보정 피치: -"
        )
        self.coverage_label = QLabel(
            "커버리지: -"
        )
        self.low_label = QLabel("최저음: -")
        self.high_label = QLabel("최고음: -")
        self.segment_label = QLabel(
            "음표 구간: -"
        )

        for label in (
            self.source_label,
            self.duration_label,
            self.gate_label,
            self.raw_label,
            self.accepted_label,
            self.processed_label,
            self.coverage_label,
            self.low_label,
            self.high_label,
            self.segment_label,
        ):
            stats_row.addWidget(label)

        stats_row.addStretch(1)
        result_root.addLayout(stats_row)

        splitter = QSplitter()

        axis = NoteAxis(orientation="left")
        self.plot = pg.PlotWidget(
            axisItems={"left": axis}
        )
        self.plot.setLabel(
            "bottom",
            "시간",
            units="s",
        )
        self.plot.setLabel("left", "음계")
        self.plot.showGrid(
            x=True,
            y=True,
            alpha=0.25,
        )
        self.plot.setMouseEnabled(
            x=True,
            y=True,
        )
        self.plot.addLegend()
        splitter.addWidget(self.plot)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "시작",
                "끝",
                "길이",
                "음계",
                "한국식",
                "대표 Hz",
                "Cent",
                "신뢰도",
            ]
        )
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(
            True
        )
        splitter.addWidget(self.table)

        splitter.setSizes([980, 590])
        result_root.addWidget(splitter, 1)

        progress_row = QHBoxLayout()

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.progress_text = QLabel("준비")

        progress_row.addWidget(
            self.progress,
            1,
        )
        progress_row.addWidget(
            self.progress_text
        )
        analysis_root.addLayout(progress_row)

        analysis_root.addStretch(1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

    def _build_menu(self):
        file_menu = self.menuBar().addMenu(
            "파일"
        )

        open_action = QAction(
            "음원/영상 열기",
            self,
        )
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(
            self.choose_audio
        )
        file_menu.addAction(open_action)

        export_action = QAction(
            "음표 CSV 저장",
            self,
        )
        export_action.setShortcut("Ctrl+S")
        export_action.triggered.connect(
            self.export_note_csv
        )
        file_menu.addAction(export_action)

        raw_action = QAction(
            "Raw CSV 저장",
            self,
        )
        raw_action.setShortcut("Ctrl+Shift+S")
        raw_action.triggered.connect(
            self.export_raw_csv
        )
        file_menu.addAction(raw_action)

        file_menu.addSeparator()

        exit_action = QAction("종료", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _release_vocal_resource(self):
        if self.current_vocal_resource is not None:
            self.current_vocal_resource.cleanup()
            self.current_vocal_resource = None

    def _is_supported_media_path(
        self,
        path: str | Path,
    ) -> bool:
        try:
            p = Path(path)
        except Exception:
            return False

        return (
            p.is_file()
            and p.suffix.lower()
            in SUPPORTED_MEDIA_EXTENSIONS
        )

    def _last_open_directory(self) -> str:
        saved = self.settings.value(
            "last_open_dir",
            "",
            type=str,
        )

        if saved:
            path = Path(saved)
            if path.is_dir():
                return str(path)

        return str(Path.home())

    def _set_audio_path(
        self,
        path: str | Path,
        *,
        source: str = "dialog",
    ) -> bool:
        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            QMessageBox.information(
                self,
                "분석 중",
                "현재 분석이 끝난 후 다른 파일을 선택하세요.",
            )
            return False

        p = Path(path)

        if not self._is_supported_media_path(p):
            QMessageBox.warning(
                self,
                "지원하지 않는 파일",
                (
                    "지원되는 음원/영상 파일이 아닙니다.\n\n"
                    f"{p}\n\n"
                    "지원 확장자:\n"
                    + ", ".join(
                        sorted(
                            SUPPORTED_MEDIA_EXTENSIONS
                        )
                    )
                ),
            )
            return False

        try:
            p = p.resolve()
        except OSError:
            p = p.absolute()

        self._release_vocal_resource()

        self.audio_path = str(p)
        self.result = None
        self.pipeline_result = None

        self.path_label.setText(
            self.audio_path
        )
        self.analyze_button.setEnabled(True)
        self.transpose_button.setEnabled(True)
        self.update_transpose_preview()
        self.export_button.setEnabled(False)
        self.export_raw_button.setEnabled(False)
        self.save_vocal_button.setEnabled(False)
        self.subtitle_generate_button.setEnabled(False)

        self.settings.setValue(
            "last_open_dir",
            str(p.parent),
        )

        self._reset_result_view()

        if hasattr(self, "main_tabs"):
            self.main_tabs.setCurrentWidget(
                self.analysis_tab
            )

        if source == "drop":
            message = (
                f"드래그 앤 드롭으로 선택: {p.name}"
            )
        else:
            message = (
                f"파일 선택 완료: {p.name}"
            )

        self.statusBar().showMessage(
            message,
            8000,
        )
        return True

    def choose_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "분석할 음원/영상 선택",
            self._last_open_directory(),
            (
                "Supported media "
                "(*.mp3 *.wav *.flac *.ogg *.m4a *.mp4 *.aac *.webm "
                "*.mkv *.mov *.wma *.opus *.m4v);;"
                "All files (*.*)"
            ),
        )
        if not path:
            return

        self._set_audio_path(
            path,
            source="dialog",
        )

    def dragEnterEvent(self, event):
        mime = event.mimeData()

        if not mime.hasUrls():
            event.ignore()
            return

        for url in mime.urls():
            if (
                url.isLocalFile()
                and self._is_supported_media_path(
                    url.toLocalFile()
                )
            ):
                event.acceptProposedAction()
                return

        event.ignore()

    def dropEvent(self, event):
        mime = event.mimeData()

        if not mime.hasUrls():
            event.ignore()
            return

        for url in mime.urls():
            if not url.isLocalFile():
                continue

            path = url.toLocalFile()

            if self._is_supported_media_path(
                path
            ):
                if self._set_audio_path(
                    path,
                    source="drop",
                ):
                    event.acceptProposedAction()
                else:
                    event.ignore()
                return

        event.ignore()

    def _reset_result_view(self):
        self.plot.clear()
        self.plot.addLegend()
        self.table.setRowCount(0)

        self.source_label.setText("소스: -")
        self.duration_label.setText("길이: -")
        self.gate_label.setText(
            "활동 게이트: -"
        )
        self.raw_label.setText("Raw 유성: -")
        self.accepted_label.setText(
            "임계값 통과: -"
        )
        self.processed_label.setText(
            "보정 피치: -"
        )
        self.coverage_label.setText(
            "커버리지: -"
        )
        self.low_label.setText("최저음: -")
        self.high_label.setText("최고음: -")
        self.segment_label.setText(
            "음표 구간: -"
        )

        self.separator_log_label.setText(
            "분리 로그: -"
        )
        self.progress.setValue(0)
        self.progress_text.setText("준비")

    def start_analysis(self):
        if not self.audio_path:
            return

        if self.worker and self.worker.isRunning():
            return

        use_vocal = (
            self.analysis_mode_combo.currentData()
            == "vocal"
        )

        self._release_vocal_resource()

        self.open_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.export_raw_button.setEnabled(False)
        self.save_vocal_button.setEnabled(False)
        self.subtitle_generate_button.setEnabled(False)

        model_filename = (
            self.separator_model_combo.currentData()
            or DEFAULT_MODEL
        )

        self.worker = AnalysisThread(
            path=self.audio_path,
            use_vocal_separation=use_vocal,
            separator_model=model_filename,
            separator_cache=self.separator_cache_check.isChecked(),
            separator_autocast=self.separator_autocast_check.isChecked(),

            fmin=self.fmin_spin.value(),
            fmax=self.fmax_spin.value(),
            threshold=self.threshold_spin.value(),
            max_dropout_ms=self.dropout_spin.value(),
            smoothing_window=self.smoothing_spin.value(),
            hysteresis_cents=self.hysteresis_spin.value(),
            min_note_ms=self.min_note_spin.value(),

            use_energy_gate=self.energy_gate_check.isChecked(),
            energy_margin_db=self.energy_margin_spin.value(),
            energy_floor_dbfs=self.energy_floor_spin.value(),
            energy_hysteresis_db=self.energy_hyst_spin.value(),
            min_activity_ms=self.min_activity_spin.value(),
            max_activity_gap_ms=self.activity_gap_spin.value(),
            range_min_note_ms=self.range_min_note_spin.value(),
            range_min_confidence=self.range_conf_spin.value(),

            parent=self,
        )

        self.worker.progress_changed.connect(
            self.on_progress
        )
        self.worker.separator_log.connect(
            self.on_separator_log
        )
        self.worker.analysis_done.connect(
            self.on_analysis_done
        )
        self.worker.analysis_failed.connect(
            self.on_analysis_failed
        )
        self.worker.finished.connect(
            self.on_worker_finished
        )
        self.worker.start()

    def on_progress(
        self,
        percent: int,
        text: str,
    ):
        self.progress.setValue(percent)
        self.progress_text.setText(text)
        self.statusBar().showMessage(text)

    def on_separator_log(self, text: str):
        compact = " ".join(text.split())
        self.separator_log_label.setText(
            compact[-180:]
        )

    def on_analysis_done(
        self,
        pipeline: PipelineResult,
    ):
        self.pipeline_result = pipeline
        self.result = pipeline.pitch
        self.current_vocal_resource = (
            pipeline.vocal_resource
        )

        self._draw_result(pipeline)

        self.export_button.setEnabled(True)
        self.export_raw_button.setEnabled(True)
        self.save_vocal_button.setEnabled(
            pipeline.used_vocal_separation
            and pipeline.vocal_resource is not None
        )
        self.subtitle_generate_button.setEnabled(
            bool(self.result and self.result.segments)
        )
        self.update_transpose_preview()

        if hasattr(self, "main_tabs"):
            self.main_tabs.setCurrentWidget(
                self.result_tab
            )

        self.statusBar().showMessage(
            "완료 - Activity Gate Engine v3",
            12000,
        )

    def on_analysis_failed(
        self,
        error_text: str,
    ):
        self.result = None
        self.pipeline_result = None

        QMessageBox.critical(
            self,
            "분석 실패",
            error_text[-7000:],
        )

    def on_worker_finished(self):
        self.open_button.setEnabled(True)
        self.analyze_button.setEnabled(
            bool(self.audio_path)
        )

    def _draw_result(
        self,
        pipeline: PipelineResult,
    ):
        result = pipeline.pitch

        source_text = (
            "분리 보컬"
            if pipeline.used_vocal_separation
            else "원본 믹스"
        )
        if (
            pipeline.vocal_resource
            and pipeline.vocal_resource.cache_hit
        ):
            source_text += "(캐시)"

        self.source_label.setText(
            f"소스: {source_text}"
        )
        self.duration_label.setText(
            f"길이: {result.duration:.1f}초"
        )

        if result.energy_gate_enabled:
            self.gate_label.setText(
                f"활동 게이트: {result.energy_threshold_dbfs:.1f} dBFS"
            )
        else:
            self.gate_label.setText(
                "활동 게이트: OFF"
            )

        self.raw_label.setText(
            f"Raw 유성: {result.raw_voiced_seconds:.1f}초"
        )
        self.accepted_label.setText(
            f"통과: {result.accepted_seconds:.1f}초"
        )
        self.processed_label.setText(
            f"보정: {result.processed_seconds:.1f}초"
        )
        self.coverage_label.setText(
            f"커버리지: {result.processed_coverage_percent:.1f}%"
        )
        self.segment_label.setText(
            f"음표: {len(result.segments)}개"
        )

        if result.min_midi is None:
            self.low_label.setText(
                "최저음: 검출 없음"
            )
            self.high_label.setText(
                "최고음: 검출 없음"
            )
        else:
            self.low_label.setText(
                f"최저음: {midi_to_note_name(result.min_midi)}"
            )
            self.high_label.setText(
                f"최고음: {midi_to_note_name(result.max_midi)}"
            )

        self.redraw_plot()

        self.table.setRowCount(
            len(result.segments)
        )

        for row, segment in enumerate(
            result.segments
        ):
            values = (
                f"{segment.start:.3f}",
                f"{segment.end:.3f}",
                f"{segment.duration:.3f}",
                segment.note,
                segment.korean_note,
                f"{segment.avg_hz:.2f}",
                f"{segment.cents:+.1f}",
                f"{segment.confidence:.2f}",
            )

            for col, value in enumerate(
                values
            ):
                self.table.setItem(
                    row,
                    col,
                    QTableWidgetItem(value),
                )

    def redraw_plot(self):
        if not hasattr(self, "plot"):
            return

        self.plot.clear()
        self.plot.addLegend()

        result = self.result
        if result is None:
            return

        if self.show_raw_check.isChecked():
            self.plot.plot(
                result.times,
                result.raw_midi_float,
                pen=pg.mkPen(
                    (150, 150, 150, 100),
                    width=1,
                ),
                connect="finite",
                name="Raw pYIN",
            )

        self.plot.plot(
            result.times,
            result.midi_float,
            pen=pg.mkPen(
                (0, 180, 255),
                width=2,
            ),
            connect="finite",
            name="Processed v3",
        )

        valid = result.midi_float[
            np.isfinite(result.midi_float)
        ]

        if valid.size:
            self.plot.setYRange(
                max(
                    0,
                    float(
                        np.floor(
                            np.nanmin(valid)
                        )
                        - 2
                    ),
                ),
                min(
                    127,
                    float(
                        np.ceil(
                            np.nanmax(valid)
                        )
                        + 2
                    ),
                ),
                padding=0.02,
            )

        self.plot.setXRange(
            0,
            min(result.duration, 30.0),
            padding=0.01,
        )

    def _export_base_stem(self) -> str:
        if self.pipeline_result:
            return Path(
                self.pipeline_result.original_path
            ).stem
        return (
            Path(self.audio_path).stem
            if self.audio_path
            else "analysis"
        )

    def export_note_csv(self):
        if not self.result:
            return

        default_name = (
            self._export_base_stem()
            + "_pitch_segments_v14.csv"
        )

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "음표 CSV 저장",
            default_name,
            "CSV (*.csv)",
        )
        if not save_path:
            return

        with open(
            save_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "start_sec",
                    "end_sec",
                    "duration_sec",
                    "midi",
                    "note",
                    "korean_note",
                    "median_hz",
                    "cents",
                    "confidence",
                    "analysis_source",
                    "energy_threshold_dbfs",
                ]
            )

            source = (
                "separated_vocals"
                if (
                    self.pipeline_result
                    and self.pipeline_result.used_vocal_separation
                )
                else "original_mix"
            )

            for s in self.result.segments:
                writer.writerow(
                    [
                        f"{s.start:.6f}",
                        f"{s.end:.6f}",
                        f"{s.duration:.6f}",
                        s.midi,
                        s.note,
                        s.korean_note,
                        f"{s.avg_hz:.6f}",
                        f"{s.cents:.3f}",
                        f"{s.confidence:.6f}",
                        source,
                        (
                            f"{self.result.energy_threshold_dbfs:.3f}"
                            if self.result.energy_gate_enabled
                            else ""
                        ),
                    ]
                )

    def export_raw_csv(self):
        if not self.result:
            return

        result = self.result

        default_name = (
            self._export_base_stem()
            + "_raw_pitch_v14.csv"
        )

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Raw Pitch CSV 저장",
            default_name,
            "CSV (*.csv)",
        )
        if not save_path:
            return

        source = (
            "separated_vocals"
            if (
                self.pipeline_result
                and self.pipeline_result.used_vocal_separation
            )
            else "original_mix"
        )

        with open(
            save_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "time_sec",
                    "rms_dbfs",
                    "energy_gate_active",
                    "energy_threshold_dbfs",
                    "raw_hz",
                    "raw_midi_float",
                    "raw_note",
                    "raw_voiced_flag",
                    "voiced_probability",
                    "accepted_by_all_gates",
                    "processed_hz",
                    "processed_midi_float",
                    "processed_note",
                    "analysis_source",
                ]
            )

            for idx, time_sec in enumerate(
                result.times
            ):
                raw_hz = result.raw_f0_hz[idx]
                raw_midi = (
                    result.raw_midi_float[idx]
                )
                processed_hz = (
                    result.processed_f0_hz[idx]
                )
                processed_midi = (
                    result.midi_float[idx]
                )

                writer.writerow(
                    [
                        f"{float(time_sec):.6f}",
                        f"{float(result.frame_rms_dbfs[idx]):.3f}",
                        int(
                            bool(
                                result.energy_gate_mask[idx]
                            )
                        ),
                        (
                            f"{result.energy_threshold_dbfs:.3f}"
                            if result.energy_gate_enabled
                            else ""
                        ),
                        (
                            f"{float(raw_hz):.6f}"
                            if np.isfinite(raw_hz)
                            else ""
                        ),
                        (
                            f"{float(raw_midi):.6f}"
                            if np.isfinite(raw_midi)
                            else ""
                        ),
                        (
                            midi_to_note_name(
                                int(round(raw_midi))
                            )
                            if np.isfinite(raw_midi)
                            else ""
                        ),
                        int(
                            bool(
                                result.raw_voiced_flag[idx]
                            )
                        ),
                        (
                            f"{float(result.voiced_probability[idx]):.6f}"
                            if np.isfinite(
                                result.voiced_probability[idx]
                            )
                            else ""
                        ),
                        int(
                            bool(
                                result.accepted_mask[idx]
                            )
                        ),
                        (
                            f"{float(processed_hz):.6f}"
                            if np.isfinite(processed_hz)
                            else ""
                        ),
                        (
                            f"{float(processed_midi):.6f}"
                            if np.isfinite(processed_midi)
                            else ""
                        ),
                        (
                            midi_to_note_name(
                                int(
                                    round(
                                        processed_midi
                                    )
                                )
                            )
                            if np.isfinite(
                                processed_midi
                            )
                            else ""
                        ),
                        source,
                    ]
                )

    def refresh_rvc_training_status(self):
        if not hasattr(
            self,
            "rvc_training_runtime_label",
        ):
            return

        ready, status = (
            training_assets_status()
        )

        self.rvc_training_runtime_label.setText(
            status
        )

        dataset_dir = self.settings.value(
            "rvc_training_dataset_dir",
            "",
            type=str,
        )

        if dataset_dir:
            path = Path(
                dataset_dir
            )

            if path.is_dir():
                self.rvc_training_dataset_label.setText(
                    str(path)
                )
                self.rvc_training_dataset_status.setText(
                    dataset_status_text(
                        path
                    )
                )

        running = bool(
            self.rvc_training_worker
            and self.rvc_training_worker.isRunning()
        )

        self.rvc_training_start_button.setEnabled(
            ready
            and not running
        )
        self.rvc_training_stop_button.setEnabled(
            running
        )

    def choose_rvc_training_dataset(self):
        current = self.settings.value(
            "rvc_training_dataset_dir",
            self._last_open_directory(),
            type=str,
        )

        folder = QFileDialog.getExistingDirectory(
            self,
            "RVC 학습 데이터셋 폴더 선택",
            current,
        )

        if not folder:
            return

        path = Path(
            folder
        ).resolve()

        self.settings.setValue(
            "rvc_training_dataset_dir",
            str(path),
        )

        self.rvc_training_dataset_label.setText(
            str(path)
        )
        self.rvc_training_dataset_status.setText(
            dataset_status_text(
                path
            )
        )

        self.refresh_rvc_training_status()

    def start_rvc_training(self):
        if (
            self.rvc_training_worker
            and self.rvc_training_worker.isRunning()
        ):
            return

        dataset_dir = self.settings.value(
            "rvc_training_dataset_dir",
            "",
            type=str,
        )

        if not dataset_dir:
            QMessageBox.information(
                self,
                "RVC 모델 학습",
                "먼저 남성/타깃 음성 데이터셋 폴더를 선택하세요.",
            )
            return

        dataset_path = Path(
            dataset_dir
        )

        if not dataset_path.is_dir():
            QMessageBox.warning(
                self,
                "RVC 모델 학습",
                "선택한 데이터셋 폴더를 찾을 수 없습니다.",
            )
            return

        name = (
            self.rvc_training_name_edit.text().strip()
        )

        if not re.fullmatch(
            r"[A-Za-z0-9_-]+",
            name,
        ):
            QMessageBox.warning(
                self,
                "RVC 모델 학습",
                "모델 이름은 영문, 숫자, _ , - 만 사용할 수 있습니다.",
            )
            return

        answer = QMessageBox.question(
            self,
            "RVC 모델 학습 시작",
            (
                "RVC 학습은 상당한 시간이 걸릴 수 있습니다.\n\n"
                "데이터셋에는 한 사람의 남성/타깃 음성만 들어 있어야 하고, "
                "반주나 다른 화자가 섞이지 않는 것이 좋습니다.\n\n"
                f"모델: {name}\n"
                f"Epochs: {self.rvc_training_epochs_spin.value()}\n"
                f"Batch: {self.rvc_training_batch_spin.value()}\n\n"
                "학습을 시작할까요?"
            ),
        )

        if (
            answer
            != QMessageBox.StandardButton.Yes
        ):
            return

        self.rvc_training_log.clear()
        self.rvc_training_progress.setValue(
            0
        )
        self.rvc_training_progress_label.setText(
            "RVC 학습 시작 중..."
        )
        self.rvc_training_start_button.setEnabled(
            False
        )
        self.rvc_training_stop_button.setEnabled(
            True
        )

        self.rvc_training_worker = RVCTrainingThread(
            dataset_dir=str(
                dataset_path
            ),
            experiment_name=name,
            epochs=(
                self.rvc_training_epochs_spin.value()
            ),
            batch_size=(
                self.rvc_training_batch_spin.value()
            ),
            save_every=(
                self.rvc_training_save_spin.value()
            ),
            workers=(
                self.rvc_training_workers_spin.value()
            ),
            gpu_id=(
                self.rvc_training_gpu_spin.value()
            ),
            cache_gpu=(
                self.rvc_training_cache_gpu_check.isChecked()
            ),
            parent=self,
        )

        self.rvc_training_worker.progress_changed.connect(
            self.on_rvc_training_progress
        )
        self.rvc_training_worker.log_line.connect(
            self.on_rvc_training_log
        )
        self.rvc_training_worker.training_done.connect(
            self.on_rvc_training_done
        )
        self.rvc_training_worker.training_failed.connect(
            self.on_rvc_training_failed
        )
        self.rvc_training_worker.finished.connect(
            self.on_rvc_training_finished
        )

        self.rvc_training_worker.start()

    def stop_rvc_training(self):
        worker = (
            self.rvc_training_worker
        )

        if (
            worker is None
            or not worker.isRunning()
        ):
            return

        answer = QMessageBox.question(
            self,
            "RVC 학습 중지",
            (
                "현재 RVC 학습 프로세스를 중지할까요?\n\n"
                "같은 모델 이름으로 다시 시작하면 RVC 체크포인트가 "
                "남아 있는 경우 자동 resume될 수 있습니다."
            ),
        )

        if (
            answer
            != QMessageBox.StandardButton.Yes
        ):
            return

        self.rvc_training_progress_label.setText(
            "학습 중지 요청 중..."
        )
        worker.cancel()

    def on_rvc_training_progress(
        self,
        percent: int,
        text: str,
    ):
        self.rvc_training_progress.setValue(
            percent
        )
        self.rvc_training_progress_label.setText(
            text
        )
        self.statusBar().showMessage(
            text,
            5000,
        )

    def on_rvc_training_log(
        self,
        text: str,
    ):
        self.rvc_training_log.appendPlainText(
            text
        )

        scrollbar = (
            self.rvc_training_log.verticalScrollBar()
        )
        scrollbar.setValue(
            scrollbar.maximum()
        )

    def on_rvc_training_done(
        self,
        model_path: str,
        index_path: str,
    ):
        self.rvc_training_progress.setValue(
            100
        )
        self.rvc_training_progress_label.setText(
            "RVC 모델 학습 완료"
        )

        self.rvc_model_path = (
            model_path
        )
        self.rvc_index_path = (
            index_path
        )

        self.settings.setValue(
            "rvc_model_path",
            self.rvc_model_path,
        )
        self.settings.setValue(
            "rvc_index_path",
            self.rvc_index_path,
        )

        self.rvc_model_label.setText(
            self.rvc_model_path
        )

        self.rvc_index_label.setText(
            (
                self.rvc_index_path
                if self.rvc_index_path
                else "Index 생성 결과 없음"
            )
        )

        index = self.transpose_engine_combo.findData(
            "rvc"
        )

        if index >= 0:
            self.transpose_engine_combo.setCurrentIndex(
                index
            )

        self.main_tabs.setCurrentWidget(
            self.transpose_tab
        )
        self.update_transpose_engine_ui()

        message = (
            "RVC 모델 학습이 완료됐습니다.\n\n"
            f"모델:\n{model_path}"
        )

        if index_path:
            message += (
                f"\n\nIndex:\n{index_path}"
            )

        message += (
            "\n\nRVC 변환 탭에 자동 선택했습니다."
        )

        QMessageBox.information(
            self,
            "RVC 학습 완료",
            message,
        )

    def on_rvc_training_failed(
        self,
        error_text: str,
    ):
        self.rvc_training_progress_label.setText(
            "RVC 학습 실패/중지"
        )

        QMessageBox.critical(
            self,
            "RVC 학습 실패",
            (
                error_text[-9000:]
                + "\n\n로그:\n"
                + str(
                    Path(__file__).resolve().parent
                    / "logs"
                    / "rvc_training_last.log"
                )
            ),
        )

    def on_rvc_training_finished(self):
        self.rvc_training_start_button.setEnabled(
            True
        )
        self.rvc_training_stop_button.setEnabled(
            False
        )
        self.refresh_rvc_training_status()

    def _current_transpose_engine(self) -> str:
        if not hasattr(
            self,
            "transpose_engine_combo",
        ):
            return "rubberband"

        return (
            self.transpose_engine_combo.currentData()
            or "rubberband"
        )

    def update_transpose_engine_ui(self):
        if not hasattr(
            self,
            "seedvc_group",
        ):
            return

        engine = (
            self._current_transpose_engine()
        )

        is_seed = (
            engine == "seed_vc"
        )
        is_rvc = (
            engine == "rvc"
        )

        self.seedvc_group.setVisible(
            is_seed
        )

        if hasattr(
            self,
            "rvc_group",
        ):
            self.rvc_group.setVisible(
                is_rvc
            )

        self.transpose_formant_check.setEnabled(
            engine == "rubberband"
        )

        if is_seed:
            tooltip = (
                "Seed-VC 모드에서는 AI 보컬이 핵심이며 "
                "반주는 내부적으로 음질 우선 DSP를 사용합니다."
            )
        elif is_rvc:
            tooltip = (
                "RVC 모드에서는 RMVPE + RVC가 보컬을 변환하고 "
                "반주는 내부적으로 음질 우선 DSP를 사용합니다."
            )
        else:
            tooltip = ""

        self.transpose_quality_combo.setToolTip(
            tooltip
        )

        self.update_transpose_preview()

    def update_seedvc_reference_ui(self):
        if not hasattr(
            self,
            "seedvc_reference_combo",
        ):
            return

        custom = (
            self.seedvc_reference_combo.currentData()
            == "custom"
        )

        self.seedvc_reference_button.setEnabled(
            custom
        )

        if custom:
            if self.seedvc_reference_path:
                self.seedvc_reference_label.setText(
                    self.seedvc_reference_path
                )
            else:
                self.seedvc_reference_label.setText(
                    "참조 음성 파일을 선택하세요. "
                    "1~30초 정도의 깨끗한 음성이 좋습니다."
                )
        else:
            self.seedvc_reference_label.setText(
                "자동 참조: 보컬 stem에서 "
                "활동량이 높은 약 12초 구간을 자동 선택"
            )

        self.update_transpose_preview()

    def choose_seedvc_reference(self):
        last_dir = self.settings.value(
            "last_seedvc_reference_dir",
            self._last_open_directory(),
            type=str,
        )

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seed-VC 참조 음성 선택",
            last_dir,
            (
                "Audio "
                "(*.wav *.flac *.mp3 *.m4a *.aac *.ogg *.mp4 *.webm);;"
                "All files (*.*)"
            ),
        )

        if not path:
            return

        self.seedvc_reference_path = (
            str(
                Path(path).resolve()
            )
        )
        self.settings.setValue(
            "last_seedvc_reference_dir",
            str(
                Path(path).resolve().parent
            ),
        )

        self.update_seedvc_reference_ui()

    def _auto_find_rvc_index(
        self,
        model_path: Path,
    ) -> Path | None:
        stem = (
            model_path.stem.lower()
        )

        roots = [
            model_path.parent,
            model_path.parent.parent / "indices",
            Path(__file__).resolve().parent
            / "tools"
            / "rvc"
            / "assets"
            / "indices",
        ]

        candidates: list[Path] = []

        for folder in roots:
            if not folder.is_dir():
                continue

            try:
                candidates.extend(
                    p
                    for p in folder.rglob(
                        "*.index"
                    )
                    if p.is_file()
                )
            except OSError:
                continue

        if not candidates:
            return None

        def score(path: Path):
            name = path.stem.lower()
            value = 0

            if "added" in name:
                value += 4

            if stem in name:
                value += 5

            tokens = [
                token
                for token in re.split(
                    r"[^a-z0-9가-힣]+",
                    stem,
                )
                if len(token) >= 2
            ]

            value += sum(
                1
                for token in tokens
                if token in name
            )

            return (
                value,
                -len(name),
            )

        best = max(
            candidates,
            key=score,
        )

        if score(best)[0] <= 0:
            return None

        return best

    def choose_rvc_model(self):
        current = (
            Path(self.rvc_model_path)
            if self.rvc_model_path
            else None
        )

        if (
            current is not None
            and current.parent.is_dir()
        ):
            start_dir = str(
                current.parent
            )
        else:
            start_dir = self.settings.value(
                "last_rvc_model_dir",
                self._last_open_directory(),
                type=str,
            )

        path, _ = QFileDialog.getOpenFileName(
            self,
            "RVC 모델 선택",
            start_dir,
            "RVC model (*.pth);;All files (*.*)",
        )

        if not path:
            return

        model = Path(
            path
        ).resolve()

        self.rvc_model_path = str(
            model
        )
        self.settings.setValue(
            "rvc_model_path",
            self.rvc_model_path,
        )
        self.settings.setValue(
            "last_rvc_model_dir",
            str(
                model.parent
            ),
        )

        self.rvc_model_label.setText(
            self.rvc_model_path
        )

        if not self.rvc_index_path:
            found = (
                self._auto_find_rvc_index(
                    model
                )
            )

            if found is not None:
                self.rvc_index_path = str(
                    found
                )
                self.settings.setValue(
                    "rvc_index_path",
                    self.rvc_index_path,
                )
                self.rvc_index_label.setText(
                    self.rvc_index_path
                    + "\n(모델명 기준 자동 발견)"
                )

        self.update_transpose_preview()

    def choose_rvc_index(self):
        model = (
            Path(self.rvc_model_path)
            if self.rvc_model_path
            else None
        )

        if (
            model is not None
            and model.parent.is_dir()
        ):
            start_dir = str(
                model.parent
            )
        else:
            start_dir = self._last_open_directory()

        path, _ = QFileDialog.getOpenFileName(
            self,
            "RVC Feature Index 선택",
            start_dir,
            "RVC index (*.index);;All files (*.*)",
        )

        if not path:
            return

        self.rvc_index_path = str(
            Path(path).resolve()
        )
        self.settings.setValue(
            "rvc_index_path",
            self.rvc_index_path,
        )
        self.rvc_index_label.setText(
            self.rvc_index_path
        )

        self.update_transpose_preview()

    def clear_rvc_index(self):
        self.rvc_index_path = ""
        self.settings.setValue(
            "rvc_index_path",
            "",
        )
        self.rvc_index_label.setText(
            "Index 미선택 - 선택하지 않으면 index rate 0으로 처리"
        )
        self.update_transpose_preview()

    def _current_transpose_input(
        self,
    ) -> Path | None:
        source_mode = (
            self.transpose_source_combo.currentData()
            if hasattr(
                self,
                "transpose_source_combo",
            )
            else "original"
        )

        if source_mode == "vocals":
            resource = (
                self.current_vocal_resource
            )

            if (
                resource is not None
                and resource.vocal_path.is_file()
            ):
                return Path(
                    resource.vocal_path
                )

            return None

        if self.audio_path:
            path = Path(
                self.audio_path
            )
            if path.is_file():
                return path

        return None

    def update_transpose_preview(self):
        if not hasattr(
            self,
            "transpose_preview_label",
        ):
            return

        engine = (
            self._current_transpose_engine()
        )
        source_mode = (
            self.transpose_source_combo.currentData()
            if hasattr(
                self,
                "transpose_source_combo",
            )
            else "original"
        )

        input_path = (
            self._current_transpose_input()
        )

        ready = True

        if input_path is None:
            if source_mode == "vocals":
                source_text = (
                    "분리된 보컬 stem이 아직 없습니다. "
                    "먼저 보컬 분리 분석을 완료하세요."
                )
            else:
                source_text = (
                    "선택된 입력 파일이 없습니다."
                )

            self.transpose_source_label.setText(
                source_text
            )
            ready = False
        else:
            self.transpose_source_label.setText(
                str(input_path)
            )

        semitones = (
            self.key_semitone_spin.value()
        )
        ratio = semitone_to_ratio(
            semitones
        )

        preview = (
            f"{semitones:+d} semitone "
            f"(pitch ratio {ratio:.4f})"
        )

        if (
            self.result is not None
            and self.result.min_midi is not None
            and self.result.max_midi is not None
        ):
            shifted_min = max(
                0,
                min(
                    127,
                    self.result.min_midi
                    + semitones,
                ),
            )
            shifted_max = max(
                0,
                min(
                    127,
                    self.result.max_midi
                    + semitones,
                ),
            )

            preview += (
                "\n현재 분석 음역: "
                f"{midi_to_note_name(self.result.min_midi)}"
                f" ~ {midi_to_note_name(self.result.max_midi)}"
                "  →  "
                f"{midi_to_note_name(shifted_min)}"
                f" ~ {midi_to_note_name(shifted_max)}"
            )

        if engine == "seed_vc":
            status = (
                seed_vc_status_text()
            )

            if source_mode == "original":
                rb_ok, rb_status = (
                    rubberband_filter_available()
                )
                status += (
                    "\n반주 Pitch Shift: "
                    + rb_status
                )
                if not rb_ok:
                    ready = False

            if not seed_vc_available():
                ready = False

            if (
                self.seedvc_reference_combo.currentData()
                == "custom"
            ):
                reference = Path(
                    self.seedvc_reference_path
                ) if self.seedvc_reference_path else None

                if (
                    reference is None
                    or not reference.is_file()
                ):
                    ready = False
                    status += (
                        "\n참조 음성 파일을 선택해야 합니다."
                    )

            preview += (
                "\n엔진: Seed-VC SVC / F0 condition ON / "
                "auto F0 adjust OFF"
            )

        elif engine == "rvc":
            status = (
                rvc_status_text()
            )

            if source_mode == "original":
                rb_ok, rb_status = (
                    rubberband_filter_available()
                )
                status += (
                    "\n반주 Pitch Shift: "
                    + rb_status
                )
                if not rb_ok:
                    ready = False

            if not rvc_available():
                ready = False

            model = (
                Path(self.rvc_model_path)
                if self.rvc_model_path
                else None
            )

            if (
                model is None
                or not model.is_file()
                or model.suffix.lower() != ".pth"
            ):
                ready = False
                status += (
                    "\nRVC .pth 모델을 선택해야 합니다."
                )
            else:
                status += (
                    "\nRVC 모델: "
                    + model.name
                )

            if self.rvc_index_path:
                index = Path(
                    self.rvc_index_path
                )

                if index.is_file():
                    status += (
                        "\nIndex: "
                        + index.name
                    )
                else:
                    status += (
                        "\nIndex 파일을 찾을 수 없어 "
                        "실행 시 index rate 0으로 처리됩니다."
                    )
            else:
                status += (
                    "\nIndex 미사용: 실행 시 index rate 0"
                )

            preview += (
                "\n엔진: RVC + RMVPE / "
                f"Index Rate {self.rvc_index_rate_spin.value():.2f} / "
                f"Protect {self.rvc_protect_spin.value():.2f}"
            )

        else:
            rb_ok, status = (
                rubberband_filter_available()
            )

            if not rb_ok:
                ready = False

            preview += (
                "\n엔진: FFmpeg RubberBand"
            )

        self.transpose_engine_status_label.setText(
            status
        )
        self.rubberband_status_label.setText(
            status
        )
        self.transpose_preview_label.setText(
            preview
        )

        running = bool(
            self.transpose_worker
            and self.transpose_worker.isRunning()
        )

        self.transpose_button.setEnabled(
            ready
            and not running
        )

    def _default_transpose_filename(
        self,
        input_path: Path,
    ) -> str:
        semitones = (
            self.key_semitone_spin.value()
        )
        suffix = (
            self.transpose_format_combo.currentData()
            or "wav"
        )

        if semitones > 0:
            key_text = f"key_plus_{semitones}"
        elif semitones < 0:
            key_text = (
                f"key_minus_{abs(semitones)}"
            )
        else:
            key_text = "key_0"

        engine = (
            self._current_transpose_engine()
        )
        if engine == "seed_vc":
            engine_text = "seedvc"
        elif engine == "rvc":
            engine_text = "rvc"
        else:
            engine_text = "rubberband"

        return (
            f"{input_path.stem}_{engine_text}_{key_text}.{suffix}"
        )

    def start_transpose(self):
        if (
            self.transpose_worker
            and self.transpose_worker.isRunning()
        ):
            return

        input_path = (
            self._current_transpose_input()
        )

        if input_path is None:
            QMessageBox.information(
                self,
                "키 변경",
                "변환할 입력 파일을 사용할 수 없습니다.",
            )
            return

        semitones = (
            self.key_semitone_spin.value()
        )

        if semitones == 0:
            answer = QMessageBox.question(
                self,
                "키 변경",
                "현재 키 변경량이 0입니다.\n"
                "그래도 새 음원으로 추출할까요?",
            )

            if (
                answer
                != QMessageBox.StandardButton.Yes
            ):
                return

        extension = (
            self.transpose_format_combo.currentData()
            or "wav"
        )

        last_dir = self.settings.value(
            "last_transpose_dir",
            str(input_path.parent),
            type=str,
        )

        default_path = (
            Path(last_dir)
            / self._default_transpose_filename(
                input_path
            )
        )

        filter_map = {
            "wav": "WAV (*.wav)",
            "flac": "FLAC (*.flac)",
            "mp3": "MP3 (*.mp3)",
            "m4a": "M4A (*.m4a)",
        }

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "키 변경 음원 저장",
            str(default_path),
            filter_map.get(
                extension,
                "Audio (*.*)",
            ),
        )

        if not output_path:
            return

        if (
            Path(output_path).suffix.lower()
            != f".{extension}"
        ):
            output_path += (
                f".{extension}"
            )

        self.settings.setValue(
            "last_transpose_dir",
            str(
                Path(output_path).parent
            ),
        )

        self.transpose_button.setEnabled(
            False
        )
        self.transpose_progress.setValue(
            0
        )
        self.transpose_progress_text.setText(
            "키 변경 준비 중..."
        )

        engine = (
            self._current_transpose_engine()
        )
        source_mode = (
            self.transpose_source_combo.currentData()
            or "original"
        )

        self.active_transpose_semitones = (
            semitones
        )
        self.active_transpose_engine = (
            engine
        )

        if engine == "seed_vc":
            auto_reference = (
                self.seedvc_reference_combo.currentData()
                != "custom"
            )

            reference_path = (
                None
                if auto_reference
                else self.seedvc_reference_path
            )

            self.transpose_worker = SeedVCTransposeThread(
                input_path=str(input_path),
                output_path=output_path,
                source_mode=source_mode,
                semitones=semitones,
                auto_reference=auto_reference,
                reference_path=reference_path,
                diffusion_steps=(
                    self.seedvc_steps_spin.value()
                ),
                cfg_rate=(
                    self.seedvc_cfg_spin.value()
                ),
                fp16=(
                    self.seedvc_fp16_check.isChecked()
                ),
                separator_model=(
                    self.separator_model_combo.currentData()
                    or DEFAULT_MODEL
                ),
                separator_cache=(
                    self.separator_cache_check.isChecked()
                ),
                separator_autocast=(
                    self.separator_autocast_check.isChecked()
                ),
                parent=self,
            )

            self.transpose_worker.log_line.connect(
                lambda text: (
                    self.statusBar().showMessage(
                        text,
                        5000,
                    )
                )
            )
        elif engine == "rvc":
            self.transpose_worker = RVCTransposeThread(
                input_path=str(input_path),
                output_path=output_path,
                source_mode=source_mode,
                semitones=semitones,
                model_path=self.rvc_model_path,
                index_path=(
                    self.rvc_index_path
                    or None
                ),
                index_rate=(
                    self.rvc_index_rate_spin.value()
                ),
                protect=(
                    self.rvc_protect_spin.value()
                ),
                rms_mix_rate=(
                    self.rvc_rms_mix_spin.value()
                ),
                speaker_id=(
                    self.rvc_speaker_spin.value()
                ),
                separator_model=(
                    self.separator_model_combo.currentData()
                    or DEFAULT_MODEL
                ),
                separator_cache=(
                    self.separator_cache_check.isChecked()
                ),
                separator_autocast=(
                    self.separator_autocast_check.isChecked()
                ),
                parent=self,
            )

            self.transpose_worker.log_line.connect(
                lambda text: (
                    self.statusBar().showMessage(
                        text,
                        5000,
                    )
                )
            )
        else:
            self.transpose_worker = TransposeThread(
                input_path=str(input_path),
                output_path=output_path,
                semitones=semitones,
                preserve_formant=(
                    self.transpose_formant_check.isChecked()
                ),
                quality=(
                    self.transpose_quality_combo.currentData()
                    or "quality"
                ),
                parent=self,
            )

        self.transpose_worker.progress_changed.connect(
            self.on_transpose_progress
        )
        self.transpose_worker.transpose_done.connect(
            self.on_transpose_done
        )
        self.transpose_worker.transpose_failed.connect(
            self.on_transpose_failed
        )
        self.transpose_worker.finished.connect(
            self.on_transpose_finished
        )

        self.transpose_progress_text.setText(
            "키 변경 작업 시작 중..."
        )
        self.transpose_worker.start()

    def on_transpose_progress(
        self,
        percent: int,
        text: str,
    ):
        self.transpose_progress.setValue(
            percent
        )
        self.transpose_progress_text.setText(
            text
        )
        self.statusBar().showMessage(
            text
        )

    def _transposed_subtitle_segments(
        self,
        semitones: int,
    ):
        if (
            self.result is None
            or not self.result.segments
        ):
            return []

        result = []

        for segment in self.result.segments:
            midi = max(
                0,
                min(
                    127,
                    int(segment.midi)
                    + semitones,
                ),
            )

            result.append(
                type(
                    "TransposedSubtitleSegment",
                    (),
                    {
                        "start": float(
                            segment.start
                        ),
                        "end": float(
                            segment.end
                        ),
                        "duration": float(
                            segment.duration
                        ),
                        "midi": midi,
                        "note": midi_to_note_name(
                            midi
                        ),
                        "korean_note": midi_to_korean_name(
                            midi
                        ),
                    },
                )()
            )

        return result

    def _write_transposed_subtitles(
        self,
        output_audio_path: str,
        semitones: int,
    ) -> list[str]:
        if not (
            self.transpose_subtitle_check.isChecked()
            and self.result is not None
            and self.result.segments
        ):
            return []

        shifted_segments = (
            self._transposed_subtitle_segments(
                semitones
            )
        )

        groups = build_subtitle_groups(
            shifted_segments,
            target_seconds=(
                self.subtitle_seconds_spin.value()
            ),
            max_notes=(
                self.subtitle_max_notes_spin.value()
            ),
            split_silence_ms=(
                self.subtitle_silence_spin.value()
            ),
        )

        if not groups:
            return []

        base = str(
            Path(
                output_audio_path
            ).with_suffix("")
        )
        output_mode = (
            self.subtitle_format_combo.currentData()
            or "ass"
        )
        display_mode = (
            self.subtitle_note_format_combo.currentData()
            or "korean"
        )
        notes_per_line = (
            self.subtitle_line_notes_spin.value()
        )
        highlight_current = (
            self.subtitle_highlight_check.isChecked()
        )

        created = []

        if output_mode in {
            "ass",
            "both",
        }:
            ass_path = (
                base + "_notes.ass"
            )
            write_ass(
                ass_path,
                groups,
                display_mode=display_mode,
                notes_per_line=notes_per_line,
                highlight_current=highlight_current,
            )
            created.append(
                ass_path
            )

        if output_mode in {
            "srt",
            "both",
        }:
            srt_path = (
                base + "_notes.srt"
            )
            write_srt(
                srt_path,
                groups,
                display_mode=display_mode,
                notes_per_line=notes_per_line,
            )
            created.append(
                srt_path
            )

        return created

    def on_transpose_done(
        self,
        output_path: str,
    ):
        semitones = (
            self.active_transpose_semitones
        )

        subtitle_files = []

        try:
            subtitle_files = (
                self._write_transposed_subtitles(
                    output_path,
                    semitones,
                )
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "음원 변환 완료 / 자막 생성 실패",
                (
                    f"음원은 정상 생성됐습니다.\n\n"
                    f"{output_path}\n\n"
                    "하지만 변환된 음계 자막 생성 중 오류가 발생했습니다.\n"
                    f"{exc}"
                ),
            )

        self.transpose_progress.setValue(
            100
        )
        self.transpose_progress_text.setText(
            f"완료: {output_path}"
        )

        message = (
            f"키 변경 음원 생성 완료\n\n"
            f"{output_path}"
        )

        if subtitle_files:
            message += (
                "\n\n변환된 음계 자막:\n"
                + "\n".join(
                    subtitle_files
                )
            )

        QMessageBox.information(
            self,
            "키 변경 완료",
            message,
        )

        self.statusBar().showMessage(
            f"키 변경 완료: {output_path}",
            15000,
        )

    def on_transpose_failed(
        self,
        error_text: str,
    ):
        self.transpose_progress_text.setText(
            "키 변경 실패"
        )

        if (
            self.active_transpose_engine
            == "seed_vc"
        ):
            error_text = (
                "Seed-VC SVC 변환 실패\n\n"
                + error_text
            )
        elif (
            self.active_transpose_engine
            == "rvc"
        ):
            error_text = (
                "RVC + RMVPE 변환 실패\n\n"
                + error_text
            )

        QMessageBox.critical(
            self,
            "키 변경 실패",
            error_text[-7000:],
        )

    def on_transpose_finished(self):
        self.update_transpose_preview()

    def generate_note_subtitles(self):
        if (
            self.result is None
            or not self.result.segments
        ):
            QMessageBox.information(
                self,
                "음계 자막 생성",
                "먼저 Pitch 분석을 완료하세요.",
            )
            return

        groups = build_subtitle_groups(
            self.result.segments,
            target_seconds=self.subtitle_seconds_spin.value(),
            max_notes=self.subtitle_max_notes_spin.value(),
            split_silence_ms=self.subtitle_silence_spin.value(),
        )

        if not groups:
            QMessageBox.information(
                self,
                "음계 자막 생성",
                "자막으로 만들 음표 구간이 없습니다.",
            )
            return

        output_mode = (
            self.subtitle_format_combo.currentData()
            or "ass"
        )
        display_mode = (
            self.subtitle_note_format_combo.currentData()
            or "note"
        )
        notes_per_line = (
            self.subtitle_line_notes_spin.value()
        )
        highlight_current = (
            self.subtitle_highlight_check.isChecked()
        )

        base_name = (
            self._export_base_stem()
            + "_note_timeline"
        )

        if output_mode == "srt":
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "음계 SRT 자막 저장",
                base_name + ".srt",
                "SubRip (*.srt)",
            )
            if not save_path:
                return

            if not save_path.lower().endswith(".srt"):
                save_path += ".srt"

            write_srt(
                save_path,
                groups,
                display_mode=display_mode,
                notes_per_line=notes_per_line,
            )
            created = [save_path]

        elif output_mode == "both":
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "음계 자막 저장 위치 선택",
                base_name + ".ass",
                "Advanced SubStation Alpha (*.ass)",
            )
            if not save_path:
                return

            base = str(
                Path(save_path).with_suffix("")
            )
            ass_path = base + ".ass"
            srt_path = base + ".srt"

            write_ass(
                ass_path,
                groups,
                display_mode=display_mode,
                notes_per_line=notes_per_line,
                highlight_current=highlight_current,
            )
            write_srt(
                srt_path,
                groups,
                display_mode=display_mode,
                notes_per_line=notes_per_line,
            )
            created = [
                ass_path,
                srt_path,
            ]

        else:
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "음계 ASS 자막 저장",
                base_name + ".ass",
                "Advanced SubStation Alpha (*.ass)",
            )
            if not save_path:
                return

            if not save_path.lower().endswith(".ass"):
                save_path += ".ass"

            write_ass(
                save_path,
                groups,
                display_mode=display_mode,
                notes_per_line=notes_per_line,
                highlight_current=highlight_current,
            )
            created = [save_path]

        avg_notes = (
            sum(
                len(group.segments)
                for group in groups
            )
            / len(groups)
        )

        self.statusBar().showMessage(
            (
                f"음계 자막 생성 완료: {len(groups)}개 묶음 / "
                f"평균 {avg_notes:.1f}음표"
            ),
            15000,
        )

        QMessageBox.information(
            self,
            "음계 자막 생성 완료",
            (
                f"자막 묶음: {len(groups)}개\n"
                f"평균 음표 수: {avg_notes:.1f}개\n\n"
                + "\n".join(created)
            ),
        )

    def save_separated_vocal(self):
        resource = self.current_vocal_resource

        if (
            resource is None
            or not resource.vocal_path.is_file()
        ):
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "분리 보컬 WAV 저장",
            self._export_base_stem()
            + "_vocals.wav",
            "WAV (*.wav)",
        )
        if not save_path:
            return

        shutil.copy2(
            resource.vocal_path,
            save_path,
        )

    def closeEvent(self, event):
        if (
            self.rvc_training_worker
            and self.rvc_training_worker.isRunning()
        ):
            answer = QMessageBox.question(
                self,
                "RVC 학습 진행 중",
                (
                    "RVC 모델 학습이 진행 중입니다.\n"
                    "종료하면 현재 학습 프로세스도 중지됩니다.\n\n"
                    "그래도 종료할까요?"
                ),
            )

            if (
                answer
                != QMessageBox.StandardButton.Yes
            ):
                event.ignore()
                return

            self.rvc_training_worker.cancel()

        if (
            self.transpose_worker
            and self.transpose_worker.isRunning()
        ):
            answer = QMessageBox.question(
                self,
                "키 변경 진행 중",
                "키 변경 작업이 진행 중입니다.\n"
                "작업이 끝난 후 종료하는 것을 권장합니다.\n\n"
                "그래도 종료할까요?",
            )

            if (
                answer
                != QMessageBox.StandardButton.Yes
            ):
                event.ignore()
                return

        self._release_vocal_resource()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)

    pg.setConfigOptions(antialias=True)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
