from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import librosa
import numpy as np

from media_input import prepare_audio_for_analysis

# V29_LEAD_MELODY_ANALYSIS_PATCH


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
KOREAN_NOTE_NAMES = ("도", "도♯", "레", "레♯", "미", "파", "파♯", "솔", "솔♯", "라", "라♯", "시")


@dataclass(slots=True)
class NoteSegment:
    start: float
    end: float
    duration: float
    midi: int
    note: str
    korean_note: str
    avg_hz: float
    cents: float
    confidence: float


@dataclass(slots=True)
class AnalysisResult:
    path: str
    duration: float
    sample_rate: int
    hop_length: int

    # Frame-level raw pYIN
    times: np.ndarray
    raw_f0_hz: np.ndarray
    raw_midi_float: np.ndarray
    raw_voiced_flag: np.ndarray
    voiced_probability: np.ndarray

    # Vocal activity / energy information
    frame_rms_dbfs: np.ndarray
    energy_gate_mask: np.ndarray
    energy_gate_enabled: bool
    energy_reference_dbfs: float
    energy_threshold_dbfs: float
    energy_active_seconds: float

    # Lead Melody Gate information
    lead_gate_mask: np.ndarray
    lead_gate_values: np.ndarray
    lead_gate_enabled: bool
    lead_gate_strength: str
    lead_gate_threshold: float
    lead_active_seconds: float
    lead_selected_seconds: float
    lead_mean_confidence: float

    # Frame-level processed pitch
    accepted_mask: np.ndarray
    processed_f0_hz: np.ndarray
    midi_float: np.ndarray

    segments: list[NoteSegment]

    min_midi: int | None
    max_midi: int | None
    range_candidate_segments: int

    raw_voiced_seconds: float
    accepted_seconds: float
    processed_seconds: float
    processed_coverage_percent: float

    input_was_converted: bool = False

    voiced_threshold: float = 0.25
    max_dropout_ms: float = 80.0
    smoothing_window: int = 5
    hysteresis_cents: float = 20.0
    min_note_ms: float = 35.0

    # Engine v3 activity-gate settings
    energy_margin_db: float = 32.0
    energy_floor_dbfs: float = -55.0
    energy_hysteresis_db: float = 4.0
    min_activity_ms: float = 80.0
    max_activity_gap_ms: float = 100.0
    range_min_note_ms: float = 100.0
    range_min_confidence: float = 0.35


ProgressCallback = Callable[[int, str], None]


def midi_to_note_name(midi: int) -> str:
    octave = midi // 12 - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def midi_to_korean_name(midi: int) -> str:
    scientific_octave = midi // 12 - 1
    korean_octave = scientific_octave - 2
    return f"{korean_octave}옥타브 {KOREAN_NOTE_NAMES[midi % 12]}"


def midi_float_to_hz(midi_values: np.ndarray) -> np.ndarray:
    result = np.full_like(midi_values, np.nan, dtype=float)
    valid = np.isfinite(midi_values)
    result[valid] = 440.0 * np.power(2.0, (midi_values[valid] - 69.0) / 12.0)
    return result


def hz_to_midi_float(freq_hz: np.ndarray) -> np.ndarray:
    result = np.full_like(freq_hz, np.nan, dtype=float)
    valid = np.isfinite(freq_hz) & (freq_hz > 0)
    result[valid] = 69.0 + 12.0 * np.log2(freq_hz[valid] / 440.0)
    return result


def _align_length(values: np.ndarray, length: int, fill_value) -> np.ndarray:
    values = np.asarray(values)
    if len(values) == length:
        return values
    if len(values) > length:
        return values[:length]

    missing = length - len(values)
    return np.concatenate(
        [
            values,
            np.full(missing, fill_value, dtype=values.dtype),
        ]
    )


def _frame_rms_dbfs(
    y: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    target_frames: int,
) -> np.ndarray:
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]

    rms = _align_length(
        np.asarray(rms, dtype=float),
        target_frames,
        0.0,
    )

    return 20.0 * np.log10(np.maximum(rms, 1e-12))


def _fill_short_false_runs(mask: np.ndarray, max_frames: int) -> np.ndarray:
    if max_frames <= 0:
        return mask.copy()

    out = mask.copy()
    n = len(out)
    i = 0

    while i < n:
        if out[i]:
            i += 1
            continue

        start = i
        while i < n and not out[i]:
            i += 1
        end = i

        if (
            0 < end - start <= max_frames
            and start > 0
            and end < n
            and out[start - 1]
            and out[end]
        ):
            out[start:end] = True

    return out


def _remove_short_true_runs(mask: np.ndarray, min_frames: int) -> np.ndarray:
    if min_frames <= 1:
        return mask.copy()

    out = mask.copy()
    n = len(out)
    i = 0

    while i < n:
        if not out[i]:
            i += 1
            continue

        start = i
        while i < n and out[i]:
            i += 1
        end = i

        if end - start < min_frames:
            out[start:end] = False

    return out


def _adaptive_energy_gate(
    dbfs: np.ndarray,
    *,
    hop_seconds: float,
    margin_db: float,
    floor_dbfs: float,
    hysteresis_db: float,
    min_activity_ms: float,
    max_activity_gap_ms: float,
) -> tuple[np.ndarray, float, float]:
    # 매우 낮은 numerical silence는 reference 계산에서 제외한다.
    reference_candidates = dbfs[
        np.isfinite(dbfs) & (dbfs > -120.0)
    ]

    if reference_candidates.size:
        # 90 percentile은 실제 보컬이 강하게 존재하는 구간을 대표하면서
        # 순간 peak에는 과도하게 영향을 받지 않는다.
        reference_dbfs = float(
            np.percentile(reference_candidates, 90.0)
        )
    else:
        reference_dbfs = float(floor_dbfs)

    # 곡마다 stem 레벨이 다르므로 reference-relative threshold를 쓰되,
    # 너무 낮게 내려가지 않도록 absolute floor로 제한한다.
    threshold_dbfs = max(
        float(floor_dbfs),
        reference_dbfs - float(margin_db),
    )

    open_threshold = threshold_dbfs
    close_threshold = threshold_dbfs - max(
        0.0,
        float(hysteresis_db),
    )

    gate = np.zeros(len(dbfs), dtype=bool)
    active = False

    for idx, value in enumerate(dbfs):
        if not np.isfinite(value):
            active = False
        elif not active:
            if value >= open_threshold:
                active = True
        else:
            if value < close_threshold:
                active = False

        gate[idx] = active

    max_gap_frames = max(
        0,
        int(round(
            (max_activity_gap_ms / 1000.0)
            / hop_seconds
        )),
    )
    min_active_frames = max(
        1,
        int(round(
            (min_activity_ms / 1000.0)
            / hop_seconds
        )),
    )

    # 자음/숨/짧은 dip 때문에 vocal activity가 잠깐 꺼지는 것은 연결.
    gate = _fill_short_false_runs(
        gate,
        max_frames=max_gap_frames,
    )

    # 아주 짧게 튀는 leakage/잔향 에너지 island는 제거.
    gate = _remove_short_true_runs(
        gate,
        min_frames=min_active_frames,
    )

    return gate, reference_dbfs, threshold_dbfs


def _bridge_short_gaps(
    values: np.ndarray,
    *,
    max_gap_frames: int,
    max_jump_semitones: float = 3.0,
) -> np.ndarray:
    out = values.copy()
    n = len(out)
    i = 0

    while i < n:
        if np.isfinite(out[i]):
            i += 1
            continue

        gap_start = i
        while i < n and not np.isfinite(out[i]):
            i += 1
        gap_end = i
        gap_len = gap_end - gap_start

        left = gap_start - 1
        right = gap_end

        if (
            0 < gap_len <= max_gap_frames
            and left >= 0
            and right < n
            and np.isfinite(out[left])
            and np.isfinite(out[right])
            and abs(float(out[right] - out[left])) <= max_jump_semitones
        ):
            out[gap_start:gap_end] = np.linspace(
                float(out[left]),
                float(out[right]),
                gap_len + 2,
            )[1:-1]

    return out


def _nanmedian_smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.copy()

    if window % 2 == 0:
        window += 1

    radius = window // 2
    out = values.copy()

    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue

        lo = max(0, idx - radius)
        hi = min(len(values), idx + radius + 1)
        chunk = values[lo:hi]
        finite = chunk[np.isfinite(chunk)]

        if finite.size:
            out[idx] = float(np.median(finite))

    return out


def _quantize_with_hysteresis(
    midi_values: np.ndarray,
    hysteresis_cents: float,
) -> np.ndarray:
    quantized = np.full(len(midi_values), -1, dtype=int)
    current_note: int | None = None
    hysteresis = max(0.0, hysteresis_cents) / 100.0

    for idx, value in enumerate(midi_values):
        if not np.isfinite(value):
            current_note = None
            continue

        if current_note is None:
            current_note = int(round(float(value)))
        else:
            upper = current_note + 0.5 + hysteresis
            lower = current_note - 0.5 - hysteresis

            if value > upper or value < lower:
                current_note = int(round(float(value)))

        quantized[idx] = current_note

    return quantized


def _runs_from_quantized(
    quantized: np.ndarray,
) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start: int | None = None
    current: int | None = None

    for idx, note in enumerate(quantized):
        if note < 0:
            if start is not None and current is not None:
                runs.append((start, idx, current))
            start = None
            current = None
            continue

        if start is None:
            start = idx
            current = int(note)
            continue

        if note != current:
            runs.append((start, idx, int(current)))
            start = idx
            current = int(note)

    if start is not None and current is not None:
        runs.append(
            (start, len(quantized), int(current))
        )

    return runs


def _merge_short_note_blips(
    quantized: np.ndarray,
    *,
    min_frames: int,
) -> np.ndarray:
    if min_frames <= 1:
        return quantized.copy()

    out = quantized.copy()

    for _ in range(4):
        runs = _runs_from_quantized(out)
        changed = False

        for run_index, (start, end, note) in enumerate(runs):
            if end - start >= min_frames:
                continue

            prev_run = (
                runs[run_index - 1]
                if run_index > 0
                else None
            )
            next_run = (
                runs[run_index + 1]
                if run_index + 1 < len(runs)
                else None
            )

            if prev_run is not None and prev_run[1] != start:
                prev_run = None
            if next_run is not None and next_run[0] != end:
                next_run = None

            replacement: int | None = None

            if (
                prev_run
                and next_run
                and prev_run[2] == next_run[2]
            ):
                replacement = prev_run[2]
            elif prev_run and next_run:
                prev_len = prev_run[1] - prev_run[0]
                next_len = next_run[1] - next_run[0]
                replacement = (
                    prev_run[2]
                    if prev_len >= next_len
                    else next_run[2]
                )
            elif prev_run:
                replacement = prev_run[2]
            elif next_run:
                replacement = next_run[2]

            if replacement is not None:
                out[start:end] = replacement
                changed = True

        if not changed:
            break

    return out


def _make_segments_v3(
    times: np.ndarray,
    processed_f0_hz: np.ndarray,
    processed_midi: np.ndarray,
    voiced_prob: np.ndarray,
    hop_seconds: float,
    *,
    hysteresis_cents: float,
    min_note_ms: float,
) -> list[NoteSegment]:
    quantized = _quantize_with_hysteresis(
        processed_midi,
        hysteresis_cents=hysteresis_cents,
    )

    min_frames = max(
        1,
        int(round(
            (min_note_ms / 1000.0)
            / hop_seconds
        )),
    )

    quantized = _merge_short_note_blips(
        quantized,
        min_frames=min_frames,
    )

    segments: list[NoteSegment] = []

    for start_idx, end_idx, midi_note in _runs_from_quantized(
        quantized
    ):
        idxs = np.arange(start_idx, end_idx)

        pitch_values = processed_midi[idxs]
        hz_values = processed_f0_hz[idxs]

        finite_pitch = np.isfinite(pitch_values)
        finite_hz = np.isfinite(hz_values)

        if not np.any(finite_pitch) or not np.any(finite_hz):
            continue

        start = max(
            0.0,
            float(times[start_idx] - hop_seconds / 2.0),
        )
        end = float(
            times[end_idx - 1] + hop_seconds / 2.0
        )
        duration = max(0.0, end - start)

        median_midi_float = float(
            np.median(pitch_values[finite_pitch])
        )
        cents = (
            median_midi_float - midi_note
        ) * 100.0

        probabilities = voiced_prob[idxs]
        probabilities = probabilities[
            np.isfinite(probabilities)
        ]
        confidence = (
            float(np.mean(probabilities))
            if probabilities.size
            else 0.0
        )

        segments.append(
            NoteSegment(
                start=start,
                end=end,
                duration=duration,
                midi=midi_note,
                note=midi_to_note_name(midi_note),
                korean_note=midi_to_korean_name(midi_note),
                avg_hz=float(
                    np.median(hz_values[finite_hz])
                ),
                cents=float(cents),
                confidence=confidence,
            )
        )

    return segments


def analyze_audio(
    file_path: str,
    *,
    fmin_hz: float = 65.4,
    fmax_hz: float = 1396.9,
    voiced_threshold: float = 0.25,
    sample_rate: int = 22050,
    hop_length: int = 256,
    max_dropout_ms: float = 80.0,
    smoothing_window: int = 5,
    hysteresis_cents: float = 20.0,
    min_note_ms: float = 35.0,

    # Engine v3
    use_energy_gate: bool = True,
    energy_margin_db: float = 32.0,
    energy_floor_dbfs: float = -55.0,
    energy_hysteresis_db: float = 4.0,
    min_activity_ms: float = 80.0,
    max_activity_gap_ms: float = 100.0,
    range_min_note_ms: float = 100.0,
    range_min_confidence: float = 0.35,

    # v2.9 Lead Melody Gate.
    # Times/values come from rvc_lead_selector.analyze_lead_frames().
    lead_gate_times: np.ndarray | None = None,
    lead_gate_values: np.ndarray | None = None,
    lead_gate_threshold: float = 0.30,
    lead_gate_strength: str = "balanced",
    lead_mean_confidence: float = 0.0,

    progress: ProgressCallback | None = None,
) -> AnalysisResult:
    original_path = Path(file_path)
    if not original_path.exists():
        raise FileNotFoundError(original_path)

    def report(percent: int, text: str) -> None:
        if progress:
            progress(percent, text)

    prepared = None

    try:
        report(3, "입력 형식을 확인하는 중...")
        prepared = prepare_audio_for_analysis(
            original_path,
            sample_rate=sample_rate,
        )

        if prepared.was_converted:
            report(
                10,
                "FFmpeg로 Pitch 분석용 오디오를 준비했습니다.",
            )
        else:
            report(10, "오디오 파일을 읽는 중...")

        y, sr = librosa.load(
            str(prepared.analysis_path),
            sr=sample_rate,
            mono=True,
        )

        if y.size == 0:
            raise ValueError("오디오 데이터가 비어 있습니다.")

        duration = len(y) / sr

        report(
            22,
            "pYIN으로 Raw 기본주파수(F0)를 분석하는 중...",
        )
        frame_length = 2048

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y=y,
            sr=sr,
            fmin=fmin_hz,
            fmax=fmax_hz,
            frame_length=frame_length,
            hop_length=hop_length,
            resolution=0.1,
        )

        times = librosa.times_like(
            f0,
            sr=sr,
            hop_length=hop_length,
        )

        raw_f0 = np.asarray(f0, dtype=float)
        voiced_prob = np.asarray(
            voiced_prob,
            dtype=float,
        )
        voiced_flag = np.asarray(
            voiced_flag,
            dtype=bool,
        )
        raw_midi = hz_to_midi_float(raw_f0)

        report(
            64,
            "보컬 stem 에너지(RMS)를 분석하는 중...",
        )
        rms_dbfs = _frame_rms_dbfs(
            y,
            frame_length=frame_length,
            hop_length=hop_length,
            target_frames=len(raw_f0),
        )

        hop_seconds = hop_length / sr

        if use_energy_gate:
            (
                energy_gate,
                energy_reference_dbfs,
                energy_threshold_dbfs,
            ) = _adaptive_energy_gate(
                rms_dbfs,
                hop_seconds=hop_seconds,
                margin_db=energy_margin_db,
                floor_dbfs=energy_floor_dbfs,
                hysteresis_db=energy_hysteresis_db,
                min_activity_ms=min_activity_ms,
                max_activity_gap_ms=max_activity_gap_ms,
            )
        else:
            energy_gate = np.ones(
                len(raw_f0),
                dtype=bool,
            )
            candidates = rms_dbfs[
                np.isfinite(rms_dbfs)
                & (rms_dbfs > -120.0)
            ]
            energy_reference_dbfs = (
                float(np.percentile(candidates, 90.0))
                if candidates.size
                else float(energy_floor_dbfs)
            )
            energy_threshold_dbfs = float("-inf")

        lead_gate_enabled = bool(
            lead_gate_times is not None
            and lead_gate_values is not None
            and len(
                np.asarray(
                    lead_gate_times
                )
            ) > 1
            and len(
                np.asarray(
                    lead_gate_values
                )
            ) > 1
        )

        if lead_gate_enabled:
            source_gate_times = np.asarray(
                lead_gate_times,
                dtype=float,
            ).reshape(
                -1
            )
            source_gate_values = np.asarray(
                lead_gate_values,
                dtype=float,
            ).reshape(
                -1
            )

            gate_count = min(
                source_gate_times.size,
                source_gate_values.size,
            )

            source_gate_times = source_gate_times[
                :gate_count
            ]
            source_gate_values = source_gate_values[
                :gate_count
            ]

            finite_gate = (
                np.isfinite(
                    source_gate_times
                )
                & np.isfinite(
                    source_gate_values
                )
            )

            source_gate_times = source_gate_times[
                finite_gate
            ]
            source_gate_values = source_gate_values[
                finite_gate
            ]

            if source_gate_times.size > 1:
                order = np.argsort(
                    source_gate_times
                )
                source_gate_times = source_gate_times[
                    order
                ]
                source_gate_values = source_gate_values[
                    order
                ]

                aligned_lead_values = np.interp(
                    times.astype(
                        float,
                        copy=False,
                    ),
                    source_gate_times,
                    source_gate_values,
                    left=0.0,
                    right=0.0,
                )

                aligned_lead_values = np.clip(
                    aligned_lead_values,
                    0.0,
                    1.0,
                )

                lead_gate_mask = (
                    aligned_lead_values
                    >= float(
                        lead_gate_threshold
                    )
                )
            else:
                lead_gate_enabled = False
                aligned_lead_values = np.ones(
                    len(
                        raw_f0
                    ),
                    dtype=float,
                )
                lead_gate_mask = np.ones(
                    len(
                        raw_f0
                    ),
                    dtype=bool,
                )
        else:
            aligned_lead_values = np.ones(
                len(
                    raw_f0
                ),
                dtype=float,
            )
            lead_gate_mask = np.ones(
                len(
                    raw_f0
                ),
                dtype=bool,
            )

        if lead_gate_enabled:
            report(
                72,
                (
                    "보컬 활동 + Lead Melody Gate 적용 중..."
                ),
            )
        else:
            report(
                72,
                (
                    "보컬 활동 게이트 적용 중..."
                    if use_energy_gate
                    else "에너지 게이트 비활성 상태..."
                ),
            )

        accepted_mask = (
            voiced_flag
            & np.isfinite(raw_midi)
            & np.isfinite(voiced_prob)
            & (voiced_prob >= voiced_threshold)
            & energy_gate
            & lead_gate_mask
        )

        working_midi = np.full_like(
            raw_midi,
            np.nan,
            dtype=float,
        )
        working_midi[accepted_mask] = (
            raw_midi[accepted_mask]
        )

        max_gap_frames = max(
            0,
            int(round(
                (max_dropout_ms / 1000.0)
                / hop_seconds
            )),
        )

        report(
            78,
            "짧은 pitch dropout을 연결하는 중...",
        )
        working_midi = _bridge_short_gaps(
            working_midi,
            max_gap_frames=max_gap_frames,
            max_jump_semitones=3.0,
        )

        report(84, "피치 곡선을 smoothing 하는 중...")

        if smoothing_window < 1:
            smoothing_window = 1
        if smoothing_window % 2 == 0:
            smoothing_window += 1

        processed_midi = _nanmedian_smooth(
            working_midi,
            window=smoothing_window,
        )
        processed_f0 = midi_float_to_hz(
            processed_midi
        )

        report(
            90,
            "hysteresis 기반으로 음표 구간을 만드는 중...",
        )
        segments = _make_segments_v3(
            times,
            processed_f0,
            processed_midi,
            voiced_prob,
            hop_seconds,
            hysteresis_cents=hysteresis_cents,
            min_note_ms=min_note_ms,
        )

        # 최고/최저음은 한두 frame 튄 note가 아니라
        # 일정 시간 이상 유지되고 confidence가 있는 segment만 사용.
        range_segments = [
            segment
            for segment in segments
            if (
                segment.duration
                >= range_min_note_ms / 1000.0
                and segment.confidence
                >= range_min_confidence
            )
        ]

        if not range_segments:
            range_segments = segments

        if range_segments:
            min_midi = min(
                segment.midi
                for segment in range_segments
            )
            max_midi = max(
                segment.midi
                for segment in range_segments
            )
        else:
            min_midi = None
            max_midi = None

        raw_voiced_seconds = float(
            np.count_nonzero(
                voiced_flag & np.isfinite(raw_f0)
            )
            * hop_seconds
        )
        accepted_seconds = float(
            np.count_nonzero(accepted_mask)
            * hop_seconds
        )
        processed_seconds = float(
            np.count_nonzero(
                np.isfinite(processed_midi)
            )
            * hop_seconds
        )
        energy_active_seconds = float(
            np.count_nonzero(energy_gate)
            * hop_seconds
        )
        lead_active_seconds = float(
            np.count_nonzero(
                lead_gate_mask
            )
            * hop_seconds
        ) if lead_gate_enabled else float(
            duration
        )
        lead_selected_seconds = float(
            np.count_nonzero(
                accepted_mask
            )
            * hop_seconds
        ) if lead_gate_enabled else float(
            accepted_seconds
        )

        coverage = (
            processed_seconds / duration * 100.0
            if duration > 0
            else 0.0
        )

        result = AnalysisResult(
            path=str(original_path),
            duration=float(duration),
            sample_rate=int(sr),
            hop_length=int(hop_length),

            times=times,
            raw_f0_hz=raw_f0,
            raw_midi_float=raw_midi,
            raw_voiced_flag=voiced_flag,
            voiced_probability=voiced_prob,

            frame_rms_dbfs=rms_dbfs,
            energy_gate_mask=energy_gate,
            energy_gate_enabled=bool(use_energy_gate),
            energy_reference_dbfs=float(
                energy_reference_dbfs
            ),
            energy_threshold_dbfs=float(
                energy_threshold_dbfs
            ),
            energy_active_seconds=energy_active_seconds,

            lead_gate_mask=lead_gate_mask,
            lead_gate_values=aligned_lead_values,
            lead_gate_enabled=bool(
                lead_gate_enabled
            ),
            lead_gate_strength=str(
                lead_gate_strength
            ),
            lead_gate_threshold=float(
                lead_gate_threshold
            ),
            lead_active_seconds=float(
                lead_active_seconds
            ),
            lead_selected_seconds=float(
                lead_selected_seconds
            ),
            lead_mean_confidence=float(
                lead_mean_confidence
            ),

            accepted_mask=accepted_mask,
            processed_f0_hz=processed_f0,
            midi_float=processed_midi,

            segments=segments,

            min_midi=min_midi,
            max_midi=max_midi,
            range_candidate_segments=len(
                range_segments
            ),

            raw_voiced_seconds=raw_voiced_seconds,
            accepted_seconds=accepted_seconds,
            processed_seconds=processed_seconds,
            processed_coverage_percent=float(
                coverage
            ),

            input_was_converted=bool(
                prepared.was_converted
            ),

            voiced_threshold=float(
                voiced_threshold
            ),
            max_dropout_ms=float(
                max_dropout_ms
            ),
            smoothing_window=int(
                smoothing_window
            ),
            hysteresis_cents=float(
                hysteresis_cents
            ),
            min_note_ms=float(
                min_note_ms
            ),

            energy_margin_db=float(
                energy_margin_db
            ),
            energy_floor_dbfs=float(
                energy_floor_dbfs
            ),
            energy_hysteresis_db=float(
                energy_hysteresis_db
            ),
            min_activity_ms=float(
                min_activity_ms
            ),
            max_activity_gap_ms=float(
                max_activity_gap_ms
            ),
            range_min_note_ms=float(
                range_min_note_ms
            ),
            range_min_confidence=float(
                range_min_confidence
            ),
        )

        report(100, "분석 완료")
        return result

    finally:
        if prepared is not None:
            prepared.cleanup()
