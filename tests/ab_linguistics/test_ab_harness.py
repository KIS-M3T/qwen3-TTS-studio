"""A/B testing harness for Italian Text Normalization (TN) evaluation (M1 WP3 + M2 WP6)."""

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import random
import secrets
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pytest
import soundfile as sf

from src.audio.metrics import analyze_acoustic_quality, detect_trailing_silence
from src.engine.types import AudioBuffer, DecodingParams, VoicePrompt
from src.linguistics import (
    BaseTextNormalizer,
    IdentityNormalizer,
    ItalianTextNormalizer,
    NormalizedText,
    TNAnnotation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AB_CORPUS_PATH = REPO_ROOT / "bench" / "m2_ab_corpus.jsonl"
DEFAULT_AB_OUTPUT_DIR = REPO_ROOT / "bench" / "ab_audio_m2"


def build_default_ab_decoding_params(seed: int = 42) -> DecodingParams:
    """Return frozen deterministic DecodingParams per M2 Acceptance Matrix § 3.1."""
    return DecodingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        repetition_penalty=1.05,
        max_new_tokens=2048,
        seed=seed,
        subtalker_params={"non_streaming_mode": True},
    )


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest for a binary or text file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_model_weights_sha256(model_dir: Path) -> Dict[str, str]:
    """Compute SHA-256 digests for safetensors/bin weight files inside model_dir."""
    if not model_dir.exists() or not model_dir.is_dir():
        return {}
    digests: Dict[str, str] = {}
    for ext in ("*.safetensors", "*.bin"):
        for weight_file in sorted(model_dir.glob(ext)):
            digests[weight_file.name] = compute_file_sha256(weight_file)
    return digests


def inspect_audio_health(audio: AudioBuffer) -> Dict[str, Any]:
    """Compute objective signal validity and acoustic health metrics without fabricating MOS."""
    samples = np.asarray(audio.samples, dtype=np.float32)
    sample_count = int(samples.size)
    all_finite = bool(sample_count > 0 and np.all(np.isfinite(samples)))
    peak_linear = float(np.max(np.abs(samples))) if all_finite else 0.0
    rms_linear = float(np.sqrt(np.mean(samples ** 2))) if all_finite else 0.0
    is_silent = bool(sample_count == 0 or not all_finite or peak_linear < 1e-4)

    quality = analyze_acoustic_quality(samples, audio.sample_rate) if all_finite else {
        "duration_sec": 0.0,
        "dc_offset": 0.0,
        "peak_dbfs": -100.0,
        "rms_dbfs": -100.0,
        "crest_factor_db": 0.0,
        "dynamic_range_db": 0.0,
        "is_clipped": False,
    }
    is_truncated, trailing_silence_ratio = (
        detect_trailing_silence(samples, audio.sample_rate)
        if all_finite
        else (True, 1.0)
    )
    technically_valid = bool(
        all_finite
        and not is_silent
        and audio.sample_rate > 0
        and quality["duration_sec"] > 0.0
    )
    return {
        "technically_valid": technically_valid,
        "sample_count": sample_count,
        "sample_rate": int(audio.sample_rate),
        "all_finite": all_finite,
        "is_silent": is_silent,
        "rms_linear": round(rms_linear, 6),
        "peak_linear": round(peak_linear, 6),
        "trailing_silence_truncated": bool(is_truncated),
        "trailing_silence_ratio": trailing_silence_ratio,
        **quality,
    }


def load_ab_corpus(corpus_path: Path = DEFAULT_AB_CORPUS_PATH) -> List[Dict[str, Any]]:
    """Load acoustic A/B JSONL test corpus preserving order."""
    records: List[Dict[str, Any]] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _parse_rating(value: Any) -> Optional[int]:
    """Parse optional integer MOS score in [1, 5]; blank/None returns None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    score = int(float(text))
    if not 1 <= score <= 5:
        raise ValueError(f"MOS rating must be within [1, 5], got {score}")
    return score


def score_blinded_evaluations(
    rows: Sequence[Dict[str, Any]],
    mapping_by_pair: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Aggregate human A/B ratings using exact M2 Acceptance Matrix § 3.3 formulas.

    Supports either direct ``score_a``/``score_b`` columns or blinded
    ``stimulus_1_score``/``stimulus_2_score`` joined against ``mapping_by_pair``.
    Control rows (``is_control`` true or test_id ending in ``-CTRL``) are excluded
    from the active denominator and summarized separately.
    """
    active_ratings: List[Dict[str, Any]] = []
    control_ratings: List[Dict[str, Any]] = []

    for row in rows:
        test_id = str(row.get("test_id", "")).strip()
        pair_label = str(row.get("pair_label", "")).strip()
        evaluator_id = str(row.get("evaluator_id", "")).strip() or "unknown"
        raw_ctrl = str(row.get("is_control", "")).strip().lower()
        is_control = raw_ctrl in {"1", "true", "yes"} or test_id.endswith("-CTRL")

        score_a = _parse_rating(row.get("score_a"))
        score_b = _parse_rating(row.get("score_b"))
        if (score_a is None or score_b is None) and mapping_by_pair is not None:
            s1 = _parse_rating(row.get("stimulus_1_score"))
            s2 = _parse_rating(row.get("stimulus_2_score"))
            map_entry = mapping_by_pair.get(pair_label) or mapping_by_pair.get(test_id)
            if s1 is not None and s2 is not None and map_entry is not None:
                if map_entry["stimulus_1"] == "A":
                    score_a, score_b = s1, s2
                else:
                    score_a, score_b = s2, s1

        if score_a is None or score_b is None:
            continue

        entry = {
            "test_id": test_id,
            "pair_label": pair_label,
            "evaluator_id": evaluator_id,
            "score_a": score_a,
            "score_b": score_b,
        }
        if is_control:
            control_ratings.append(entry)
        else:
            active_ratings.append(entry)

    if not active_ratings:
        return {
            "status": "PENDING_HUMAN_EVALUATION",
            "active_comparisons": 0,
            "control_comparisons": len(control_ratings),
            "mos_a": None,
            "mos_b": None,
            "delta_mos": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "standard_win_rate": None,
            "non_loss_rate": None,
            "b_mos1_count": 0,
            "disagreements": [],
            "gate_checks": {
                "delta_mos_ge_minus_0_10": None,
                "standard_win_rate_ge_0_65": None,
                "non_loss_rate_ge_0_85": None,
                "zero_b_mos1": None,
            },
        }

    total = len(active_ratings)
    wins = sum(1 for r in active_ratings if r["score_b"] > r["score_a"])
    ties = sum(1 for r in active_ratings if r["score_b"] == r["score_a"])
    losses = sum(1 for r in active_ratings if r["score_b"] < r["score_a"])

    mos_a = round(sum(r["score_a"] for r in active_ratings) / total, 4)
    mos_b = round(sum(r["score_b"] for r in active_ratings) / total, 4)
    delta_mos = round(mos_b - mos_a, 4)
    standard_win_rate = round(wins / total, 4)
    non_loss_rate = round((wins + ties) / total, 4)
    b_mos1_count = sum(1 for r in active_ratings if r["score_b"] == 1)

    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for r in active_ratings:
        key = r["test_id"] or r["pair_label"]
        by_pair.setdefault(key, []).append(r)

    disagreements: List[Dict[str, Any]] = []
    for key, pair_rows in sorted(by_pair.items()):
        b_scores = [p["score_b"] for p in pair_rows]
        if b_scores and (max(b_scores) - min(b_scores) >= 2):
            disagreements.append({
                "test_id": key,
                "b_scores": b_scores,
                "spread": max(b_scores) - min(b_scores),
                "evaluators": [p["evaluator_id"] for p in pair_rows],
            })

    gate_checks = {
        "delta_mos_ge_minus_0_10": delta_mos >= -0.10,
        "standard_win_rate_ge_0_65": standard_win_rate >= 0.65,
        "non_loss_rate_ge_0_85": non_loss_rate >= 0.85,
        "zero_b_mos1": b_mos1_count == 0,
    }
    all_passed = all(gate_checks.values())

    return {
        "status": "PASSED" if all_passed else "FAILED",
        "active_comparisons": total,
        "control_comparisons": len(control_ratings),
        "mos_a": mos_a,
        "mos_b": mos_b,
        "delta_mos": delta_mos,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "standard_win_rate": standard_win_rate,
        "non_loss_rate": non_loss_rate,
        "b_mos1_count": b_mos1_count,
        "disagreements": disagreements,
        "gate_checks": gate_checks,
    }


@dataclass(frozen=True)
class ABComparisonResult:
    """Differential evaluation report between Condition A (raw) and Condition B (normalized)."""
    test_id: str
    condition_a_raw: str
    condition_b_norm: str
    duration_sec_a: float
    duration_sec_b: float
    delta_duration_sec: float
    asr_diagnostic_transcript_a: Optional[str]
    asr_diagnostic_transcript_b: Optional[str]
    human_evaluation_rubric: Dict[str, Optional[int]]


class ABTestingHarness:
    """Harness to coordinate pairwise A/B generation and differential reporting."""

    HUMAN_EVAL_CRITERIA = [
        "general_intelligibility",
        "numbers_dates_accuracy",
        "acronyms_pronunciation",
        "prosody_and_pauses",
        "absence_of_truncation",
        "absence_of_artifacts_hallucinations",
        "voice_naturalness",
    ]

    def __init__(self, normalizer: BaseTextNormalizer):
        self.normalizer = normalizer

    def evaluate_pair(
        self,
        test_id: str,
        raw_text: str,
        audio_a: AudioBuffer,
        audio_b: AudioBuffer,
        asr_a: Optional[str] = None,
        asr_b: Optional[str] = None,
    ) -> ABComparisonResult:
        """Produce pairwise differential report."""
        norm_result = self.normalizer.normalize(raw_text)

        delta_dur = round(audio_b.duration_sec - audio_a.duration_sec, 4)

        rubric = {criterion: None for criterion in self.HUMAN_EVAL_CRITERIA}

        return ABComparisonResult(
            test_id=test_id,
            condition_a_raw=raw_text,
            condition_b_norm=norm_result.normalized_text,
            duration_sec_a=round(audio_a.duration_sec, 4),
            duration_sec_b=round(audio_b.duration_sec, 4),
            delta_duration_sec=delta_dur,
            asr_diagnostic_transcript_a=asr_a,
            asr_diagnostic_transcript_b=asr_b,
            human_evaluation_rubric=rubric,
        )

    def run_synthesis_pair(
        self,
        test_id: str,
        raw_text: str,
        facade: Any,
        output_dir: Path,
        model_id: str = "1.7B-Base",
        mode: str = "clone",
        voice_prompt: Optional[VoicePrompt] = None,
        speaker_id: Optional[str] = None,
        language: str = "italian",
        params: Optional[DecodingParams] = None,
        pair_label: Optional[str] = None,
        category: str = "",
        weights_sha256: Optional[Dict[str, str]] = None,
        blind_rng: Optional[random.Random] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Synthesize Path A (raw) and Path B (normalized) identically and create blinded stimuli."""
        effective_params = params or build_default_ab_decoding_params()
        raw_dir = output_dir / "raw_pairs"
        blind_dir = output_dir / "blinded"
        raw_dir.mkdir(parents=True, exist_ok=True)
        blind_dir.mkdir(parents=True, exist_ok=True)

        norm_result = self.normalizer.normalize(raw_text)
        normalized_text = norm_result.normalized_text

        res_a = facade.synthesize(
            text=raw_text,
            model_id=model_id,
            mode=mode,
            language=language,
            voice_prompt=voice_prompt,
            speaker_id=speaker_id,
            params=effective_params,
        )
        res_b = facade.synthesize(
            text=normalized_text,
            model_id=model_id,
            mode=mode,
            language=language,
            voice_prompt=voice_prompt,
            speaker_id=speaker_id,
            params=effective_params,
        )

        wav_a_path = raw_dir / f"{test_id}_A_raw.wav"
        wav_b_path = raw_dir / f"{test_id}_B_norm.wav"
        sf.write(str(wav_a_path), res_a.audio.samples, res_a.audio.sample_rate)
        sf.write(str(wav_b_path), res_b.audio.samples, res_b.audio.sample_rate)

        sha_a = compute_file_sha256(wav_a_path)
        sha_b = compute_file_sha256(wav_b_path)
        health_a = inspect_audio_health(res_a.audio)
        health_b = inspect_audio_health(res_b.audio)

        rng = blind_rng or secrets.SystemRandom()
        a_first = bool(rng.randint(0, 1) == 0)
        label = pair_label or test_id.lower().replace("-", "_")
        stim_1_path = blind_dir / f"{label}_stimulus_1.wav"
        stim_2_path = blind_dir / f"{label}_stimulus_2.wav"

        if a_first:
            shutil.copy2(wav_a_path, stim_1_path)
            shutil.copy2(wav_b_path, stim_2_path)
            mapping = {"stimulus_1": "A", "stimulus_2": "B"}
        else:
            shutil.copy2(wav_b_path, stim_1_path)
            shutil.copy2(wav_a_path, stim_2_path)
            mapping = {"stimulus_1": "B", "stimulus_2": "A"}

        nonce = secrets.token_hex(8) if blind_rng is None else f"det_{rng.getrandbits(32):08x}"
        is_control = test_id.endswith("-CTRL") or category == "acoustic_ab_control_noop"

        manifest_record: Dict[str, Any] = {
            "test_id": test_id,
            "pair_label": label,
            "category": category,
            "is_control": is_control,
            "model_id": model_id,
            "mode": mode,
            "language": language,
            "seed": effective_params.seed,
            "decoding_params": {
                "temperature": effective_params.temperature,
                "top_p": effective_params.top_p,
                "top_k": effective_params.top_k,
                "repetition_penalty": effective_params.repetition_penalty,
                "max_new_tokens": effective_params.max_new_tokens,
            },
            "weights_sha256": weights_sha256 or {},
            "raw_text": raw_text,
            "normalized_text": normalized_text,
            "text_modified": normalized_text != raw_text,
            "annotations_count": len(norm_result.annotations),
            "wav_a_relpath": wav_a_path.relative_to(output_dir).as_posix(),
            "wav_b_relpath": wav_b_path.relative_to(output_dir).as_posix(),
            "stimulus_1_relpath": stim_1_path.relative_to(output_dir).as_posix(),
            "stimulus_2_relpath": stim_2_path.relative_to(output_dir).as_posix(),
            "sha256_a": sha_a,
            "sha256_b": sha_b,
            "duration_sec_a": round(res_a.audio.duration_sec, 4),
            "duration_sec_b": round(res_b.audio.duration_sec, 4),
            "delta_duration_sec": round(res_b.audio.duration_sec - res_a.audio.duration_sec, 4),
            "inference_time_sec_a": res_a.inference_time_sec,
            "inference_time_sec_b": res_b.inference_time_sec,
            "rtf_a": res_a.rtf,
            "rtf_b": res_b.rtf,
            "audio_health_a": health_a,
            "audio_health_b": health_b,
            "human_mos_status": "PENDING_HUMAN_EVALUATION",
        }
        secret_record: Dict[str, Any] = {
            "pair_label": label,
            "test_id": test_id,
            "is_control": is_control,
            "nonce": nonce,
            "stimulus_1": mapping["stimulus_1"],
            "stimulus_2": mapping["stimulus_2"],
            "stimulus_1_sha256": compute_file_sha256(stim_1_path),
            "stimulus_2_sha256": compute_file_sha256(stim_2_path),
        }
        return manifest_record, secret_record

    def run_corpus(
        self,
        records: Sequence[Dict[str, Any]],
        facade: Any,
        output_dir: Path,
        model_id: str = "1.7B-Base",
        mode: str = "clone",
        voice_prompt: Optional[VoicePrompt] = None,
        speaker_id: Optional[str] = None,
        language: str = "italian",
        params: Optional[DecodingParams] = None,
        weights_sha256: Optional[Dict[str, str]] = None,
        blind_rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        """Run full A/B acoustic generation across corpus and emit manifest + blinded templates."""
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_records: List[Dict[str, Any]] = []
        secret_mappings: List[Dict[str, Any]] = []

        for idx, rec in enumerate(records, start=1):
            test_id = str(rec["id"])
            pair_label = f"pair_{idx:02d}"
            manifest_rec, secret_rec = self.run_synthesis_pair(
                test_id=test_id,
                raw_text=str(rec["input_text"]),
                facade=facade,
                output_dir=output_dir,
                model_id=model_id,
                mode=mode,
                voice_prompt=voice_prompt,
                speaker_id=speaker_id,
                language=language,
                params=params,
                pair_label=pair_label,
                category=str(rec.get("category", "")),
                weights_sha256=weights_sha256,
                blind_rng=blind_rng,
            )
            manifest_records.append(manifest_rec)
            secret_mappings.append(secret_rec)

        manifest_path = output_dir / "ab_manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for rec in manifest_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        mapping_path = output_dir / "blinded_mapping_secret.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump({"pairs": secret_mappings}, f, indent=2, ensure_ascii=False)

        template_path = output_dir / "ab_scoring_template.csv"
        fieldnames = [
            "pair_label",
            "test_id",
            "is_control",
            "stimulus_1_relpath",
            "stimulus_2_relpath",
            "evaluator_id",
            "stimulus_1_score",
            "stimulus_2_score",
            "notes",
        ]
        with open(template_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for eval_idx in range(1, 4):
                for rec in manifest_records:
                    writer.writerow({
                        "pair_label": rec["pair_label"],
                        "test_id": rec["test_id"],
                        "is_control": rec["is_control"],
                        "stimulus_1_relpath": rec["stimulus_1_relpath"],
                        "stimulus_2_relpath": rec["stimulus_2_relpath"],
                        "evaluator_id": f"eval_{eval_idx}",
                        "stimulus_1_score": "",
                        "stimulus_2_score": "",
                        "notes": "",
                    })

        active = [r for r in manifest_records if not r["is_control"]]
        controls = [r for r in manifest_records if r["is_control"]]
        all_valid = all(
            r["audio_health_a"]["technically_valid"] and r["audio_health_b"]["technically_valid"]
            for r in manifest_records
        )
        summary = {
            "model_id": model_id,
            "seed": (params or build_default_ab_decoding_params()).seed,
            "total_records": len(manifest_records),
            "active_pairs": len(active),
            "control_pairs": len(controls),
            "active_pairs_all_modified": all(r["text_modified"] for r in active),
            "control_pairs_noop": all(not r["text_modified"] for r in controls),
            "all_wavs_technically_valid": all_valid,
            "human_mos_evaluation": score_blinded_evaluations([]),
            "manifest_path": str(manifest_path),
            "template_csv_path": str(template_path),
            "mapping_secret_path": str(mapping_path),
        }
        summary_path = output_dir / "ab_objective_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    @staticmethod
    def score_csv(csv_path: Path, mapping_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load evaluator ratings CSV (and optional secret mapping) and compute A/B metrics."""
        mapping_by_pair: Optional[Dict[str, Dict[str, Any]]] = None
        if mapping_path is not None and mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mapping_by_pair = {}
            for entry in data.get("pairs", []):
                mapping_by_pair[entry["pair_label"]] = entry
                mapping_by_pair[entry["test_id"]] = entry

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return score_blinded_evaluations(rows, mapping_by_pair=mapping_by_pair)


def test_identity_normalizer_contract():
    """Verify IdentityNormalizer preserves input text unchanged."""
    normalizer = IdentityNormalizer()
    res = normalizer.normalize("Il 15/05 alle 14:30")
    assert res.raw_text == "Il 15/05 alle 14:30"
    assert res.normalized_text == "Il 15/05 alle 14:30"
    assert len(res.annotations) == 0
    assert res.metadata["modified"] is False


def test_ab_harness_differential_reporting():
    """Verify ABTestingHarness produces compliant pairwise differential report."""
    normalizer = IdentityNormalizer()
    harness = ABTestingHarness(normalizer)

    # Condition A: 1.0s audio
    audio_a = AudioBuffer(samples=np.zeros(24000, dtype=np.float32), sample_rate=24000)
    # Condition B: 1.25s audio
    audio_b = AudioBuffer(samples=np.zeros(30000, dtype=np.float32), sample_rate=24000)

    report = harness.evaluate_pair(
        test_id="NUM_01",
        raw_text="Il treno arriva alle 14:30.",
        audio_a=audio_a,
        audio_b=audio_b,
        asr_a="il treno arriva alle quattordici e trenta",
        asr_b="il treno arriva alle quattordici e trenta",
    )

    assert report.test_id == "NUM_01"
    assert report.delta_duration_sec == pytest.approx(0.25)
    assert len(report.human_evaluation_rubric) == 7
    assert "absence_of_truncation" in report.human_evaluation_rubric
    assert "numbers_dates_accuracy" in report.human_evaluation_rubric


def test_ab_corpus_rev2_text_contract():
    """Verify all 20 active A/B pairs transform cleanly and M2-AB-017-CTRL stays No-Op."""
    records = load_ab_corpus(DEFAULT_AB_CORPUS_PATH)
    assert len(records) == 21

    normalizer = ItalianTextNormalizer()
    active = [r for r in records if r["id"] != "M2-AB-017-CTRL"]
    ctrl = [r for r in records if r["id"] == "M2-AB-017-CTRL"]
    assert len(active) == 20
    assert len(ctrl) == 1

    for rec in records:
        res = normalizer.normalize(rec["input_text"])
        assert res.normalized_text == rec["expected_normalized_text"], rec["id"]

    for rec in active:
        assert rec["expected_normalized_text"] != rec["input_text"], rec["id"]
    assert ctrl[0]["expected_normalized_text"] == ctrl[0]["input_text"]


def test_ab_harness_runner_and_blinded_manifest(tmp_path):
    """Verify Path A raw vs Path B normalized synthesis, blinded bijection, and manifest SHA-256."""
    from unittest.mock import MagicMock
    from src.engine.types import GenerationResult, MemorySnapshot

    calls = []

    def _fake_synthesize(**kwargs):
        calls.append(kwargs)
        freq = 220.0 if len(calls) % 2 == 1 else 440.0
        t = np.linspace(0.0, 0.5, 12000, endpoint=False, dtype=np.float32)
        samples = (0.2 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        buf = AudioBuffer(samples=samples, sample_rate=24000)
        mem = MemorySnapshot(100, 200, 150, 300)
        return GenerationResult(
            audio=buf,
            inference_time_sec=0.1,
            rtf=0.2,
            memory=mem,
            model_version=kwargs["model_id"],
            decoding_params=kwargs["params"],
            environment_manifest_sha256="test_sha",
        )

    mock_facade = MagicMock()
    mock_facade.synthesize.side_effect = _fake_synthesize

    harness = ABTestingHarness(ItalianTextNormalizer())
    sample_records = [
        {"id": "M2-AB-001", "input_text": "La seduta è fissata per il 1° ottobre alle 09:00.", "category": "acoustic_ab_dates"},
        {"id": "M2-AB-017-CTRL", "input_text": "Il trattato istitutivo della NATO è stato ratificato a Washington.", "category": "acoustic_ab_control_noop"},
    ]
    summary = harness.run_corpus(
        records=sample_records,
        facade=mock_facade,
        output_dir=tmp_path,
        model_id="1.7B-Base",
        mode="clone",
        params=build_default_ab_decoding_params(seed=42),
        blind_rng=random.Random(123),
    )

    assert len(calls) == 4
    # Identical decoding params & seed across Path A and Path B
    assert calls[0]["text"] == "La seduta è fissata per il 1° ottobre alle 09:00."
    assert calls[1]["text"] == "La seduta è fissata per il primo ottobre alle nove."
    assert calls[0]["params"].seed == 42 and calls[1]["params"].seed == 42
    assert calls[2]["text"] == calls[3]["text"]  # Control No-Op

    assert summary["active_pairs"] == 1
    assert summary["control_pairs"] == 1
    assert summary["all_wavs_technically_valid"] is True
    assert summary["human_mos_evaluation"]["status"] == "PENDING_HUMAN_EVALUATION"

    manifest_lines = (tmp_path / "ab_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 2
    rec0 = json.loads(manifest_lines[0])
    assert len(rec0["sha256_a"]) == 64 and len(rec0["sha256_b"]) == 64

    mapping = json.loads((tmp_path / "blinded_mapping_secret.json").read_text(encoding="utf-8"))
    pair0_map = mapping["pairs"][0]
    stim1_sha = pair0_map["stimulus_1_sha256"]
    stim2_sha = pair0_map["stimulus_2_sha256"]
    if pair0_map["stimulus_1"] == "A":
        assert stim1_sha == rec0["sha256_a"] and stim2_sha == rec0["sha256_b"]
    else:
        assert stim1_sha == rec0["sha256_b"] and stim2_sha == rec0["sha256_a"]


def test_ab_scoring_formulas_include_ties_and_exclude_control(tmp_path):
    """Verify MOS, Delta-MOS, Standard Win Rate (ties in denominator), Non-Loss Rate, and disagreements."""
    mapping_path = tmp_path / "blinded_mapping_secret.json"
    mapping_path.write_text(
        json.dumps({
            "pairs": [
                {"pair_label": "pair_01", "test_id": "M2-AB-001", "stimulus_1": "B", "stimulus_2": "A"},
                {"pair_label": "pair_02", "test_id": "M2-AB-002", "stimulus_1": "A", "stimulus_2": "B"},
                {"pair_label": "pair_03", "test_id": "M2-AB-017-CTRL", "stimulus_1": "A", "stimulus_2": "B"},
            ]
        }),
        encoding="utf-8",
    )
    csv_path = tmp_path / "ratings.csv"
    csv_path.write_text(
        "\n".join([
            "pair_label,test_id,is_control,evaluator_id,stimulus_1_score,stimulus_2_score",
            "pair_01,M2-AB-001,False,eval_1,5,3",  # B=5, A=3 -> Win
            "pair_01,M2-AB-001,False,eval_2,4,4",  # B=4, A=4 -> Tie
            "pair_01,M2-AB-001,False,eval_3,2,4",  # B=2, A=4 -> Loss & disagreement (5-2=3 >= 2)
            "pair_02,M2-AB-002,False,eval_1,3,5",  # A=3, B=5 -> Win
            "pair_03,M2-AB-017-CTRL,True,eval_1,1,1",  # Control row excluded from active MOS & b_mos1_count
        ]),
        encoding="utf-8",
    )

    result = ABTestingHarness.score_csv(csv_path, mapping_path=mapping_path)
    assert result["active_comparisons"] == 4
    assert result["control_comparisons"] == 1
    assert result["wins"] == 2
    assert result["ties"] == 1
    assert result["losses"] == 1
    # Standard Win Rate = 2 / (2+1+1) = 0.50 (ties MUST remain in denominator)
    assert result["standard_win_rate"] == pytest.approx(0.50)
    # Non-Loss Rate = (2 + 1) / 4 = 0.75
    assert result["non_loss_rate"] == pytest.approx(0.75)
    assert result["mos_a"] == pytest.approx((3 + 4 + 4 + 3) / 4)
    assert result["mos_b"] == pytest.approx((5 + 4 + 2 + 5) / 4)
    assert result["delta_mos"] == pytest.approx(0.5)
    assert result["b_mos1_count"] == 0
    assert len(result["disagreements"]) == 1
    assert result["disagreements"][0]["test_id"] == "M2-AB-001"

