"""CLI runner for M2 WP6 Acoustic A/B Evaluation and blinded MOS scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engine.adapters.qwen3_adapter import KNOWN_REF_TRANSCRIPTS, MODEL_PATHS
from src.engine.facade import TTSEngineFacade
from src.linguistics import ItalianTextNormalizer
from tests.ab_linguistics.test_ab_harness import (
    ABTestingHarness,
    DEFAULT_AB_CORPUS_PATH,
    DEFAULT_AB_OUTPUT_DIR,
    DEFAULT_COORDINATOR_KEY_PATH,
    DEFAULT_EVALUATOR_PACK_DIR,
    build_default_ab_decoding_params,
    compute_model_weights_sha256,
    load_ab_corpus,
)


def _format_objective_markdown(manifest_records: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    """Render an objective markdown summary strictly from measured generation outputs."""
    lines: List[str] = [
        "# M2 WP6 Acoustic A/B Objective Evaluation Report",
        "",
        f"- **Model:** `{summary['model_id']}`",
        f"- **Seed:** `{summary['seed']}`",
        f"- **Active Pairs:** `{summary['active_pairs']}`",
        f"- **Control Pairs (No-Op):** `{summary['control_pairs']}`",
        f"- **All WAVs Technically Valid:** `{summary['all_wavs_technically_valid']}`",
        f"- **Human Listening Panel Status:** `{summary['human_mos_evaluation']['status']}` *(subjective MOS ratings must be supplied by human native-Italian listeners via `ab_scoring_template.csv`)*",
        "",
        "## Pairwise Objective Measurements",
        "",
        "| Pair ID | Category | Modified | Dur A (s) | Dur B (s) | Delta Dur (s) | RTF A | RTF B | RMS A (dBFS) | RMS B (dBFS) | Valid A/B |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for rec in manifest_records:
        ha = rec["audio_health_a"]
        hb = rec["audio_health_b"]
        valid_pair = ha["technically_valid"] and hb["technically_valid"]
        lines.append(
            f"| `{rec['test_id']}` | `{rec['category']}` | {rec['text_modified']} | "
            f"{rec['duration_sec_a']:.3f} | {rec['duration_sec_b']:.3f} | {rec['delta_duration_sec']:+.3f} | "
            f"{rec['rtf_a']:.3f} | {rec['rtf_b']:.3f} | {ha['rms_dbfs']:.2f} | {hb['rms_dbfs']:.2f} | {valid_pair} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_generate(args: argparse.Namespace) -> int:
    """Execute reproducible CUDA A/B generation across the corpus."""
    corpus_path = Path(args.corpus)
    output_dir = Path(args.output_dir)
    ref_audio = Path(args.ref_audio)
    model_id = args.model_id

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    records = load_ab_corpus(corpus_path)
    model_dir = Path(MODEL_PATHS.get(model_id, model_id))
    weights_sha256 = compute_model_weights_sha256(model_dir)
    params = build_default_ab_decoding_params(seed=args.seed)
    ref_transcript = KNOWN_REF_TRANSCRIPTS.get(ref_audio.name)

    facade = TTSEngineFacade()
    harness = ABTestingHarness(ItalianTextNormalizer())

    try:
        with facade.use_adapter(model_id) as adapter:
            voice_prompt = adapter.create_voice_prompt(
                ref_audio_path=str(ref_audio),
                ref_transcript=ref_transcript,
                x_vector_only=(ref_transcript is None),
            )
            summary = harness.run_corpus(
                records=records,
                facade=facade,
                output_dir=output_dir,
                model_id=model_id,
                mode="clone",
                voice_prompt=voice_prompt,
                language="italian",
                params=params,
                weights_sha256=weights_sha256,
            )
    finally:
        facade.unload()

    manifest_records: List[Dict[str, Any]] = []
    with open(output_dir / "ab_manifest.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                manifest_records.append(json.loads(line))

    report_path = output_dir / "AB_OBJECTIVE_REPORT.md"
    report_path.write_text(_format_objective_markdown(manifest_records, summary), encoding="utf-8")
    summary["objective_report_path"] = str(report_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def run_prepare_blind_pack(args: argparse.Namespace) -> int:
    """Create a freshly randomized, stripped evaluator pack and isolated coordinator key."""
    info = ABTestingHarness.create_blind_evaluator_package(
        source_dir=Path(args.source_dir),
        pack_dir=Path(args.pack_dir),
        key_output_path=Path(args.key_out),
        num_evaluators=args.evaluators,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def run_score(args: argparse.Namespace) -> int:
    """Compute MOS, Delta-MOS, Standard Win Rate, and Non-Loss Rate from evaluator CSV."""
    csv_path = Path(args.csv)
    mapping_path = Path(args.mapping) if args.mapping else None
    result = ABTestingHarness.score_csv(csv_path, mapping_path=mapping_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M2 WP6 Acoustic A/B Evaluation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    gen_p = sub.add_parser("generate", help="Generate real CUDA A/B audio pairs and manifest")
    gen_p.add_argument("--model-id", default="1.7B-Base", choices=["1.7B-Base", "0.6B-Base"])
    gen_p.add_argument("--seed", type=int, default=42)
    gen_p.add_argument("--corpus", default=str(DEFAULT_AB_CORPUS_PATH))
    gen_p.add_argument("--output-dir", default=str(DEFAULT_AB_OUTPUT_DIR))
    gen_p.add_argument("--ref-audio", default=str(REPO_ROOT / "bench" / "reference_voices" / "ref_voice_it_01.wav"))

    pack_p = sub.add_parser("prepare-blind-pack", help="Build clean anonymous evaluator pack")
    pack_p.add_argument("--source-dir", default=str(DEFAULT_AB_OUTPUT_DIR))
    pack_p.add_argument("--pack-dir", default=str(DEFAULT_EVALUATOR_PACK_DIR))
    pack_p.add_argument("--key-out", default=str(DEFAULT_COORDINATOR_KEY_PATH))
    pack_p.add_argument("--evaluators", type=int, default=3)

    score_p = sub.add_parser("score", help="Score blinded evaluator CSV ratings")
    score_p.add_argument("--csv", required=True)
    score_p.add_argument("--mapping", default=str(DEFAULT_COORDINATOR_KEY_PATH))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "generate":
        return run_generate(args)
    if args.command == "prepare-blind-pack":
        return run_prepare_blind_pack(args)
    if args.command == "score":
        return run_score(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
