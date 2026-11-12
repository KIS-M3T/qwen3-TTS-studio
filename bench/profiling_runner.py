"""Profiling runner supporting Tier 2 Smoke and Tier 3 Full Matrix benchmarks (R5, C6, B4)."""

import argparse
import hashlib
from pathlib import Path
import sys
import time

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Optional
import soundfile as sf

from src.engine.adapters.qwen3_adapter import Qwen3EngineAdapter
from src.engine.types import DecodingParams
from src.instrumentation.tracker import BenchmarkSession


SMOKE_TEST_CASES = [
    ("NUM_01", "Il treno parte dal binario 3 alle ore 14:30."),
]

PROFILING_TEST_CASES = [
    ("NUM_01", "Il treno parte dal binario 3 alle ore 14:30."),
    ("NUM_02", "Nel 1998 il costo stimato era di circa 2500000 lire."),
    ("ACR_01", "La sede dell'INPS e dell'ISTAT si trova a Roma."),
    ("DATE_01", "La conferenza si terrà il 15/05/2026 a Milano."),
    ("CURR_01", "Il prezzo promozionale è di 49,99 € IVA inclusa."),
    ("PROS_01", "Sebbene fosse tardi, decise comunque di proseguire il viaggio."),
    ("LEN_01", "Sì."),
    ("LEN_02", "No, grazie."),
    ("TECH_01", "Il protocollo TCP/IP garantisce una connessione affidabile a 100 Mb/s."),
    ("MIX_01", "Il 3º classificato riceverà un premio di 150 € il 22/08/2026."),
]


def run_profiling(
    tier: str = "smoke",
    models: Optional[List[str]] = None,
    iterations: int = 5,
    session_id: Optional[str] = None,
    ref_audio_path: str = "bench/reference_voices/ref_voice_it_01.wav",
) -> None:
    """Execute performance profiling across configured execution tiers.

    Tiers:
      - smoke: 1 model (0.6B), 1 case, 1 iteration (~2 min)
      - full: 2 models, 10 cases, 5 iterations (~90-100 min)
    """
    selected_models = models or (["0.6B-Base"] if tier == "smoke" else ["0.6B-Base", "1.7B-Base"])
    test_cases = SMOKE_TEST_CASES if tier == "smoke" else PROFILING_TEST_CASES
    n_iters = 1 if tier == "smoke" else max(1, iterations)

    session = BenchmarkSession(session_id=session_id)
    print(f"\n==========================================")
    print(f"PROFILING RUNNER — TIER: {tier.upper()}")
    print(f"Session ID: {session.session_id}")
    print(f"Directory:  {session.session_dir}")
    print(f"Models:     {selected_models}")
    print(f"Cases:      {len(test_cases)}")
    print(f"Iterations: {n_iters}")
    print(f"==========================================\n")

    for model_id in selected_models:
        print(f"[{model_id}] Initializing adapter and loading weights...")
        adapter = Qwen3EngineAdapter(model_id)

        try:
            # G4: Measure real process memory baseline prior to model load
            from src.instrumentation.memory import get_memory_snapshot
            init_mem = get_memory_snapshot()
            proc_baseline_mb = init_mem.get("torch_allocated_mb")

            t_load_0 = time.perf_counter()
            adapter.load(device="cuda:0", dtype_str="bfloat16")
            load_sec = round(time.perf_counter() - t_load_0, 3)
            print(f"[{model_id}] Loaded successfully in {load_sec}s.")

            # G4: Checkpoint, Config and Weights SHA-256 identification
            config_file = Path(adapter.model_path) / "config.json"
            config_sha256 = hashlib.sha256(config_file.read_bytes()).hexdigest() if config_file.exists() else None

            weights_file = Path(adapter.model_path) / "model.safetensors"
            weights_sha256 = None
            if weights_file.exists():
                h_w = hashlib.sha256()
                with open(weights_file, "rb") as wf:
                    while chunk := wf.read(2 * 1024 * 1024):
                        h_w.update(chunk)
                weights_sha256 = h_w.hexdigest()
            checkpoint_revision = "qwen3-tts-12hz-official"

            # Warmup run (mandatory separation)
            print(f"[{model_id}] Executing warmup inference...")
            warm_res = adapter.synthesize_base(text="Warmup di prova per inizializzazione runtime.")
            session.record_run(
                model_id=model_id,
                test_case_id="WARM_01",
                iteration_index=0,
                input_text_raw="Warmup di prova",
                run_type="warmup",
                measured=False,
                audio_duration_sec=warm_res.audio.duration_sec,
                inference_time_sec=warm_res.inference_time_sec,
                baseline_process_mb=proc_baseline_mb,
                checkpoint_revision=checkpoint_revision,
                config_sha256=config_sha256,
                weights_sha256=weights_sha256,
            )
            print(f"[{model_id}] Warmup complete (RTF: {warm_res.rtf}).")

            # Measured iterations
            for case_id, text in test_cases:
                for it in range(1, n_iters + 1):
                    print(f"[{model_id}][{case_id}] Iteration {it}/{n_iters}...")
                    dec_params = DecodingParams(seed=42 + it)
                    res = adapter.synthesize_base(
                        text=text,
                        language="italian",
                        params=dec_params,
                    )

                    # Save WAV and compute SHA-256 (G4, G6, H4)
                    wav_out_path = session.get_wav_path_for_run(model_id, case_id, it)
                    t_save_0 = time.perf_counter()
                    sf.write(str(wav_out_path), res.audio.samples, res.audio.sample_rate)
                    save_time_sec = time.perf_counter() - t_save_0
                    wav_sha256 = hashlib.sha256(wav_out_path.read_bytes()).hexdigest()

                    session.record_run(
                        model_id=model_id,
                        test_case_id=case_id,
                        iteration_index=it,
                        input_text_raw=text,
                        run_type="measured",
                        measured=True,
                        seed=42 + it,
                        temperature=dec_params.temperature,
                        top_p=dec_params.top_p,
                        top_k=dec_params.top_k,
                        repetition_penalty=dec_params.repetition_penalty,
                        min_new_tokens=dec_params.min_new_tokens,
                        max_new_tokens=dec_params.max_new_tokens,
                        subtalker_params=dec_params.subtalker_params,
                        applied_params=res.metadata.get("applied_params"),
                        audio_duration_sec=res.audio.duration_sec,
                        inference_time_sec=res.inference_time_sec,
                        save_time_sec=round(save_time_sec, 4),
                        baseline_process_mb=proc_baseline_mb,
                        wav_path=str(wav_out_path),
                        wav_sha256=wav_sha256,
                        checkpoint_revision=checkpoint_revision,
                        config_sha256=config_sha256,
                        weights_sha256=weights_sha256,
                    )
                    print(f"  -> Generated {res.audio.duration_sec:.2f}s in {res.inference_time_sec:.2f}s (RTF: {res.rtf}, WAV SHA: {wav_sha256[:8]}...)")

        except Exception as e:
            print(f"[{model_id}] Error during profiling: {e}")
        finally:
            print(f"[{model_id}] Unloading model...")
            adapter.unload()

    reports = session.generate_reports()
    print(f"\n==========================================")
    print(f"PROFILING COMPLETED")
    print(f"Events:  {session.events_file}")
    print(f"Summary: {session.summary_file}")
    print(f"CSV:     {session.csv_file}")
    print(f"==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profiling runner for Italian TTS Studio")
    parser.add_argument("--tier", choices=["smoke", "full"], default="smoke", help="Execution tier")
    parser.add_argument("--models", nargs="+", default=None, help="Models to profile")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per case (default: 5)")
    parser.add_argument("--session-id", default=None, help="Optional session identifier")
    args = parser.parse_args()

    run_profiling(
        tier=args.tier,
        models=args.models,
        iterations=args.iterations,
        session_id=args.session_id,
    )
