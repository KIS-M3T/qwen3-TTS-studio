"""Profiling runner supporting Tier 2 Smoke and Tier 3 Full Matrix benchmarks (R5, C6, B4)."""

import argparse
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
            t_load_0 = time.perf_counter()
            adapter.load(device="cuda:0", dtype_str="bfloat16")
            load_sec = round(time.perf_counter() - t_load_0, 3)
            print(f"[{model_id}] Loaded successfully in {load_sec}s.")

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
            )
            print(f"[{model_id}] Warmup complete (RTF: {warm_res.rtf}).")

            # Measured iterations
            for case_id, text in test_cases:
                for it in range(1, n_iters + 1):
                    print(f"[{model_id}][{case_id}] Iteration {it}/{n_iters}...")
                    res = adapter.synthesize_base(
                        text=text,
                        language="italian",
                        params=DecodingParams(seed=42 + it),
                    )

                    # Save WAV
                    wav_filename = f"{case_id}_{model_id}_iter{it}.wav"
                    wav_out_path = session.session_dir / wav_filename
                    sf.write(str(wav_out_path), res.audio.samples, res.audio.sample_rate)

                    session.record_run(
                        model_id=model_id,
                        test_case_id=case_id,
                        iteration_index=it,
                        input_text_raw=text,
                        run_type="measured",
                        measured=True,
                        seed=42 + it,
                        audio_duration_sec=res.audio.duration_sec,
                        inference_time_sec=res.inference_time_sec,
                        wav_path=str(wav_out_path),
                    )
                    print(f"  -> Generated {res.audio.duration_sec:.2f}s in {res.inference_time_sec:.2f}s (RTF: {res.rtf})")

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
