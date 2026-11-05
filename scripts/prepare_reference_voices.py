import csv
from pathlib import Path
import soundfile as sf
import numpy as np

tsv_path = Path("cv-corpus-22.0-delta-2025-06-20-it/cv-corpus-22.0-delta-2025-06-20/it/validated.tsv")
clips_dir = Path("cv-corpus-22.0-delta-2025-06-20-it/cv-corpus-22.0-delta-2025-06-20/it/clips")
ref_dir = Path("bench/reference_voices")
ref_dir.mkdir(parents=True, exist_ok=True)

with open(tsv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

valid_clips = []
for r in rows:
    p = clips_dir / r["path"]
    if p.exists():
        try:
            info = sf.info(str(p))
            if 3.5 <= info.duration <= 7.0:
                valid_clips.append((r, info.duration))
        except Exception:
            pass

print(f"Found candidates with 3.5-7.0s duration: {len(valid_clips)}")

selected = [valid_clips[0], valid_clips[1]]

datasheet_lines = [
    "# Reference Voices Datasheet (Milestone M0)",
    "> Riferimenti vocali italiani legali per benchmark e test di clonazione zero-shot.",
    "",
    "| File | Fonte Originale | Durata | Trascrizione Ufficiale | Licenza Audio | Licenza Testo |",
    "| :--- | :--- | :--- | :--- | :--- | :--- |",
]

for idx, (row, dur) in enumerate(selected, 1):
    src_path = clips_dir / row["path"]
    data, sr = sf.read(str(src_path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    target_sr = 24000
    if sr != target_sr:
        num_target_samples = int(len(data) * target_sr / sr)
        resampled = np.interp(
            np.linspace(0, len(data), num_target_samples, endpoint=False),
            np.arange(len(data)),
            data,
        ).astype(np.float32)
    else:
        resampled = data.astype(np.float32)

    peak = np.max(np.abs(resampled))
    if peak > 0:
        resampled = resampled * (0.89 / peak)

    out_file = ref_dir / f"ref_voice_it_{idx:02d}.wav"
    sf.write(str(out_file), resampled, target_sr, subtype="PCM_16")
    print(f"Wrote {out_file}: {len(resampled)/target_sr:.2f}s, SR={target_sr}")

    datasheet_lines.append(
        f"| `ref_voice_it_{idx:02d}.wav` | Common Voice ({row['path']}) | {dur:.2f}s | \"{row['sentence']}\" | CC0 (Public Domain) | CC0 / Permissive |"
    )

with open(ref_dir / "VOICE_DATASHEET.md", "w", encoding="utf-8") as f:
    f.write("\n".join(datasheet_lines) + "\n")
print("VOICE_DATASHEET.md written successfully.")
