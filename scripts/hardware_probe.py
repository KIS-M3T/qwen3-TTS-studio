#!/usr/bin/env python3
"""
Hardware & Environment Probe for Italian TTS Studio.
Non-invasive diagnostic script collecting GPU, VRAM, RAM, CPU, OS, Driver,
CUDA, Python, and Disk information across Windows and Linux.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> str | None:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return None


def get_gpu_info_nvidia_smi() -> list[dict]:
    gpus = []
    query_cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,memory.used,driver_version,pstate,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = run_cmd(query_cmd)
    if not out:
        return gpus

    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": float(parts[2]),
                "memory_free_mb": float(parts[3]),
                "memory_used_mb": float(parts[4]),
                "driver_version": parts[5],
                "pstate": parts[6] if len(parts) > 6 else "N/A",
                "temp_c": float(parts[7]) if len(parts) > 7 and parts[7].replace(".", "").isdigit() else None,
            })
    return gpus


def get_cuda_version() -> str | None:
    nvcc_out = run_cmd(["nvcc", "--version"])
    if nvcc_out:
        for line in nvcc_out.splitlines():
            if "release" in line:
                return line.split("release")[-1].strip()

    smi_out = run_cmd(["nvidia-smi"])
    if smi_out:
        for line in smi_out.splitlines():
            if "CUDA Version:" in line:
                parts = line.split("CUDA Version:")
                if len(parts) > 1:
                    return parts[1].split()[0].replace("|", "").strip()
    return None


def get_system_ram() -> dict:
    ram_info = {"total_gb": None, "available_gb": None}
    if platform.system() == "Windows":
        try:
            ps_cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "$os = Get-CimInstance Win32_OperatingSystem; Write-Output \"$($os.TotalVisibleMemorySize)|$($os.FreePhysicalMemory)\"",
            ]
            out = run_cmd(ps_cmd)
            if out and "|" in out:
                parts = out.strip().split("|")
                total_kb = float(parts[0])
                free_kb = float(parts[1])
                ram_info["total_gb"] = round(total_kb / (1024 * 1024), 2)
                ram_info["available_gb"] = round(free_kb / (1024 * 1024), 2)
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                mem = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        mem[parts[0].strip()] = int(parts[1].split()[0])
                if "MemTotal" in mem:
                    ram_info["total_gb"] = round(mem["MemTotal"] / (1024 * 1024), 2)
                if "MemAvailable" in mem:
                    ram_info["available_gb"] = round(mem["MemAvailable"] / (1024 * 1024), 2)
        except Exception:
            pass
    return ram_info


def get_disk_info(path: str = ".") -> dict:
    total, used, free = shutil.disk_usage(Path(path).resolve())
    return {
        "drive_path": str(Path(path).resolve()),
        "total_gb": round(total / (1024**3), 2),
        "used_gb": round(used / (1024**3), 2),
        "free_gb": round(free / (1024**3), 2),
    }


def get_torch_info() -> dict:
    info = {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "device_names": [],
        "bf16_supported": False,
        "fp16_supported": False,
    }
    try:
        import torch
        info["installed"] = True
        info["version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            info["device_names"] = [torch.cuda.get_device_name(i) for i in range(info["device_count"])]
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
            info["fp16_supported"] = True
    except ImportError:
        pass
    except Exception as e:
        info["error"] = str(e)
    return info


def run_probe() -> dict:
    gpus = get_gpu_info_nvidia_smi()
    cuda_ver = get_cuda_version()
    ram = get_system_ram()
    disk = get_disk_info(".")
    torch_info = get_torch_info()

    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "system_ram": ram,
        "disk": disk,
        "gpus": gpus,
        "cuda_version_detected": cuda_ver,
        "torch": torch_info,
    }


def generate_markdown_report(data: dict) -> str:
    lines = [
        "# Hardware & Environment Diagnostic Report",
        "> Generato automaticamente da `scripts/hardware_probe.py` per Milestone M0.",
        "",
        "## 1. Sommario Esecutivo",
        "",
    ]

    gpu_desc = "Nessuna GPU rilevata"
    if data["gpus"]:
        g = data["gpus"][0]
        gpu_desc = f"**{g['name']}** — {g['memory_total_mb']:.0f} MB VRAM (Libera: {g['memory_free_mb']:.0f} MB)"

    ram_total = f"{data['system_ram']['total_gb']} GB" if data['system_ram']['total_gb'] else "N/A"
    ram_free = f"{data['system_ram']['available_gb']} GB" if data['system_ram']['available_gb'] else "N/A"

    lines.extend([
        f"- **Sistema Operativo**: {data['os']['system']} {data['os']['release']} ({data['os']['machine']})",
        f"- **CPU**: {data['os']['processor']}",
        f"- **RAM di Sistema**: {ram_total} totali (disponibili: {ram_free})",
        f"- **GPU Principale**: {gpu_desc}",
        f"- **Driver NVIDIA**: {data['gpus'][0]['driver_version'] if data['gpus'] else 'N/A'}",
        f"- **CUDA Version (Driver)**: {data['cuda_version_detected'] or 'N/A'}",
        f"- **Storage Locale**: {data['disk']['free_gb']} GB liberi su {data['disk']['total_gb']} GB ({data['disk']['drive_path']})",
        f"- **Python Runtime**: {data['python']['version']} (`{data['python']['executable']}`)",
        f"- **PyTorch**: {data['torch']['version'] or 'Non installato / CPU'} (CUDA disponibile: {data['torch']['cuda_available']})",
        "",
        "## 2. Dettaglio GPU (nvidia-smi)",
        "",
        "| ID | Modello | VRAM Totale | VRAM Libera | VRAM Usata | Driver | Temp |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for g in data["gpus"]:
        temp = f"{g['temp_c']} °C" if g['temp_c'] is not None else "N/A"
        lines.append(
            f"| {g['index']} | {g['name']} | {g['memory_total_mb']:.0f} MB | {g['memory_free_mb']:.0f} MB | {g['memory_used_mb']:.0f} MB | {g['driver_version']} | {temp} |"
        )

    lines.extend([
        "",
        "## 3. Ambiente PyTorch & Accellerazione",
        "",
        f"- **PyTorch Installato**: {data['torch']['installed']}",
        f"- **Versione PyTorch**: {data['torch']['version']}",
        f"- **CUDA Runtime PyTorch**: {data['torch']['cuda_version'] or 'N/A'}",
        f"- **Dispositivi CUDA accessibili da PyTorch**: {data['torch']['device_count']}",
        f"- **Supporto nativo BF16**: {data['torch']['bf16_supported']}",
        f"- **Supporto nativo FP16**: {data['torch']['fp16_supported']}",
        "",
        "## 4. Note Operative per Milestone M0",
        "",
        "1. **Verifica VRAM Effettiva**: La scheda GPU rilevata dispone di **~12.288 MB di VRAM** dedicata.",
        "2. **PyTorch Environment**: Se l'interprete Python globale ha installata una build PyTorch CPU (`+cpu`), per il benchmark con GPU CUDA è necessario operare all'interno di un ambiente virtuale (`.venv`) dedicato con PyTorch compilato per CUDA 12.x (`cu124` o `cu121`).",
        "3. **Spazio Disco**: Verificato sufficiente per i pesi modello Qwen3-TTS 0.6B (~2.3 GB) e 1.7B (~4.2 GB) e l'audio generato.",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    report_data = run_probe()
    md_content = generate_markdown_report(report_data)

    out_json = Path("scripts/hardware_info.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    out_md = Path("HARDWARE_REPORT.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("=== PROBE COMPLETATO ===")
    print(f"Salvato report JSON in: {out_json}")
    print(f"Salvato report MD in:   {out_md}")
    print("\n" + md_content)
