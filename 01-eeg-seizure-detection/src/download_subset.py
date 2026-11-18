"""Unduh subset kecil CHB-MIT (satu pasien) dari PhysioNet.

Mengambil semua rekaman yang berisi kejang ditambah beberapa rekaman tanpa kejang,
supaya total ukuran tetap kecil (~40 MB per file).
"""
import argparse
import re
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/chbmit/1.0.0"
ROOT = Path(__file__).resolve().parents[1]


def parse_summary(text: str) -> dict[str, list[tuple[float, float]]]:
    """Kembalikan {nama_file: [(mulai_detik, selesai_detik), ...]} dari file summary."""
    seizures: dict[str, list[tuple[float, float]]] = {}
    current = None
    starts: list[float] = []
    for line in text.splitlines():
        m = re.match(r"File Name:\s*(\S+\.edf)", line)
        if m:
            current = m.group(1)
            seizures[current] = []
            starts = []
            continue
        m = re.match(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)", line)
        if m and current:
            starts.append(float(m.group(1)))
            continue
        m = re.match(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)", line)
        if m and current and starts:
            seizures[current].append((starts.pop(0), float(m.group(1))))
    return seizures


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"sudah ada: {dest.name}")
        return
    print(f"mengunduh {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--patient", default="chb01")
    p.add_argument("--n-normal", type=int, default=3, help="jumlah rekaman tanpa kejang")
    args = p.parse_args()

    out = ROOT / "data" / args.patient
    out.mkdir(parents=True, exist_ok=True)

    summary_name = f"{args.patient}-summary.txt"
    download(f"{BASE}/{args.patient}/{summary_name}", out / summary_name)
    seizures = parse_summary((out / summary_name).read_text())

    with_sz = [f for f, s in seizures.items() if s]
    without_sz = [f for f, s in seizures.items() if not s][: args.n_normal]
    files = with_sz + without_sz
    print(f"{len(with_sz)} rekaman dengan kejang, {len(without_sz)} tanpa kejang")

    for f in files:
        download(f"{BASE}/{args.patient}/{f}", out / f)
    print(f"selesai. data di {out}")


if __name__ == "__main__":
    main()
