import argparse
import ctypes
from datetime import datetime, timezone
import os
import re
import sqlite3
import time
from pathlib import Path


PROGRESS_RE = re.compile(r"(?P<step>\d+)/(?P<total>\d+).*?(?P<speed>\d+(?:\.\d+)?)s/it")


def process_exists(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def parse_progress(log_path: Path) -> tuple[int | None, str | None]:
    if not log_path.exists():
        return None, None
    # Read the tail only; tqdm logs are repeated and can get large.
    with log_path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 64 * 1024), os.SEEK_SET)
        text = f.read().decode("utf-8", errors="ignore")

    matches = list(PROGRESS_RE.finditer(text))
    if not matches:
        return None, None
    match = matches[-1]
    step = int(match.group("step"))
    speed = f"{float(match.group('speed')):.2f} sec/iter"
    return step, speed


def update_job(db_path: Path, job_id: str, step: int, speed: str) -> None:
    with sqlite3.connect(db_path, timeout=30.0) as conn:
        conn.execute(
            "UPDATE Job SET step = ?, speed_string = ?, info = ?, updated_at = ? WHERE id = ?",
            (step, speed, "Training", datetime.now(timezone.utc).isoformat(), job_id),
        )
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()

    db_path = Path(args.db)
    log_path = Path(args.log)
    last_step = None
    while True:
        step, speed = parse_progress(log_path)
        if step is not None and speed is not None and step != last_step:
            update_job(db_path, args.job_id, step, speed)
            last_step = step
        if args.pid and not process_exists(args.pid):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
