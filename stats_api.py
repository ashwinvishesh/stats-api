import time
import re
import os
import threading
from flask import Flask, jsonify

# =========================
# V7 Script
# =========================

LOG_FILE = "/root/miner.log"
STALE_THRESHOLD_SEC = 30
READ_BACK_LINES = 300
LOG_POLL_INTERVAL = 0.2

# =========================
# App & State
# =========================

app = Flask(__name__)
state_lock = threading.Lock()

state = {
    "gpu_hashrates": {},
    "last_update":   None
}

# =========================
# ANSI stripper
# Screen -L logs raw terminal output including color escape codes.
# Strip them before any parsing.
# =========================

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

def strip_ansi(s: str) -> str:
    return ANSI_ESCAPE.sub("", s)

# =========================
# Regex
# =========================

GPU_REGEX = re.compile(
    r"gpu=(\d+):(.+?)(?=\s+component=).*?hashrate_th_s=([\d.]+)",
    re.IGNORECASE
)

# =========================
# Debug helpers
# =========================

def dbg(tag: str, msg: str):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    print(f"[{ts}][{tag}] {msg}", flush=True)

# =========================
# Log Processing
# =========================

def process_line(line: str):
    now     = time.time()
    clean   = strip_ansi(line).rstrip("\n")

    gpu_match = GPU_REGEX.search(clean)
    if gpu_match:
        gpu_id   = int(gpu_match.group(1))
        gpu_name = gpu_match.group(2).strip()
        hashrate = float(gpu_match.group(3))
        dbg("MATCH", f"gpu_id={gpu_id} name={gpu_name!r} hashrate={hashrate}")

        with state_lock:
            state["gpu_hashrates"][gpu_id] = {
                "name":      gpu_name,
                "hashrate":  hashrate,
                "last_seen": now
            }
            state["last_update"] = now

# =========================
# Log Follower
# =========================

def follow_log():
    last_inode = None
    dbg("FOLLOWER", f"starting, watching {LOG_FILE!r}")

    while True:
        if not os.path.exists(LOG_FILE):
            dbg("FOLLOWER", "log file does not exist yet, waiting...")
            time.sleep(1)
            continue

        try:
            stat  = os.stat(LOG_FILE)
            inode = stat.st_ino

            if inode != last_inode:
                last_inode = inode
                dbg("FOLLOWER", f"new/rotated file detected (inode={inode}), backfilling last {READ_BACK_LINES} lines")

                with open(LOG_FILE, "r", errors="replace") as f:
                    all_lines  = f.readlines()
                    tail_lines = all_lines[-READ_BACK_LINES:]
                    dbg("FOLLOWER", f"total lines in file={len(all_lines)}, backfilling {len(tail_lines)} lines")

                    for line in tail_lines:
                        process_line(line)

                    f.seek(0, 2)
                    dbg("FOLLOWER", f"backfill done, tailing from offset={f.tell()}")

                    while True:
                        line = f.readline()
                        if not line:
                            time.sleep(LOG_POLL_INTERVAL)
                            if not os.path.exists(LOG_FILE):
                                dbg("FOLLOWER", "file disappeared, restarting")
                                break
                            try:
                                if os.stat(LOG_FILE).st_ino != inode:
                                    dbg("FOLLOWER", "inode changed (rotation), restarting")
                                    break
                            except OSError:
                                break
                            continue

                        process_line(line)

        except Exception as e:
            dbg("FOLLOWER", f"exception: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

# =========================
# API
# =========================

@app.route("/stats")
def stats():
    now = time.time()

    with state_lock:
        active_gpus = {
            gpu_id: gpu
            for gpu_id, gpu in state["gpu_hashrates"].items()
            if now - gpu["last_seen"] <= STALE_THRESHOLD_SEC
        }

        gpus_array = [
            {
                "id":        gpu_id,
                "gpu_name":  gpu["name"],
                "hashrate":  gpu["hashrate"],
                "last_seen": gpu["last_seen"]
            }
            for gpu_id, gpu in sorted(active_gpus.items())
        ]

        last_update    = state["last_update"]
        total_hashrate = sum(gpu["hashrate"] for gpu in active_gpus.values())

    stale = (
        last_update is None or
        now - last_update > STALE_THRESHOLD_SEC
    )

    return jsonify({
        "total_hashrate": total_hashrate,
        "gpu_count":      len(gpus_array),
        "gpus":           gpus_array,
        "last_update":    last_update,
        "stale":          stale
    })

# =========================
# Main
# =========================

if __name__ == "__main__":
    threading.Thread(target=follow_log, daemon=True).start()
    app.run(host="0.0.0.0", port=8080)