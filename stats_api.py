#!/usr/bin/env python3
"""Miner watcher for the Nock ZK miner (gigahash.cloud).

Tails the miner's screen log, parses total and per-GPU hashrate and serves them
as JSON on GET / and GET /stats (standard library only, no pip). The last few
log lines are included too, so extra fields a miner prints (power, VRAM, ...)
can be checked by eye and new miners' parsers can be written from real output.

Only total_hashrate and gpus[].hashrate are part of the contract, because every
miner prints those. To support another miner, change HEADER_REGEX and
parse_gpu_row.

Nock table the parser reads:

    | gigahash.cloud | NOCK ZK | Total 38.97 Mn/s | 0m | Accepted 0 | Stale 0 | Errors 0 |
    | GPU | Device         | Rate       | Util | Temp | ...
    | 0   | RTX 4070 SUPER | 38.97 Mn/s | 100% | 86 C | ...

Environment: MINER_LOG (default /root/miner.log), WATCHER_PORT (8080),
STALE_THRESHOLD_SEC (30), LOG_TAIL_LINES (30).
"""
import json
import os
import re
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

LOG_FILE = os.environ.get("MINER_LOG", "/root/miner.log")
PORT = int(os.environ.get("WATCHER_PORT", "8080"))
STALE_THRESHOLD_SEC = float(os.environ.get("STALE_THRESHOLD_SEC", "30"))
LOG_TAIL_LINES = int(os.environ.get("LOG_TAIL_LINES", "30"))
MAX_TAIL_LINE_CHARS = 300
READ_BACK_LINES = 300
LOG_POLL_INTERVAL = 0.2

state_lock = threading.Lock()
state = {
    "total_hashrate": None,
    "gpus": {},          # gpu_id -> {id, gpu_name, hashrate}, replaced per table
    "last_update": None,
}
log_tail = deque(maxlen=LOG_TAIL_LINES)

# screen -L logs raw terminal output, so strip colour and cursor escapes.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# | gigahash.cloud | NOCK ZK | Total 38.97 Mn/s | 0m | ...
HEADER_REGEX = re.compile(r"^\|.*?\|\s*Total\s+([\d.]+)\s*\S+", re.IGNORECASE)
RATE_REGEX = re.compile(r"^([\d.]+)\s*\S+$")

_pending_gpus = {}


def dbg(tag, msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{ts}][{tag}] {msg}", flush=True)


def parse_gpu_row(line):
    """Parse '| 0 | RTX 4070 SUPER | 38.97 Mn/s | ...' into id, name and hashrate."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) < 3 or not cells[0].isdigit():
        return None
    rate = RATE_REGEX.match(cells[2])
    if not rate:
        return None
    return {"id": int(cells[0]), "gpu_name": cells[1], "hashrate": float(rate.group(1))}


def process_line(line, now=None):
    now = now if now is not None else time.time()
    clean = ANSI_ESCAPE.sub("", line).replace("\r", "").strip()
    if not clean:
        return

    with state_lock:
        log_tail.append(clean[:MAX_TAIL_LINE_CHARS])

    if not clean.startswith("|"):
        return

    header = HEADER_REGEX.match(clean)
    if header:
        _pending_gpus.clear()
        with state_lock:
            state["total_hashrate"] = float(header.group(1))
            state["last_update"] = now
        return

    gpu = parse_gpu_row(clean)
    if gpu:
        _pending_gpus[gpu["id"]] = gpu
        with state_lock:
            # Swap in the whole table so a GPU that drops out disappears.
            state["gpus"] = dict(_pending_gpus)
            state["last_update"] = now


def tail_lines(path, count):
    with open(path, "r", errors="replace") as f:
        return list(deque(f, maxlen=count))


def follow_log():
    last_inode = None
    dbg("FOLLOWER", f"watching {LOG_FILE!r}")
    while True:
        if not os.path.exists(LOG_FILE):
            time.sleep(1)
            continue
        try:
            inode = os.stat(LOG_FILE).st_ino
            if inode != last_inode:
                last_inode = inode
                dbg("FOLLOWER", f"new file (inode={inode}), backfilling {READ_BACK_LINES} lines")
                for line in tail_lines(LOG_FILE, READ_BACK_LINES):
                    process_line(line)
            with open(LOG_FILE, "r", errors="replace") as f:
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if line:
                        process_line(line)
                        continue
                    time.sleep(LOG_POLL_INTERVAL)
                    try:
                        st = os.stat(LOG_FILE)
                    except OSError:
                        break
                    if st.st_ino != inode or st.st_size < f.tell():
                        dbg("FOLLOWER", "rotated or truncated, restarting")
                        last_inode = None
                        break
        except Exception as e:  # keep the watcher alive whatever happens
            dbg("FOLLOWER", f"exception: {e!r}")
            time.sleep(1)


def snapshot(tail=LOG_TAIL_LINES, now=None):
    now = now if now is not None else time.time()
    with state_lock:
        total = state["total_hashrate"]
        last_update = state["last_update"]
        gpus = [dict(g) for _, g in sorted(state["gpus"].items())]
        lines = list(log_tail)[-tail:] if tail > 0 else []
    stale = last_update is None or now - last_update > STALE_THRESHOLD_SEC
    result = {
        "total_hashrate": total,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "last_update": last_update,
        "age_sec": None if last_update is None else round(now - last_update, 1),
        "stale": stale,
    }
    if tail > 0:
        result["log_tail"] = lines
    return result


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path not in ("/", "/stats"):
            self.send_error(404)
            return
        try:
            tail = int(parse_qs(url.query).get("tail", [LOG_TAIL_LINES])[0])
        except ValueError:
            tail = LOG_TAIL_LINES
        body = json.dumps(snapshot(tail)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    threading.Thread(target=follow_log, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
