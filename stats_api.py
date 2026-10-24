#!/usr/bin/env python3
"""Miner watcher.

Tails the miner's screen log, parses total and per-GPU hashrate and serves them
as JSON on GET / and GET /stats (standard library only, no pip). The last few
log lines are included too, so extra fields a miner prints (power, VRAM, ...)
can be checked by eye and new miners' parsers can be written from real output.

Only total_hashrate and gpus[].hashrate are part of the contract, because every
miner prints those. Each miner application (not each coin: the log format belongs to
the miner, and a coin can be mined with several) has a parser in PARSERS, chosen with
MINER, e.g. gigahash, srbminer or onezerominer; a
parser turns one clean log line into an event, a list of events, or None. Events:
("table", None) starts a new stats table (forgets the previous GPU rows),
("total", rate) and ("gpu", row). Hashrates must be converted to the coin's
baseline unit by the parser.

The watcher also restarts a hung miner: when no usable hashrate has been seen
for HANG_THRESHOLD_SEC it kills the miner process (matched by
MINER_PROCESS_PATTERN, a regex on the full command line) and relies on the
startup script's shell loop to start it again. After MAX_RESTARTS it gives up,
so a miner that cannot run does not burn the rental. The JSON reports
restart_count, last_restart_at and restart_gave_up.

Gigahash (Nock ZK) table the "gigahash" parser reads:

    | gigahash.cloud | NOCK ZK | Total 38.97 Mn/s | 0m | Accepted 0 | Stale 0 | Errors 0 |
    | GPU | Device         | Rate       | Util | Temp | ...
    | 0   | RTX 4070 SUPER | 38.97 Mn/s | 100% | 86 C | ...

Environment: MINER (required: a key of PARSERS), MINER_LOG (default /root/miner.log), WATCHER_PORT
(8080), STALE_THRESHOLD_SEC (per miner, see STALE_THRESHOLDS), LOG_TAIL_LINES (30), HANG_THRESHOLD_SEC (240,
0 turns restarting off), MAX_RESTARTS (5), MINER_PROCESS_PATTERN
(^\\./gigahash-zk, which skips the shell loop that holds the same name).
"""
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

LOG_FILE = os.environ.get("MINER_LOG", "/root/miner.log")
PORT = int(os.environ.get("WATCHER_PORT", "8080"))
LOG_TAIL_LINES = int(os.environ.get("LOG_TAIL_LINES", "30"))
MINER = os.environ.get("MINER", "")
# How long without a stats table before the numbers count as old. It must be well above the
# miner's print interval (SRBMiner prints every 30 s, so 30 would flap).
STALE_THRESHOLDS = {"gigahash": 30, "srbminer": 90, "onezerominer": 90}
STALE_THRESHOLD_SEC = float(os.environ.get("STALE_THRESHOLD_SEC", STALE_THRESHOLDS.get(MINER, 30)))
HANG_THRESHOLD_SEC = float(os.environ.get("HANG_THRESHOLD_SEC", "240"))
MAX_RESTARTS = int(os.environ.get("MAX_RESTARTS", "5"))
HANG_CHECK_INTERVAL = 5
MAX_TAIL_LINE_CHARS = 300
READ_BACK_LINES = 300
LOG_POLL_INTERVAL = 0.2

state_lock = threading.Lock()
state = {
    "total_hashrate": None,
    "gpus": {},          # gpu_id -> {id, gpu_name, hashrate}, replaced per table
    "last_update": None,
    "last_good": None,   # last time a total above zero was seen
    "started_at": None,  # when the watcher, or the last restart, began waiting for the miner
    "restart_count": 0,
    "last_restart_at": None,
    "restart_gave_up": False,
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


def parse_gigahash_line(line):
    """The gigahash.cloud miner (Nock ZK) prints a header with the total, then one row per GPU."""
    if not line.startswith("|"):
        return None
    header = HEADER_REGEX.match(line)
    if header:
        return [("table", None), ("total", float(header.group(1)))]
    gpu = parse_gpu_row(line)
    if gpu:
        return ("gpu", gpu)
    return None


# SRBMiner-MULTI prints, every 30 s, after a timestamp:
#  ID Name                         Hashrate    Power ...
# #0  RTX 4070 SUPER            141.16 TH/s   241.7W ...
#     Hashrate                  141.16 TH/s   241.7W ...        <- the total, after the rows
SRB_TIMESTAMP = re.compile(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\]\s*")
SRB_TABLE_HEADER = re.compile(r"^ID\s+Name\s+Hashrate\b")
SRB_GPU_ROW = re.compile(r"^#(\d+)\s+(.+?)\s+([\d.]+)\s*([kMGTP]?H/s)\b")
SRB_TOTAL = re.compile(r"^Hashrate\s+([\d.]+)\s*([kMGTP]?H/s)\b")
TERAHASH_PER_UNIT = {"H/s": 1e-12, "kH/s": 1e-9, "MH/s": 1e-6, "GH/s": 1e-3, "TH/s": 1.0, "PH/s": 1e3}


def to_terahash(value, unit):
    """Convert a rate to TH/s, the unit the Pearl baselines use; None for an unknown unit."""
    factor = TERAHASH_PER_UNIT.get(unit)
    return None if factor is None else float(value) * factor


def parse_srbminer_line(line):
    body = SRB_TIMESTAMP.sub("", line).strip()
    if SRB_TABLE_HEADER.match(body):
        return ("table", None)
    row = SRB_GPU_ROW.match(body)
    if row:
        rate = to_terahash(row.group(3), row.group(4))
        if rate is not None:
            return ("gpu", {"id": int(row.group(1)), "gpu_name": row.group(2), "hashrate": rate})
        return None
    total = SRB_TOTAL.match(body)
    if total:
        rate = to_terahash(total.group(1), total.group(2))
        if rate is not None:
            return ("total", rate)
    return None


# OneZeroMiner prints, every 30 s, after a timestamp:
#  #   Name         Hashrate      Shares   Eff           Temp ...
#  0   4070 SUPER   145.96 TH/s   0/0/0    629.50 GH/W   87°C ...
#      Total        145.96 TH/s   0/0/0    629.50 GH/W          <- the total, after the rows
OZ_TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s+")
OZ_TABLE_HEADER = re.compile(r"^#\s+Name\s+Hashrate\b")
OZ_GPU_ROW = re.compile(r"^(\d+)\s+(.+?)\s{2,}([\d.]+)\s*([kMGTP]?H/s)\b")
OZ_TOTAL = re.compile(r"^Total\s+([\d.]+)\s*([kMGTP]?H/s)\b")


def parse_onezerominer_line(line):
    body = OZ_TIMESTAMP.sub("", line).strip()
    if OZ_TABLE_HEADER.match(body):
        return ("table", None)
    row = OZ_GPU_ROW.match(body)
    if row:
        rate = to_terahash(row.group(3), row.group(4))
        if rate is not None:
            return ("gpu", {"id": int(row.group(1)), "gpu_name": row.group(2), "hashrate": rate})
        return None
    total = OZ_TOTAL.match(body)
    if total:
        rate = to_terahash(total.group(1), total.group(2))
        if rate is not None:
            return ("total", rate)
    return None


PARSERS = {"gigahash": parse_gigahash_line, "srbminer": parse_srbminer_line,
           "onezerominer": parse_onezerominer_line}
# Regexes on the full command line of each miner. Anchored so the shell loop that launches the
# miner, whose command line merely contains the name, is not matched.
PROCESS_PATTERNS = {"gigahash": r"^\./gigahash-zk", "srbminer": r"^\./SRBMiner-MULTI",
                    "onezerominer": r"^\./onezerominer"}
MINER_PROCESS_PATTERN = os.environ.get("MINER_PROCESS_PATTERN", PROCESS_PATTERNS.get(MINER, ""))


def process_line(line, now=None, parse=None):
    now = now if now is not None else time.time()
    parse = parse or PARSERS[MINER]
    clean = ANSI_ESCAPE.sub("", line).replace("\r", "").strip()
    if not clean:
        return

    with state_lock:
        log_tail.append(clean[:MAX_TAIL_LINE_CHARS])

    events = parse(clean)
    if events is None:
        return
    if isinstance(events, tuple):
        events = [events]
    for kind, value in events:
        if kind == "table":
            _pending_gpus.clear()
        elif kind == "total":
            with state_lock:
                state["total_hashrate"] = value
                state["last_update"] = now
                if value > 0:
                    state["last_good"] = now
        else:
            _pending_gpus[value["id"]] = value
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


def kill_miner():
    """Kill the miner process; the startup script's shell loop starts it again."""
    if not MINER_PROCESS_PATTERN:
        return False  # an empty pattern would match every process
    # SIGKILL, not the default SIGTERM: a miner stuck in the driver or stopped does not act on SIGTERM.
    result = subprocess.run(["pkill", "-KILL", "-f", MINER_PROCESS_PATTERN], capture_output=True)
    return result.returncode == 0


def check_hang(now=None, kill=kill_miner):
    """Restart the miner when it has shown no usable hashrate for HANG_THRESHOLD_SEC.

    The clock starts when the watcher starts or at the last restart, so a miner that
    is still starting up gets the whole threshold. Returns True when it killed the miner.
    """
    if HANG_THRESHOLD_SEC <= 0:
        return False
    now = now if now is not None else time.time()
    with state_lock:
        if state["started_at"] is None:
            state["started_at"] = now
        if state["restart_gave_up"]:
            return False
        since = max(state["started_at"], state["last_good"] or 0)
        if now - since <= HANG_THRESHOLD_SEC:
            return False
        if state["restart_count"] >= MAX_RESTARTS:
            state["restart_gave_up"] = True
            dbg("HANG", f"no hashrate for {now - since:.0f}s and {MAX_RESTARTS} restarts used, giving up")
            return False
        state["restart_count"] += 1
        state["last_restart_at"] = now
        state["started_at"] = now
        state["total_hashrate"] = None
        state["gpus"] = {}
        count = state["restart_count"]
    dbg("HANG", f"no hashrate for {now - since:.0f}s, killing the miner (restart {count} of {MAX_RESTARTS})")
    if not kill():
        dbg("HANG", "no miner process matched " + repr(MINER_PROCESS_PATTERN))
    return True


def watch_for_hang():
    while True:
        time.sleep(HANG_CHECK_INTERVAL)
        try:
            check_hang()
        except Exception as e:  # keep the watcher alive whatever happens
            dbg("HANG", f"exception: {e!r}")


def snapshot(tail=LOG_TAIL_LINES, now=None):
    now = now if now is not None else time.time()
    with state_lock:
        total = state["total_hashrate"]
        last_update = state["last_update"]
        gpus = [dict(g) for _, g in sorted(state["gpus"].items())]
        lines = list(log_tail)[-tail:] if tail > 0 else []
        restart_count = state["restart_count"]
        last_restart_at = state["last_restart_at"]
        gave_up = state["restart_gave_up"]
    stale = last_update is None or now - last_update > STALE_THRESHOLD_SEC
    result = {
        "total_hashrate": total,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "last_update": last_update,
        "age_sec": None if last_update is None else round(now - last_update, 1),
        "stale": stale,
        "restart_count": restart_count,
        "last_restart_at": last_restart_at,
        "restart_gave_up": gave_up,
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
    if MINER not in PARSERS:
        raise SystemExit(f"MINER must be one of {sorted(PARSERS)}, got {MINER!r}")
    threading.Thread(target=follow_log, daemon=True).start()
    threading.Thread(target=watch_for_hang, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
