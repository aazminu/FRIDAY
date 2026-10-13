"""Console start-up screen for RUN.bat: banner and progress bars for the soul and the brain.

Deliberately imports nothing heavy, so the window shows something the moment it opens.
"""
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMES = ROOT / 'state/logs/startup-times.json'

ART = Path(__file__).with_name('friday_art.txt')  # the RUN.bat intro art, unescaped

def _supports_blocks():
    try:
        '█░'.encode(sys.stdout.encoding or 'utf-8')
        return True
    except (UnicodeEncodeError, LookupError):
        return False

FULL, EMPTY = ('█', '░') if _supports_blocks() else ('#', '-')

# The window stays black with light green text (RUN.bat sets `color 0A`). The art alone is red.
# cmd's `color` recolors the whole window, so per-line ANSI colors are used for the art instead.
RED, RESET = '\x1b[31m', '\x1b[0m'
GREEN = '\x1b[92m'  # light green, same as `color 0A`

def enable_ansi():
    """Turn on ANSI color handling in the Windows console; returns False if unavailable."""
    if sys.platform != 'win32':
        return sys.stdout.isatty()
    try:
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            return False  # output is redirected (tests, logs): no color codes
        return bool(kernel.SetConsoleMode(handle, mode.value | 0x0004))  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except (AttributeError, OSError):
        return False

def banner(delay=.04):
    """Reveal the intro art line by line in red, then continue in light green on black."""
    color = enable_ansi()
    try:
        lines = ART.read_text(encoding='utf-8').splitlines()
    except OSError:
        lines = ['  F R I D A Y']
    for line in lines:
        print((RED + line + RESET + GREEN) if color else line, flush=True)
        time.sleep(delay)

def remembered(name, default):
    try:
        return float(json.loads(TIMES.read_text(encoding='utf-8')).get(name, default))
    except (OSError, ValueError, TypeError):
        return default

def remember(name, seconds):
    try:
        data = json.loads(TIMES.read_text(encoding='utf-8')) if TIMES.exists() else {}
    except (OSError, ValueError):
        data = {}
    previous = data.get(name)
    data[name] = round(seconds if previous is None else .5 * previous + .5 * seconds, 1)
    try:
        TIMES.parent.mkdir(parents=True, exist_ok=True)
        TIMES.write_text(json.dumps(data, indent=2), encoding='utf-8')
    except OSError:
        pass

class LoadingBar:
    """Animated bar for a step whose length is only estimable.

    Real milestones (reach) set a floor; between them the bar creeps toward the next
    milestone using the remembered duration, and never shows 100% until finish().
    An optional probe() returns (fraction, note) from real evidence such as a log file.
    """
    def __init__(self, label, name, expected_seconds, width=30, probe=None):
        self.label, self.name, self.width, self.probe = label, name, width, probe
        self.expected = max(3., remembered(name, expected_seconds))
        self.floor, self.note, self.started = 0., 'starting', time.perf_counter()
        # Re-entrant: reach() may be called from inside a probe while drawing.
        self._stop = threading.Event(); self._lock = threading.RLock(); self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True); self._thread.start(); return self

    def reach(self, fraction, note):
        with self._lock:
            self.floor = max(self.floor, min(.97, fraction)); self.note = note

    def _fraction(self):
        elapsed = time.perf_counter() - self.started
        estimate = .95 * (1 - pow(2.718, -2.2 * elapsed / self.expected))
        return max(self.floor, min(.95, estimate)), elapsed

    def _probe(self):
        if not self.probe: return
        try:
            found = self.probe()
        except Exception:
            found = None
        if found: self.reach(*found)

    def _draw(self, fraction, elapsed, final=False):
        filled = int(round(fraction * self.width))
        bar = FULL * filled + EMPTY * (self.width - filled)
        note = (self.note if not final else self.note)[:34]
        sys.stdout.write(f'\r  {self.label:<7} [{bar}] {int(fraction * 100):3d}%  {elapsed:5.1f}s  {note:<34}')
        sys.stdout.flush()

    def _run(self):
        while not self._stop.wait(.12):
            self._probe()  # outside the lock: reach() takes it
            with self._lock:
                fraction, elapsed = self._fraction()
                self._draw(fraction, elapsed)

    def finish(self, note='ready'):
        self._stop.set()
        if self._thread: self._thread.join()
        elapsed = time.perf_counter() - self.started
        self.note = note
        self._draw(1., elapsed, final=True)
        sys.stdout.write('\n'); sys.stdout.flush()
        remember(self.name, elapsed)

    def fail(self, note):
        self._stop.set()
        if self._thread: self._thread.join()
        self.note = note
        self._draw(self.floor, time.perf_counter() - self.started)
        sys.stdout.write('\n'); sys.stdout.flush()

def server_log_probe(log_path):
    """Map real llama-server log milestones to progress."""
    marks = (('load_tensors', .25, 'reading model weights'), ('llama_context', .6, 'building context'),
             ('repack', .7, 'optimizing for this CPU'), ('warming up', .8, 'warming up'),
             ('model loaded', .92, 'model loaded'), ('listening on', .95, 'almost ready'))
    created = time.time() - 1
    def probe():
        try:
            path = Path(log_path)
            if path.stat().st_mtime < created:
                return None  # still the previous run's log
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            return None
        best = None
        for key, fraction, note in marks:
            if key in text: best = (fraction, note)
        return best
    return probe
