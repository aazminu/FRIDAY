"""The double-click launcher must never hang (RUN.bat 'jams', 2026-10-12)."""
import io
import sys
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import launcher_ui

class LauncherTests(unittest.TestCase):
    def test_bar_with_reporting_probe_finishes_instead_of_deadlocking(self):
        calls = []
        def probe():
            calls.append(1)
            return (.6, 'building context')
        out = io.StringIO()
        done = threading.Event()
        def run():
            with redirect_stdout(out):
                bar = launcher_ui.LoadingBar('Brain', 'test-brain', 5, probe=probe).start()
                time.sleep(.5)
                bar.finish('language online')
            done.set()
        original = launcher_ui.remember
        launcher_ui.remember = lambda *a: None  # don't write timing files from tests
        try:
            threading.Thread(target=run, daemon=True).start()
            self.assertTrue(done.wait(5), 'progress bar deadlocked')
        finally:
            launcher_ui.remember = original
        self.assertTrue(calls)
        self.assertIn('100%', out.getvalue())
        self.assertIn('building context', out.getvalue())

    def test_art_is_red_then_text_returns_to_light_green(self):
        out = io.StringIO()
        original = launcher_ui.enable_ansi
        try:
            launcher_ui.enable_ansi = lambda: True
            with redirect_stdout(out): launcher_ui.banner(delay=0)
            first = out.getvalue().splitlines()[0]
            self.assertTrue(first.startswith(launcher_ui.RED))
            self.assertTrue(first.endswith(launcher_ui.RESET + launcher_ui.GREEN))
            launcher_ui.enable_ansi = lambda: False  # redirected output: plain text only
            out = io.StringIO()
            with redirect_stdout(out): launcher_ui.banner(delay=0)
            self.assertNotIn('\x1b[', out.getvalue())
        finally:
            launcher_ui.enable_ansi = original

    def test_fresh_copy_starts_from_base_and_never_overwrites_memories(self):
        import tempfile
        import friday
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'release').mkdir()
            (root / 'release/friday-base.pth').write_bytes(b'base')
            original = friday.ROOT
            friday.ROOT = root
            try:
                with redirect_stdout(io.StringIO()):
                    friday.first_run_checkpoint()
                live = root / 'state/checkpoints/v14-language.pth'
                self.assertEqual(live.read_bytes(), b'base')
                live.write_bytes(b'my memories')
                with redirect_stdout(io.StringIO()):
                    friday.first_run_checkpoint()
                self.assertEqual(live.read_bytes(), b'my memories')
            finally:
                friday.ROOT = original

    def test_intro_art_is_the_friday_lettering_only(self):
        art = launcher_ui.ART.read_text(encoding='utf-8')
        self.assertIn('|_|    |_|  \\_\\_____||_____//_/    \\_\\  |_|', art)
        self.assertNotIn('^~~~^~~~^', art)  # the skull was removed at the user's request (2026-10-12)
        self.assertEqual(len(art.splitlines()), 6)

if __name__ == '__main__':
    unittest.main()
