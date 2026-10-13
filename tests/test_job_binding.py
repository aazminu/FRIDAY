"""Closing the Friday window must not leave the language server running (orphaned 4 GB server, 2026-10-12)."""
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

PARENT = r'''
import subprocess, sys, time
sys.path.insert(0, {src!r})
from cpu_runtime import bind_child_to_this_process
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
job = bind_child_to_this_process(child.pid)
print(child.pid, bool(job), flush=True)
time.sleep(60)
'''

@unittest.skipUnless(os.name == 'nt', 'Windows job objects')
class JobBindingTests(unittest.TestCase):
    def test_child_dies_when_parent_is_killed(self):
        src = str(Path(__file__).resolve().parents[1] / 'src')
        parent = subprocess.Popen([sys.executable, '-c', PARENT.format(src=src)], stdout=subprocess.PIPE, text=True)
        pid, bound = parent.stdout.readline().split()
        self.assertEqual(bound, 'True')
        parent.kill()  # like closing the window: no atexit, no cleanup code
        parent.wait(10)
        deadline = time.time() + 10
        alive = True
        while time.time() < deadline:
            alive = str(pid) in subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], capture_output=True, text=True).stdout
            if not alive:
                break
            time.sleep(.3)
        self.assertFalse(alive, 'child outlived its parent')

class RuntimeSelectionTests(unittest.TestCase):
    def test_native_arch_is_real_hardware_not_the_python_build(self):
        # An x64 Python emulated on an ARM64 PC must still pick the native ARM64 llama-server (2026-10-12).
        import platform
        import cpu_runtime
        arch = cpu_runtime.native_arch()
        self.assertIn(arch, ('arm64', 'x64'))
        self.assertTrue(str(cpu_runtime.DEFAULT_RUNTIME).endswith(f'llama-{arch}' + os.sep + 'llama-server.exe'))
        if os.environ.get('PROCESSOR_IDENTIFIER', '').upper().startswith('ARM'):
            self.assertEqual(arch, 'arm64', f'python reports {platform.machine()}')

if __name__ == '__main__':
    unittest.main()
