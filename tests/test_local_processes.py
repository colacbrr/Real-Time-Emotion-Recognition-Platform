import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stop_local_app import belongs_to_project, stop


class LocalProcessTests(unittest.TestCase):
    def test_other_project_server_is_not_selected(self):
        self.assertFalse(belongs_to_project('/work/emotion', '/work/rag/backend', ['uvicorn', 'server:app']))
        self.assertFalse(belongs_to_project('/work/emotion', '/work/rag/frontend', ['node', '/work/rag/node_modules/.bin/vite']))

    def test_similarly_named_commands_and_nested_folders_are_not_selected(self):
        self.assertFalse(belongs_to_project('/work/emotion', '/work/emotion/frontend', ['node', 'vitest']))
        self.assertFalse(belongs_to_project('/work/emotion', '/work/emotion/other/frontend', ['vite']))
        self.assertFalse(belongs_to_project('/work/emotion', '/work/emotion/backend', ['uvicorn', 'different:app']))

    def test_only_current_project_pids_receive_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            root, other, proc = folder / 'emotion', folder / 'other', folder / 'proc'
            for project in [root, other]:
                for component in ['backend', 'frontend']:
                    (project / component).mkdir(parents=True)
            proc.mkdir()
            for pid, cwd, args in [(100001, root / 'backend', ['uvicorn', 'server:app']),
                                   (100002, root / 'frontend', ['node', '/some/path/vite']),
                                   (100003, other / 'frontend', ['node', '/some/path/vite'])]:
                entry = proc / str(pid)
                entry.mkdir()
                (entry / 'cwd').symlink_to(cwd, target_is_directory=True)
                (entry / 'cmdline').write_bytes(('\0'.join(args) + '\0').encode())
            killed = Mock()
            self.assertEqual(set(stop(root, proc, killed)), {100001, 100002})
            self.assertEqual({tuple(call.args) for call in killed.call_args_list},
                             {(100001, signal.SIGTERM), (100002, signal.SIGTERM)})
            killed.reset_mock()
            self.assertEqual(set(stop(root, proc, killed, dry_run=True)), {100001, 100002})
            killed.assert_not_called()


if __name__ == '__main__':
    unittest.main()
