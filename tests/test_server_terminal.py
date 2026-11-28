"""Pure terminal-contract tests for the WebSocket worker state machine."""
import unittest

from server import worker_terminal_error


class WorkerTerminalContractTests(unittest.TestCase):
    def test_normal_completion_requires_success_status_and_closed(self):
        self.assertIsNone(worker_terminal_error(
            worker_status_seen=True, worker_status_ok=True,
            closed_seen=True, exitcode=0))

    def test_closed_without_success_status_is_protocol_failure(self):
        self.assertEqual(worker_terminal_error(
            worker_status_seen=False, worker_status_ok=False,
            closed_seen=True, exitcode=0), "native_worker_protocol")
        self.assertEqual(worker_terminal_error(
            worker_status_seen=True, worker_status_ok=False,
            closed_seen=True, exitcode=0), "recognition_failed")

    def test_nonzero_exit_preserves_typed_native_error(self):
        self.assertEqual(worker_terminal_error(
            worker_status_seen=False, worker_status_ok=False,
            closed_seen=False, exitcode=17), "native_worker_exited")

    def test_success_status_without_closed_is_not_success(self):
        self.assertEqual(worker_terminal_error(
            worker_status_seen=True, worker_status_ok=True,
            closed_seen=False, exitcode=0), "native_worker_protocol")


if __name__ == "__main__":
    unittest.main()
