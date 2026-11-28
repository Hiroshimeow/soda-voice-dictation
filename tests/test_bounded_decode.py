"""Regression for observed native-helper hangs during file evaluation."""
import pathlib,tempfile,time,unittest
from unittest.mock import patch
from dataclasses import dataclass
from training.bounded_decode import recognize_pcm
@dataclass
class Event:
    text:str
    is_final:bool
class FiniteRecognizer:
    def stream_pcm(self,chunks,*,realtime_factor):
        size=sum(len(c) for c in chunks)
        if size!=16000 or realtime_factor!=2:raise ValueError("audio contract changed")
        yield Event("xin",False)
        yield Event("xin chào",True)
class HangingRecognizer:
    def stream_pcm(self,chunks,*,realtime_factor):
        time.sleep(10)
        yield Event("too late",True)


class FailedRecognizer:
    def stream_pcm(self, chunks, *, realtime_factor):
        raise RuntimeError("native helper emitted a truncated frame")


class FiniteEmptyRecognizer:
    def stream_pcm(self, chunks, *, realtime_factor):
        if sum(len(c) for c in chunks) != 16000 or realtime_factor != 2:
            raise ValueError("audio contract changed")
        if False:
            yield None


class StatusRecognizer:
    def __init__(self, status_position, status):
        self.status_position = status_position
        self.status = status

    def stream_pcm_status(self, chunks, *, realtime_factor):
        if self.status_position == "before":
            yield self.status
        yield Event("before", True)
        if self.status_position == "after":
            yield self.status


@dataclass
class ConfigHangingRecognizer:
    marker:str
    def stream_pcm(self,chunks,*,realtime_factor):
        with tempfile.NamedTemporaryFile(suffix=".cfg",delete=False) as config:
            config.write(b"private context test fixture")
            path=config.name
        pathlib.Path(self.marker).write_text(path,encoding="utf-8")
        time.sleep(10)
        yield Event("too late",True)
class Tests(unittest.TestCase):
    def test_clean_empty_decode_is_a_successful_no_final(self):
        result = recognize_pcm(FiniteEmptyRecognizer(), bytes(16000), pace=2, timeout=5)
        self.assertEqual(result["text"], "")
        self.assertEqual(result["finals"], [])
        self.assertEqual(result["final_segments"], 0)

    def test_truncated_status_before_final_is_not_empty_success(self):
        recognizer = StatusRecognizer("before", {"type": "adapter_status", "truncated": True})
        with self.assertRaisesRegex(RuntimeError, "truncated frame"):
            recognize_pcm(recognizer, bytes(16000), pace=2, timeout=5)

    def test_nonzero_status_before_final_is_not_success(self):
        recognizer = StatusRecognizer("before", {"type": "adapter_status", "returncode": 17})
        with self.assertRaisesRegex(RuntimeError, "status 17"):
            recognize_pcm(recognizer, bytes(16000), pace=2, timeout=5)

    def test_truncated_status_after_final_is_not_success(self):
        recognizer = StatusRecognizer("after", {"type": "adapter_status", "truncated": True})
        with self.assertRaisesRegex(RuntimeError, "truncated frame"):
            recognize_pcm(recognizer, bytes(16000), pace=2, timeout=5)

    def test_nonzero_status_after_final_is_not_success(self):
        recognizer = StatusRecognizer("after", {"type": "adapter_status", "returncode": 17})
        with self.assertRaisesRegex(RuntimeError, "status 17"):
            recognize_pcm(recognizer, bytes(16000), pace=2, timeout=5)

    def test_final_and_partial_contract(self):
        r=recognize_pcm(FiniteRecognizer(),bytes(16000),pace=2,timeout=5)
        self.assertEqual(r["text"],"xin chào")
        self.assertEqual(r["finals"],["xin chào"])
        self.assertEqual(r["partial_events"],1)
        self.assertEqual(r["final_segments"],1)
    def test_failed_decode_is_not_an_empty_success(self):
        with self.assertRaisesRegex(RuntimeError, "native helper emitted a truncated frame"):
            recognize_pcm(FailedRecognizer(), bytes(16000), pace=2, timeout=5)

    def test_hung_worker_is_bounded(self):
        start=time.monotonic()
        with self.assertRaises(TimeoutError):
            recognize_pcm(HangingRecognizer(),bytes(16000),pace=2,timeout=0.3)
        self.assertLess(time.monotonic()-start,3)
    def test_timeout_removes_owned_temp_configuration(self):
        with tempfile.TemporaryDirectory() as td:
            marker=pathlib.Path(td)/"marker.txt"
            with patch("training.bounded_decode.tempfile.tempdir",td):
                with self.assertRaises(TimeoutError):
                    recognize_pcm(ConfigHangingRecognizer(str(marker)),bytes(16000),timeout=1.5)
            self.assertTrue(marker.exists(),"Child must have created the temporary config before timeout")
            config=pathlib.Path(marker.read_text(encoding="utf-8"))
            self.assertTrue(config.is_relative_to(pathlib.Path(td)))
            self.assertFalse(config.exists())
            self.assertFalse(config.parent.exists())
if __name__=="__main__":unittest.main()
