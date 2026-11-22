"""Small cross-process lock and atomic UTF-8 writes for local dataset state."""
from __future__ import annotations
import contextlib,os,pathlib,tempfile,time

@contextlib.contextmanager
def exclusive_lock(directory:pathlib.Path,timeout:float=10):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/".dataset.lock").open("a+b") as handle:
        handle.seek(0,2)
        if handle.tell()==0:handle.write(b"0");handle.flush()
        deadline=time.monotonic()+timeout
        while True:
            try:
                handle.seek(0)
                if os.name=="nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except (OSError,BlockingIOError):
                if time.monotonic()>=deadline:raise TimeoutError("Dataset is busy; retry later")
                time.sleep(0.05)
        try:yield
        finally:
            handle.seek(0)
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)

def atomic_bytes(path:pathlib.Path,content:bytes):
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temp=tempfile.mkstemp(prefix=path.name+".",suffix=".pending",dir=path.parent)
    try:
        with os.fdopen(descriptor,"wb") as stream:
            stream.write(content);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)

def stable_split(pcm_sha256:str)->str:
    if len(pcm_sha256)!=64 or any(c not in "0123456789abcdef" for c in pcm_sha256):
        raise ValueError("Invalid PCM identity")
    return "holdout" if int(pcm_sha256[:8],16)%5==0 else "train"
