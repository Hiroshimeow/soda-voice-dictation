"""Transcribe local audio files through the same local SODA backend."""
from __future__ import annotations
import asyncio,json,shutil,subprocess
from pathlib import Path
from typing import Callable
import websockets

SUPPORTED_EXTENSIONS={".wav",".mp3",".m4a",".aac",".flac",".ogg"}
PCM_CHUNK=3200
SEND_DELAY=.05
MAX_FILE_DURATION_SECONDS=4*60*60

def validate_audio_file(path:str|Path)->Path:
    p=Path(path).expanduser().resolve()
    if not p.is_file(): raise ValueError("Audio file does not exist")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS: raise ValueError("Unsupported audio format")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe is required")
    return p

def probe_duration(path:str|Path)->float:
    p=validate_audio_file(path)
    r=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(p)],capture_output=True,text=True,timeout=20)
    if r.returncode!=0: raise RuntimeError("ffprobe could not read the audio file")
    try:d=float(r.stdout.strip())
    except ValueError as exc: raise RuntimeError("Audio duration is unavailable") from exc
    if not 0<d<=MAX_FILE_DURATION_SECONDS: raise ValueError("Audio duration must be between 0 and 4 hours")
    return d

async def transcribe_file(path:str|Path,ws_url:str,on_event:Callable[[dict],None])->dict:
    p=validate_audio_file(path);duration=probe_duration(p)
    proc=await asyncio.create_subprocess_exec("ffmpeg","-hide_banner","-loglevel","error","-i",str(p),"-vn","-ac","1","-ar","16000","-f","s16le","-",stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    finals=[];partial=""
    try:
        async with websockets.connect(ws_url,open_timeout=30,close_timeout=10,max_size=1048576) as ws:
            first=json.loads(await asyncio.wait_for(ws.recv(),timeout=35))
            if first.get("type")!="ready": raise RuntimeError(first.get("detail") or first.get("code") or "SODA is not ready")
            on_event({"type":"file_ready","duration":duration})
            async def rx():
                nonlocal partial
                async for raw in ws:
                    event=json.loads(raw);kind=event.get("type")
                    if kind=="partial":
                        partial=event.get("text") or ""
                        on_event({"type":"file_transcript","text":" ".join(finals+([partial] if partial else [])),"partial":True})
                    elif kind=="final":
                        text=(event.get("text") or "").strip()
                        if text:finals.append(text)
                        partial=""
                        on_event({"type":"file_transcript","text":" ".join(finals),"partial":False})
                    elif kind=="error":
                        raise RuntimeError(event.get("detail") or event.get("code") or "SODA error")
                    elif kind=="closed": return
            task=asyncio.create_task(rx());sent=0
            assert proc.stdout is not None
            while True:
                chunk=await proc.stdout.read(PCM_CHUNK)
                if not chunk: break
                await ws.send(chunk);sent+=len(chunk)
                on_event({"type":"file_progress","seconds":sent/32000.0,"duration":duration})
                await asyncio.sleep(SEND_DELAY)
            await ws.send("stop")
            await asyncio.wait_for(task,timeout=45)
        code=await asyncio.wait_for(proc.wait(),timeout=10)
        if code!=0:
            err=(await proc.stderr.read()).decode("utf-8","replace") if proc.stderr else ""
            raise RuntimeError((err or "ffmpeg failed")[:300])
        text=" ".join(finals).strip()
        if not text: raise RuntimeError("No speech was recognized in the file")
        return {"text":text,"duration":duration}
    finally:
        if proc.returncode is None:
            proc.kill()
            try: await proc.wait()
            except Exception: pass
