"""Real SODA lifecycle checks, over an optionally remote TLS WebSocket."""
import asyncio
import json
import os
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import websockets

BASE=os.environ.get("STT_TEST_BASE","http://127.0.0.1:18765").rstrip("/")
URL=BASE.replace("https:","wss:").replace("http:","ws:")+"/v1/stt?lang=vi-VN"
WAV=Path(os.environ.get("STT_TEST_WAV","test-vi.wav"))


def request(path,body=None,origin=None):
    req=urllib.request.Request(BASE+path,data=None if body is None else json.dumps(body).encode(),
        headers={"Origin":origin or BASE,"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=10) as response:
            return response.status,json.load(response)
    except urllib.error.HTTPError as error:
        return error.code,json.loads(error.read())


async def idle():
    for _ in range(100):
        status=await asyncio.to_thread(request,"/health")
        if status[1]["active_sessions"]==0:return
        await asyncio.sleep(.1)
    raise AssertionError(("Session not cleaned up",status))


async def connect():
    ws=await websockets.connect(URL,origin=BASE,open_timeout=15)
    ready=json.loads(await asyncio.wait_for(ws.recv(),20))
    assert ready["type"]=="ready",ready
    return ws


async def normal(pcm):
    ws=await connect()
    final=[];partial=[];closed=False
    async def receive():
        nonlocal closed
        async for raw in ws:
            event=json.loads(raw)
            assert event["type"]!="error",event
            if event["type"]=="partial":partial.append(event["text"])
            if event["type"]=="final":final.append(event["text"])
            if event["type"]=="closed":closed=True;return
    read=asyncio.create_task(receive())
    for pos in range(0,len(pcm),3200):
        await ws.send(pcm[pos:pos+3200]);await asyncio.sleep(.1)
    await ws.send("stop")
    await asyncio.wait_for(read,15)
    await ws.close()
    assert final and partial and closed,(final,len(partial),closed)
    await idle()
    return " ".join(final)


async def main():
    await idle()
    with wave.open(str(WAV),"rb") as wav:
        assert (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())==(16000,1,2)
        pcm=wav.readframes(wav.getnframes())
    assert (await asyncio.to_thread(request,"/v1/dataset/ingest",[]))[0]==422
    assert (await asyncio.to_thread(request,"/v1/dataset/ingest",{},"https://invalid.example"))[0]==403
    try:
        async with websockets.connect(URL,origin="https://invalid.example"):
            raise AssertionError("Invalid origin accepted")
    except websockets.exceptions.InvalidStatus as error:
        assert error.response.status_code==403
    print("ORIGIN_AND_SCHEMA_GUARDS pass",flush=True)
    transcript=await normal(pcm)
    print("REAL_AUDIO",json.dumps(transcript,ensure_ascii=False),flush=True)
    for i in range(6):
        ws=await connect();await ws.send(pcm[:3200]);await ws.close();await idle()
    print("ABRUPT_DISCONNECTS 6 cleaned",flush=True)
    ws=await connect();await ws.send(b"x")
    events=[]
    async for raw in ws:
        event=json.loads(raw);events.append(event)
        if event["type"]=="closed":break
    await ws.close();await idle()
    assert any(x.get("code")=="invalid_pcm_frame" for x in events),events
    print("INVALID_FRAME cleaned",flush=True)
    one=await connect();two=await connect()
    async with websockets.connect(URL,origin=BASE) as third:
        value=json.loads(await third.recv());assert value["code"]=="capacity_exceeded",value
    await one.close();await two.close();await idle()
    print("CAPACITY_AND_RECOVERY pass",flush=True)
    ws=await connect();events=[]
    async def flood_reader():
        async for raw in ws:
            event=json.loads(raw);events.append(event)
            if event["type"]=="closed":return
    reader=asyncio.create_task(flood_reader())
    try:
        for _ in range(400):await ws.send(b"\0\0"*1600)
    except websockets.exceptions.ConnectionClosed:
        pass
    await asyncio.wait_for(reader,15)
    await ws.close();await idle()
    assert any(x.get("code")=="audio_backpressure" for x in events),events
    print("FULL_QUEUE_ABORT cleaned",flush=True)
    final=await normal(pcm)
    print("RECOVERY_REAL_AUDIO",json.dumps(final,ensure_ascii=False),flush=True)
    print("PASS all sessions returned to zero; TLS verified by default",flush=True)

asyncio.run(main())
