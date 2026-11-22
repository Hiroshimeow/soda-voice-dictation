"""Isolated Chrome + synthetic microphone test; no saved personal recordings."""
import json
import os
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

URL=os.environ.get("STT_BROWSER_URL","http://127.0.0.1:18765/")
options=Options()
options.add_argument("--headless=new")
options.add_argument("--window-size=600,1000")
options.add_experimental_option("mobileEmulation", {"deviceMetrics":{"width":412,"height":915,"pixelRatio":1}})
options.add_argument("--use-fake-ui-for-media-stream")
options.add_argument("--use-fake-device-for-media-stream")
options.add_argument("--autoplay-policy=no-user-gesture-required")
options.set_capability("goog:loggingPrefs",{"browser":"ALL"})
driver=webdriver.Chrome(options=options)
try:
    driver.get(URL)
    assert driver.execute_script("return window.isSecureContext && !!navigator.mediaDevices")
    assert driver.execute_script("return window.innerWidth")==412
    assert driver.execute_script("return [...document.querySelectorAll('#lang option')].map(x=>x.value)") == ["vi-VN"]
    result=driver.execute_script("""
        const outcomes=[];
        for(const rate of [44100,48000,16000]){
            const frames=[];let Processor;
            class Base{constructor(){this.port={postMessage:m=>{if(m.type==='pcm')frames.push(new Int16Array(m.buffer));}};}}
            new Function('sampleRate','AudioWorkletProcessor','registerProcessor',processorCode)(rate,Base,(_,Ctor)=>{Processor=Ctor});
            const processor=new Processor();const size=rate*5+137;
            for(let pos=0;pos<size;pos+=128){const samples=new Float32Array(Math.min(128,size-pos));samples.fill(.2);processor.process([[samples]]);}
            processor.port.onmessage({data:'flush'});
            const all=frames.flatMap(x=>Array.from(x));
            outcomes.push({rate,actual:all.length,expected:Math.ceil(size*16000/rate),amplitudeOkay:all.every(x=>Math.abs(x-6553)<=1),lastFrame:frames.at(-1).length});
        }
        return outcomes;
    """)
    assert all(x["actual"]==x["expected"] and x["amplitudeOkay"] for x in result),result
    print("RESAMPLER",json.dumps(result),flush=True)
    for i in range(3):
        driver.find_element("id","start").click()
        WebDriverWait(driver,30).until(lambda d:d.execute_script("return phase === 'listening'"))
        time.sleep(1.25+i*.15)
        driver.find_element("id","stop").click()
        WebDriverWait(driver,25).until(lambda d:d.execute_script("return phase === 'idle'"))
        outcome=driver.execute_script("return {bytes:recentBytes,state:document.getElementById('state').textContent,contextReleased:context===null,micReleased:stream===null,startEnabled:!document.getElementById('start').disabled,partialFrame:recentBytes%3200};")
        assert outcome["bytes"]>0 and outcome["contextReleased"] and outcome["micReleased"] and outcome["startEnabled"],outcome
        print("CYCLE",i,json.dumps(outcome,ensure_ascii=False),flush=True)
    oversize=driver.execute_script("""
        recentPCM=[];recentBytes=0;capturedBytes=0;captureOverflow=false;
        for(let i=0;i<301;i++)recordFrame(new ArrayBuffer(3200));
        setPhase('idle');
        return {overflow:captureOverflow,retained:recentBytes,disabled:document.getElementById('save_sample').disabled};
    """)
    assert oversize=={"overflow":True,"retained":0,"disabled":True},oversize
    print("LONG_SAMPLE_GUARD",oversize,flush=True)
    driver.save_screenshot(str(Path(__file__).resolve().parents[1]/"data"/"mobile-ui.png"))
    errors=[item for item in driver.get_log("browser") if item["level"]=="SEVERE" and "favicon.ico" not in item["message"]]
    assert not errors,errors
    print("PASS mobile browser HTTPS, worklet, flush, repeated stop, no truncated training pairs",flush=True)
finally:
    driver.quit()
