"""End-to-end browser/microphone smoke test using Chrome's synthetic microphone.

No real user recording is captured or saved. This tests UI -> AudioWorklet ->
WebSocket -> native Google SODA init -> stop, not speech accuracy.
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

o=Options()
o.add_argument("--headless=new")
o.add_argument("--use-fake-ui-for-media-stream")
o.add_argument("--use-fake-device-for-media-stream")
o.add_argument("--autoplay-policy=no-user-gesture-required")
d=webdriver.Chrome(options=o)
try:
    d.get("http://127.0.0.1:18765/")
    d.find_element("id","start").click()
    WebDriverWait(d,30).until(lambda b:"Đang nghe" in b.find_element("id","state").text)
    print("START_OK",d.find_element("id","state").text)
    d.find_element("id","stop").click()
    WebDriverWait(d,25).until(lambda b:"Đã dừng" in b.find_element("id","state").text)
    print("STOP_OK",d.find_element("id","state").text)
    assert d.find_element("id","save_sample").is_displayed()
    status=d.execute_async_script("""
        const done=arguments[arguments.length-1];
        fetch('/v1/dataset/status').then(r=>r.json()).then(done)
        .catch(e=>done({error:String(e)}));
    """)
    print("DATASET_STATUS",status)
    assert status["saved_samples"]==0
finally:
    d.quit()
