from selenium import webdriver
from selenium.webdriver.chrome.options import Options
o = Options()
o.add_argument("--headless=new")
d = webdriver.Chrome(options=o)
try:
    d.get("http://127.0.0.1:18765/")
    print("TITLE", d.title)
    print("H1", d.find_element("tag name","h1").text)
    print("JAVASCRIPT_LOADED", d.execute_script("return typeof datasetStatus === 'function' && !!document.getElementById('save_sample') && !!document.getElementById('consent')"))
    assert d.execute_script("return typeof datasetStatus === 'function' && !!document.getElementById('save_sample') && !!document.getElementById('consent')")
    print("APIS", d.execute_script("""return {
        microphone: !!navigator.mediaDevices,
        worklet: !!window.AudioWorkletNode,
        start: !!document.getElementById('start'),
        stop: !!document.getElementById('stop'),
        languages: Array.from(document.querySelectorAll('#lang option')).map(o => o.value)
    }"""))
finally:
    d.quit()
