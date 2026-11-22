"""Focused UI race check with delayed local responses; no live dataset writes."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
MOCK = r"""
window.__test={requests:[],pending:[],savedSamples:0,nextHash:"baseline"};
const test=window.__test;
const response=value=>new Response(JSON.stringify(value),{status:200,headers:{"Content-Type":"application/json"}});
window.fetch=(path,options={})=>{
 if(path==="/v1/dataset/ingest"){
  test.requests.push(JSON.parse(options.body));
  return new Promise((resolve,reject)=>test.pending.push({resolve,reject}));
 }
 if(path==="/v1/dataset/status")return Promise.resolve(response({saved_samples:test.savedSamples}));
 if(path==="/v1/adaptation/status")return new Promise(resolve=>{test.initialBias=()=>resolve(response({enabled:false,phrase_count:0}));});
 if(path==="/health")return Promise.resolve(response({ok:true}));
 throw new Error("Unexpected request: "+path);
};
window.WebSocket=class {
 static OPEN=1;
 constructor(){
  this.readyState=0;this.bufferedAmount=0;
  const hash=test.nextHash;
  setTimeout(()=>{this.readyState=1;this.onmessage?.({data:JSON.stringify({type:"ready",locale:"vi-VN",bias_profile_sha256:hash})});},5);
 }
 send(value){
  if(value==="stop")setTimeout(()=>{
   this.onmessage?.({data:JSON.stringify({type:"final",text:"Nội dung thử nghiệm."})});
   this.onmessage?.({data:JSON.stringify({type:"closed"})});
  },5);
 }
 close(){this.readyState=3;queueMicrotask(()=>this.onclose?.());}
};
"""
HTML = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
HTML = HTML.replace("<script>", "<script>" + MOCK + "</script><script>", 1)


class Handler(BaseHTTPRequestHandler):
    post_count = 0

    def do_GET(self):
        body = HTML.encode() if self.path == "/" else b""
        self.send_response(200 if self.path == "/" else 204)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        Handler.post_count += 1
        self.send_response(500)
        self.end_headers()

    def log_message(self, *args):
        pass


def main():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--use-fake-ui-for-media-stream")
    options.add_argument("--use-fake-device-for-media-stream")
    options.add_argument("--autoplay-policy=no-user-gesture-required")
    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 15)

    def start(hash_value="baseline"):
        driver.execute_script("window.__test.nextHash=arguments[0]", hash_value)
        driver.find_element("id", "start").click()
        wait.until(lambda d: d.execute_script("return phase==='listening'"))
        wait.until(lambda d: d.execute_script("return recentBytes>=32000"))

    def stop():
        driver.find_element("id", "stop").click()
        wait.until(lambda d: d.execute_script("return phase==='idle'"))

    def prepare_save(reference):
        driver.execute_script("""
            document.getElementById('correction').value=arguments[0];
            document.getElementById('consent').checked=true;
        """, reference)

    def snapshot():
        return driver.execute_script("""
            return {generation,bytes:recentBytes,reference:document.getElementById('correction').value,
              consent:document.getElementById('consent').checked,
              state:document.getElementById('state').textContent,
              firstFrame:Array.from(new Uint8Array(recentPCM[0]))};
        """)

    def assert_preserved(before):
        after = snapshot()
        assert after["generation"] == before["generation"]
        assert after["bytes"] >= before["bytes"], (before["bytes"], after["bytes"])
        for key in ("reference", "consent", "state", "firstFrame"):
            assert after[key] == before[key], key
        assert driver.execute_script("return phase==='listening'")
        assert "Âm thanh tạm trên trang đã được xóa" not in driver.find_element("id", "capture_note").text

    try:
        driver.get(f"http://127.0.0.1:{httpd.server_port}/")
        start()
        assert "nguyên bản" in driver.find_element("id", "bias_status").text
        stop()
        prepare_save("Bản chép của mẫu cũ.")
        original_bytes = driver.execute_script("return recentBytes")
        driver.find_element("id", "save_sample").click()
        wait.until(lambda d: d.execute_script("return __test.requests.length===1"))
        request = driver.execute_script("return __test.requests[0]")
        assert request["reference"] == "Bản chép của mẫu cũ."
        assert len(base64.b64decode(request["pcm_base64"])) == original_bytes
        assert driver.find_element("id", "start").is_enabled()

        start("isolated-test-profile")
        prepare_save("Bản chép của đoạn đang ghi mới.")
        before = snapshot()
        driver.execute_script("""
            __test.savedSamples=1;
            __test.pending[0].resolve(new Response(JSON.stringify({sample_id:'mock-1',count:1}),{status:200}));
            __test.initialBias();
        """)
        wait.until(lambda d: "phiên trước" in d.find_element("id", "save_result").text)
        wait.until(lambda d: d.find_element("id", "sample_count").text.startswith("1 "))
        assert_preserved(before)
        assert "cụm từ ưu tiên" in driver.find_element("id", "bias_status").text
        print("PASS delayed old success preserves active PCM, reference, consent and state", flush=True)
        print("PASS ready profile label survives stale initial baseline response", flush=True)

        stop()
        driver.find_element("id", "save_sample").click()
        wait.until(lambda d: d.execute_script("return __test.requests.length===2"))
        start()
        prepare_save("Bản chép của phiên thứ ba.")
        before = snapshot()
        driver.execute_script("__test.pending[1].reject(new Error('Lỗi mạng giả lập'))")
        wait.until(lambda d: "Chưa lưu được mẫu của phiên trước" in d.find_element("id", "save_result").text)
        assert_preserved(before)
        assert "nguyên bản" in driver.find_element("id", "bias_status").text
        print("PASS delayed old failure preserves newer recording and reports its own failure", flush=True)

        stop()
        driver.find_element("id", "save_sample").click()
        wait.until(lambda d: d.execute_script("return __test.requests.length===3"))
        driver.execute_script("""
            __test.savedSamples=2;
            __test.pending[2].resolve(new Response(JSON.stringify({sample_id:'mock-2',count:2}),{status:200}));
        """)
        wait.until(lambda d: d.execute_script("return recentBytes===0 && !document.getElementById('consent').checked"))
        wait.until(lambda d: d.find_element("id", "sample_count").text.startswith("2 "))
        assert Handler.post_count == 0
        print("PASS same-session success clears only its own buffer; simulated sample count=2; actual HTTP POSTs=0", flush=True)
    finally:
        driver.quit()
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
