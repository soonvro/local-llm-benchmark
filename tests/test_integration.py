"""Integration smoke using a fake local OpenAI-compatible SSE server; not an L4 measurement."""
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark import completion_stream, run_benchmark

class FakeHandler(BaseHTTPRequestHandler):
    def log_message(self,fmt,*args): pass
    def do_GET(self):
        if self.path == '/v1/models':
            data=json.dumps({'data':[{'id':'fake-local'}]}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        else:self.send_error(404)
    def do_POST(self):
        if self.path == '/tokenize':
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(b'{"count":20}')
            return
        n=int(self.headers.get('Content-Length','0'))
        body=json.loads(self.rfile.read(n))
        # Single case with R1 and known evidence.
        value=json.dumps({'requirements':[{'id':'R1','level':'required','user_status':'have',
            'resume_quote':'Java로 앱을 개발했다','jd_quote':'Java 개발 경험'}]},ensure_ascii=False)
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
        for c in [value[:15],value[15:]]:
            chunk={'choices':[{'delta':{'content':c},'finish_reason':None}]}
            self.wfile.write(('data: '+json.dumps(chunk,ensure_ascii=False)+'\n\n').encode());self.wfile.flush()
        self.wfile.write(('data: '+json.dumps({'choices':[],'usage':{'completion_tokens':80}})+'\n\n').encode())
        self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()

class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s=ThreadingHTTPServer(('127.0.0.1',0),FakeHandler)
        cls.thread=threading.Thread(target=cls.s.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.s.server_port}/v1'
    @classmethod
    def tearDownClass(cls):cls.s.shutdown();cls.s.server_close()
    def test_sse(self):
        x=completion_stream(self.url,'fake-local',[{'role':'user','content':'hello'}],timeout=5)
        self.assertEqual(x['tokens'],80)
        self.assertEqual(x['token_source'],'server_usage')
        self.assertTrue(x['text'].startswith('{'))
        self.assertGreater(x['total_s'],0)
    def test_full_benchmark(self):
        case={'id':'mock','domain':'it','jd':'R1 필수: Java 개발 경험.',
              'resume':'Java로 앱을 개발했다.', 'expected':{'R1':{'level':'required','status':'have'}}}
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            (p/'cases.json').write_text(json.dumps([case],ensure_ascii=False),encoding='utf-8')
            summary=run_benchmark(self.url,'fake-local',p/'cases.json',p/'out',1,1,256,5,0)
            self.assertEqual(summary['status_accuracy'],1.0)
            self.assertEqual(summary['evidence_valid_rate'],1.0)
            self.assertEqual(len((p/'out'/'results.jsonl').read_text().splitlines()),2)

if __name__=='__main__':unittest.main()
