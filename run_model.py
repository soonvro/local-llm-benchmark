#!/usr/bin/env python3
"""Optional orchestration: load ONE model, measure GPU memory, run benchmark, shut down."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from benchmark import check_server, run_benchmark, vram_used_mib

ROOT=Path(__file__).resolve().parent

class GPUMonitor:
    def __init__(self,index:int, interval:float=0.25):
        self.index=index; self.interval=interval; self.samples=[]
        self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self._loop, daemon=True)
    def _loop(self):
        while not self.stop_event.is_set():
            x=vram_used_mib(self.index)
            if x is not None:self.samples.append([time.time(),x])
            self.stop_event.wait(self.interval)
    def start(self):self.thread.start()
    def stop(self):self.stop_event.set();self.thread.join(timeout=3)
    @property
    def peak(self):return max((x[1] for x in self.samples),default=None)

def main():
    p=argparse.ArgumentParser(description='Download (opt-in), launch, benchmark and stop one local LLM')
    p.add_argument('--profile',required=True)
    p.add_argument('--profile-file',type=Path,default=ROOT/'profiles.json')
    p.add_argument('--model-dir',type=Path,default=ROOT/'models')
    p.add_argument('--download',action='store_true',help='EXPLICITLY download GGUF or vLLM HF weights')
    p.add_argument('--llama-server-bin',default='llama-server')
    p.add_argument('--vllm-bin',default='vllm')
    p.add_argument('--gpu-index',type=int,default=0)
    p.add_argument('--ready-timeout',type=int,default=240)
    p.add_argument('--timeout',type=int,default=240)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--perf-repeats',type=int,default=3)
    p.add_argument('--max-tokens',type=int,default=700)
    p.add_argument('--out',type=Path,default=None)
    args=p.parse_args()
    profiles=json.loads(args.profile_file.read_text(encoding='utf-8'))
    if args.profile not in profiles: p.error(f'Unknown profile: {args.profile} (available: {list(profiles)})')
    x=profiles[args.profile]; kind=x['kind']
    out=args.out or ROOT/'results'/args.profile
    out.mkdir(parents=True,exist_ok=True)
    base_url=f'http://127.0.0.1:{x["port"]}/v1'
    before=vram_used_mib(args.gpu_index)
    if before is not None and before > 1500:
        print(f'[WARN] GPU already uses {before} MiB; dedicated-GPU VRAM figures may be inaccurate',flush=True)
    if kind=='llama_cpp':
        model_path=args.model_dir/args.profile/x['file']
        if not model_path.is_file():
            if not args.download:
                print(f'[ERROR] Missing {model_path}\nDownload manually using: '
                      f'hf download {x["model_repo"]} {x["file"]} --local-dir {model_path.parent}\n'
                      'Or rerun with --download.',file=sys.stderr)
                sys.exit(2)
            model_path.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['hf','download',x['model_repo'],x['file'],'--local-dir',str(model_path.parent)],check=True)
        cmd=[args.llama_server_bin,'-m',str(model_path),'--host','127.0.0.1','--port',str(x['port']),
             '--alias',x['model_id'],*x.get('args',[])]
    else:
        repo=x['model_repo']
        if args.download:
            print('[INFO] Downloading vLLM repository explicitly requested',flush=True)
            subprocess.run(['hf','download',repo],check=True)
        cmd=[args.vllm_bin,'serve',repo,'--host','127.0.0.1','--port',str(x['port']),
             '--served-model-name',x['model_id'],*x.get('args',[])]
    exe=shutil.which(cmd[0])
    if not exe and not Path(cmd[0]).is_file():
        print(f'[ERROR] Executable not found: {cmd[0]}',file=sys.stderr)
        sys.exit(2)
    print('[LAUNCH]', ' '.join(cmd),flush=True)
    logpath=out/'server.log'
    log=logpath.open('w',encoding='utf-8')
    monitor=GPUMonitor(args.gpu_index)
    monitor.start()
    proc=None
    ready_used=None
    try:
        # Separate process group allows terminating vLLM worker subprocesses.
        proc=subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              env={**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu_index)})
        started=time.monotonic()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f'Model process exited {proc.returncode}. Inspect {logpath}')
            try:
                available=check_server(base_url, timeout=2)
                served_ids=[m.get("id") for m in available.get("data",[]) if isinstance(m,dict)]
                if x["model_id"] not in served_ids:
                    raise RuntimeError(
                        f"Unexpected model on {base_url}: {served_ids}; expected {x['model_id']}. "
                        "Another server may be using this port. Stop it before benchmarking.")
                break
            except RuntimeError:
                raise
            except Exception:
                if time.monotonic()-started > args.ready_timeout:
                    raise TimeoutError(f'Server not ready in {args.ready_timeout}s; inspect {logpath}')
                time.sleep(2)
        ready_used=vram_used_mib(args.gpu_index)
        print(f'[READY] VRAM baseline: {before} MiB, model loaded: {ready_used} MiB',flush=True)
        summary=run_benchmark(base_url,x['model_id'],ROOT/'cases.json',out,args.repeats,
                              args.perf_repeats,args.max_tokens,args.timeout,args.gpu_index)
        summary['hardware']={'gpu_index':args.gpu_index,'before_server_mib':before,
                             'ready_server_mib':ready_used,
                             'estimated_delta_mib':(ready_used-before) if before is not None and ready_used is not None else None,
                             'peak_total_used_mib':monitor.peak,
                             'peak_delta_mib':(monitor.peak-before) if monitor.peak is not None and before is not None else None,
                             'note':'GPU total used, measured by nvidia-smi; do not equate server reservation with model weights.'}
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print('[GPU]',json.dumps(summary['hardware'],ensure_ascii=False))
    except Exception as ex:
        print(f'[FATAL] {ex}',file=sys.stderr)
        sys.exit(2)
    finally:
        if proc and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        monitor.stop()
        log.close()

if __name__=='__main__': main()
