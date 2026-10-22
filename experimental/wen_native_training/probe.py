"""Bounded native child lifetime: close contexts and release the GPU on exit."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import numpy as np


class Probe:
    def __init__(self, binary, model, directory, profile, gpu):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True,exist_ok=False)
        self.profile = self.directory/'profile.json'
        self.profile.write_text(json.dumps(profile,sort_keys=True,indent=2)+'\n')
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID')
        # This experiment targets native arithmetic. It does not enable the
        # separate PyTorch-matching reference profile to manufacture parity.
        env.update(GGML_CUDA_F32_ACTIVATIONS='0',COW_QWEN35_F32_ATTENTION='0',
                   GGML_CUDA_CUBLAS_COMPUTE_TYPE='auto',NVIDIA_TF32_OVERRIDE='0')
        self.log = (self.directory/'native.log').open('w')
        self.child = subprocess.Popen([str(Path(binary).resolve()),str(Path(model).resolve()),
                                      str(self.directory),str(self.profile)],env=env,
                                     stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.child.stdout,selectors.EVENT_READ)
        try:
            self.ready = self.read()
            assert self.ready.get('ready') is True
        except BaseException:
            self.close()
            raise
        self.counter = 0

    def read(self):
        if not self.selector.select(timeout=300):
            raise TimeoutError('Native probe stalled; caller must close it')
        line = self.child.stdout.readline()
        if not line:
            raise RuntimeError('Native probe exited; inspect native.log')
        return json.loads(line)

    def request(self, value):
        self.child.stdin.write(json.dumps(value,separators=(',',':'))+'\n')
        self.child.stdin.flush()
        return self.read()

    def tokenize(self, text):
        return self.request(dict(op='tokenize',text=text))['tokens']

    def forward(self, tokens, positions=None, adapter=None, capture=True, chunk=64):
        self.counter += 1
        name = 'forward_'+str(self.counter)
        request = dict(op='forward',name=name,tokens=tokens,capture=capture,chunk=chunk)
        if positions is not None: request['positions'] = positions
        if adapter is not None: request['adapter'] = str(Path(adapter).resolve())
        row = self.request(request)
        scores = self.directory/(name+'_logits.f32')
        logits = np.fromfile(scores,dtype='<f4'); scores.unlink()
        assert len(logits) == row['vocab']
        hidden = None
        if capture:
            path = self.directory/(name+'_hidden.f32')
            hidden = np.fromfile(path,dtype='<f4'); path.unlink()
            assert hidden.shape == (4096,)
        return logits, hidden, row

    def generate(self,tokens,positions=None,adapter=None,maximum=256):
        self.counter += 1
        request = dict(op='generate',name='generate_'+str(self.counter),tokens=tokens,
                       max_new_tokens=maximum,capture=False)
        if positions is not None: request['positions'] = positions
        if adapter is not None: request['adapter'] = str(Path(adapter).resolve())
        return self.request(request)

    def close(self):
        try:
            if self.child.poll() is None:
                self.child.stdin.close()
                try: self.child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.child.terminate()
                    try: self.child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.child.kill(); self.child.wait(timeout=10)
        finally:
            self.selector.close()
            self.log.close()

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
