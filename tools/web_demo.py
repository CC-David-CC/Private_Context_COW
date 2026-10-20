"""Start the web app with a fixed sample answer. No model, GPU or model download."""
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

SAMPLE = ('This is a fixed sample answer, not an AI model.\n\n'
          r'One half is \(\frac{1}{2}=0.5\).'+ '\n\n'
          'Enable Render math in Sampling to see the fraction.')


class SampleEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        prefix = '' if self.tok.decode(ids).endswith('</think>\n\n') else '</think>\n\n'
        self.script = self.tok.encode(prefix + SAMPLE) + self.tok.encode('<|im_end|>',parse_special=True)
        yield from super().generate(ids,max_new,sampling,cancel,embeddings)


def sample_service():
    tok=ByteTokenizer()
    return Service(SampleEngine(tok,'sample'),tok,
                   ChatTemplate(Path(__file__).resolve().parents[1]/'serve/chat_template.jinja'),
                   model_name='Scripted sample (no AI model)')

if __name__ == '__main__':
    from serve.chat_workspace import ChatWorkspace
    service=sample_service()
    service.workspace=ChatWorkspace(Path(__file__).resolve().parents[1]/'context-cow-data/sample.chats.sqlite3')
    server=serve(service,host='127.0.0.1',port=8095)
    print('Sample app ready: http://127.0.0.1:8095 (fixed answer, no model). Ctrl+C stops it.',flush=True)
    webbrowser.open('http://127.0.0.1:8095')
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        service.workspace.close()
