"""Optional native BPE, inspired by fastokens; COW retains token boundaries.

The GGUF remains the vocabulary/merge authority. Existing Unicode splitting,
control versus user-defined literals, protected plain spans and piece-cache
limits remain in the reference tokenizer. No tokenizer is downloaded and no
transformers/tokenizers process-wide patch is installed.
"""
import importlib.metadata
import json
import os

from tools.strata_tokenizer import Tokenizer,UNICODE_TO_BYTE

FASTOKENS_VERSION='0.3.4'


def load_tokenizer(path,*,backend='python'):
    if backend=='python':return Tokenizer.from_gguf(path)
    if backend=='fastokens':return FastokensTokenizer.from_gguf(path)
    raise ValueError('Unknown tokenizer backend')


class FastokensTokenizer(Tokenizer):
    backend_name='fastokens'

    def __init__(self,*args,**kwargs):
        try:version=importlib.metadata.version('fastokens')
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError('Optional tokenizer requires fastokens=='+FASTOKENS_VERSION) from None
        if version!=FASTOKENS_VERSION:
            raise RuntimeError('Qualified adapter requires fastokens=='+FASTOKENS_VERSION)
        for name in ('FASTOKENS_INPUT_CACHE','FASTOKENS_SEGMENT_CACHE'):
            if os.environ.get(name,'0') not in ('','0'):
                raise ValueError('Use the bounded COW piece cache; native cache overrides are unsupported')
        from fastokens._native import Tokenizer as NativeTokenizer
        super().__init__(*args,**kwargs)
        document=dict(version='1.0',truncation=None,padding=None,added_tokens=[],normalizer=None,
            pre_tokenizer=dict(type='ByteLevel',add_prefix_space=False,trim_offsets=False,use_regex=False),
            post_processor=None,decoder=dict(type='ByteLevel',add_prefix_space=False,trim_offsets=False,use_regex=False),
            model=dict(type='BPE',dropout=None,unk_token=None,continuing_subword_prefix=None,
                end_of_word_suffix=None,fuse_unk=False,byte_fallback=False,ignore_merges=False,
                vocab=self.ids,merges=[list(pair) for pair in sorted(self.ranks,key=self.ranks.__getitem__)]))
        self._native=NativeTokenizer.from_json_str(json.dumps(document,ensure_ascii=False))
        if self._native.vocab_size!=len(self.tokens):raise ValueError('Native tokenizer vocabulary extent differs')

    def _bpe_heap(self,word):
        # The reference chooses this path only for a previously uncached piece
        # longer than HEAP_MIN bytes. Short pieces and all warm hits retain the
        # measured reference path, including its Unicode and literal handling.
        text=bytes(UNICODE_TO_BYTE[c] for c in word).decode('utf-8')
        encoded=self._native.encode_batch([text],add_special_tokens=False)
        if len(encoded)!=1:raise ValueError('Native tokenizer batch extent differs')
        ids=encoded[0].ids
        if (type(ids) is not list or not 0<len(ids)<=len(word)
                or any(type(i) is not int or not 0<=i<len(self.tokens) for i in ids)):
            raise ValueError('Native tokenizer returned invalid token IDs')
        pieces=[self.tokens[i] for i in ids]
        if ''.join(pieces)!=word:raise ValueError('Native BPE changed the input bytes')
        return pieces
