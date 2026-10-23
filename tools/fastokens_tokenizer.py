"""Optional native BPE, inspired by fastokens; COW retains token boundaries.

The GGUF remains the vocabulary/merge authority. Existing Unicode splitting,
control versus user-defined literals, protected plain spans and piece-cache
limits remain in the reference tokenizer. No tokenizer is downloaded and no
transformers/tokenizers process-wide patch is installed.
"""
import importlib.metadata
import json
import os

from tools.strata_tokenizer import Tokenizer

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

    def _encode_piece(self,piece):
        # Avoid an encode/decode round trip through the byte alphabet. The
        # reference still owns piece boundaries and its bounded result cache.
        if len(piece)<=self.HEAP_MIN//4:return super()._encode_piece(piece)
        size=len(piece.encode('utf-8'))
        if size<=self.HEAP_MIN:return super()._encode_piece(piece)
        encoded=self._native.encode_batch([piece],add_special_tokens=False)
        if len(encoded)!=1:raise ValueError('Native tokenizer batch extent differs')
        ids=encoded[0].ids
        if (type(ids) is not list or not 0<len(ids)<=size
                or any(type(i) is not int or not 0<=i<len(self.tokens) for i in ids)):
            raise ValueError('Native tokenizer returned invalid token IDs')
        return ids
