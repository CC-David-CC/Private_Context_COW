"""One bounded reasoning/closing/final matcher over canonical consumed tokens.

The immutable grammar definition owns the budget. Mutable phase/cursor state is
derived by GrammarEngine.restore replay, just like the existing final parser.
No tokens are inserted, erased, or consumed without the model's masked selection.
This first profile requires a dedicated </think> control/user-defined token. Ordinary-text
delimiters require another explicitly tested contract and are rejected here.
"""
import codecs

from .store import InvalidOperation


def normalize(source, normalize_final):
    if (not isinstance(source, dict) or set(source) != {'schema', 'max_tokens', 'final'}
            or type(source['schema']) is not int or source['schema'] != 1
            or type(source['max_tokens']) is not int or not 0 <= source['max_tokens'] <= 32768):
        raise InvalidOperation('Thinking grammar requires schema 1 and a 0..32768 token budget')
    final = source['final']
    if (not isinstance(final, dict) or set(final) != {'format', 'source'}
            or final['format'] not in ('gbnf', 'lark', 'json_schema')):
        raise InvalidOperation('Thinking final must be one GBNF or JSON Schema definition')
    clean, _ = normalize_final(final['format'], final['source'])
    return dict(schema=1, max_tokens=source['max_tokens'],
                final=dict(format=final['format'], source=clean))


class ThinkingMatcher:
    def __init__(self, engine, final, budget, close_token, free):
        self.engine, self.final, self.budget = engine, final, budget
        self.close_token, self.free = close_token, free
        self.phase = 'reasoning'
        self.reasoning_tokens = self.final_tokens = 0
        self.closed_at_budget = None
        self.decoder = codecs.getincrementaldecoder('utf8')(errors='strict')

    def is_error(self):
        return self.final.is_error() or self.free.is_error()

    def is_accepting(self):
        return self.phase == 'final' and self.final.is_accepting()

    def is_stopped(self):
        return self.phase == 'final' and self.final.is_stopped()

    def compute_bitmask(self):
        if self.phase == 'final':
            return self.final.compute_bitmask()
        width = (self.engine.vocab_size + 7) // 8
        if self.reasoning_tokens == self.budget:
            mask = bytearray(width)
        else:
            mask = bytearray(self.free.compute_bitmask()[:width])
            for token in self.engine._special_ids | frozenset(self.engine.eos_tokens) | {self.close_token}:
                mask[token >> 3] &= ~(1 << (token & 7))
            if self.reasoning_tokens + 1 == self.budget:
                # Do not exhaust the budget halfway through a UTF-8 codepoint.
                # Ordinary earlier tokens may split UTF-8; closure may not.
                pending = self.decoder.getstate()[0]
                for token, piece in enumerate(self.engine._adapter.tokens):
                    if mask[token >> 3] & (1 << (token & 7)):
                        try:
                            (pending + piece).decode('utf8', errors='strict')
                        except UnicodeDecodeError:
                            mask[token >> 3] &= ~(1 << (token & 7))
        if not self.decoder.getstate()[0]:
            mask[self.close_token >> 3] |= 1 << (self.close_token & 7)
        return bytes(mask)

    def consume_token(self, token):
        if self.phase == 'final':
            if not self.final.consume_token(token):
                return False
            if token not in self.engine.eos_tokens:
                self.final_tokens += 1
            return True
        if token == self.close_token and not self.decoder.getstate()[0]:
            self.closed_at_budget = self.reasoning_tokens == self.budget
            self.phase = 'final'
            return True
        if self.reasoning_tokens >= self.budget:
            return False
        if not self.free.consume_token(token):
            return False
        self.decoder.decode(self.engine._adapter.tokens[token], final=False)
        self.reasoning_tokens += 1
        return True

    def progress(self):
        return dict(schema=1, phase=self.phase, max_tokens=self.budget,
                    reasoning_tokens=self.reasoning_tokens,
                    closing_tokens=int(self.phase == 'final'), final_tokens=self.final_tokens,
                    close_reason=(None if self.closed_at_budget is None else
                                  'budget' if self.closed_at_budget else 'natural'))
