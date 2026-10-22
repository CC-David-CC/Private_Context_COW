"""Native output-head backward binding, without torch or an alternate model."""
import ctypes as C
from pathlib import Path
import numpy as np


class HeadBackward:
    def __init__(self, library):
        self.library = C.CDLL(str(Path(library).resolve()))
        self.function = self.library.wen_head_backward
        pointer = C.POINTER(C.c_float)
        self.function.argtypes = [C.c_int,C.c_int,C.c_int,pointer,pointer,pointer,pointer,
                                  C.c_float,C.c_int,pointer,pointer]
        self.function.restype = C.c_double

    def __call__(self, hidden, logits, a, b, alpha, target):
        hidden, logits, a, b = [np.ascontiguousarray(v, dtype=np.float32) for v in (hidden,logits,a,b)]
        if (hidden.ndim != 1 or logits.ndim != 1 or a.ndim != 2 or b.ndim != 2
                or a.shape[1] != len(hidden) or b.shape != (len(logits),a.shape[0])
                or a.shape[0] <= 0 or not 0 <= target < len(logits)
                or not all(np.isfinite(v).all() for v in (hidden,logits,a,b))):
            raise ValueError('Finite compatible head tensors required')
        da, db = np.empty_like(a), np.empty_like(b)
        pointer = lambda v: v.ctypes.data_as(C.POINTER(C.c_float))
        value = self.function(len(hidden),len(logits),len(a),pointer(hidden),pointer(logits),
                              pointer(a),pointer(b),alpha/len(a),target,pointer(da),pointer(db))
        if not np.isfinite(value) or not np.isfinite(da).all() or not np.isfinite(db).all():
            raise ValueError('Native head backward returned nonfinite results')
        return value, da, db


def export_adapter(path, a, b, alpha):
    import gguf
    path = Path(path)
    if path.exists():
        raise FileExistsError('Adapter versions are immutable')
    writer = gguf.GGUFWriter(str(path), 'wen1')
    writer.add_type('adapter')
    writer.add_string('adapter.type','lora')
    writer.add_float32('adapter.lora.alpha',alpha)
    writer.add_tensor('output.weight.lora_a',np.ascontiguousarray(a,dtype=np.float32))
    writer.add_tensor('output.weight.lora_b',np.ascontiguousarray(b,dtype=np.float32))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    actual = {t.name: np.array(t.data, copy=True) for t in gguf.GGUFReader(str(path)).tensors}
    assert set(actual) == {'output.weight.lora_a','output.weight.lora_b'}
    assert np.array_equal(actual['output.weight.lora_a'],a)
    assert np.array_equal(actual['output.weight.lora_b'],b)


def verify_backward(library):
    """Native F32 adjoints versus FP64 finite differences of the smooth head."""
    from math_reference import cross_entropy
    rng = np.random.default_rng(20261022)
    h, z, a, b = [rng.normal(size=s).astype(np.float32) for s in ((5,),(7,),(3,5),(7,3))]
    backward = HeadBackward(library)
    loss, da, db = backward(h,z,a,b,3.,2)
    aa, bb, hh = a.astype(np.float64), b.astype(np.float64), h.astype(np.float64)
    original_update = bb @ aa @ hh
    def value():
        scores = z.astype(np.float64)+(bb @ aa @ hh-original_update)
        return cross_entropy(scores[None,:],[2])[0]
    errors = []
    for variable, analytic in ((aa,da),(bb,db)):
        numeric = np.empty_like(variable)
        for index in np.ndindex(variable.shape):
            old = variable[index]
            variable[index] = old+1e-6; plus = value()
            variable[index] = old-1e-6; minus = value()
            variable[index] = old
            numeric[index] = (plus-minus)/2e-6
        np.testing.assert_allclose(analytic,numeric,rtol=1e-5,atol=2e-7)
        errors.append(float(np.max(np.abs(analytic-numeric))))
    _, zero_da, zero_db = backward(h,z,a,np.zeros_like(b),3.,2)
    assert np.array_equal(zero_da,np.zeros_like(a)) and np.linalg.norm(zero_db)>0
    return dict(loss=loss,max_abs_da=errors[0],max_abs_db=errors[1],zero_b_control=True,
                scope='smooth output-head VJP only; F32 rounding treated as identity; no quantizer derivative')


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('library')
    args = parser.parse_args()
    print(json.dumps(verify_backward(args.library)))
