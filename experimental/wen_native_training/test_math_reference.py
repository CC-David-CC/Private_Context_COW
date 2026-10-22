"""Finite-difference and adversarial checks for the new, independent graph."""
import math
import unittest
import numpy as np
from math_reference import (Schedule, Chunk, rotate, compose, compose_vjp,
                            lora, lora_vjp, cross_entropy, distance_weights)


def numerical_grad(function, value, step=1e-6):
    grad = np.empty_like(value)
    for index in np.ndindex(value.shape):
        old = value[index]
        value[index] = old+step
        plus = function()
        value[index] = old-step
        minus = function()
        value[index] = old
        grad[index] = (plus-minus)/(2*step)
    return grad


class MathTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261022)
        self.schedule = Schedule.yarn([1., .1], 8., [0., 1.])

    def close(self, left, right, atol=2e-8):
        np.testing.assert_allclose(left, right, rtol=2e-6, atol=atol)

    def test_rotation_composition_inverse_partial_and_layouts(self):
        x = self.rng.normal(size=(3, 6))
        for layout in ('split-half', 'interleaved'):
            s = Schedule(self.schedule.frequencies, layout, attention_scale=1.2)
            self.close(rotate(rotate(x, 128, s), 1048576, s), rotate(x, 1048704, s), 1e-9)
            self.close(rotate(rotate(x, 1048576, s), -1048576, s), x)
            np.testing.assert_array_equal(rotate(x, 17, s)[:, 4:], x[:, 4:])

    def test_rotation_adjoint(self):
        x, g = self.rng.normal(size=(2, 4)), self.rng.normal(size=(2, 4))
        actual = numerical_grad(lambda: np.sum(rotate(x, 12000, self.schedule)*g), x)
        self.close(actual, rotate(g, -12000, self.schedule))

    def fixture(self, k=None, v=None):
        k = self.rng.normal(size=(2, 4)) if k is None else k
        v = self.rng.normal(size=(2, 3)) if v is None else v
        return Chunk.create(k, v, self.schedule, 'writer-v1')

    def test_inverse_query_matches_explicit_global_keys_shared_softmax(self):
        q = self.rng.normal(size=(2, 4)); a, b = self.fixture(), self.fixture()
        occurrences = [(a, 0), (b, 1000000)]
        out, _ = compose(q, occurrences, self.schedule, [1000001, 1000002], 'writer-v1')
        keys = np.concatenate([rotate(c.keys, o, self.schedule) for c, o in occurrences])
        values = np.concatenate([c.values for c, _ in occurrences])
        z = q @ keys.T/math.sqrt(4)
        p = np.exp(z-z.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
        self.close(out, p@values)
        # Averaging separately normalized chunk answers is a wrong implementation.
        wrong = sum(compose(q, [(c, o)], self.schedule, [1000001, 1000002], 'writer-v1')[0]
                    for c, o in occurrences)/2
        self.assertGreater(float(np.max(np.abs(out-wrong))), 1e-3)

    def test_attention_vjp_repeated_chunk_multiple_distances(self):
        q, k, v = (self.rng.normal(size=s) for s in [(2,4), (2,4), (2,3)])
        g = self.rng.normal(size=(2,3))
        def forward():
            c = self.fixture(k, v)
            return compose(q, [(c,0),(c,128),(c,1000000)], self.schedule,
                           [1000001,1000002], 'writer-v1')
        output, tape = forward()
        dq, chunks = compose_vjp(tape, g)
        dk, dv = next(iter(chunks.values()))
        self.assertEqual(len(chunks), 1)
        loss = lambda: np.sum(forward()[0]*g)
        for value, grad in ((q,dq),(k,dk),(v,dv)):
            self.close(numerical_grad(loss, value), grad)

    def test_masks_and_identity_admission_do_not_mutate_cache(self):
        q = self.rng.normal(size=(1,4)); c = self.fixture()
        before = c.keys.tobytes(), c.values.tobytes()
        out, tape = compose(q, [(c,128)], self.schedule, [128], 'writer-v1')
        self.close(out[0], c.values[0])
        _, grads = compose_vjp(tape, np.ones_like(out))
        np.testing.assert_array_equal(grads[c.key][1][1], 0.)
        self.assertEqual(before, (c.keys.tobytes(), c.values.tobytes()))
        with self.assertRaises(ValueError): c.keys[0,0] = 3
        for schedule, writer, origin in [(Schedule((1., .2)), 'writer-v1', 0),
                                         (self.schedule, 'writer-v2', 0),
                                         (self.schedule, 'writer-v1', -1)]:
            with self.assertRaises(ValueError): compose(q, [(c,origin)], schedule, [1000], writer)
        with self.assertRaises(ValueError):
            compose(q, [(c,0),(c,1)], self.schedule, [1000], 'writer-v1')
        with self.assertRaises(ValueError):
            compose(q, [(c,128)], self.schedule, [127], 'writer-v1')

    def test_lora_native_formula_vjp(self):
        x,w,a,b = (self.rng.normal(size=s) for s in [(2,4),(3,4),(2,4),(3,2)])
        g = self.rng.normal(size=(2,3))
        out,tape = lora(x,w,a,b,4.)
        self.close(out, x @ (w+2*b@a).T)
        dx,da,db = lora_vjp(tape,g)
        loss = lambda: np.sum(lora(x,w,a,b,4.)[0]*g)
        for value,grad in ((x,dx),(a,da),(b,db)):
            self.close(numerical_grad(loss,value),grad)

    def test_split_writer_backward_with_shared_occurrences(self):
        x,w,a,b = (self.rng.normal(size=s) for s in [(2,4),(4,4),(2,4),(4,2)])
        q = self.rng.normal(size=(1,4)); values = self.rng.normal(size=(2,3))
        g = self.rng.normal(size=(1,3)); positions = np.arange(2)
        def forward():
            raw, writer_tape = lora(x,w,a,b,2.)
            c = self.fixture(rotate(raw,positions,self.schedule),values)
            y, reader_tape = compose(q,[(c,0),(c,128)],self.schedule,[256],'writer-v1')
            return y,writer_tape,reader_tape,c
        y,wt,rt,c = forward()
        _, chunk_grads = compose_vjp(rt,g)
        raw_grad = rotate(chunk_grads[c.key][0],-positions,self.schedule)
        _,da,db = lora_vjp(wt,raw_grad)
        # This finite difference differentiates the complete same graph without
        # a detached Writer boundary. It is not ordinary full-causal prefill.
        loss = lambda: np.sum(forward()[0]*g)
        self.close(numerical_grad(loss,a),da)
        self.close(numerical_grad(loss,b),db)

    def test_loss_and_distance_normalization(self):
        logits = self.rng.normal(size=(3,5)); targets = [1,3,0]
        value, grad = cross_entropy(logits,targets)
        self.close(numerical_grad(lambda: cross_entropy(logits,targets)[0],logits),grad)
        weights = distance_weights([0,128,1048576])
        self.assertAlmostEqual(float(weights.mean()),1.)
        losses = np.array([1.,2.,3.])
        self.assertAlmostEqual(float(np.mean(weights*losses)),float(np.dot(weights,losses)/3))
        with self.assertRaises(ValueError): distance_weights([-1])


if __name__ == '__main__':
    unittest.main()
