"""CPU scheduler controls for retaining a just-created root; no GPU claims."""
import tempfile
import threading
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from test_pool_coordinator import Model


class RootResidencyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.models = {}
        self.releases = []
        self.execution_releases = []
        gate = threading.Event()
        gate.set()

        def factory(gpu):
            model = Model(directory.name, gpu, (0,), [], gate)
            model.release_checkpoints = lambda: self.releases.append(gpu)
            execute = model.run_greedy_request

            def observed(*args, **kwargs):
                self.execution_releases.append(len(self.releases))
                return execute(*args, **kwargs)

            model.run_greedy_request = observed
            self.models[gpu] = model
            return model

        self.coordinator = ContextCoordinator(directory.name,
            PoolConfig(gpus=(0,), ram_bytes=4096, disk_bytes=16384,
                       min_available_bytes=0), factory)
        self.addCleanup(self.coordinator.close)
        self.coordinator.ready(timeout=5)

    def idle(self):
        # Durable publication and worker cleanup are separate operations.
        with self.coordinator._condition:
            self.assertTrue(self.coordinator._condition.wait_for(
                lambda: not self.coordinator._workers[0]['busy'], timeout=5))

    def test_first_ask_keeps_just_prepared_root_without_release(self):
        c = self.coordinator
        root = c.create('base', (1, 2), 'create').durable.result(5)['snapshot']
        self.idle()
        model = self.models[0]
        prepared = model.cache[root]
        observed = []
        execute = model.run_greedy_request

        def record(*args, **kwargs):
            observed.append(model.cache[args[0]] is prepared)
            return execute(*args, **kwargs)

        model.run_greedy_request = record
        released_before = len(self.releases)
        c.submit('base', (3,), 1, 'ask').durable.result(5)
        self.assertEqual(self.execution_releases[-1], released_before,
                         'First ask discarded the root that was just prepared')
        self.assertEqual(observed, [True],
                         'First ask replaced the resident root with a loaded image')

    def test_switching_sibling_branches_does_not_release_same_root(self):
        c = self.coordinator
        root = c.create('base', (1, 2), 'create').durable.result(5)['snapshot']
        c.fork('a', root)
        c.fork('b', root)
        c.submit('a', (3,), 1, 'ask-a').durable.result(5)
        self.idle()
        released_before = len(self.releases)
        c.submit('b', (4,), 1, 'ask-b').durable.result(5)
        self.assertEqual(self.execution_releases[-1], released_before)
        self.assertEqual(c.store.head('a').root, c.store.head('b').root)

    def test_returning_to_different_root_releases_previous_tree(self):
        c = self.coordinator
        c.create('first', (1, 2), 'create-first').durable.result(5)
        c.create('second', (8, 9), 'create-second').durable.result(5)
        c.submit('second', (10,), 1, 'ask-second').durable.result(5)
        self.idle()
        released_before = len(self.releases)
        c.submit('first', (3,), 1, 'ask-first').durable.result(5)
        self.assertEqual(self.execution_releases[-1], released_before + 1)
        self.assertNotEqual(c.store.head('first').root, c.store.head('second').root)


if __name__ == '__main__':
    unittest.main()
