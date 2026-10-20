import importlib
import os
from pathlib import Path
import sys
import unittest

from experimental.branching_context.scripts.checkpoint_lifecycle_diagnostics import (
    LifecycleProbe,
    install_lifecycle_observers,
)


class PinnedLifecycleBindingTests(unittest.TestCase):
    def test_observers_install_and_restore_against_pinned_source_without_model_start(self):
        source = Path(os.environ['CTT_DIAGNOSTIC_SOURCE_ROOT']).resolve(strict=True)
        source_modules = source/'experimental/branching_context/src'
        self.assertTrue(source_modules.is_dir())
        sys.path.insert(0, str(source_modules))
        try:
            real = importlib.import_module('context_time_travel.real')
            backend = importlib.import_module('context_time_travel.strata_backend')
            pools = importlib.import_module('context_time_travel.pools')
            targets = (
                (backend.StrataBackend, '_request'),
                (backend.StrataBackend, 'serialize'),
                (real.Checkpoint, 'digest'),
                (pools, 'checkpoint_digest'),
                (pools.SharedCheckpointPool, 'admit'),
                (pools.SharedCheckpointPool, '_admit'),
                (real.BranchingModel, '_prepared_checkpoint'),
                (real.BranchingModel, '_publish_checkpoint_parts'),
                (real.BranchingModel, '_publish_checkpoint_file'),
                (real.BranchingModel, 'validate_checkpoint'),
                (real.BranchingModel, '_load_checkpoint'),
            )
            originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
            probe = LifecycleProbe()
            try:
                install_lifecycle_observers(probe, real, backend, pools)
                self.assertTrue(all(getattr(owner, name) is not original
                                    for owner, name, original in originals))
                self.assertEqual(probe.snapshot()['events'], [])
            finally:
                probe.close()
            self.assertTrue(all(getattr(owner, name) is original
                                for owner, name, original in originals))
        finally:
            sys.path.remove(str(source_modules))


if __name__ == '__main__':
    unittest.main()
