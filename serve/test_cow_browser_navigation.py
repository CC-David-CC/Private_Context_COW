"""CPU checks for waiting for the real document before relative URL requests."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('cow_smoke_navigation',
    Path(__file__).parent/'web/cow_native_smoke.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class BrowserNavigationTests(unittest.TestCase):
    def ready(self, **changes):
        value = dict(url='http://127.0.0.1:8189/', readyState='complete', hasWorkspace=True)
        value.update(changes)
        return value

    def test_about_blank_is_not_ready_even_if_document_complete(self):
        self.assertFalse(smoke._application_page_ready(self.ready(url='about:blank'),
                                                       'http://127.0.0.1:8189/'))

    def test_wrong_server_or_path_is_not_ready(self):
        for url in ('http://127.0.0.1:8190/', 'http://localhost:8189/',
                    'http://127.0.0.1:8189/other', 'chrome-error://chromewebdata/'):
            with self.subTest(url=url):
                self.assertFalse(smoke._application_page_ready(self.ready(url=url),
                    'http://127.0.0.1:8189/'))

    def test_loaded_app_is_required_at_the_expected_url(self):
        for change in ({'readyState':'loading'}, {'hasWorkspace':False}, {'hasWorkspace':1}):
            self.assertFalse(smoke._application_page_ready(self.ready(**change),
                'http://127.0.0.1:8189/'))
        self.assertTrue(smoke._application_page_ready(self.ready(), 'http://127.0.0.1:8189/'))

    def test_waits_past_blank_and_loading_and_preserves_observations(self):
        devtools = mock.Mock()
        observations = [self.ready(url='about:blank'), self.ready(readyState='loading'), self.ready()]
        devtools.evaluate.side_effect = observations
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(smoke.time, 'sleep'):
            result = smoke._wait_for_application_page(devtools, 'http://127.0.0.1:8189', Path(directory))
            self.assertEqual(result['observations'], observations)
            self.assertEqual(result['status'], 'PASS_APPLICATION_DOCUMENT_READY')
            self.assertTrue((Path(directory)/'browser-navigation.json').is_file())

    def test_timeout_preserves_wrong_document_without_fetch(self):
        devtools = mock.Mock()
        devtools.evaluate.return_value = self.ready(url='about:blank')
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(smoke.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'application document'):
                smoke._wait_for_application_page(devtools, 'http://127.0.0.1:8189',
                    Path(directory), timeout=0)
            self.assertTrue((Path(directory)/'browser-navigation.json').is_file())
            self.assertFalse(any('fetch(' in str(call) for call in devtools.evaluate.call_args_list))


if __name__ == '__main__': unittest.main()
