"""Already-uploaded image references use real CLI/HTTP authority checks."""
import hashlib
import json
import unittest
from unittest import mock

from tools import cow_cli
import tools.test_cow_cli_images as fixture


class CowCliImageReferenceTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.CowCliImagesTests()
        self.addCleanup(self.case.doCleanups)
        self.case.setUp()
        self.flow = self.case.flow
        code, raw, error = self.case.cli("upload-image", "--namespace-id", self.flow.namespace,
                                       "--file", str(self.case.image))
        self.assertEqual(code, 0, error)
        self.image_id = json.loads(raw)["image_id"]

    def prepare(self, parts, operation="reference", namespace=None):
        path = self.case.folder / "references.json"
        path.write_text(json.dumps(parts))
        return self.case.cli("prepare", "--namespace-id", namespace or self.flow.namespace,
            "--branch", "reference", "--parts-file", str(path), "--operation-id", operation)

    def test_preuploaded_reference_preserves_order_without_upload_or_local_file(self):
        self.case.image.unlink()
        parts = [dict(type="text", text="Before "), dict(type="image", image_id=self.image_id),
                 dict(type="text", text=" after")]
        with mock.patch.object(cow_cli.CowClient, "upload_image", side_effect=AssertionError("must not upload again")):
            code, _, error = self.prepare(parts)
        self.assertEqual(code, 0, error)
        body = json.loads(cow_cli.RequestJournal(self.case.journal).get("reference")["body_json"])
        self.assertEqual(body["messages"][0]["content"], parts)
        self.assertEqual(len(self.flow.encoder.calls), 1)

    def test_bad_reference_rejects_before_prepare_or_upload(self):
        before = self.case.fixture.inventory()
        for bad in ("", "A" * 64, "../private", "http://example.invalid/image"):
            with self.subTest(bad=bad):
                code, _, error = self.prepare([dict(type="image", image_id=bad)], "invalid")
                self.assertEqual(code, 2)
                self.assertEqual(self.case.fixture.inventory(), before)
                self.assertIsNone(cow_cli.RequestJournal(self.case.journal).get("invalid"))

    def test_existing_reference_cannot_borrow_another_namespace_grant(self):
        other = self.flow.tokens.create_namespace("Other", "other-namespace").namespace_id
        code, _, error = self.prepare([dict(type="image", image_id=self.image_id),
                                      dict(type="text", text="Inspect.")], namespace=other)
        self.assertEqual(code, 2)
        self.assertIn("not uploaded in this namespace", error.lower())
        self.assertEqual(self.flow.encoder.calls, [])


if __name__ == "__main__":
    unittest.main()
