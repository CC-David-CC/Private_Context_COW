"""Independent reference packet fixtures, CPU only; not model execution."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from run_thinking_suffix_flow import reference_ingest, FlowError
from context_time_travel.input_spans import InputHistory
from test_vision_execution import spans

class ThinkingImageReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.payload=self.root/'image.sve'
        raw=struct.pack('<5i',0x31455653,6,3,2,4)+struct.pack('<24f',*[n/31 for n in range(24)])
        self.payload.write_bytes(raw);self.digest=hashlib.sha256(raw).hexdigest()
        self.inputs=spans(embedding_sha256=self.digest);self.history=InputHistory(self.inputs)
        self.paths={self.digest:self.payload}
    def packet(self,history=None,start=0,label='case',paths=None):
        return reference_ingest(history or self.history,start,self.paths if paths is None else paths,32,self.root,label)
    def test_text_extension_needs_no_image_payload_or_position_rewrite(self):
        packet=self.packet(self.history+(4,5),len(self.history),paths={})
        self.assertEqual(packet['command'],'feed 4 5');self.assertEqual(packet['image_count'],0)
        self.assertEqual(sorted(x.name for x in self.root.iterdir()),['image.sve'])
    def test_root_manifest_and_coordinates_match_hand_encoded_asymmetric_grid(self):
        packet=self.packet();self.assertEqual(packet['image_count'],1)
        lines=Path(packet['manifest']).read_text().splitlines()
        self.assertEqual(lines[0],'COW_VISION_INPUT_V1 0 9 1');self.assertEqual(lines[1],'1 2 7 7 7 7 7 7 3')
        self.assertTrue(lines[2].startswith('2 6 3 2 4 7 '+self.digest+' '))
        raw=Path(packet['positions']).read_bytes();coords=list(struct.iter_unpack('<3i',raw))
        self.assertEqual(coords[:9],[(0,0,0),(1,1,1),(2,2,2),(2,2,3),(2,2,4),(2,3,2),(2,3,3),(2,3,4),(5,5,5)])
        self.assertEqual(coords[9],(6,6,6));self.assertEqual(len(coords),32)
    def test_append_uses_absolute_new_rows_and_keeps_prior_rotary_extent(self):
        history=self.history+InputHistory(self.inputs);packet=self.packet(history,9)
        line=Path(packet['manifest']).read_text().splitlines()[2]
        self.assertTrue(line.startswith('11 6 3 2 4 7 '+self.digest+' '))
        coords=list(struct.iter_unpack('<3i',Path(packet['positions']).read_bytes()))
        self.assertEqual(coords[9:12],[(6,6,6),(7,7,7),(8,8,8)])
        self.assertEqual(coords[17],(11,11,11));self.assertEqual(packet['count'],9)
    def test_repeat_placement_consumes_two_spans_using_one_immutable_payload(self):
        packet=self.packet(InputHistory(self.inputs+self.inputs));lines=Path(packet['manifest']).read_text().splitlines()
        self.assertEqual(packet['image_count'],2);self.assertEqual(lines[0],'COW_VISION_INPUT_V1 0 18 2')
        self.assertEqual(len(lines),4);self.assertIn(json.dumps(str(self.payload)),lines[2]);self.assertIn(json.dumps(str(self.payload)),lines[3])
    def test_bad_payload_rejects_before_publishing_any_packet(self):
        for label,payload in [('missing',None),('corrupt',b'bad'),('nonfinite',struct.pack('<5i',0x31455653,6,3,2,4)+struct.pack('<24f',float('nan'),*range(23)))]:
            with self.subTest(label=label):
                if payload is not None:self.payload.write_bytes(payload)
                paths={} if payload is None else self.paths;history=self.history
                if label=='nonfinite':
                    digest=hashlib.sha256(payload).hexdigest();paths={digest:self.payload};history=InputHistory(spans(embedding_sha256=digest))
                with self.assertRaisesRegex(FlowError,'nonfinite' if label=='nonfinite' else 'embedding'):
                    self.packet(history=history,label=label,paths=paths)
                self.assertFalse((self.root/(label+'.manifest')).exists())
                self.assertFalse((self.root/(label+'-positions.i32')).exists())
    def test_partial_image_boundary_and_unsafe_label_are_rejected(self):
        for start,label in [(3,'partial'),(True,'boolean'),(0,'../escape')]:
            with self.subTest(start=start,label=label),self.assertRaises(FlowError):self.packet(start=start,label=label)
        with self.assertRaises(FlowError):self.packet(InputHistory(self.inputs[:-1]))

    def test_matching_checksum_cannot_hide_wrong_sve_shape(self):
        raw=self.payload.read_bytes();raw=struct.pack('<5i',0x31455653,6,2,3,4)+raw[20:]
        self.payload.write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
        history=InputHistory(spans(embedding_sha256=digest))
        with self.assertRaisesRegex(FlowError,'shape'):
            self.packet(history,paths={digest:self.payload})
    def test_native_protocol_limit_rejects_more_than_eight_new_images(self):
        history=InputHistory(self.inputs*9)
        with self.assertRaisesRegex(FlowError,'eight'):
            reference_ingest(history,0,self.paths,128,self.root,'too-many')

    def test_user_boundary_and_ordered_originals_cannot_be_spoofed(self):
        from run_thinking_suffix_flow import validate_suffix_images
        validate_suffix_images(self.history,['a'*64],(1,2))
        for originals,header in [(['b'*64],(1,2)),(['a'*64],(9,9)),(['a'*64],())]:
            with self.subTest(originals=originals,header=header),self.assertRaises(FlowError):
                validate_suffix_images(self.history,originals,header)
    def test_image_observer_requires_one_real_entrypoint_and_parses(self):
        import ast
        from run_thinking_suffix_flow import install_image_observer
        entry="importlib.import_module('serve.cow_http').main()\n"
        ast.parse(install_image_observer(entry,self.root))
        for source in ('',entry+entry):
            with self.assertRaises(FlowError):install_image_observer(source,self.root)

    def test_question_command_requires_the_actual_controller_phase(self):
        from run_thinking_suffix_flow import question_command
        self.assertEqual(question_command('idle',False),'ask')
        self.assertEqual(question_command('stopped',True),'append-user')
        for status,append in [('idle',True),('stopped',False),('active',False),('active',True)]:
            with self.subTest(status=status,append=append),self.assertRaises(FlowError):question_command(status,append)

if __name__=='__main__':unittest.main()
