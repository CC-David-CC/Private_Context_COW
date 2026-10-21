"""Model-selection policy fixtures; real inference is measured separately."""
from pathlib import Path
import sys,json,tempfile,unittest
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import cow_prepare_model as prepare
import cow_easy_setup as setup

def fixture_pack(stage):
    for name in ('dense.bin','index.txt','native_experts.txt','tokenizer/vocab.json'):
        p=stage/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'unit fixture only')

class ModelSelectionTests(unittest.TestCase):
    def test_q4_default_assets_stay_identical(self):
        self.assertEqual(prepare.model_plan(),prepare.model_plan('UD-Q4_K_XL'))
        self.assertEqual(len(prepare.model_plan()),4)
        self.assertEqual(prepare.pack_name('UD-Q4_K_XL'),prepare.PACK_NAME)
    def test_iq3_uses_pinned_two_shard_gsq_model(self):
        assets=prepare.model_plan('IQ3_XXS')
        self.assertEqual(len(assets),2);self.assertEqual(sum(a.size for a in assets),75839998528)
        self.assertTrue(all('ISTA-DASLab' in a.url and 'GSQ-RCO' in a.name and 'ed59f92082b1e93c0e96d60a8b11aab089b52f09' in a.url for a in assets))
        self.assertTrue(all(len(a.sha256)==64 for a in assets))
    def test_iq2_uses_pinned_two_shard_gsq_model(self):
        assets=prepare.model_plan('IQ2_XS')
        self.assertEqual(len(assets),2);self.assertEqual(sum(a.size for a in assets),68026093024)
        self.assertTrue(all('IQ2_XS' in a.name for a in assets))
    def test_unsupported_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unsupported'):prepare.model_plan('../other')
    def test_model_pack_paths_are_distinct(self):
        self.assertEqual(len({prepare.pack_name(m) for m in prepare.MODEL_CHOICES}),3)
        for model in prepare.MODEL_CHOICES:self.assertEqual(Path(prepare.pack_name(model)).name,prepare.pack_name(model))
    def test_distinct_model_packs_coexist_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(prepare,'recipe',return_value={'unit':'recipe'}):
            root=Path(tmp)
            def build(stage):fixture_pack(stage)
            a=prepare.publish_pack(root,prepare.model_plan('IQ3_XXS'),build,model='IQ3_XXS')
            before=(a/prepare.RECEIPT).read_bytes()
            b=prepare.publish_pack(root,prepare.model_plan('IQ2_XS'),build,model='IQ2_XS')
            self.assertNotEqual(a,b);self.assertEqual((a/prepare.RECEIPT).read_bytes(),before)
    def test_wrong_shard_receipt_cannot_be_reused_for_other_quant(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(prepare,'recipe',return_value={'unit':'recipe'}):
            root=Path(tmp)
            def build(stage):fixture_pack(stage)
            a=prepare.publish_pack(root,prepare.model_plan('IQ3_XXS'),build,model='IQ3_XXS');a.rename(root/prepare.pack_name('IQ2_XS'))
            with self.assertRaisesRegex(ValueError,'changed|recipe'):
                prepare.publish_pack(root,prepare.model_plan('IQ2_XS'),build,model='IQ2_XS')
    def config(self,root):
        (root/'server').mkdir();profile=root/'profile.json';key=root/'key';key.write_text('fixture key')
        profile.write_text(json.dumps({'model_path':'/models/'+prepare.model_plan()[0].name}))
        config=root/'server/server.json';config.write_text(json.dumps({'profile':str(profile),'api_key_file':str(key)}));return config
    def test_explicit_different_quant_rejected_before_installing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.config(root);before=config.read_bytes()
            with patch.object(setup,'inspect_host',side_effect=AssertionError('must preserve configured host')):
                with self.assertRaisesRegex(setup.SetupError,'different model'):
                    setup.main(['--data-dir',str(root),'--model','IQ2_XS'])
            self.assertEqual(config.read_bytes(),before)
    def test_repeating_setup_without_model_keeps_existing_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=self.config(root);before=config.read_bytes()
            self.assertEqual(setup.main(['--data-dir',str(root)]),0)
            self.assertEqual(config.read_bytes(),before)
    def test_guided_setup_forwards_model_to_packer_and_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new';calls=[]
            h=setup.Hardware(32*1024**3,24*1024**3,24*1024**3,24*1024**3,12,'Fixture GPU','/cuda/nvcc')
            with patch.object(setup,'inspect_host',return_value=h),patch.object(setup.shutil,'disk_usage',return_value=__import__('collections').namedtuple('Disk','total used free')(1024**4,0,1024**4)),patch.object(setup,'run_step',side_effect=lambda label,command,*args:calls.append(command)):
                self.assertEqual(setup.main(['--data-dir',str(root),'--model','IQ2_XS','--yes']),0)
            packer=next(c for c in calls if any(str(a).endswith('cow_prepare_model.py') for a in c))
            self.assertEqual(packer[packer.index('--model')+1],'IQ2_XS')
            server=next(c for c in calls if any(str(a).endswith('cow_server_setup.py') for a in c))
            self.assertEqual(server[server.index('--pack')+1],str(root/prepare.pack_name('IQ2_XS')))
            self.assertEqual(Path(server[server.index('--model')+1]).name,prepare.model_plan('IQ2_XS')[0].name)

if __name__=='__main__':unittest.main(verbosity=2)
