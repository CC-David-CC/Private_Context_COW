"""Compile the actual mixed V4 handler; CPU protocol/math evidence, not GPU."""
import inspect
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest

import build_min_p_selector_probe as retained
from context_time_travel import min_p
from test_vision_dry_native import reference

ROOT=Path(__file__).resolve().parents[2]
SCORES=[0.,1.,5.,-2.,-9.,-9.,-9.,-9.]


class VisionDryEntropyNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_vision_dry_combinations_probe as builder
        cls.builder=builder
        matches=re.findall(r"command = r'''(.*?)'''",inspect.getsource(retained.instrument_min_p_selector_v4),re.DOTALL)
        if len(matches)!=1:raise RuntimeError('Expected one retained V4 handler template')
        cls.original=matches[0];command=builder.command_v4(cls.original)
        start=command.index('} else if (command == "sample_mask_image_v4") {')
        end=command.rindex('} else if (command == "generate") {')
        body=command[command.index('{',start)+1:end]
        cls.temp=tempfile.TemporaryDirectory(prefix='vision-dry-entropy-');cls.addClassCleanup(cls.temp.cleanup)
        root=Path(cls.temp.name);source=root/'handler.cpp';cls.binary=root/'handler'
        source.write_text(r'''#define MIN_P_SELECT_ENABLE_V4
#include "min_p_select.h"
#include "vision_dry.h"
#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
struct Placement { int64_t start, rows; };
struct TestSession {
 std::vector<int32_t> values{0,1,7,7,2,0,1};
 std::vector<Placement> images{{2,2}};
 std::vector<float> scores{0,1,5,-2,-9,-9,-9,-9};
 const auto& ids() const {return values;}
 const auto& input_images() const {return images;}
 const auto& logits() const {return scores;}
} session;
struct TestForward {size_t vocab() const {return 8;}} forward;
uint64_t sampling_ns=0,sampling_v4_ns=0,sampling_calls=0,sampling_v4_calls=0;
uint64_t sampling_logit_bytes=0,sampling_v4_logit_bytes=0,sampling_mask_bytes=0,sampling_v4_mask_bytes=0;
uint64_t typical_filter_calls=0,dynatemp_filter_calls=0,dry_penalized_candidates_v4=0;
uint64_t dry_penalized_candidates=0;
uint64_t sampling_image_dry_combination_calls=0;
void error_reply(const std::string& message){std::cerr<<"ERROR "<<message<<"\n";}
int main(int argc,char**argv){
 if(argc>1 && std::string(argv[1])=="penalty"){
  session.values={7,7,0,1,2,0,1};session.images={{0,2}};
 }
 std::string line;
 while(std::getline(std::cin,line)){
  std::istringstream in(line);std::string command,extra;in>>command;
  if(command=="sample_mask_image_v4"){
'''+body+r'''
  }else{error_reply("unexpected command");}
 }
 std::cerr<<"COUNTERS "<<sampling_calls<<" "<<sampling_v4_calls<<" "<<sampling_image_dry_combination_calls<<"\n";
}''',newline='\n')
        built=subprocess.run(['g++','-std=c++17','-O2','-fno-fast-math','-I',str(ROOT/'native'),str(source),'-o',str(cls.binary)],capture_output=True,text=True)
        if built.returncode:raise RuntimeError(built.stderr)

    def invoke(self,*,penalty=False,history=(0,1,2,0,1),draw=0,last_n=256):
        with tempfile.TemporaryDirectory(prefix='vision-dry-files-') as temp:
            root=Path(temp);mask=root/'mask';tail=root/'history';breakers=root/'breakers'
            mask.write_bytes(b'\x0f');tail.write_bytes(struct.pack('<'+'I'*len(history),*history));breakers.write_bytes(b'')
            fields=[7,.7,.05,0,1.,draw,json.dumps(str(mask)),4.,2.,1,last_n,json.dumps(str(tail)),json.dumps(str(breakers)),.95,.35,1.]
            run=subprocess.run([str(self.binary),'penalty' if penalty else 'barrier'],input='sample_mask_image_v4 '+' '.join(map(str,fields))+'\n',text=True,capture_output=True)
            self.assertEqual(run.returncode,0,run.stderr);return run

    def test_actual_text_projection_and_barrier_receipt(self):
        result=self.invoke();row=json.loads(result.stdout)
        self.assertEqual((row['consumed'],row['text_tokens'],row['dry_history_count']),(7,5,5))
        self.assertEqual(row['dry_boundaries'],[2]);self.assertEqual(row['dry_boundary_count'],1)
        self.assertEqual(row['vision_dry_sampling_protocol'],self.builder.PROTOCOL)
        self.assertEqual(row['protocol'],min_p.SAMPLING_PROTOCOL_V4)
        self.assertEqual(result.stderr.strip(),'COUNTERS 1 1 1')

    def test_current_or_prior_match_cannot_cross_image_boundary(self):
        for draw in (0,1<<52,(1<<53)-1):
            with self.subTest(draw=draw):
                row=json.loads(self.invoke(draw=draw).stdout)
                adjusted=reference(SCORES,[0,1,2,0,1],[],[2],1,4.,2.)
                self.assertEqual(adjusted,SCORES);self.assertEqual(row['dry_penalized_candidates'],0)
                expected=min_p.choose(adjusted,b'\x0f',temperature=.7,min_p=.05,top_k=0,top_p=1.,
                    typical_p=.95,dynatemp_range=.35,dynatemp_exponent=1.,draw53=draw)
                for key in ('token','kept_count','typical_kept_count','effective_temperature'):
                    self.assertEqual(row[key],expected[key])

    def test_actual_prior_text_run_receives_penalty_before_entropy_filters(self):
        for draw in (0,1<<52,(1<<53)-1):
            with self.subTest(draw=draw):
                row=json.loads(self.invoke(penalty=True,draw=draw).stdout)
                adjusted=reference(SCORES,[0,1,2,0,1],[],[0],1,4.,2.)
                self.assertLess(adjusted[2],SCORES[2]);self.assertGreater(row['dry_penalized_candidates'],0)
                expected=min_p.choose(adjusted,b'\x0f',temperature=.7,min_p=.05,top_k=0,top_p=1.,
                    typical_p=.95,dynatemp_range=.35,dynatemp_exponent=1.,draw53=draw)
                for key in ('token','kept_count','typical_kept_count','effective_temperature'):
                    self.assertEqual(row[key],expected[key])

    def test_model_cell_tail_is_rejected_without_sampling(self):
        result=self.invoke(history=(0,1,7,7,2,0,1))
        self.assertEqual(result.stdout,'');self.assertIn('missing or malformed',result.stderr)
        self.assertTrue(result.stderr.rstrip().endswith('COUNTERS 0 0 0'))

    def test_same_size_forged_text_tail_is_rejected_without_sampling(self):
        result=self.invoke(history=(0,1,2,0,0))
        self.assertEqual(result.stdout,'');self.assertIn('native session tail',result.stderr)
        self.assertTrue(result.stderr.rstrip().endswith('COUNTERS 0 0 0'))

    def test_unexpected_template_is_refused(self):
        with self.assertRaises(ValueError):self.builder.command_v4('not the retained command')


if __name__=='__main__':unittest.main()
