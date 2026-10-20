"""Compile/run the segmented DRY helper on CPU; not GPU qualification.

The independent bounded scan treats image positions as sequence boundaries,
without inventing vocabulary IDs or changing the frozen text DRY reference.
"""
import math
import random
import struct
import subprocess
import tempfile
from pathlib import Path
import unittest

HEADER=Path(__file__).resolve().parents[2]/'native'/'min_p_select.h'
CPP=r'''#include "min_p_select.h"
#include <iostream>
#include <iomanip>
int main() {
    size_t n,h,b,c; double mult,base; unsigned allowed,last_n;
    if (!(std::cin>>n>>h>>b>>c>>mult>>base>>allowed>>last_n)) return 2;
    std::vector<float> scores(n);std::vector<uint32_t> history(h),breakers(b);
    std::vector<size_t> boundaries(c);
    for(auto &v:scores)if(!(std::cin>>v))return 3;
    for(auto &v:history)if(!(std::cin>>v))return 4;
    for(auto &v:breakers)if(!(std::cin>>v))return 5;
    for(auto &v:boundaries)if(!(std::cin>>v))return 6;
    try {
        auto before=scores;size_t affected=0;
        auto out=min_p_select::apply_dry_penalties(scores.data(),n,history,breakers,
            min_p_select::DryConfig{mult,base,allowed,last_n},&affected,&boundaries);
        if(scores!=before)return 7;
        std::cout<<std::setprecision(17);
        for(auto v:out)std::cout<<v<<" ";
        std::cout<<"\nCOUNT "<<affected<<"\n";
    }catch(const std::exception &e){std::cout<<"ERROR "<<e.what()<<"\n";}
}
'''


def reference(scores,history,breakers,boundaries,allowed,mult,base):
    """Brute-force each prior run against the current run's suffix."""
    out=list(scores);best={};start=0
    current_start=max(boundaries,default=0)
    for index,token in enumerate(history):
        if token in breakers:current_start=max(current_start,index+1)
    for index,token in enumerate(history):
        if index in boundaries:start=index
        if index==start or token in breakers or (boundaries and boundaries[-1]==len(history)):continue
        longest=0
        for count in range(1,min(index-start,len(history)-current_start)+1):
            if history[index-count:index]==history[len(history)-count:]:longest=count
        if longest>=allowed:best[token]=max(best.get(token,0),longest)
    for token,length in best.items():
        penalty=min(mult*math.pow(base,length-allowed),3.4028234663852886e38)
        out[token]=struct.unpack('<f',struct.pack('<f',max(scores[token]-penalty,-3.4028234663852886e38)))[0]
    return out


class VisionDryNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();root=Path(cls.temp.name)
        cls.addClassCleanup(cls.temp.cleanup)
        cpp=root/'segmented.cpp';cpp.write_text(CPP);cls.binary=root/'segmented'
        subprocess.run(['g++','-std=c++17','-O2','-I',str(HEADER.parent),str(cpp),'-o',str(cls.binary)],check=True)

    def run_case(self,history,boundaries=(),*,breakers=(),allowed=2,mult=1.,base=1.75):
        scores=[4.]*8
        fields=[8,len(history),len(breakers),len(boundaries),mult,base,allowed,64]
        raw=' '.join(map(str,fields))+'\n'+' '.join(map(str,scores+list(history)+list(breakers)+list(boundaries)))+'\n'
        result=subprocess.run([str(self.binary)],input=raw,text=True,capture_output=True,check=True).stdout
        return result,scores

    def test_current_suffix_cannot_match_through_an_image(self):
        output,_=self.run_case([0,1,2,0,1],[4])
        self.assertEqual(list(map(float,output.splitlines()[0].split())),[4.]*8)

    def test_prior_occurrence_cannot_cross_an_image(self):
        output,_=self.run_case([0,1,2,0,1],[1,3])
        self.assertEqual(list(map(float,output.splitlines()[0].split())),[4.]*8)

    def test_valid_prior_run_is_still_searchable_across_an_image(self):
        history=[0,1,2,0,1];output,scores=self.run_case(history,[3])
        self.assertEqual(list(map(float,output.splitlines()[0].split())),reference(scores,history,(),[3],2,1.,1.75))
        self.assertEqual(float(output.splitlines()[0].split()[2]),3.)

    def test_empty_current_text_and_zero_allowed_do_not_bridge_an_image(self):
        output,_=self.run_case([0,1,2],[3],allowed=0)
        self.assertEqual(list(map(float,output.splitlines()[0].split())),[4.]*8)

    def test_invalid_boundaries_are_rejected(self):
        for boundaries in ([4],[2,1],[1,1]):
            with self.subTest(boundaries=boundaries):
                output,_=self.run_case([0,1,2],boundaries)
                self.assertTrue(output.startswith('ERROR '),output)

    def test_randomized_runs_match_independent_scan(self):
        rng=random.Random(1938)
        for index in range(160):
            history=[rng.randrange(7) for _ in range(rng.randrange(1,40))]
            boundaries=sorted(rng.sample(range(len(history)+1),rng.randrange(min(5,len(history))+1)))
            breakers=sorted(set(history)&{6});allowed=rng.randrange(4)
            output,scores=self.run_case(history,boundaries,breakers=breakers,allowed=allowed)
            expected=reference(scores,history,breakers,boundaries,allowed,1.,1.75)
            actual=list(map(float,output.splitlines()[0].split()))
            self.assertEqual(actual,expected,(index,history,boundaries))

    def test_no_images_preserves_the_frozen_text_reference(self):
        from context_time_travel.dry import penalties_from_ids
        rng=random.Random(1515)
        for _ in range(80):
            history=[rng.randrange(7) for _ in range(rng.randrange(1,40))]
            breakers=sorted(set(history)&{6});allowed=rng.randrange(4)
            output,scores=self.run_case(history,breakers=breakers,allowed=allowed)
            penalties=penalties_from_ids(history,breakers,dry_multiplier=1.,dry_base=1.75,
                dry_allowed_length=allowed,dry_penalty_last_n=64)
            expected=[struct.unpack('<f',struct.pack('<f',score-penalties.get(token,0.)))[0]
                for token,score in enumerate(scores)]
            self.assertEqual(list(map(float,output.splitlines()[0].split())),expected)


if __name__=='__main__':unittest.main()
