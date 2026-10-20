"""Native bounded text projection checked against a separate cell-walk oracle."""
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

NATIVE=Path(__file__).resolve().parents[2]/'native'
CPP=r'''#include "vision_dry.h"
#include <iostream>
struct Image{int64_t start,rows;};
int main(){
    size_t n,k,last;if(!(std::cin>>n>>k>>last))return 2;
    std::vector<int32_t> ids(n);std::vector<Image> images(k);
    for(auto &v:ids)if(!(std::cin>>v))return 3;
    for(auto &v:images)if(!(std::cin>>v.start>>v.rows))return 4;
    try{
        auto window=cow_vision_dry::project(ids,images,last);
        std::cout<<window.text_count<<"\n";
        for(auto v:window.tokens)std::cout<<v<<" ";std::cout<<"\n";
        for(auto v:window.boundaries)std::cout<<v<<" ";std::cout<<"\n";
    }catch(const std::exception&e){std::cout<<"ERROR "<<e.what()<<"\n";}
}'''


def reference(ids,images,last):
    text=[];boundaries=[];covered=set()
    for start,count in images:covered.update(range(start,start+count))
    starts={start for start,count in images}
    for position,token in enumerate(ids):
        if position in starts:boundaries.append(len(text))
        if position not in covered:text.append(token)
    offset=max(0,len(text)-last)
    return (len(text),text[-last:] if last else [],
        sorted({v-offset for v in boundaries if v>=offset}) if last else [])


class VisionDryProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        root=Path(cls.temp.name);cpp=root/'project.cpp';cpp.write_text(CPP);cls.binary=root/'project'
        subprocess.run(['g++','-std=c++17','-O2','-I',str(NATIVE),str(cpp),'-o',str(cls.binary)],check=True)

    def project(self,ids,images,last):
        raw=f'{len(ids)} {len(images)} {last}\n'+' '.join(map(str,ids))+'\n'
        raw+=' '.join(str(v) for image in images for v in image)+'\n'
        out=subprocess.run([str(self.binary)],input=raw,text=True,capture_output=True,check=True).stdout
        if out.startswith('ERROR '):return out
        lines=out.splitlines();return int(lines[0]),list(map(int,lines[1].split())),list(map(int,lines[2].split()))

    def test_literal_pad_id_remains_text_outside_recorded_image_rows(self):
        self.assertEqual(self.project([7,7,7,7,3],[(1,2)],4),(3,[7,7,3],[1]))

    def test_tail_cut_and_adjacent_images_keep_exact_unique_boundaries(self):
        ids=list(range(20));images=[(3,4),(7,2),(14,3)]
        for last in (0,1,3,8,64):
            with self.subTest(last=last):self.assertEqual(self.project(ids,images,last),reference(ids,images,last))

    def test_leading_trailing_images_and_image_only_prefix(self):
        for ids,images in [([7,7,1,2,7],[(0,2),(4,1)]),([7,7],[(0,2)])]:
            with self.subTest(images=images):self.assertEqual(self.project(ids,images,64),reference(ids,images,64))

    def test_invalid_or_overlapping_placements_are_rejected(self):
        for images in ([(-1,1)],[(1,0)],[(2,2)],[(0,2),(1,1)],[(2,1),(0,1)]):
            with self.subTest(images=images):self.assertTrue(self.project([0,1,2],images,64).startswith('ERROR '))

    def test_randomized_projection_matches_cell_walk(self):
        rng=random.Random(1962)
        for index in range(160):
            ids=[rng.randrange(8) for _ in range(rng.randrange(1,100))];images=[];position=0
            while position<len(ids):
                if rng.randrange(4)==0:
                    count=rng.randrange(1,min(9,len(ids)-position)+1);images.append((position,count));position+=count
                else:position+=1
            last=rng.randrange(100)
            self.assertEqual(self.project(ids,images,last),reference(ids,images,last),(index,images,last))


if __name__=='__main__':unittest.main()
