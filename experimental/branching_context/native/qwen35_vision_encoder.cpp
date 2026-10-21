// Adapter for the existing COW ENC/SVE1 boundary, using the pinned mtmd API.
// This process loads decoder metadata with no_alloc, not decoder weights.
// The projector itself is fully allocated and executes the real image encoder.
// No decoder context is created and no text prefix is evaluated.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

int main(int argc, char ** argv) {
 try {
  std::string model_path, projector_path, flash="on"; int threads=0, max_tokens=0; bool gpu=false;
  for (int i=1;i<argc;++i) {
   std::string arg=argv[i];
   if(arg=="--gpu") { gpu=true;continue; }
   if(i+1>=argc) throw std::runtime_error("Missing option value");
   std::string value=argv[++i];
   if(arg=="--model") model_path=value;
   else if(arg=="--mmproj") projector_path=value;
   else if(arg=="--threads") threads=std::stoi(value);
   else if(arg=="--max-tokens") max_tokens=std::stoi(value);
   else if(arg=="--flash-attn") flash=value;
   else throw std::runtime_error("Unknown option");
  }
  if(model_path.empty()||projector_path.empty()||threads<=0||max_tokens<=0||max_tokens>1024||
       (flash!="on"&&flash!="off"&&flash!="auto")) throw std::runtime_error("Invalid encoder configuration");
  uint32_t endian=1; if(*reinterpret_cast<unsigned char *>(&endian)!=1) throw std::runtime_error("SVE1 requires little-endian host");
  static_assert(sizeof(float)==4,"SVE1 float width");
  llama_backend_init();
  auto mp=llama_model_default_params();mp.no_alloc=true;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.n_gpu_layers=0;
  std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(model_path.c_str(),mp),llama_model_free);
  if(!model) throw std::runtime_error("Cannot load decoder metadata");
  char arch[64]={};llama_model_meta_val_str(model.get(),"general.architecture",arch,sizeof(arch));
  int width=llama_model_n_embd_inp(model.get());
  if(std::string(arch)!="qwen35"||width!=4096) throw std::runtime_error("Only the tested Qwen35 4096-wide model is supported");
  auto params=mtmd_context_params_default();params.use_gpu=gpu;params.n_threads=threads;
  params.warmup=false;params.print_timings=false;params.image_min_tokens=1;params.image_max_tokens=max_tokens;
  params.flash_attn_type=flash=="on"?LLAMA_FLASH_ATTN_TYPE_ENABLED:flash=="off"?LLAMA_FLASH_ATTN_TYPE_DISABLED:LLAMA_FLASH_ATTN_TYPE_AUTO;
  std::unique_ptr<mtmd_context,decltype(&mtmd_free)> encoder(mtmd_init_from_file(projector_path.c_str(),model.get(),params),mtmd_free);
  if(!encoder||!mtmd_decode_use_mrope(encoder.get())) throw std::runtime_error("Cannot load compatible M-RoPE image projector");
  std::cout<<"READY "<<width<<std::endl;
  std::string line;
  while(std::getline(std::cin,line)) {
   if(line=="QUIT") break;
   std::istringstream in(line);std::string command,input,output,extra;
   if(!(in>>command>>input>>output)||command!="ENC"||(in>>extra)) throw std::runtime_error("Invalid ENC request");
   auto started=std::chrono::steady_clock::now();
   auto loaded=mtmd_helper_bitmap_init_from_file(encoder.get(),input.c_str(),false,mtmd_helper_init_opt_default());
   std::unique_ptr<mtmd_bitmap,decltype(&mtmd_bitmap_free)> bitmap(loaded.bitmap,mtmd_bitmap_free);
   if(!bitmap||loaded.video_ctx||mtmd_bitmap_is_audio(bitmap.get())) throw std::runtime_error("Expected one still image");
   std::unique_ptr<mtmd_input_chunks,decltype(&mtmd_input_chunks_free)> chunks(mtmd_input_chunks_init(),mtmd_input_chunks_free);
   std::string marker=mtmd_default_marker();mtmd_input_text text{marker.data(),marker.size(),false,true};
   const mtmd_bitmap * bitmaps[]={bitmap.get()};
   if(mtmd_tokenize(encoder.get(),chunks.get(),&text,bitmaps,1)!=0) throw std::runtime_error("Image preprocessing failed");
   const mtmd_input_chunk * image=nullptr;
   for(size_t i=0;i<mtmd_input_chunks_size(chunks.get());++i) {
    auto chunk=mtmd_input_chunks_get(chunks.get(),i);
    if(mtmd_input_chunk_get_type(chunk)==MTMD_INPUT_CHUNK_TYPE_IMAGE) {
     if(image) throw std::runtime_error("Multiple image chunks are unsupported");image=chunk;
    } else if(mtmd_input_chunk_get_type(chunk)!=MTMD_INPUT_CHUNK_TYPE_TEXT) throw std::runtime_error("Unexpected media chunk");
   }
   if(!image) throw std::runtime_error("Missing image chunk");
   const auto tokens=mtmd_input_chunk_get_tokens_image(image);int rows=mtmd_input_chunk_get_n_tokens(image);
   if(rows<=0||rows>max_tokens) throw std::runtime_error("Unsupported image row count");
   auto last=mtmd_image_tokens_get_decoder_pos(tokens,17,rows-1);
   int nx=int(last.x)-17+1,ny=int(last.y)-17+1;
   if(nx<=0||ny<=0||int64_t(nx)*ny!=rows) throw std::runtime_error("Unsupported rectangular image geometry");
   // Verify this encoder's grid against COW's existing three-axis convention,
   // rather than assuming every multimodal projector uses the same positions.
   for(int i=0;i<rows;++i) {
    auto pos=mtmd_image_tokens_get_decoder_pos(tokens,17,i);
    if(pos.t!=17||pos.y!=uint32_t(17+i/nx)||pos.x!=uint32_t(17+i%nx)||pos.z!=0)
     throw std::runtime_error("Image positions differ from the COW grid convention");
   }
   if(mtmd_input_chunk_get_n_pos(image)!=std::max(nx,ny)) throw std::runtime_error("Unexpected image temporal extent");
   if(mtmd_encode_chunk(encoder.get(),image)!=0) throw std::runtime_error("Image encoding failed");
   const float * values=mtmd_get_output_embd(encoder.get());if(!values) throw std::runtime_error("Missing projected embeddings");
   for(size_t i=0;i<size_t(rows)*width;++i) if(!std::isfinite(values[i])) throw std::runtime_error("Non-finite image embedding");
   const int32_t header[]={0x31455653,rows,nx,ny,width};std::ofstream out(output,std::ios::binary|std::ios::trunc);
   out.write(reinterpret_cast<const char *>(header),sizeof(header));out.write(reinterpret_cast<const char *>(values),size_t(rows)*width*sizeof(float));out.close();
   if(!out) throw std::runtime_error("Cannot write complete SVE1 image");
   double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
   std::cout<<"OK "<<rows<<" "<<nx<<" "<<ny<<" "<<ms<<std::endl;
  }
  return 0;
 } catch(const std::exception & e) {std::cerr<<"Qwen35 encoder: "<<e.what()<<std::endl;return 1;}
}
