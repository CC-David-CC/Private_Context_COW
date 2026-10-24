// PCC native ordinary-backprop training. No Python tensor/autograd runtime.
// The differentiable graph is explicit: unsupported gradient paths fail closed.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <fcntl.h>
#include <execinfo.h>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
using json=nlohmann::json;
static void need(bool ok,const std::string &why){if(!ok)throw std::runtime_error(why);}
using T=ggml_tensor;
static ggml_backend_t(*cuda_init)(int)=nullptr;
static void load_cuda(){
    if(cuda_init)return;
    need(std::strlen(PCC_CUDA_BACKEND)>0,"This CPU-only build cannot execute CUDA requests");
    // The inspected FP32 profile bypasses inference MMF/TF32 operand rounding.
    // Its actual binary is fingerprinted by build.py and qualified by VJPs.
    setenv("GGML_CUDA_F32_ACTIVATIONS","1",1);
    setenv("NVIDIA_TF32_OVERRIDE","0",1);
    // The pinned engine builds CUDA as a directly linked backend rather than
    // a GGML_BACKEND_DL plugin. Use its exported native initializer explicitly.
    auto*library=dlopen(PCC_CUDA_BACKEND,RTLD_NOW|RTLD_GLOBAL);need(library,"Cannot open pinned native CUDA backend");
    cuda_init=reinterpret_cast<ggml_backend_t(*)(int)>(dlsym(library,"ggml_backend_cuda_init"));
    need(cuda_init,"Pinned backend does not export native CUDA initialization");
}
static std::vector<float> read_tensor(T *t){
    need(t&&t->type==GGML_TYPE_F32,"Expected F32 tensor");
    need(t->buffer||(t->view_src&&t->view_src->buffer),"Unallocated result "+std::string(t->name)+" op="+ggml_op_name(t->op));
    std::vector<float> v(ggml_nelements(t));ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;
}
static void put(T*t,const std::vector<float>&v){need(size_t(ggml_nelements(t))==v.size(),"Tensor size mismatch");ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}
struct Context {
    ggml_context *p;
    explicit Context(size_t bytes):p(ggml_init({bytes,nullptr,true})){need(p,"Metadata allocation failed");}
    ~Context(){ggml_free(p);}
    Context(const Context&)=delete;Context&operator=(const Context&)=delete;
};
struct Device {
    ggml_backend_t backend=nullptr;
    std::string name;
    bool allocation_guard=false;
    size_t fixed_bytes=0,scratch_bytes=0,reserve_bytes=0,budget_bytes=0,host_reserve_bytes=0;
    size_t peak_planned_graph=0,checked_graphs=0;
    static size_t host_available(){
        std::ifstream in("/proc/meminfo");std::string line;
        while(std::getline(in,line))if(line.rfind("MemAvailable:",0)==0)return std::stoull(line.substr(13))*1024;
        throw std::runtime_error("Cannot establish current host memory");
    }
    static size_t room(size_t total,size_t charged){return total>charged?total-charged:0;}
    size_t graph_limit(){
        size_t free=0,total=0;ggml_backend_dev_memory(ggml_backend_get_device(backend),&free,&total);
        if(name=="CPU")free=host_available();
        return std::min(room(free,scratch_bytes+reserve_bytes),room(budget_bytes,fixed_bytes+scratch_bytes+reserve_bytes));
    }
    void observed_graph(size_t bytes,size_t limit){
        ++checked_graphs;peak_planned_graph=std::max(peak_planned_graph,bytes);
        std::cerr<<"GRAPH_ALLOCATION_PLAN "<<json({{"device",name},{"buffer_bytes",bytes},{"limit_bytes",limit},{"admitted",bytes<=limit}}).dump()<<std::endl;
    }
    explicit Device(const std::string &n):name(n){
        if(n=="CPU")backend=ggml_backend_cpu_init();
        else if(n.rfind("CUDA",0)==0){load_cuda();auto id=n.substr(4);need(!id.empty()&&std::all_of(id.begin(),id.end(),::isdigit)&&id==std::to_string(std::stoi(id)),"Canonical explicit CUDA device index required");backend=cuda_init(std::stoi(id));}
        else throw std::runtime_error("Unsupported training backend "+n);
        need(backend,"Missing native backend "+n);
        if(n=="CPU")ggml_backend_cpu_set_n_threads(backend,8);
    }
    ~Device(){ggml_backend_free(backend);}
};
struct Parameter {T*w=nullptr,*grad=nullptr,*a=nullptr,*b=nullptr,*da=nullptr,*db=nullptr;};
struct Block {
    Device &device;Context ctx{8*1024*1024};
    std::map<std::string,Parameter> params;
    ggml_backend_buffer_t buffer=nullptr;
    bool full;int rank;
    Block(Device&d,bool f,int r):device(d),full(f),rank(r){}
    ~Block(){ggml_backend_buffer_free(buffer);}
    T *create(const std::string &name,int64_t n0,int64_t n1=1){
        auto*t=ggml_new_tensor_2d(ctx.p,GGML_TYPE_F32,n0,n1);ggml_set_name(t,name.c_str());return t;
    }
    void add(const std::string &name,int64_t n0,int64_t n1,bool linear){
        auto&p=params[name];p.w=create(name,n0,n1);
        if(full){ggml_set_param(p.w);p.grad=create(name+".gradient",n0,n1);}
        else if(linear){
            p.a=create(name+".lora_a",n0,rank);p.b=create(name+".lora_b",rank,n1);
            ggml_set_param(p.a);ggml_set_param(p.b);
            p.da=create(name+".lora_a.gradient",n0,rank);p.db=create(name+".lora_b.gradient",rank,n1);
        }
    }
    void allocate(){
        buffer=ggml_backend_alloc_ctx_tensors(ctx.p,device.backend);need(buffer,"Parameter allocation failed on "+device.name);
        ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    std::vector<std::pair<T*,T*>> trainable(){
        std::vector<std::pair<T*,T*>> out;
        for(auto &[n,p]:params){if(p.grad)out.push_back({p.w,p.grad});if(p.a){out.push_back({p.a,p.da});out.push_back({p.b,p.db});}}
        return out;
    }
    void zero(){for(auto[p,g]:trainable())ggml_backend_tensor_memset(g,0,0,ggml_nbytes(g));}
    void initialize_adapters(){
        std::mt19937 rng(20261023);std::normal_distribution<float>normal(0,.01f);
        for(auto &[n,p]:params)if(p.a){std::vector<float>a(ggml_nelements(p.a));for(auto&v:a)v=normal(rng);put(p.a,a);put(p.b,std::vector<float>(ggml_nelements(p.b),0));}
        zero();
    }
};
struct Graph {
    // Mixed 32-token recurrent examples can exceed 65,536 nodes after backward
    // expansion. This bounds graph bookkeeping; tensor operations are unchanged.
    static constexpr size_t default_capacity=262144;
    size_t capacity;bool efficient_sequence;
    Device &device;Context ctx;ggml_cgraph*g;
    static size_t bounded_capacity(size_t tokens,size_t layers){
        need((tokens<=32768&&layers<=2)||(tokens<=512&&layers<=32),"Bounded sequence graph limits exceeded");
        size_t wanted=tokens*layers*256+65536,result=default_capacity;
        while(result<wanted)result*=2;
        return result;
    }
    static size_t metadata_bytes(size_t nodes){
        if(nodes==default_capacity)return 128*1024*1024;
        return ggml_graph_overhead_custom(nodes,true)+2*nodes*ggml_tensor_overhead()+16*1024*1024;
    }
    std::vector<Device*> devices;
    std::map<T*,Device*> placements;
    std::unique_ptr<Device>scheduler_cpu;
    ggml_backend_sched_t sched=nullptr;
    ggml_gallocr_t direct_allocator=nullptr;
    std::vector<std::pair<T*,std::vector<float>>> data;
    std::vector<std::pair<T*,std::vector<int32_t>>> ids;
    explicit Graph(Device&d):Graph(std::vector<Device*>{&d}){}
    explicit Graph(const std::vector<Device*>&ds,size_t nodes=default_capacity,bool efficient=false):
        capacity(nodes),efficient_sequence(efficient),device(*ds.at(0)),ctx(metadata_bytes(nodes)),devices(ds){
        g=ggml_new_graph_custom(ctx.p,capacity,true);
    }
    ~Graph(){ggml_backend_sched_free(sched);ggml_gallocr_free(direct_allocator);}
    T*input(int64_t n0,int64_t n1,const std::vector<float>&v){
        auto*t=ggml_new_tensor_2d(ctx.p,GGML_TYPE_F32,n0,n1);ggml_set_input(t);data.push_back({t,v});return t;
    }
    T*zeros(int64_t a,int64_t b=1,int64_t c=1){
        auto*t=ggml_new_tensor_3d(ctx.p,GGML_TYPE_F32,a,b,c);ggml_set_input(t);data.push_back({t,std::vector<float>(a*b*c,0)});return t;
    }
    T*constant(float v){return input(1,1,{v});}
    T*indices(const std::vector<int32_t>&v){auto*t=ggml_new_tensor_1d(ctx.p,GGML_TYPE_I32,v.size());ggml_set_input(t);ids.push_back({t,v});return t;}
    T*sigmoid(T*x){auto*one=ggml_repeat(ctx.p,constant(1),x);return ggml_div(ctx.p,one,ggml_add(ctx.p,one,ggml_exp(ctx.p,ggml_neg(ctx.p,x))));}
    T*norm(T*x,T*w,float eps){return ggml_mul(ctx.p,ggml_rms_norm(ctx.p,x,eps),w);}
    T*linear(Block&block,const std::string&name,T*x){
        auto it=block.params.find(name);need(it!=block.params.end(),"Missing weight "+name);auto&p=it->second;
        auto*y=ggml_mul_mat(ctx.p,p.w,x);ggml_mul_mat_set_prec(y,GGML_PREC_F32);
        if(p.a){auto*z=ggml_mul_mat(ctx.p,p.a,x);ggml_mul_mat_set_prec(z,GGML_PREC_F32);z=ggml_mul_mat(ctx.p,p.b,z);ggml_mul_mat_set_prec(z,GGML_PREC_F32);y=ggml_add(ctx.p,y,z);}
        return y;
    }
    void forward(T*out){ggml_set_output(out);ggml_build_forward_expand(g,out);}
    void stage(T*out,Device&where){
        const int begin=ggml_graph_n_nodes(g);forward(out);
        for(int i=begin;i<ggml_graph_n_nodes(g);++i)placements.emplace(ggml_graph_node(g,i),&where);
    }
    void backward(T*loss,Block &block){
        ggml_set_loss(loss);forward(loss);
        const int n=ggml_graph_n_nodes(g);std::vector<T*>acc(n,nullptr);
        auto params=block.trainable();
        for(int i=0;i<n;++i)for(auto[w,dw]:params)if(ggml_graph_node(g,i)==w)acc[i]=dw;
        ggml_build_backward_expand(ctx.p,g,acc.data());
        for(auto[w,dw]:params)need(ggml_graph_get_grad(g,w),"Missing gradient for "+std::string(w->name));
    }
    void backward(T*loss,const std::vector<std::unique_ptr<Block>>&blocks){
        ggml_set_loss(loss);forward(loss);
        const int n=ggml_graph_n_nodes(g);std::vector<T*>acc(n,nullptr);std::vector<std::pair<T*,T*>>params;
        for(auto&block:blocks){auto ps=block->trainable();params.insert(params.end(),ps.begin(),ps.end());}
        for(int i=0;i<n;++i)for(auto[w,dw]:params)if(ggml_graph_node(g,i)==w)acc[i]=dw;
        // One reverse pass through the complete graph. No section boundary
        // receives an externally manufactured adjoint in this standard path.
        ggml_build_backward_expand(ctx.p,g,acc.data());
        for(int i=0;i<n;++i){auto*t=ggml_graph_node(g,i);auto*d=ggml_graph_get_grad(g,t);auto it=placements.find(t);if(d&&it!=placements.end())placements.emplace(d,it->second);}
        for(auto[w,dw]:params)need(ggml_graph_get_grad(g,w),"Missing full-graph gradient for "+std::string(w->name));
    }
    void compute(){
        // No silent CPU fallback for a selected GPU training graph.
        for(int i=0;i<ggml_graph_n_nodes(g);++i){auto*t=ggml_graph_node(g,i);bool supported=false;for(auto*d:devices)supported|=ggml_backend_supports_op(d->backend,t);need(supported,"Unsupported native gradient op "+std::string(ggml_op_name(t->op)));}
        const bool direct_backend=efficient_sequence&&devices.size()==1;
        if(direct_backend){
            // A single selected backend has no cross-device splits. The generic scheduler
            // reserves metadata for every possible split; avoid that irrelevant
            // upper bound while executing the identical complete graph.
            direct_allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(device.backend));
            need(direct_allocator,"Single-backend allocator construction failed");
            if(device.allocation_guard){
                size_t bytes=0;ggml_gallocr_reserve_n_size(direct_allocator,g,nullptr,nullptr,&bytes);
                const size_t limit=device.graph_limit();device.observed_graph(bytes,limit);
                need(bytes<=limit,"Graph allocation admission refused on "+device.name);
                need(ggml_gallocr_reserve(direct_allocator,g),"Admitted graph buffer allocation failed");
            }
            need(ggml_gallocr_alloc_graph(direct_allocator,g),"Single-backend graph allocation failed");
            std::cerr<<"DIRECT_GRAPH_ALLOCATION bytes="<<ggml_gallocr_get_buffer_size(direct_allocator,0)<<std::endl;
        }else{
        std::vector<ggml_backend_t>backends;for(auto*d:devices)backends.push_back(d->backend);
        // GGML requires a final CPU backend for scheduler administration. Every
        // actual tensor op is explicitly placed on an admitted device below;
        // this is not permission to offload an unsupported gradient to the CPU.
        if(devices.back()->name!="CPU"){scheduler_cpu=std::make_unique<Device>("CPU");backends.push_back(scheduler_cpu->backend);}
        // The built graph is final. Count its actual nodes and leaves instead
        // of reserving scheduler splits for every unused bookkeeping slot.
        size_t scheduler_nodes=capacity;
        if(efficient_sequence){
            // The pinned public API exposes nodes but no leaf-count accessor.
            // Walk their actual source/view tensors to include every live leaf.
            std::set<T*>live;std::vector<T*>pending;
            for(int i=0;i<ggml_graph_n_nodes(g);++i)pending.push_back(ggml_graph_node(g,i));
            while(!pending.empty()){
                auto*t=pending.back();pending.pop_back();if(!live.insert(t).second)continue;
                for(int s=0;s<GGML_MAX_SRC;++s)if(t->src[s])pending.push_back(t->src[s]);
                if(t->view_src)pending.push_back(t->view_src);
            }
            scheduler_nodes=live.size()+16;
        }
        sched=ggml_backend_sched_new(backends.data(),nullptr,backends.size(),scheduler_nodes,false,false);
        for(int i=0;i<ggml_graph_n_nodes(g);++i){
            auto*t=ggml_graph_node(g,i);Device*owner=nullptr;auto existing=placements.find(t);
            if(existing!=placements.end())owner=existing->second;
            if(!owner&&t->buffer){auto*dev=ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));for(auto*d:devices)if(ggml_backend_get_device(d->backend)==dev)owner=d;}
            for(int s=0;!owner&&s<GGML_MAX_SRC;++s)if(t->src[s]){
                auto it=placements.find(t->src[s]);if(it!=placements.end())owner=it->second;
                if(!owner&&t->src[s]->buffer){auto*dev=ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->src[s]->buffer));for(auto*d:devices)if(ggml_backend_get_device(d->backend)==dev)owner=d;}
            }
            if(!owner)owner=&device;
            need(ggml_backend_supports_op(owner->backend,t),"Assigned GPU cannot execute "+std::string(ggml_op_name(t->op)));
            placements[t]=owner;ggml_backend_sched_set_tensor_backend(sched,t,owner->backend);
        }
        need(sched,"Scheduler construction failed");
        if(device.allocation_guard){
            std::vector<size_t>limits(backends.size()),sizes(backends.size());
            for(size_t i=0;i<devices.size();++i)limits[i]=devices[i]->graph_limit();
            if(backends.size()>devices.size())limits.back()=Device::room(Device::host_available(),device.host_reserve_bytes);
            const bool allocated=ggml_backend_sched_alloc_graph_bounded(sched,g,limits.data(),sizes.data());
            bool admitted=true;
            for(size_t i=0;i<devices.size();++i){devices[i]->observed_graph(sizes[i],limits[i]);admitted=admitted&&sizes[i]<=limits[i];}
            if(backends.size()>devices.size()){
                admitted=admitted&&sizes.back()<=limits.back();
                std::cerr<<"GRAPH_HOST_TRANSFER_PLAN "<<json({{"buffer_bytes",sizes.back()},{"limit_bytes",limits.back()}}).dump()<<std::endl;
            }
            need(admitted,"Graph allocation admission refused on assigned backends");
            need(allocated,"Bounded graph allocation failed on "+device.name);
        }else need(ggml_backend_sched_alloc_graph(sched,g),"Graph allocation failed on "+device.name);
        for(int i=0;i<ggml_graph_n_nodes(g);++i){auto*t=ggml_graph_node(g,i);if(t->op!=GGML_OP_NONE&&device.name!="CPU")need(ggml_backend_sched_get_tensor_backend(sched,t)!=scheduler_cpu->backend,"Unexpected CPU compute in GPU training graph");}
        }
        for(auto &[t,v]:data){need(t->buffer,"Unallocated/unused floating graph input");put(t,v);}
        for(auto &[t,v]:ids){need(t->buffer,"Unallocated/unused integer graph input");ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}
        for(int i=0;i<ggml_graph_n_nodes(g);++i){auto*t=ggml_graph_node(g,i);if(t->flags&GGML_TENSOR_FLAG_LOSS){auto*d=ggml_graph_get_grad_acc(g,t);if(d){float one=1;ggml_backend_tensor_set(d,&one,0,4);}}}
        need((direct_backend?ggml_backend_graph_compute(device.backend,g):ggml_backend_sched_graph_compute(sched,g))==GGML_STATUS_SUCCESS,"Native graph execution failed");
        for(auto*d:devices)ggml_backend_synchronize(d->backend);
    }
};
struct Geometry {
    int hidden=4,ffn=8,vocab=7,layers=2,heads=2,kvheads=1,head=2;
    int state=2,kheads=1,vheads=2,kernel=3,interval=2,rotary=2,rope_type=0;
    int context=32768;float eps=1e-6f,freq=10000;
    int sections[4]={0,0,0,0};
};
static T*column(Graph&g,T*x,int t){return ggml_view_1d(g.ctx.p,x,x->ne[0],size_t(t)*x->nb[1]);}
static T*place(Graph&g,T*dst,T*src,int t){return ggml_set(g.ctx.p,dst,src,dst->nb[1],dst->nb[2],dst->nb[3],size_t(t)*dst->nb[1]);}
static T*join_columns(Graph&g,std::vector<T*> columns,int rows){
    need(!columns.empty(),"Empty sequence assembly");
    for(auto &column:columns)column=ggml_reshape_2d(g.ctx.p,column,rows,1);
    while(columns.size()>1){
        std::vector<T*>next;next.reserve((columns.size()+1)/2);
        for(size_t i=0;i<columns.size();i+=2)next.push_back(i+1<columns.size()?ggml_concat(g.ctx.p,columns[i],columns[i+1],1):columns[i]);
        columns=std::move(next);
    }
    return columns.front();
}
static T*recurrent(Graph&g,Block&b,const std::string&p,T*x,const Geometry&h,int n){
    auto*c=g.ctx.p;int d=h.state,hk=h.kheads,hv=h.vheads,channels=d*(2*hk+hv);
    auto*qkv=g.linear(b,p+"attn_qkv.weight",x);auto*z=g.linear(b,p+"attn_gate.weight",x);
    auto*beta=g.sigmoid(g.linear(b,p+"ssm_beta.weight",x));
    auto*alpha=ggml_add(c,g.linear(b,p+"ssm_alpha.weight",x),b.params.at(p+"ssm_dt.bias").w);
    auto*decay=ggml_exp(c,ggml_mul(c,ggml_softplus(c,alpha),b.params.at(p+"ssm_a").w));
    auto*w=b.params.at(p+"ssm_conv1d.weight").w;
    auto*convolved=g.efficient_sequence?nullptr:g.zeros(channels,n);std::vector<T*>convolution_columns;
    for(int t=0;t<n;++t){
        T*sum=nullptr;
        for(int lag=0;lag<h.kernel;++lag){int pos=t-h.kernel+1+lag;if(pos<0)continue;
            auto*weight=ggml_reshape_1d(c,ggml_cont(c,ggml_view_2d(c,w,1,channels,w->nb[1],size_t(lag)*4)),channels);
            auto*term=ggml_mul(c,column(g,qkv,pos),weight);sum=sum?ggml_add(c,sum,term):term;
        }
        need(sum,"Empty causal convolution");if(g.efficient_sequence)convolution_columns.push_back(sum);else convolved=place(g,convolved,sum,t);
    }
    if(g.efficient_sequence)convolved=join_columns(g,std::move(convolution_columns),channels);
    convolved=ggml_silu(c,convolved);
    auto*qt=ggml_view_3d(c,convolved,d,hk,n,d*4,channels*4,0);
    auto*kt=ggml_view_3d(c,convolved,d,hk,n,d*4,channels*4,d*hk*4);
    auto*vt=ggml_view_3d(c,convolved,d,hv,n,d*4,channels*4,d*hk*2*4);
    // RMS_NORM_BACK's CUDA implementation consumes dense rows. Preserve its
    // input layout explicitly, rather than relying on forward-only view support.
    qt=ggml_cont(c,qt);kt=ggml_cont(c,kt);
    // Nested token views must use the materialized V layout. GGML adjoint
    // buffers are dense, so retaining the parent Q/K/V channel stride here
    // would scatter a child's gradient beyond the V adjoint allocation.
    vt=ggml_cont(c,vt);
    qt=ggml_scale(c,ggml_rms_norm(c,qt,h.eps/d),1/std::sqrt(float(d)));
    kt=ggml_scale(c,ggml_rms_norm(c,kt,h.eps/d),1/std::sqrt(float(d)));
    if(hk!=hv){qt=ggml_repeat_4d(c,qt,d,hv,n,1);kt=ggml_repeat_4d(c,kt,d,hv,n,1);}
    auto*state=g.zeros(d,d,hv);auto*out=g.efficient_sequence?nullptr:g.zeros(d*hv,n);std::vector<T*>output_columns;
    for(int t=0;t<n;++t){
        auto*q=ggml_reshape_3d(c,ggml_cont(c,ggml_view_2d(c,qt,d,hv,qt->nb[1],size_t(t)*qt->nb[2])),d,1,hv);
        auto*k=ggml_reshape_3d(c,ggml_cont(c,ggml_view_2d(c,kt,d,hv,kt->nb[1],size_t(t)*kt->nb[2])),d,1,hv);
        auto*v=ggml_reshape_3d(c,ggml_cont(c,ggml_view_2d(c,vt,d,hv,vt->nb[1],size_t(t)*vt->nb[2])),d,1,hv);
        auto*dg=ggml_reshape_3d(c,column(g,decay,t),1,1,hv);
        auto*bt=ggml_reshape_3d(c,column(g,beta,t),1,1,hv);
        state=ggml_mul(c,state,dg);
        auto*pred=ggml_mul_mat(c,state,k);ggml_mul_mat_set_prec(pred,GGML_PREC_F32);
        auto*delta=ggml_mul(c,ggml_sub(c,v,pred),bt);
        auto*outer=ggml_mul(c,ggml_repeat_4d(c,k,d,d,hv,1),ggml_reshape_3d(c,delta,1,d,hv));
        state=ggml_add(c,state,outer);
        auto*y=ggml_mul_mat(c,state,q);ggml_mul_mat_set_prec(y,GGML_PREC_F32);
        y=ggml_scale(c,y,1/std::sqrt(float(d)));
        if(g.efficient_sequence)output_columns.push_back(ggml_reshape_1d(c,y,d*hv));else out=place(g,out,ggml_reshape_1d(c,y,d*hv),t);
    }
    if(g.efficient_sequence)out=join_columns(g,std::move(output_columns),d*hv);
    out=ggml_reshape_3d(c,out,d,hv,n);
    out=g.norm(out,b.params.at(p+"ssm_norm.weight").w,h.eps);
    out=ggml_mul(c,out,ggml_silu(c,ggml_reshape_3d(c,z,d,hv,n)));
    return g.linear(b,p+"ssm_out.weight",ggml_reshape_2d(c,out,d*hv,n));
}
static T*attention(Graph&g,Block&b,const std::string&p,T*x,const Geometry&h,int n){
    auto*c=g.ctx.p;int d=h.head,hq=h.heads,hk=h.kvheads;
    auto*qg=g.linear(b,p+"attn_q.weight",x);
    auto*q=ggml_cont(c,ggml_view_3d(c,qg,d,hq,n,d*2*4,d*2*hq*4,0));
    auto*gate=ggml_cont_2d(c,ggml_view_3d(c,qg,d,hq,n,d*2*4,d*2*hq*4,d*4),d*hq,n);
    auto*k=ggml_reshape_3d(c,g.linear(b,p+"attn_k.weight",x),d,hk,n);
    auto*v=ggml_reshape_3d(c,g.linear(b,p+"attn_v.weight",x),d,hk,n);
    q=g.norm(q,b.params.at(p+"attn_q_norm.weight").w,h.eps);
    k=g.norm(k,b.params.at(p+"attn_k_norm.weight").w,h.eps);
    std::vector<int32_t>positions(n*4);for(int i=0;i<n*4;++i)positions[i]=i%n;
    if(h.rope_type){
        auto*pos=g.indices(positions);
        int sections[4];std::copy(h.sections,h.sections+4,sections);
        q=ggml_rope_multi(c,q,pos,nullptr,h.rotary,sections,h.rope_type,h.context,h.freq,1,0,1,32,1);
        k=ggml_rope_multi(c,k,pos,nullptr,h.rotary,sections,h.rope_type,h.context,h.freq,1,0,1,32,1);
    }else{
        auto*single=g.indices(std::vector<int32_t>(positions.begin(),positions.begin()+n));
        q=ggml_rope_ext(c,q,single,nullptr,h.rotary,0,h.context,h.freq,1,0,1,32,1);
        k=ggml_rope_ext(c,k,single,nullptr,h.rotary,0,h.context,h.freq,1,0,1,32,1);
    }
    q=ggml_permute(c,q,0,2,1,3);k=ggml_permute(c,k,0,2,1,3);
    auto*scores=ggml_mul_mat(c,k,q);ggml_mul_mat_set_prec(scores,GGML_PREC_F32);
    // A constant additive causal mask has an ordinary softmax derivative and
    // avoids the pinned CUDA backend's unsupported DIAG_MASK_ZERO backward op.
    std::vector<float>mask(n*n,0);for(int query=0;query<n;++query)for(int key=query+1;key<n;++key)mask[query*n+key]=-INFINITY;
    scores=ggml_soft_max(c,ggml_add(c,ggml_scale(c,scores,1/std::sqrt(float(d))),g.input(n,n,mask)));
    auto*vp=ggml_cont(c,ggml_permute(c,v,1,2,0,3));
    auto*y=ggml_mul_mat(c,vp,scores);ggml_mul_mat_set_prec(y,GGML_PREC_F32);
    y=ggml_cont_2d(c,ggml_permute(c,y,0,2,1,3),d*hq,n);
    y=ggml_mul(c,y,g.sigmoid(gate));return g.linear(b,p+"attn_output.weight",y);
}
static T*layer(Graph&g,Block&b,int index,T*x,const Geometry&h,int n){
    auto*c=g.ctx.p;std::string p="blk."+std::to_string(index)+".";
    auto*y=g.norm(x,b.params.at(p+"attn_norm.weight").w,h.eps);
    y=(index+1)%h.interval?recurrent(g,b,p,y,h,n):attention(g,b,p,y,h,n);
    auto*res=ggml_add(c,x,y);
    y=g.norm(res,b.params.at(p+"post_attention_norm.weight").w,h.eps);
    auto*up=g.linear(b,p+"ffn_up.weight",y);auto*gate=g.linear(b,p+"ffn_gate.weight",y);
    y=g.linear(b,p+"ffn_down.weight",ggml_mul(c,up,ggml_silu(c,gate)));
    return ggml_add(c,res,y);
}

// The first executable gate is a real GGML gradient check through both layer
// types. Large-model loading/sharding is added only after this gate succeeds.
static void populate(Block&b,const Geometry&h,int i){
    std::string p="blk."+std::to_string(i)+".";
    auto add=[&](std::string n,int a,int c=1,bool linear=false){b.add(p+n,a,c,linear);};
    add("attn_norm.weight",h.hidden);add("post_attention_norm.weight",h.hidden);
    add("ffn_up.weight",h.hidden,h.ffn,true);add("ffn_gate.weight",h.hidden,h.ffn,true);add("ffn_down.weight",h.ffn,h.hidden,true);
    if((i+1)%h.interval){
        int channels=h.state*(2*h.kheads+h.vheads);
        add("attn_qkv.weight",h.hidden,channels,true);add("attn_gate.weight",h.hidden,h.state*h.vheads,true);
        add("ssm_beta.weight",h.hidden,h.vheads,true);add("ssm_alpha.weight",h.hidden,h.vheads,true);
        add("ssm_dt.bias",h.vheads);add("ssm_a",h.vheads);add("ssm_conv1d.weight",h.kernel,channels);
        add("ssm_norm.weight",h.state);add("ssm_out.weight",h.state*h.vheads,h.hidden,true);
    }else{
        add("attn_q.weight",h.hidden,2*h.head*h.heads,true);add("attn_k.weight",h.hidden,h.head*h.kvheads,true);
        add("attn_v.weight",h.hidden,h.head*h.kvheads,true);add("attn_output.weight",h.head*h.heads,h.hidden,true);
        add("attn_q_norm.weight",h.head);add("attn_k_norm.weight",h.head);
    }
}
struct Chain {
    Geometry h;
    std::vector<std::unique_ptr<Device>> devices;
    std::vector<std::unique_ptr<Block>> blocks;
    bool full;int rank;int maximum_sequence=8192;bool efficient_sequence=false;
    Chain(const Geometry&geometry,const std::vector<std::string>&names,bool train_full,int r,const std::vector<int>&placement={}):h(geometry),full(train_full),rank(r){
        need(!names.empty()&&rank>0&&rank<=256,"Explicit devices and bounded positive rank required");
        need(std::set<std::string>(names.begin(),names.end()).size()==names.size(),"Duplicate training devices are forbidden");
        need(names.size()==1||std::find(names.begin(),names.end(),"CPU")==names.end(),"CPU training uses one CPU backend; mixed CPU/CUDA placement is not qualified");
        for(auto&name:names)devices.push_back(std::make_unique<Device>(name));
        need(placement.empty()||placement.size()==size_t(h.layers+2),"Explicit placement must cover embedding, every layer and head");
        for(int i=0;i<h.layers+2;++i){
            const int target=placement.empty()?i%devices.size():placement[i];
            need(target>=0&&target<int(devices.size()),"Explicit block placement names an unavailable device");
            blocks.push_back(std::make_unique<Block>(*devices[target],full,rank));
        }
        blocks.front()->add("token_embd.weight",h.hidden,h.vocab,false);
        for(int i=0;i<h.layers;++i)populate(*blocks[i+1],h,i);
        blocks.back()->add("output_norm.weight",h.hidden,1,false);
        blocks.back()->add("output.weight",h.hidden,h.vocab,true);
    }
    void allocate(){for(auto&b:blocks){b->allocate();b->initialize_adapters();}}
    void zero(){for(auto&b:blocks)b->zero();}
    std::vector<float> embed(const std::vector<int32_t>&tokens,const std::vector<float>*up=nullptr){
        auto&b=*blocks.front();Graph g(b.device);auto*out=ggml_get_rows(g.ctx.p,b.params.at("token_embd.weight").w,g.indices(tokens));
        if(up&&full){auto*dy=g.input(h.hidden,tokens.size(),*up);g.backward(ggml_sum(g.ctx.p,ggml_mul(g.ctx.p,out,dy)),b);}else g.forward(out);
        g.compute();return read_tensor(out);
    }
    std::vector<float> body(int i,const std::vector<float>&x,int n,const std::vector<float>*up=nullptr){
        auto&b=*blocks[i+1];Graph g(b.device);auto*in=g.input(h.hidden,n,x);if(up)ggml_set_param(in);
        auto*out=layer(g,b,i,in,h,n);
        if(up){auto*dy=g.input(h.hidden,n,*up);g.backward(ggml_sum(g.ctx.p,ggml_mul(g.ctx.p,out,dy)),b);}else g.forward(out);
        g.compute();return read_tensor(up?ggml_graph_get_grad(g.g,in):out);
    }
    std::pair<float,std::vector<float>> head(const std::vector<float>&x,const json&sample,bool backward,float weight){
        auto&b=*blocks.back();int n=sample.at("tokens").size();Graph g(b.device);auto*c=g.ctx.p;
        auto*in=g.input(h.hidden,n,x);if(backward)ggml_set_param(in);
        auto*norm=g.norm(in,b.params.at("output_norm.weight").w,h.eps);auto*logits=g.linear(b,"output.weight",norm);
        auto targets=sample.at("targets").get<std::vector<int32_t>>();auto mask=sample.at("loss_mask").get<std::vector<bool>>();
        need(int(targets.size())==n&&int(mask.size())==n,"Misaligned targets/mask");
        T*loss=nullptr;int selected=0;
        for(int t=0;t<n;++t)if(mask[t]){
            need(targets[t]>=0&&targets[t]<h.vocab,"Invalid target ID");
            std::vector<float>labels(h.vocab,0);labels[targets[t]]=1;
            auto*ce=ggml_cross_entropy_loss(c,column(g,logits,t),g.input(h.vocab,1,labels));
            loss=loss?ggml_add(c,loss,ce):ce;++selected;
        }
        need(selected>0,"Empty completion loss");loss=ggml_scale(c,loss,weight/selected);
        if(backward)g.backward(loss,b);else g.forward(loss);g.compute();
        float value=read_tensor(loss)[0]/weight;need(std::isfinite(value),"Nonfinite native loss");
        return {value,backward?read_tensor(ggml_graph_get_grad(g.g,in)):std::vector<float>()};
    }
    float experimental_section_sample(const json&sample,bool backward,float weight=1){
        auto tokens=sample.at("tokens").get<std::vector<int32_t>>();need(tokens.size()>=2&&tokens.size()<=8192,"Bounded sequence required");
        for(auto t:tokens)need(t>=0&&t<h.vocab,"Invalid input token");
        std::vector<std::vector<float>>saved;saved.push_back(embed(tokens));
        for(int i=0;i<h.layers;++i)saved.push_back(body(i,saved.back(),tokens.size()));
        auto result=head(saved.back(),sample,backward,weight);
        if(backward){auto dy=std::move(result.second);for(int i=h.layers-1;i>=0;--i)dy=body(i,saved[i],tokens.size(),&dy);if(full)embed(tokens,&dy);}
        return result.first;
    }
    float standard_sample(const json&sample,bool backward,float weight=1,std::vector<float>*observed_logits=nullptr){
        auto tokens=sample.at("tokens").get<std::vector<int32_t>>();int n=tokens.size();need(n>=2&&n<=maximum_sequence,"Bounded sequence required");
        for(auto t:tokens)need(t>=0&&t<h.vocab,"Invalid input token");
        std::vector<Device*>ds;for(auto&d:devices)ds.push_back(d.get());Graph g(ds,efficient_sequence?Graph::bounded_capacity(n,h.layers):Graph::default_capacity,efficient_sequence);auto*c=g.ctx.p;
        auto*x=ggml_get_rows(c,blocks.front()->params.at("token_embd.weight").w,g.indices(tokens));g.stage(x,blocks.front()->device);
        for(int i=0;i<h.layers;++i){x=layer(g,*blocks[i+1],i,x,h,n);g.stage(x,blocks[i+1]->device);}
        auto&head=*blocks.back();x=g.norm(x,head.params.at("output_norm.weight").w,h.eps);auto*logits=g.linear(head,"output.weight",x);g.stage(logits,head.device);
        auto targets=sample.at("targets").get<std::vector<int32_t>>();auto mask=sample.at("loss_mask").get<std::vector<bool>>();
        need(int(targets.size())==n&&int(mask.size())==n,"Misaligned targets/mask");T*loss=nullptr;int selected=0;
        for(int t=0;t<n;++t)if(mask[t]){need(targets[t]>=0&&targets[t]<h.vocab,"Invalid target");std::vector<float>labels(h.vocab,0);labels[targets[t]]=1;
            auto*ce=ggml_cross_entropy_loss(c,column(g,logits,t),g.input(h.vocab,1,labels));loss=loss?ggml_add(c,loss,ce):ce;++selected;}
        need(selected>0,"Empty completion loss");loss=ggml_scale(c,loss,weight/selected);g.stage(loss,head.device);
        if(backward)g.backward(loss,blocks);
        if(efficient_sequence)std::cerr<<"LONG_GRAPH nodes="<<ggml_graph_n_nodes(g.g)<<" capacity="<<g.capacity<<" metadata="<<Graph::metadata_bytes(g.capacity)<<std::endl;
        g.compute();
        if(observed_logits)*observed_logits=read_tensor(logits);
        float value=read_tensor(loss)[0]/weight;need(std::isfinite(value),"Nonfinite standard loss");return value;
    }
    json gradients(){
        json stats=json::array();
        for(size_t i=0;i<blocks.size();++i){auto&b=*blocks[i];Graph g(b.device);std::vector<T*>sums;std::vector<std::string>names;
            for(auto[w,dw]:b.trainable()){auto*s=ggml_sum(g.ctx.p,ggml_sqr(g.ctx.p,dw));g.forward(s);sums.push_back(s);names.push_back(w->name);}
            if(sums.empty())continue;g.compute();
            for(size_t j=0;j<sums.size();++j){float v=read_tensor(sums[j])[0];need(std::isfinite(v)&&v>=0,"Nonfinite gradient "+names[j]);stats.push_back({{"name",names[j]},{"l2",std::sqrt(v)},{"device",b.device.name},{"block",i}});}
        }
        return stats;
    }
    void update(float learning_rate){
        need(learning_rate>0&&std::isfinite(learning_rate),"Positive finite learning rate required");
        for(auto&bp:blocks){auto&b=*bp;auto params=b.trainable();if(params.empty())continue;
            Graph g(b.device);auto*opt=g.input(2,1,{learning_rate,0});
            for(auto[w,dw]:params)g.forward(ggml_opt_step_sgd(g.ctx.p,w,dw,opt));g.compute();
        }
    }
    void randomize(){
        std::mt19937 rng(241);std::normal_distribution<float>normal(0,.15f);
        for(auto&b:blocks)for(auto&[name,p]:b->params){std::vector<float>v(ggml_nelements(p.w));for(auto&x:v)x=normal(rng);
            if(name.find("norm.weight")!=std::string::npos)for(auto&x:v)x+=1;
            if(name.find("ssm_a")!=std::string::npos)for(auto&x:v)x=-std::exp(x);put(p.w,v);
        }
    }
};
static json chain_test(const std::vector<std::string>&names,bool efficient=false){
    Geometry h;json reports=json::array();
    json example={{"tokens",{0,1,2,3}},{"targets",{1,2,3,4}},{"loss_mask",{false,true,true,true}}};
    for(bool full:{true,false}){
        Chain m(h,names,full,2);m.efficient_sequence=efficient;m.allocate();m.randomize();m.zero();float before=m.standard_sample(example,true);
        auto gradients=m.gradients();
        if(full){auto grad=read_tensor(m.blocks.front()->params.at("token_embd.weight").grad);double norm=0;for(int i=0;i<h.hidden;++i)norm+=double(grad[i])*grad[i];need(norm>1e-16,"Masked prompt context lost its gradient");}
        // Check one parameter direction in every stage of the whole chain.
        double worst=0;int checks=0;std::mt19937 rng(98);std::normal_distribution<float>normal(0,1);
        for(auto&b:m.blocks){auto ps=b->trainable();if(ps.empty())continue;
            auto[w,dw]=ps.back();auto values=read_tensor(w),grad=read_tensor(dw);
            std::vector<float>direction(values.size());double norm=0;for(auto&v:direction){v=normal(rng);norm+=v*v;}norm=std::sqrt(norm);for(auto&v:direction)v/=norm;
            double analytic=0;for(size_t j=0;j<values.size();++j)analytic+=double(direction[j])*grad[j];
            constexpr float eps=.003f;auto changed=values;
            for(size_t j=0;j<values.size();++j)changed[j]=values[j]+eps*direction[j];put(w,changed);float plus=m.standard_sample(example,false);
            for(size_t j=0;j<values.size();++j)changed[j]=values[j]-eps*direction[j];put(w,changed);float minus=m.standard_sample(example,false);put(w,values);
            double numeric=(plus-minus)/(2*eps),error=std::abs(numeric-analytic);worst=std::max(worst,error);
            need(error<.0004+.03*std::abs(numeric),"Chain directional gradient mismatch "+std::string(w->name));++checks;
        }
        // Check accumulation itself, independently of the Python batch helper:
        // two half-weight reverse passes must equal the mean of separate grads.
        std::vector<std::pair<T*,T*>>all;for(auto&b:m.blocks){auto ps=b->trainable();all.insert(all.end(),ps.begin(),ps.end());}
        std::vector<std::vector<float>>first,second;for(auto[w,dw]:all)first.push_back(read_tensor(dw));
        json other={{"tokens",{3,2,1,0}},{"targets",{2,1,0,6}},{"loss_mask",{true,false,true,true}}};
        m.zero();m.standard_sample(other,true);for(auto[w,dw]:all)second.push_back(read_tensor(dw));
        m.zero();m.standard_sample(example,true,.5f);m.standard_sample(other,true,.5f);
        double batch_error=0;size_t batch_scalars=0;
        for(size_t i=0;i<all.size();++i){auto actual=read_tensor(all[i].second);
            for(size_t j=0;j<actual.size();++j){double expected=(double(first[i][j])+second[i][j])*.5,error=std::abs(actual[j]-expected);
                batch_error=std::max(batch_error,error);++batch_scalars;need(error<=1e-6+1e-4*std::abs(expected),"Native gradient accumulation mismatch");}}
        m.zero();m.standard_sample(example,true);m.update(.01f);m.zero();float after=m.standard_sample(example,false);need(after<before,"Tiny-chain update did not reduce objective");
        reports.push_back({{"mode",full?"full":"lora"},{"before",before},{"after",after},{"stage_directions",checks},{"worst_absolute_error",worst},{"batch_scalar_gradients",batch_scalars},{"batch_gradient_max_error",batch_error},{"gradients",gradients}});
    }
    return {{"status","PASS"},{"devices",names},{"scope","tiny complete hybrid chain; model weights have not been trained"},{"reports",reports}};
}
static json self_test(const std::string&backend,bool diagnostic=false){
    Device d(backend);Geometry h;std::mt19937 rng(183);std::normal_distribution<float>normal(0,.15f);
    json records=json::array();
    for(int which=0;which<2;++which){
        Block b(d,true,2);populate(b,h,which);b.allocate();
        for(auto&[name,p]:b.params){std::vector<float>v(ggml_nelements(p.w));for(auto&x:v)x=normal(rng);
            if(name.find("norm.weight")!=std::string::npos)for(auto&x:v)x+=1;
            if(name.find("ssm_a")!=std::string::npos)for(auto&x:v)x=-std::exp(x);
            put(p.w,v);
        }
        b.zero();int n=4;std::vector<float>x(h.hidden*n),dy(x.size());for(auto&v:x)v=normal(rng);for(auto&v:dy)v=normal(rng);
        auto evaluate=[&](bool backward){
            Graph g(d);auto*in=g.input(h.hidden,n,x);if(backward)ggml_set_param(in);
            auto*out=layer(g,b,which,in,h,n);auto*up=g.input(h.hidden,n,dy);auto*loss=ggml_sum(g.ctx.p,ggml_mul(g.ctx.p,out,up));
            if(backward)g.backward(loss,b);else g.forward(loss);g.compute();return read_tensor(loss)[0];
        };
        float baseline=evaluate(true);double worst=0;int checks=0;json directions=json::array();int failures=0;
        for(auto&[name,p]:b.params){
            auto values=read_tensor(p.w),grad=read_tensor(p.grad);
            // A direction through every tensor; predeclared epsilon and bounds.
            std::vector<float>direction(values.size());for(auto&v:direction)v=normal(rng);
            double norm=0;for(float v:direction)norm+=double(v)*v;norm=std::sqrt(norm);for(auto&v:direction)v/=norm;
            double analytic=0;for(size_t j=0;j<values.size();++j)analytic+=double(grad[j])*direction[j];
            constexpr float eps=.002f;auto changed=values;
            for(size_t j=0;j<values.size();++j)changed[j]=values[j]+eps*direction[j];put(p.w,changed);float plus=evaluate(false);
            for(size_t j=0;j<values.size();++j)changed[j]=values[j]-eps*direction[j];put(p.w,changed);float minus=evaluate(false);put(p.w,values);
            double numeric=(plus-minus)/(2*eps),error=std::abs(numeric-analytic);worst=std::max(worst,error);
            bool passed=error<.0003+.02*std::abs(numeric);failures+=!passed;
            if(diagnostic)directions.push_back({{"name",name},{"analytic",analytic},{"numeric",numeric},{"plus",plus},{"minus",minus},{"gradient",grad},{"passed",passed}});
            if(!diagnostic)need(passed,"Directional gradient mismatch "+name+" analytic="+std::to_string(analytic)+" numeric="+std::to_string(numeric));++checks;
        }
        records.push_back({{"layer",which},{"tensor_directions",checks},{"baseline",baseline},{"worst_absolute_error",worst}});
        if(diagnostic){
            records.back()["directions"]=directions;records.back()["failures"]=failures;
            json weights=json::object();for(auto&[name,p]:b.params)weights[name]={{"shape",{p.w->ne[0],p.w->ne[1]}},{"values",read_tensor(p.w)}};
            records.back()["fixture"]={{"input",x},{"upstream",dy},{"tokens",n},{"weights",weights}};
        }
        std::cerr<<"LAYER_GRADIENT_GATE "<<records.back().dump()<<std::endl;
    }
    bool passed=true;for(auto&r:records)if(r.value("failures",0))passed=false;
    return {{"status",passed?"PASS":"FAIL"},{"backend",backend},{"scope","tiny recurrent and attention layer VJPs; not full model qualification"},{"checks",records}};
}
static json inspect_model(const std::string&path){
    ggml_context*ctx=nullptr;auto*f=gguf_init_from_file(path.c_str(),{true,&ctx});need(f,"Cannot read GGUF");
    json meta=json::object(),tensors=json::array();
    for(int64_t i=0;i<gguf_get_n_kv(f);++i){std::string key=gguf_get_key(f,i);if(key.rfind("wen1.",0)&&key.rfind("qwen35.",0)&&key!="general.architecture")continue;
        switch(gguf_get_kv_type(f,i)){
            case GGUF_TYPE_UINT32:meta[key]=gguf_get_val_u32(f,i);break;
            case GGUF_TYPE_INT32:meta[key]=gguf_get_val_i32(f,i);break;
            case GGUF_TYPE_FLOAT32:meta[key]=gguf_get_val_f32(f,i);break;
            case GGUF_TYPE_STRING:meta[key]=gguf_get_val_str(f,i);break;
            case GGUF_TYPE_ARRAY:{if(gguf_get_arr_type(f,i)==GGUF_TYPE_INT32){auto*a=(const int32_t*)gguf_get_arr_data(f,i);meta[key]=std::vector<int32_t>(a,a+gguf_get_arr_n(f,i));}break;}
            default:break;
        }
    }
    for(int64_t i=0;i<gguf_get_n_tensors(f);++i){std::string name=gguf_get_tensor_name(f,i);if(i>3&&name.rfind("blk.0.",0)&&name.rfind("blk.3.",0))continue;
        auto*ne=gguf_get_tensor_ne(f,i);tensors.push_back({{"name",name},{"shape",{ne[0],ne[1],ne[2],ne[3]}},{"type",ggml_type_name(gguf_get_tensor_type(f,i))}});
    }
    ggml_free(ctx);gguf_free(f);return {{"metadata",meta},{"tensors",tensors}};
}
struct GGUFSource {
    std::string path;ggml_context*ctx=nullptr;gguf_context*file=nullptr;
    explicit GGUFSource(const std::string&p):path(p){file=gguf_init_from_file(path.c_str(),{true,&ctx});need(file,"Cannot read source GGUF");}
    ~GGUFSource(){gguf_free(file);ggml_free(ctx);}
    std::vector<float>values(const std::string&name){
        int64_t i=gguf_find_tensor(file,name.c_str());need(i>=0,"Missing source tensor "+name);
        auto type=gguf_get_tensor_type(file,i);need(type==GGML_TYPE_F32||type==GGML_TYPE_BF16,"Native training requires F32/BF16 weights: "+name);
        const auto*ne=gguf_get_tensor_ne(file,i);size_t count=ne[0]*ne[1]*ne[2]*ne[3];std::vector<float>out(count);
        std::ifstream stream(path,std::ios::binary);need(bool(stream),"Cannot open immutable source");
        stream.seekg(gguf_get_data_offset(file)+gguf_get_tensor_offset(file,i));
        if(type==GGML_TYPE_F32)stream.read(reinterpret_cast<char*>(out.data()),count*4);
        else{std::vector<uint16_t>raw(count);stream.read(reinterpret_cast<char*>(raw.data()),count*2);
            for(size_t j=0;j<count;++j){uint32_t bits=uint32_t(raw[j])<<16;std::memcpy(&out[j],&bits,4);}}
        need(bool(stream),"Incomplete source tensor "+name);
        for(auto v:out)if(!std::isfinite(v))throw std::runtime_error("Nonfinite source tensor "+name);
        return out;
    }
    Geometry geometry(){
        Geometry h;auto arch=std::string(gguf_get_val_str(file,gguf_find_key(file,"general.architecture")));
        need(arch=="wen1"||arch=="qwen35","This trainer currently implements dense Wen/Qwen3.5 only");
        auto integer=[&](const std::string&key){auto i=gguf_find_key(file,(arch+"."+key).c_str());need(i>=0,"Missing geometry "+key);return int(gguf_get_val_u32(file,i));};
        auto real=[&](const std::string&key){auto i=gguf_find_key(file,(arch+"."+key).c_str());need(i>=0,"Missing geometry "+key);return gguf_get_val_f32(file,i);};
        h.hidden=integer("embedding_length");h.ffn=integer("feed_forward_length");h.layers=integer("block_count");
        h.heads=integer("attention.head_count");h.kvheads=integer("attention.head_count_kv");h.head=integer("attention.key_length");
        h.state=integer("ssm.state_size");h.vheads=integer("ssm.time_step_rank");h.kheads=integer("ssm.group_count");h.kernel=integer("ssm.conv_kernel");
        h.interval=integer("full_attention_interval");h.context=integer("context_length");h.rotary=integer("rope.dimension_count");
        h.freq=real("rope.freq_base");h.eps=real("attention.layer_norm_rms_epsilon");h.rope_type=GGML_ROPE_TYPE_IMROPE;
        auto sections=gguf_find_key(file,(arch+".rope.dimension_sections").c_str());need(sections>=0&&gguf_get_arr_n(file,sections)==4,"Four explicit rotary sections required");
        auto*s=(const int32_t*)gguf_get_arr_data(file,sections);std::copy(s,s+4,h.sections);
        auto emb=gguf_find_tensor(file,"token_embd.weight");need(emb>=0,"Missing embeddings");h.vocab=gguf_get_tensor_ne(file,emb)[1];
        need(h.layers>0&&h.layers<=128&&h.hidden>0&&h.state*h.vheads>0&&h.heads%h.kvheads==0&&h.vheads%h.kheads==0,"Unsupported model geometry");return h;
    }
    void load(Chain&model){
        // Validate complete parameter coverage before allocating any model buffers.
        for(auto&b:model.blocks)for(auto&[name,p]:b->params){
            auto i=gguf_find_tensor(file,name.c_str());need(i>=0,"Missing active decoder parameter "+name);auto*ne=gguf_get_tensor_ne(file,i);
            need(ne[0]==p.w->ne[0]&&ne[1]==p.w->ne[1]&&ne[2]==1&&ne[3]==1,"Source geometry mismatch "+name);
            auto type=gguf_get_tensor_type(file,i);need(type==GGML_TYPE_F32||type==GGML_TYPE_BF16,"Unsupported trainable source dtype");
        }
        model.allocate();
        for(size_t i=0;i<model.blocks.size();++i){auto&b=*model.blocks[i];for(auto&[name,p]:b.params)put(p.w,values(name));
            std::cerr<<"MODEL_BLOCK_LOADED "<<i<<" "<<b.device.name<<std::endl;}
    }
    void load_adapter(Chain&model,const std::string&adapter_path,const std::string&expected_base){
        need(!model.full,"Full-weight mode does not silently merge an adapter");
        GGUFSource adapter(adapter_path);auto key=gguf_find_key(adapter.file,"pcc.training.base_sha256");
        need(key>=0&&expected_base==gguf_get_val_str(adapter.file,key),"Training adapter/base identity mismatch");
        auto alpha=gguf_find_key(adapter.file,"adapter.lora.alpha");need(alpha>=0&&gguf_get_val_f32(adapter.file,alpha)==model.rank,"Adapter rank/scale mismatch");
        size_t count=0;
        for(auto&b:model.blocks)for(auto&[name,p]:b->params)if(p.a)for(auto*w:{p.a,p.b}){
            auto index=gguf_find_tensor(adapter.file,w->name);need(index>=0,"Missing warm-start adapter tensor "+std::string(w->name));auto*ne=gguf_get_tensor_ne(adapter.file,index);
            need(ne[0]==w->ne[0]&&ne[1]==w->ne[1]&&ne[2]==1&&ne[3]==1,"Warm-start adapter geometry mismatch");put(w,adapter.values(w->name));++count;
        }
        need(count==size_t(gguf_get_n_tensors(adapter.file)),"Unexpected warm-start adapter tensors");
    }
};
static json memory_plan(Chain&model,const json&config,bool inspect_only=false){
    std::map<std::string,size_t>persistent,scratch,activations;
    size_t length=0;for(auto&e:config.at("examples"))length=std::max(length,e.at("tokens").size());
    const auto&h=model.h;
    for(size_t index=0;index<model.blocks.size();++index){auto&b=model.blocks[index];size_t bytes=0;for(auto&[name,p]:b->params)bytes+=ggml_nbytes(p.w);
        size_t train=0;for(auto[w,dw]:b->trainable())train+=ggml_nbytes(dw);
        for(auto&[name,p]:b->params)if(p.a)bytes+=ggml_nbytes(p.a)+ggml_nbytes(p.b);
        persistent[b->device.name]+=bytes+train;scratch[b->device.name]=std::max(scratch[b->device.name],bytes*2);
        size_t active=0;
        if(index>0&&index+1<model.blocks.size()){
            active=length*size_t(h.ffn)*4*20;
            if(index%h.interval)active+=length*size_t(h.state)*h.state*h.vheads*4*16;
            else active+=size_t(h.heads)*length*length*4*12;
        }else if(index+1==model.blocks.size())active=length*size_t(h.vocab)*4*12;
        activations[b->device.name]+=active;
    }
    size_t graph_host=0;
    if(model.efficient_sequence){
        size_t nodes=Graph::bounded_capacity(length,h.layers);
        // Metadata, scheduler tables/placements, host masks and input copies.
        size_t labels=0;for(auto&e:config.at("examples")){size_t count=0;for(auto&m:e.at("loss_mask"))if(m.get<bool>())++count;labels=std::max(labels,count);}
        graph_host=Graph::metadata_bytes(nodes)+nodes*128+length*length*8+length*size_t(h.vocab)*8;
        // Cross-entropy column adjoints and accumulation are explicitly bounded.
        activations[model.blocks.back()->device.name]+=labels*length*size_t(h.vocab)*8;
        if(model.devices.size()>1){
            // Pinned GGML scheduler reserves at most one split/node with 30
            // inputs. Include its worst-case host tables/tensor descriptors.
            graph_host+=nodes*(128*ggml_tensor_overhead()+512);
        }
        if(model.devices.size()==1&&model.devices.front()->name=="CPU")activations["CPU"]+=graph_host;
        else{
            std::ifstream meminfo("/proc/meminfo");std::string line;size_t available=0;
            while(std::getline(meminfo,line))if(line.rfind("MemAvailable:",0)==0){available=std::stoull(line.substr(13))*1024;break;}
            const size_t host_reserve=config.value("host_reserve_bytes",size_t(8ull<<30));
            need(available>0&&graph_host+host_reserve<=available,"Host graph metadata admission refused");
        }
    }
    json plans=json::array();size_t reserve=config.value("reserve_bytes",size_t(4ull<<30));
    for(auto&d:model.devices){size_t free=0,total=0;ggml_backend_dev_memory(ggml_backend_get_device(d->backend),&free,&total);
        if(d->name=="CPU"){
            // The pinned GGML CPU backend reports total RAM as free on Linux.
            // Admission must respect current host pressure instead.
            std::ifstream meminfo("/proc/meminfo");std::string line;free=0;
            while(std::getline(meminfo,line))if(line.rfind("MemAvailable:",0)==0){free=std::stoull(line.substr(13))*1024;break;}
            need(free>0,"Cannot establish available host memory for CPU training");
        }
        const size_t conservative=persistent[d->name]+scratch[d->name]+activations[d->name]+reserve;
        const bool allocator=config.value("memory_profile",std::string("conservative-v1"))=="allocator-v1";
        const size_t fixed=persistent[d->name]+(d->name=="CPU"?graph_host:0);
        size_t estimate=allocator?fixed+scratch[d->name]+reserve:conservative;
        size_t cap=config.contains("device_budget_bytes")?config.at("device_budget_bytes").at(d->name).get<size_t>():free;
        if(!inspect_only)need(estimate<=free&&estimate<=cap,"Training memory admission refused on "+d->name);
        plans.push_back({{"device",d->name},{"free",free},{"total",total},{"persistent",persistent[d->name]},{"scratch_allowance",scratch[d->name]},{"activation_allowance",activations[d->name]},{"host_graph_allowance",graph_host},{"reserve",reserve},{"admission_estimate",estimate}});
        if(inspect_only)plans.back()["admitted"]=estimate<=free&&estimate<=cap;
        if(allocator){
            plans.back()["admission_stage"]="persistent-and-host; graph checked before allocation";
            plans.back()["conservative_estimate"]=conservative;
            d->allocation_guard=true;d->fixed_bytes=fixed;d->scratch_bytes=scratch[d->name];d->reserve_bytes=reserve;d->budget_bytes=cap;
            d->host_reserve_bytes=config.value("host_reserve_bytes",size_t(8ull<<30));
        }
    }
    return plans;
}
static void write_all(int fd,const void*bytes,size_t count){auto*p=(const char*)bytes;while(count){auto n=::write(fd,p,count);need(n>0,"Candidate write failed");p+=n;count-=n;}}
static void save_candidate(Chain&model,GGUFSource&source,const std::string&path,const std::string&base_sha256){
    need(!std::filesystem::exists(path),"Candidate already exists");
    Context metadata(8*1024*1024);auto*out=gguf_init_empty();need(out,"GGUF output allocation failed");
    std::vector<T*>tensors;std::vector<std::pair<T*,T*>>mapping;
    gguf_set_val_str(out,"pcc.training.base_sha256",base_sha256.c_str());
    if(model.full){
        gguf_set_kv(out,source.file);gguf_set_val_u32(out,"general.file_type",0);
        // Full mode trains the source's active decoder parameters directly.
        // Extra inactive tensors are preserved unchanged, never called trained.
        for(int64_t i=0;i<gguf_get_n_tensors(source.file);++i){std::string name=gguf_get_tensor_name(source.file,i);T*parameter=nullptr;
            for(auto&b:model.blocks){auto it=b->params.find(name);if(it!=b->params.end()){parameter=it->second.w;break;}}
            auto*original=ggml_get_tensor(source.ctx,name.c_str());need(original,"Missing source metadata tensor");
            T*desc=parameter?ggml_new_tensor(metadata.p,GGML_TYPE_F32,4,parameter->ne):ggml_dup_tensor(metadata.p,original);
            ggml_set_name(desc,name.c_str());gguf_add_tensor(out,desc);tensors.push_back(desc);mapping.push_back({desc,parameter});
        }
    }else{
        auto architecture=gguf_get_val_str(source.file,gguf_find_key(source.file,"general.architecture"));
        gguf_set_val_str(out,"general.architecture",architecture);gguf_set_val_str(out,"general.type","adapter");
        gguf_set_val_str(out,"adapter.type","lora");gguf_set_val_f32(out,"adapter.lora.alpha",float(model.rank));
        for(auto&b:model.blocks)for(auto&[name,p]:b->params)if(p.a)for(auto*w:{p.a,p.b}){
            auto*desc=ggml_dup_tensor(metadata.p,w);ggml_set_name(desc,w->name);gguf_add_tensor(out,desc);tensors.push_back(desc);mapping.push_back({desc,w});
        }
    }
    gguf_set_val_str(out,"pcc.training.base_sha256",base_sha256.c_str());
    int fd=::open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);need(fd>=0,"Cannot create immutable candidate");
    try{
        size_t size=gguf_get_meta_size(out);std::vector<char>header(size);gguf_get_meta_data(out,header.data());write_all(fd,header.data(),header.size());size_t position=header.size();
        for(size_t i=0;i<mapping.size();++i){auto[desc,parameter]=mapping[i];size_t offset=header.size()+gguf_get_tensor_offset(out,i);
            if(position<offset){std::vector<char>pad(offset-position,0);write_all(fd,pad.data(),pad.size());position=offset;}
            if(parameter){auto values=read_tensor(parameter);write_all(fd,values.data(),values.size()*4);position+=values.size()*4;}
            else{
                auto old=gguf_find_tensor(source.file,desc->name);size_t bytes=gguf_get_tensor_size(source.file,old);std::ifstream in(source.path,std::ios::binary);
                in.seekg(gguf_get_data_offset(source.file)+gguf_get_tensor_offset(source.file,old));std::vector<char>chunk(1<<20);
                while(bytes){size_t n=std::min(bytes,chunk.size());in.read(chunk.data(),n);need(bool(in),"Source preservation read failed");write_all(fd,chunk.data(),n);bytes-=n;position+=n;}
            }
        }
        need(::fsync(fd)==0,"Candidate fsync failed");::close(fd);fd=-1;
    }catch(...){if(fd>=0)::close(fd);gguf_free(out);throw;}
    gguf_free(out);
}
static json concat_test(const std::string&name){
    Device device(name);json checks=json::array();
    for(int dim=0;dim<4;++dim){
        Block block(device,true,1);std::vector<std::vector<float>>values;
        for(int part=0;part<2;++part){
            int64_t shape[4]={3,2,2,1};shape[dim]=part?3:2;
            auto&p=block.params["part"+std::to_string(part)];
            p.w=ggml_new_tensor(block.ctx.p,GGML_TYPE_F32,4,shape);ggml_set_param(p.w);
            p.grad=ggml_dup_tensor(block.ctx.p,p.w);
            std::vector<float>v(ggml_nelements(p.w));for(size_t i=0;i<v.size();++i)v[i]=.01f*(i+1+part*100);
            values.push_back(v);
        }
        block.allocate();int index=0;for(auto&[key,p]:block.params){auto&v=values[index++];ggml_backend_tensor_set(p.w,v.data(),0,v.size()*4);ggml_backend_tensor_memset(p.grad,0,0,ggml_nbytes(p.grad));}
        Graph g(device);auto*joined=ggml_concat(g.ctx.p,block.params.at("part0").w,block.params.at("part1").w,dim);
        auto*loss=ggml_sum(g.ctx.p,ggml_sqr(g.ctx.p,joined));g.backward(loss,block);g.compute();
        double worst=0;index=0;for(auto&[key,p]:block.params){auto grads=read_tensor(p.grad);auto&v=values[index++];for(size_t i=0;i<v.size();++i)worst=std::max(worst,std::abs(double(grads[i])-2*double(v[i])));}
        need(worst<1e-6,"Concatenation ordinary adjoint mismatch");checks.push_back({{"axis",dim},{"maximum_absolute_gradient_error",worst}});
    }
    return {{"status","PASS"},{"scope","odd concatenation shapes, every axis; ordinary backward"},{"checks",checks}};
}
static json train_model(const json&config){
    // The native child retains its own deadline if its Python owner disconnects.
    const int deadline=config.value("deadline_seconds",3600);
    need(deadline>=1&&deadline<=3600,"Native deadline must be 1..3600 seconds");
    alarm(unsigned(deadline));
    need(config.value("backprop_mode",std::string("standard"))=="standard","This entry point requires ordinary end-to-end backpropagation");
    need(config.value("optimizer",std::string("sgd"))=="sgd","Only native SGD is currently implemented");
    auto names=config.at("devices").get<std::vector<std::string>>();need(!names.empty(),"Explicit devices required");
    for(auto&name:names)if(name!="CPU"){load_cuda();break;}
    const std::string mode=config.at("mode");need(mode=="full"||mode=="lora","Explicit full/lora mode required");
    GGUFSource source(config.at("model").get<std::string>());Chain model(source.geometry(),names,mode=="full",config.value("rank",4),config.value("block_devices",std::vector<int>{}));
    const std::string sequence_profile=config.value("sequence_profile",std::string("short-v1"));
    need(sequence_profile=="short-v1"||sequence_profile=="tiny-cpu-long-v1"||sequence_profile=="dense-short-v1","Unknown sequence qualification profile");
    if(sequence_profile=="tiny-cpu-long-v1"){
        const auto&h=model.h;
        need(names==std::vector<std::string>{"CPU"}&&h.layers<=2&&h.hidden<=64&&h.ffn<=128&&h.vocab<=512&&h.heads<=2&&h.state<=8&&h.kernel<=4,
             "Long sequence profile is qualified only for bounded tiny CPU fixtures, not Wen");
        model.maximum_sequence=32768;model.efficient_sequence=true;
    }
    if(sequence_profile=="dense-short-v1"){
        const auto&h=model.h;
        need(h.layers<=32&&h.hidden<=4096&&h.ffn<=12288&&h.vocab<=262144&&h.heads<=16&&h.kvheads<=4&&h.head<=256&&h.state<=128&&h.vheads<=32&&h.kheads<=16&&h.kernel<=4,
             "Dense short profile exceeds its inspected geometry bounds");
        model.maximum_sequence=512;model.efficient_sequence=true;
    }
    for(const auto&e:config.at("examples")){
        need(e.at("tokens").is_array()&&e.at("targets").is_array()&&e.at("loss_mask").is_array(),"Token arrays required");
        const size_t n=e.at("tokens").size();
        need(n>=2&&n<=size_t(model.maximum_sequence)&&e.at("targets").size()==n&&e.at("loss_mask").size()==n,"Sequence admission refused");
        if(model.efficient_sequence){
            need(n<=size_t(model.h.context),"Model context metadata is shorter than the training sequence");
            size_t selected=0;for(const auto&m:e.at("loss_mask")){need(m.is_boolean(),"Boolean loss mask required");if(m.get<bool>())++selected;}
            need(selected>0&&selected<=256,"Balanced sequence profiles support at most 256 selected target positions");
        }
    }
    const std::string memory_profile=config.value("memory_profile",std::string("conservative-v1"));
    need(memory_profile=="conservative-v1"||memory_profile=="allocator-v1","Unknown memory admission profile");
    const bool allocator=memory_profile=="allocator-v1";
    need(!allocator||sequence_profile=="dense-short-v1","Allocator admission is limited to the bounded dense profile");
    const bool admission_only=config.value("admission_only",false);
    auto plan=memory_plan(model,config,admission_only);std::cerr<<"MEMORY_ADMISSION "<<plan.dump()<<std::endl;
    auto finish=[&](json result){
        if(allocator){
            result["memory_profile"]=memory_profile;result["allocation_peaks"]=json::array();
            for(auto&d:model.devices)result["allocation_peaks"].push_back({{"device",d->name},{"planned_graph_buffer_bytes",d->peak_planned_graph},{"checked_graphs",d->checked_graphs}});
        }
        return result;
    };
    if(admission_only){
        bool admitted=true;for(const auto&row:plan)admitted=admitted&&row.at("admitted").get<bool>();
        return finish({{"native",true},{"status",admitted?(allocator?"PERSISTENT_ADMISSION_PASS":"MEMORY_ADMISSION_PASS"):(allocator?"PERSISTENT_ADMISSION_REFUSED":"MEMORY_ADMISSION_REFUSED")},
            {"memory_plan",plan},{"model_layers",model.h.layers},{"weights_loaded",false},{"weights_updated",false},
            {"graph_allocation_qualified",false},{"promoted",false}});
    }
    source.load(model);
    if(config.contains("initial_adapter"))source.load_adapter(model,config.at("initial_adapter"),config.at("base_sha256"));
    auto examples=config.at("examples");need(examples.is_array()&&!examples.empty(),"Explicit training examples required");
    int batch=config.value("batch_size",1),steps=config.value("steps",1);float lr=config.value("learning_rate",.0001f);
    need(batch>0&&steps>0&&steps<=1000&&std::isfinite(lr)&&lr>0,"Invalid training budget");
    for(auto&e:examples)need(e.value("verified",false)&&e.contains("provenance_sha256"),"Verified owned training examples required");
    if(config.value("gradient_probe_only",false)){
        need(examples.size()==1,"Gradient probe admits one exact example");
        model.zero();const float loss=model.standard_sample(examples[0],true);
        auto gradients=model.gradients();int nonzero=0;for(const auto&row:gradients)nonzero+=row.at("l2").get<float>()>0;
        need(nonzero>0,"Gradient probe produced no nonzero parameter derivatives");
        return finish({{"native",true},{"status","GRADIENT_PROBE_PASS"},{"backprop_mode","standard"},{"mode",mode},
            {"loss",loss},{"gradients",gradients},{"memory_plan",plan},{"model_layers",model.h.layers},
            {"weights_updated",false},{"candidate_written",false},{"promoted",false}});
    }
    if(config.value("evaluate_only",false)){
        json losses=json::array();for(auto&e:examples){std::vector<float>logits;losses.push_back(model.standard_sample(e,false,1,config.contains("logits_output")?&logits:nullptr));
            if(config.contains("logits_output")){need(examples.size()==1,"One example per explicit logits artifact");auto path=config.at("logits_output").get<std::string>();int fd=::open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);need(fd>=0,"Logits output already exists or is inaccessible");write_all(fd,logits.data(),logits.size()*4);need(::fsync(fd)==0,"Logits fsync failed");::close(fd);}}
        return finish({{"native",true},{"status","FORWARD_EXECUTED"},{"backprop_mode","standard"},{"losses",losses},{"memory_plan",plan},{"model_layers",model.h.layers},{"promoted",false}});
    }
    json reports=json::array();size_t cursor=config.value("example_cursor",size_t(0));
    for(int step=0;step<steps;++step){
        model.zero();int count=std::min(size_t(batch),examples.size()-(cursor%examples.size()));float before=0;
        for(int j=0;j<count;++j)before+=model.standard_sample(examples[(cursor+j)%examples.size()],true,1.f/count)/count;
        auto gradients=model.gradients();model.update(lr);float after=0;
        for(int j=0;j<count;++j)after+=model.standard_sample(examples[(cursor+j)%examples.size()],false)/count;
        cursor+=count;reports.push_back({{"step",step},{"before",before},{"after",after},{"examples",count},{"example_cursor",cursor},{"gradients",gradients}});
        std::cerr<<"NATIVE_STANDARD_STEP "<<step<<" "<<before<<" "<<after<<std::endl;
    }
    auto output=config.at("candidate").get<std::string>();need(std::filesystem::absolute(output)!=std::filesystem::absolute(source.path),"Base overwrite forbidden");save_candidate(model,source,output,config.at("base_sha256"));
    return finish({{"status","MECHANICS_EXECUTED"},{"native",true},{"backprop_mode","standard"},{"mode",mode},{"optimizer","sgd"},{"model_layers",model.h.layers},{"memory_plan",plan},{"steps",reports},{"candidate",output},{"example_cursor",cursor},{"promoted",false},{"save_reload_qualified",false}});
}
int main(int argc,char**argv){try{
    setenv("GGML_NO_BACKTRACE","1",1);
    ggml_set_abort_callback([](const char*message){std::cerr<<"NATIVE_ABORT "<<message<<std::endl;void*frames[32];int n=backtrace(frames,32);backtrace_symbols_fd(frames,n,2);});
    if(argc==3&&std::string(argv[1])=="--inspect"){std::cout<<inspect_model(argv[2]).dump(2)<<std::endl;return 0;}
    if(argc==3&&(std::string(argv[1])=="--train"||std::string(argv[1])=="--admit-training"||std::string(argv[1])=="--probe-gradients")){
        std::ifstream in(argv[2]);need(bool(in),"Cannot read private training request");json config;in>>config;
        if(std::string(argv[1])=="--admit-training")config["admission_only"]=true;
        if(std::string(argv[1])=="--probe-gradients")config["gradient_probe_only"]=true;
        std::cout<<train_model(config).dump(2)<<std::endl;return 0;
    }
    if(argc==3&&std::string(argv[1])=="--concat-test"){if(std::string(argv[2])!="CPU")load_cuda();std::cout<<concat_test(argv[2]).dump(2)<<std::endl;return 0;}
    if(argc==3&&std::string(argv[1])=="--gradient-diagnostic"){
        if(std::string(argv[2])!="CPU")load_cuda();auto report=self_test(argv[2],true);std::cout<<report.dump(2)<<std::endl;return report.at("status")=="PASS"?0:3;
    }
    if(argc==3&&(std::string(argv[1])=="--chain-test"||std::string(argv[1])=="--chain-test-efficient")){
        std::vector<std::string>names;std::string value=argv[2];size_t begin=0,end;
        while((end=value.find(',',begin))!=std::string::npos){names.push_back(value.substr(begin,end-begin));begin=end+1;}names.push_back(value.substr(begin));
        for(auto&name:names)if(name!="CPU"){load_cuda();break;}
        std::cout<<chain_test(names,std::string(argv[1])=="--chain-test-efficient").dump(2)<<std::endl;return 0;
    }
    need(argc==3&&std::string(argv[1])=="--self-test","Usage: pcc-native-train --self-test CPU|CUDA0");
    if(std::string(argv[2])!="CPU")load_cuda();
    std::cout<<self_test(argv[2]).dump(2)<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<"PCC NATIVE TRAINING: "<<e.what()<<std::endl;return 2;}}
