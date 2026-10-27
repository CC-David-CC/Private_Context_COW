// Native optimizer mechanics. Independently written GGML graphs; no PyTorch.
// Equations: KellerJordan/Muon f98f1cacc0263b04290753e32be8d498c1efc806;
// shape/RMS scaling: Liu et al., arXiv:2502.16982. F32 math is explicit.
#pragma once
struct OptimizerConfig {
    std::string kind="sgd",projection="jordan5",scaling="original";
    float lr=0,aux_lr=.0003f,beta1=.9f,beta2=.95f,eps=1e-8f,momentum=.95f;
    float decay=0,aux_decay=0,normalization_eps=1e-7f;
    bool nesterov=true,factorwise_lora=false;int ns_steps=5,refine_steps=5;uint64_t refine_start=1;
    static OptimizerConfig parse(const json&request){
        OptimizerConfig c;c.kind=request.value("optimizer",std::string("sgd"));
        if(request.contains("lora_update_space")){
            need(request["lora_update_space"].is_string()&&request["lora_update_space"]=="factorwise-v1","Unknown LoRA update space");
            need(request.value("mode",std::string())=="lora"&&(c.kind=="adamw"||c.kind=="muon"),"Factorwise routing requires adaptive LoRA");
            c.factorwise_lora=true;
        }
        need(c.kind=="sgd"||c.kind=="adamw"||c.kind=="muon","Unsupported native optimizer");
        c.lr=request.value("learning_rate",.0001f);
        if(c.kind=="sgd"){for(const auto&key:{"optimizer_settings","initial_optimizer_state","initial_optimizer_state_sha256","optimizer_state_mode","optimizer_state_output"})need(!request.contains(key),"SGD does not consume adaptive optimizer state/settings");return c;}
        auto s=request.value("optimizer_settings",json::object());need(s.is_object(),"Optimizer settings must be an object");
        const std::set<std::string>keys={"schema","aux_learning_rate","beta1","beta2","epsilon","momentum","nesterov","ns_steps","projection","scaling","weight_decay","aux_weight_decay","normalization_epsilon","refine_start_step","refine_steps","precision"};
        for(auto it=s.begin();it!=s.end();++it)need(keys.count(it.key()),"Unknown optimizer setting "+it.key());
        need(s.value("schema",1)==1,"Unknown optimizer settings schema");
        need(s.value("precision",std::string("native-f32-v1"))=="native-f32-v1","Only explicit native F32 optimizer math is qualified");
        auto real=[&](const char*k,float fallback){if(!s.contains(k))return fallback;need(s[k].is_number(),std::string("Numeric optimizer setting required: ")+k);float v=s[k].get<float>();need(std::isfinite(v),"Nonfinite optimizer setting");return v;};
        c.aux_lr=real("aux_learning_rate",c.aux_lr);c.beta1=real("beta1",c.beta1);c.beta2=real("beta2",c.beta2);c.eps=real("epsilon",c.eps);
        c.momentum=real("momentum",c.momentum);c.decay=real("weight_decay",0);c.aux_decay=real("aux_weight_decay",0);c.normalization_eps=real("normalization_epsilon",c.normalization_eps);
        if(s.contains("nesterov"))need(s["nesterov"].is_boolean(),"Boolean Nesterov selection required");c.nesterov=s.value("nesterov",true);
        for(const auto&k:{"ns_steps","refine_steps","refine_start_step"})if(s.contains(k))need(s[k].is_number_integer()&&!s[k].is_boolean(),"Integer optimizer schedule required");
        c.ns_steps=s.value("ns_steps",5);c.refine_steps=s.value("refine_steps",5);c.refine_start=s.value("refine_start_step",uint64_t(1));
        c.projection=s.value("projection",std::string("jordan5"));c.scaling=s.value("scaling",std::string("original"));
        need(std::isfinite(c.lr)&&c.lr>0&&c.aux_lr>0&&c.eps>0&&c.normalization_eps>0,"Positive optimizer rates and epsilon required");
        need(c.beta1>=0&&c.beta1<1&&c.beta2>=0&&c.beta2<1&&c.momentum>=0&&c.momentum<1,"Optimizer moment coefficient outside [0,1)");
        need(c.decay>=0&&c.aux_decay>=0&&c.lr*c.decay<1&&c.aux_lr*c.aux_decay<1,"Invalid decoupled weight decay");
        need(c.ns_steps>=1&&c.ns_steps<=20&&c.refine_steps>=1&&c.refine_steps<=20&&c.refine_start>=1&&c.refine_start<=1000000000,"Bounded optimizer projection schedule required");
        need(c.projection=="jordan5"||c.projection=="fast-refine-v1","Unknown Muon projection");
        need(c.scaling=="original"||c.scaling=="rms","Unknown Muon update scaling");
        if(c.kind=="adamw")for(const auto&k:{"momentum","nesterov","ns_steps","projection","scaling","normalization_epsilon","refine_start_step","refine_steps","aux_learning_rate"})need(!s.contains(k),"AdamW cannot silently ignore a Muon-only setting");
        if(c.projection=="jordan5")need(!s.contains("refine_start_step")&&!s.contains("refine_steps"),"Jordan projection cannot silently ignore a refinement schedule");
        return c;
    }
    json canonical_base()const{return {{"schema",1},{"optimizer",kind},{"precision","native-f32-v1"},{"learning_rate",lr},{"aux_learning_rate",aux_lr},{"beta1",beta1},{"beta2",beta2},{"epsilon",eps},{"momentum",momentum},{"nesterov",nesterov},{"ns_steps",ns_steps},{"projection",projection},{"scaling",scaling},{"weight_decay",decay},{"aux_weight_decay",aux_decay},{"normalization_epsilon",normalization_eps},{"refine_start_step",refine_start},{"refine_steps",refine_steps}};}
    json canonical()const{auto value=canonical_base();if(factorwise_lora)value["lora_update_space"]="factorwise-v1";return value;}
};
static bool optimizer_hidden_matrix(T*w){
    std::string n=w->name;if(n.rfind("blk.",0))return false;
    auto dot=n.find('.',4);if(dot==std::string::npos||dot==4||!std::all_of(n.begin()+4,n.begin()+dot,::isdigit))return false;
    const std::set<std::string>suffixes={"attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight","ffn_up.weight","ffn_gate.weight","ffn_down.weight"};
    return suffixes.count(n.substr(dot+1));
}
static std::string optimizer_role(T*w){
    const std::string n=w->name;if(optimizer_hidden_matrix(w))return "hidden_matrix";
    if(n=="token_embd.weight")return "embedding";if(n=="output.weight")return "prediction_head";
    return "auxiliary_gain_bias_gate_or_other";
}
static bool optimizer_factor_hidden_matrix(T*w){
    std::string n=w->name;
    for(const std::string suffix:{".lora_a",".lora_b"}){
        if(n.size()>suffix.size()&&n.compare(n.size()-suffix.size(),suffix.size(),suffix)==0){
            n.resize(n.size()-suffix.size());
            // Match the original weight name without mutating the live tensor.
            if(n.rfind("blk.",0))return false;
            auto dot=n.find('.',4);
            if(dot==std::string::npos||dot==4||!std::all_of(n.begin()+4,n.begin()+dot,::isdigit))return false;
            const std::set<std::string>allowed={"attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight","ffn_up.weight","ffn_gate.weight","ffn_down.weight"};
            return allowed.count(n.substr(dot+1));
        }
    }
    return false;
}
static bool optimizer_uses_muon(T*w,const OptimizerConfig&c){
    return c.kind=="muon"&&(optimizer_hidden_matrix(w)||(c.factorwise_lora&&optimizer_factor_hidden_matrix(w)));
}
static T*optimizer_mm(Graph&g,T*a,T*b){auto*y=ggml_mul_mat(g.ctx.p,a,b);ggml_mul_mat_set_prec(y,GGML_PREC_F32);return y;}
static T*optimizer_polar(Graph&g,T*x,const OptimizerConfig&c,uint64_t step){
    auto*ctx=g.ctx.p;const bool transpose=x->ne[1]>x->ne[0];
    if(transpose)x=ggml_cont(ctx,ggml_transpose(ctx,x));
    auto*norm=ggml_add(ctx,ggml_sqrt(ctx,ggml_sum(ctx,ggml_sqr(ctx,x))),g.constant(c.normalization_eps));
    x=ggml_div(ctx,x,ggml_repeat(ctx,norm,x));
    for(int i=0;i<c.ns_steps;++i){
        auto*a=optimizer_mm(g,x,x);
        auto*b=ggml_add(ctx,ggml_scale(ctx,a,-4.7750f),ggml_scale(ctx,optimizer_mm(g,a,a),2.0315f));
        x=ggml_add(ctx,ggml_scale(ctx,x,3.4445f),optimizer_mm(g,ggml_cont(ctx,ggml_transpose(ctx,x)),b));
    }
    if(c.projection=="fast-refine-v1"&&step>=c.refine_start){
        // Explicit experimental schedule: Jordan's fast finite polynomial first;
        // rescale its bounded singular interval by 1/1.5, then cubic refinement.
        // This is not labelled Polar Express or presumed to improve learning.
        x=ggml_scale(ctx,x,1.f/1.5f);
        for(int i=0;i<c.refine_steps;++i){auto*a=optimizer_mm(g,x,x);
            x=ggml_sub(ctx,ggml_scale(ctx,x,1.5f),ggml_scale(ctx,optimizer_mm(g,ggml_cont(ctx,ggml_transpose(ctx,x)),a),.5f));}
    }
    return transpose?ggml_cont(ctx,ggml_transpose(ctx,x)):x;
}
struct OptimizerBlock {
    Device&device;Context ctx{1024*1024};ggml_backend_buffer_t buffer=nullptr;
    struct State{T*w;T*gradient;T*m;T*v;bool muon;std::string role;};std::vector<State>states;
    OptimizerBlock(Block&b,const OptimizerConfig&c):device(b.device){
        for(auto[w,dw]:b.trainable()){
            const bool factor=c.factorwise_lora&&optimizer_factor_hidden_matrix(w);
            bool muon=optimizer_uses_muon(w,c);
            auto*m=ggml_dup_tensor(ctx.p,w);ggml_set_name(m,(std::string(w->name)+".opt_m").c_str());
            T*v=muon?nullptr:ggml_dup_tensor(ctx.p,w);if(v)ggml_set_name(v,(std::string(w->name)+".opt_v").c_str());
            states.push_back({w,dw,m,v,muon,factor?"hidden_matrix_lora_factor":optimizer_role(w)});
        }
    }
    void allocate(){
        if(states.empty())return;buffer=ggml_backend_alloc_ctx_tensors(ctx.p,device.backend);need(buffer,"Optimizer state allocation failed on "+device.name);
        for(auto&s:states)for(auto*t:{s.m,s.v})if(t)ggml_backend_tensor_memset(t,0,0,ggml_nbytes(t));
    }
    ~OptimizerBlock(){ggml_backend_buffer_free(buffer);}
};
struct NativeOptimizer {
    OptimizerConfig config;uint64_t completed_steps=0;std::vector<std::unique_ptr<OptimizerBlock>>blocks;
    NativeOptimizer(const std::vector<Block*>&model,const OptimizerConfig&c):config(c){need(c.kind!="sgd","SGD retains its existing stateless path");for(auto*b:model)blocks.push_back(std::make_unique<OptimizerBlock>(*b,c));}
    void allocate(){for(auto&b:blocks)b->allocate();}
    json groups()const{json rows=json::array();for(auto&b:blocks)for(auto&s:b->states)rows.push_back({{"name",s.w->name},{"role",s.role},{"optimizer",s.muon?"muon":"adamw"},{"rows",s.w->ne[1]},{"cols",s.w->ne[0]},{"device",b->device.name},{"state_bytes",ggml_nbytes(s.m)+(s.v?ggml_nbytes(s.v):0)}});return rows;}
    json step(){
        need(completed_steps<1000000000,"Optimizer step counter limit");const uint64_t next=completed_steps+1;
        const auto start=std::chrono::steady_clock::now();size_t matrices=0,aux=0;
        for(auto&b:blocks)for(auto&s:b->states){
            Graph g(std::vector<Device*>{&b->device},4096,true);auto*ctx=g.ctx.p;
            if(s.muon){
                auto*m=ggml_add(ctx,ggml_scale(ctx,s.m,config.momentum),ggml_scale(ctx,s.gradient,1-config.momentum));
                auto*u=config.nesterov?ggml_add(ctx,ggml_scale(ctx,m,config.momentum),ggml_scale(ctx,s.gradient,1-config.momentum)):m;
                u=optimizer_polar(g,u,config,next);
                const float rows=s.w->ne[1],cols=s.w->ne[0];
                const float scale=config.scaling=="original"?std::sqrt(std::max(1.f,rows/cols)):.2f*std::sqrt(std::max(rows,cols));
                auto*w=ggml_sub(ctx,ggml_scale(ctx,s.w,1-config.lr*config.decay),ggml_scale(ctx,u,config.lr*scale));
                g.forward(ggml_cpy(ctx,w,s.w));g.forward(ggml_cpy(ctx,m,s.m));++matrices;
            }else{
                const float lr=config.kind=="adamw"?config.lr:config.aux_lr;
                const float wd=(optimizer_hidden_matrix(s.w)||(config.factorwise_lora&&optimizer_factor_hidden_matrix(s.w)))?config.decay:config.aux_decay;
                auto*params=g.input(7,1,{lr,config.beta1,config.beta2,config.eps,wd,float(1/(1-std::pow(double(config.beta1),next))),float(1/(1-std::pow(double(config.beta2),next)))});
                g.forward(ggml_opt_step_adamw(ctx,s.w,s.gradient,s.m,s.v,params));++aux;
            }
            g.compute();
        }
        completed_steps=next;
        return {{"completed_steps",completed_steps},{"muon_matrices",matrices},{"adamw_tensors",aux},{"projection_refinement_active",config.kind=="muon"&&config.projection=="fast-refine-v1"&&next>=config.refine_start},{"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
    }
    size_t state_bytes()const{size_t bytes=0;for(auto&b:blocks)for(auto&s:b->states)bytes+=ggml_nbytes(s.m)+(s.v?ggml_nbytes(s.v):0);return bytes;}
};
