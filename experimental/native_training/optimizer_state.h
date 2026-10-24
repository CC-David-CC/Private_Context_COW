// Completed-update optimizer checkpoint. No periodic/mid-step transaction claim.
#pragma once
static std::string native_digest_hex(const unsigned char*bytes,size_t size){
    sha256_t state;sha256_init(&state);sha256_update(&state,bytes,size);unsigned char raw[32];sha256_final(&state,raw);
    static const char hex[]="0123456789abcdef";std::string out;for(auto v:raw){out+=hex[v>>4];out+=hex[v&15];}return out;
}
static std::string native_digest_final(sha256_t state){unsigned char raw[32];sha256_final(&state,raw);static const char h[]="0123456789abcdef";std::string out;for(auto v:raw){out+=h[v>>4];out+=h[v&15];}return out;}
static bool native_digest_valid(const std::string&s){return s.size()==64&&std::all_of(s.begin(),s.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});}
static std::string native_file_sha256(const std::string&path){
    int fd=::open(path.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC);need(fd>=0,"Cannot read pinned regular artifact");
    struct stat info;if(fstat(fd,&info)!=0||!S_ISREG(info.st_mode)){::close(fd);throw std::runtime_error("Pinned artifact must be a regular file");}
    sha256_t digest;sha256_init(&digest);std::vector<unsigned char>chunk(1<<20);
    try{for(;;){ssize_t n=::read(fd,chunk.data(),chunk.size());need(n>=0,"Pinned artifact read failed");if(!n)break;sha256_update(&digest,chunk.data(),size_t(n));}}
    catch(...){::close(fd);throw;}::close(fd);return native_digest_final(digest);
}
static json optimizer_state_layout(const NativeOptimizer&optimizer){
    json rows=json::array();for(auto&b:optimizer.blocks)for(auto&s:b->states)for(auto*t:{s.m,s.v})if(t)
        rows.push_back({{"name",t->name},{"shape",{t->ne[0],t->ne[1]}},{"role",s.role},{"optimizer",s.muon?"muon":"adamw"},{"bytes",ggml_nbytes(t)}});
    return rows;
}
static void optimizer_little_endian(){uint16_t value=1;need(*reinterpret_cast<unsigned char*>(&value)==1&&sizeof(float)==4,"Optimizer state requires native little-endian IEEE F32");}
static std::string save_optimizer_state(const NativeOptimizer&optimizer,const std::string&path,const std::string&model_hash,const std::string&dataset_hash,uint64_t cursor){
    optimizer_little_endian();need(native_digest_valid(model_hash)&&native_digest_valid(dataset_hash),"Pinned optimizer candidate/dataset hashes required");
    json header={{"schema",1},{"kind","pcc.completed-optimizer-state.v1"},{"storage","le-f32"},{"model_sha256",model_hash},{"dataset_sha256",dataset_hash},{"example_cursor",cursor},{"completed_steps",optimizer.completed_steps},{"config",optimizer.config.canonical()},{"tensors",optimizer_state_layout(optimizer)}};
    std::string metadata=header.dump();need(metadata.size()<=4*1024*1024,"Optimizer state metadata too large");
    int fd=::open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW|O_CLOEXEC,0600);need(fd>=0,"Optimizer output already exists or cannot be created");
    sha256_t digest;sha256_init(&digest);
    auto write=[&](const void*data,size_t count){auto*p=static_cast<const unsigned char*>(data);sha256_update(&digest,p,count);while(count){ssize_t n=::write(fd,p,count);need(n>0,"Optimizer state write failed");p+=n;count-=n;}};
    try{
        write("PCCOPT1\n",8);uint64_t count=metadata.size();unsigned char length[8];for(int i=0;i<8;++i)length[i]=(count>>(8*i))&255;write(length,8);write(metadata.data(),metadata.size());
        for(auto&b:optimizer.blocks)for(auto&s:b->states)for(auto*t:{s.m,s.v})if(t){auto values=read_tensor(t);need(std::all_of(values.begin(),values.end(),[](float v){return std::isfinite(v);}),"Nonfinite optimizer state cannot be checkpointed");write(values.data(),values.size()*4);}
        need(::fsync(fd)==0,"Optimizer checkpoint fsync failed");::close(fd);fd=-1;
        auto parent=std::filesystem::absolute(path).parent_path();int directory=::open(parent.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC);need(directory>=0,"Cannot open checkpoint directory");int result=::fsync(directory);::close(directory);need(result==0,"Checkpoint directory fsync failed");
    }catch(...){if(fd>=0)::close(fd);throw;}
    return native_digest_final(digest);
}
static void load_optimizer_state(NativeOptimizer&optimizer,const std::string&path,const std::string&expected_hash,const std::string&model_hash,const std::string&dataset_hash,uint64_t cursor){
    optimizer_little_endian();need(native_digest_valid(expected_hash)&&native_digest_valid(model_hash)&&native_digest_valid(dataset_hash),"Pinned optimizer resume identities required");
    int fd=::open(path.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC);need(fd>=0,"Cannot open optimizer resume state");
    struct stat info;if(fstat(fd,&info)!=0||!S_ISREG(info.st_mode)){::close(fd);throw std::runtime_error("Optimizer resume state must be regular");}sha256_t digest;sha256_init(&digest);
    auto read=[&](void*data,size_t count){auto*p=static_cast<unsigned char*>(data);size_t total=count;while(count){ssize_t n=::read(fd,p,count);need(n>0,"Truncated optimizer resume state");p+=n;count-=n;}sha256_update(&digest,static_cast<unsigned char*>(data),total);};
    try{
        unsigned char magic[8],length[8];read(magic,8);need(std::memcmp(magic,"PCCOPT1\n",8)==0,"Unknown optimizer state format");read(length,8);uint64_t size=0;for(int i=0;i<8;++i)size|=uint64_t(length[i])<<(8*i);need(size>0&&size<=4*1024*1024,"Bounded optimizer header required");
        std::string metadata(size,'\0');read(metadata.data(),metadata.size());auto header=json::parse(metadata);
        const std::set<std::string>keys={"schema","kind","storage","model_sha256","dataset_sha256","example_cursor","completed_steps","config","tensors"};
        need(header.is_object()&&header.size()==keys.size(),"Unexpected optimizer checkpoint metadata");for(auto&key:keys)need(header.contains(key),"Missing optimizer checkpoint metadata");
        need(header.at("schema")==1&&header.at("kind")=="pcc.completed-optimizer-state.v1"&&header.at("storage")=="le-f32","Unknown optimizer state schema");
        need(header.at("model_sha256")==model_hash,"Optimizer state belongs to different candidate weights");
        need(header.at("dataset_sha256")==dataset_hash,"Optimizer state belongs to a different ordered dataset");
        need(header.at("example_cursor").is_number_unsigned()&&header.at("example_cursor").get<uint64_t>()==cursor,"Optimizer resume data cursor mismatch");
        need(header.at("config")==optimizer.config.canonical(),"Optimizer resume configuration mismatch");
        need(header.at("tensors")==optimizer_state_layout(optimizer),"Optimizer resume tensor coverage/roles mismatch");
        need(header.at("completed_steps").is_number_unsigned(),"Optimizer step must be an unsigned integer");uint64_t step=header.at("completed_steps").get<uint64_t>();need(step>0&&step<=1000000000,"Invalid completed optimizer step");
        for(auto&b:optimizer.blocks)for(auto&s:b->states)for(auto*t:{s.m,s.v})if(t){std::vector<float>values(ggml_nelements(t));read(values.data(),values.size()*4);need(std::all_of(values.begin(),values.end(),[](float v){return std::isfinite(v);}),"Nonfinite optimizer resume state");put(t,values);}
        unsigned char extra;need(::read(fd,&extra,1)==0,"Unexpected trailing optimizer state bytes");::close(fd);fd=-1;
        need(native_digest_final(digest)==expected_hash,"Optimizer resume SHA-256 mismatch");optimizer.completed_steps=step;
    }catch(...){if(fd>=0)::close(fd);throw;}
}
