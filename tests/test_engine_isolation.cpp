#include "vonnx/vonnx_engine.hpp"
#include <cassert>
#include <iostream>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
using namespace vonnx;
int run(int argc,char **argv) {
    if(argc!=5)return 2;
    EngineConfig c;c.model_path=argv[1];c.precision_python=argv[2];c.precision_tool=argv[3];c.model_cache_dir=argv[4];
    c.input_height=32;c.input_width=64;c.max_input_height=32;c.max_input_width=64;
    c.max_batch_size=1;c.batch_mode="latency";c.pipeline_depth=1;c.concurrency=4;c.graph_specialization="auto";
    vONNXEngine x(c),n(c);
    std::vector<uint8_t> image(32*64*3);
    FrameTask t{};t.width=64;t.height=32;t.step=64*3;t.data_ptr=image.data();t.data_size=image.size();t.format=PixelFormat::RGB24;
    std::promise<void> entered,release;
    auto signal=release.get_future().share();
    auto xf=x.predict_async(t,[&](auto&){entered.set_value();signal.wait();throw std::runtime_error("isolated callback");});
    bool ok=entered.get_future().wait_for(std::chrono::seconds(5))==std::future_status::ready;
    auto begin=std::chrono::steady_clock::now();
    t.stream_id=9;t.frame_id=73;
    auto nf=n.predict_async(t);
    ok=ok && nf.wait_for(std::chrono::seconds(2))==std::future_status::ready;
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    if(ok) {auto r=nf.get();ok=r.size()==1 && r[0].stream_id==9 && r[0].frame_id==73;}
    release.set_value();xf.get();x.close();n.pause();n.resume();
    auto stats=n.resource_stats();ok=ok && stats.initialization_count==1;n.close();
    std::cout << "native engine isolation " << (ok?"PASS":"FAIL") << "; N completed in " << ms << " ms while X callback was blocked\n";
    return ok?0:1;
}
#ifdef _WIN32
int wmain(int argc,wchar_t **argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    std::vector<std::string> values;std::vector<char*> pointers;
    for(int i=0;i<argc;++i) {
        int size=WideCharToMultiByte(CP_UTF8,0,argv[i],-1,nullptr,0,nullptr,nullptr);
        std::string value(size,'\0');WideCharToMultiByte(CP_UTF8,0,argv[i],-1,value.data(),size,nullptr,nullptr);
        value.pop_back();values.push_back(std::move(value));
    }
    for(auto &value:values)pointers.push_back(value.data());
    try{return run(argc,pointers.data());}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
#else
int main(int argc,char **argv) {return run(argc,argv);}
#endif
