#pragma once
#include <filesystem>
#include <fstream>
#include <set>
#include <cstring>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace vonnx {
// A bounded protobuf wire reader. Tensor payloads are skipped, never interpreted.
struct ProtoField { unsigned id{}, wire{}; uint64_t number{}; const unsigned char* data{}; size_t size{}; };
class ProtoReader {
    const unsigned char *p_, *end_;
    uint64_t varint() {
        uint64_t v=0;
        for (int shift=0; shift<64; shift+=7) {
            if (p_==end_) throw std::invalid_argument("Truncated ONNX protobuf");
            unsigned b=*p_++;
            if (shift==63 && b>1) throw std::invalid_argument("ONNX varint overflow");
            v |= uint64_t(b&127)<<shift;
            if (!(b&128)) return v;
        }
        throw std::invalid_argument("Invalid ONNX varint");
    }
public:
    ProtoReader(const unsigned char* p,size_t n):p_(p),end_(p+n){}
    bool next(ProtoField &f) {
        if (p_==end_) return false;
        auto key=varint(); f={}; f.id=unsigned(key>>3); f.wire=unsigned(key&7);
        if (!f.id) throw std::invalid_argument("Invalid ONNX field");
        if (f.wire==0) {f.number=varint();return true;}
        size_t n=f.wire==1?8:f.wire==5?4:f.wire==2?size_t(varint()):SIZE_MAX;
        if(n>size_t(end_-p_)) throw std::invalid_argument("Invalid ONNX field length");
        f.data=p_; f.size=n; p_+=n; return true;
    }
};
inline void precision_type(std::set<int>& types,uint64_t t) {
    if(t==1 || t==10) types.insert(int(t));
    if(t==11 || t==16 || (t>=17 && t<=24))
        throw std::invalid_argument("Unsupported ONNX floating point type");
}
inline void scan_precision(const unsigned char* data,size_t n,int kind,std::set<int>& types,int depth=0) {
    if(depth>128) throw std::invalid_argument("ONNX graph nesting limit exceeded");
    ProtoReader reader(data,n); ProtoField f;
    // kind: model=0 graph=1 node=2 attribute=3 tensor=4 value=5 type=6 tensor_type=7 sparse=8
    while(reader.next(f)) {
        if(kind==4 && f.id==2 && f.wire==0) precision_type(types,f.number);
        if(kind==7 && f.id==1 && f.wire==0) precision_type(types,f.number);
        // AttributeProto.i also includes Cast.to. Only dtype IDs of interest count
        // when the attribute name is "to" (handled below).
        if(f.wire!=2) continue;
        int child=-1;
        if(kind==0 && f.id==7) child=1;
        if(kind==0 && f.id==25) throw std::invalid_argument("ONNX local functions are not supported");
        if(kind==1) child=f.id==1?2:f.id==5?4:(f.id==11||f.id==12||f.id==13)?5:f.id==15?8:-1;
        if(kind==2 && f.id==5) child=3;
        if(kind==3) child=(f.id==5||f.id==10)?4:(f.id==6||f.id==11)?1:(f.id==22||f.id==23)?8:-1;
        if(kind==5 && f.id==2) child=6;
        if(kind==6 && f.id==1) child=7;
        if(kind==8 && (f.id==1||f.id==2)) child=4;
        if(child>=0) scan_precision(f.data,f.size,child,types,depth+1);
    }
    if(kind==3) {
        ProtoReader attrs(data,n); std::string name; uint64_t to=0;
        while(attrs.next(f)) {
            if(f.id==1 && f.wire==2) name.assign(reinterpret_cast<const char*>(f.data),f.size);
            if(f.id==3 && f.wire==0) to=f.number;
        }
        if(name=="to") precision_type(types,to);
    }
}
inline std::string inspect_precision(const std::string& path) {
    std::ifstream f(std::filesystem::u8path(path),std::ios::binary);
    if(!f) throw std::invalid_argument("Cannot open ONNX model");
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(f)),{});
    std::set<int> types; scan_precision(data.data(),data.size(),0,types);
    if(types.empty()) throw std::invalid_argument("Cannot determine ONNX precision");
    return types.size()==2?"mixed":*types.begin()==10?"fp16":"fp32";
}
#ifdef _WIN32
inline std::wstring win_quote(const std::wstring& text) {
    std::wstring out=L"\""; size_t slashes=0;
    for(auto c:text) {
        if(c==L'\\') {++slashes;continue;}
        out.append(slashes*(c==L'"'?2:1),L'\\'); slashes=0;
        if(c==L'"') out+=L'\\'; out+=c;
    }
    out.append(slashes*2,L'\\'); return out+L'"';
}
inline std::string convert_precision(const std::string& path,const std::string& cache,
                                      const std::string& python,const std::string& tool) {
    wchar_t temp[MAX_PATH],response[MAX_PATH];
    if(!GetTempPathW(MAX_PATH,temp)||!GetTempFileNameW(temp,L"vfp",0,response))
        throw std::runtime_error("Cannot create precision conversion response file");
    struct Cleanup {const wchar_t* p; ~Cleanup(){DeleteFileW(p);}} cleanup{response};
    auto helper=std::filesystem::u8path(tool);
    if(tool.empty()) {
        HMODULE module{}; wchar_t name[32768];
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&convert_precision),&module);
        GetModuleFileNameW(module,name,32768);
        helper=std::filesystem::path(name).parent_path()/L"precision.py";
    }
    if(!std::filesystem::is_regular_file(helper)) throw std::runtime_error("FP16 conversion requires precision.py next to the module, or precision_tool");
    std::wstring exe=python.empty()?L"python.exe":std::filesystem::u8path(python).wstring();
    std::wstring cmd=win_quote(exe)+L" "+win_quote(helper.wstring())+L" "+win_quote(std::filesystem::u8path(path).wstring())+L" "+win_quote(response);
    if(!cache.empty()) cmd+=L" --cache-dir "+win_quote(std::filesystem::u8path(cache).wstring());
    STARTUPINFOW si{};si.cb=sizeof(si); PROCESS_INFORMATION pi{};
    if(!CreateProcessW(nullptr,cmd.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW,nullptr,nullptr,&si,&pi))
        throw std::runtime_error("Cannot start Python FP16 converter; set precision_python to an interpreter with onnx and onnxruntime");
    CloseHandle(pi.hThread);
    auto wait=WaitForSingleObject(pi.hProcess,180000);
    if(wait!=WAIT_OBJECT_0) {TerminateProcess(pi.hProcess,1);WaitForSingleObject(pi.hProcess,5000);CloseHandle(pi.hProcess);throw std::runtime_error("FP16 conversion timed out");}
    DWORD exit{};GetExitCodeProcess(pi.hProcess,&exit);CloseHandle(pi.hProcess);
    std::ifstream result(response,std::ios::binary);
    std::string value((std::istreambuf_iterator<char>(result)),{});
    if(exit || value.empty() || value.rfind("ERROR: ",0)==0) throw std::runtime_error("FP16 conversion failed: "+value);
    return value;
}
#else
inline std::string convert_precision(const std::string&,const std::string&,const std::string&,const std::string&) {
    throw std::runtime_error("Native automatic FP16 conversion currently supports Windows only");
}
#endif
}
