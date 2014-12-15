#include <iostream>

using namespace std;

extern "C" void __scrt_dump_i8(const char* label, int8_t field){
  cerr << label << ": " << int(field) << endl;
}

extern "C" void __scrt_dump_i16(const char* label, int16_t field){
  cerr << label << ": " << field << endl;
}

extern "C" void __scrt_dump_i32(const char* label, int32_t field){
  cerr << label << ": " << field << endl;
}

extern "C" void __scrt_dump_i64(const char* label, int64_t field){
  cerr << label << ": " << field << endl;
}
