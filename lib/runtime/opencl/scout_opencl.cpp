#include "runtime/opencl/scout_opencl.h"

#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <elf.h>

using namespace std;

bool __sc_opencl = false;
cl_context __sc_opencl_context;
cl_program __sc_opencl_program;
cl_command_queue __sc_opencl_command_queue;
cl_device_id __sc_opencl_device_id;

namespace{

  struct ELFSection{
    string name;
    size_t offset;
    size_t size;
    const char* data;
  };

  typedef vector<ELFSection> ELFSectionVec;

  // assumes elfImage is a 32-bit AMD OpenCL ELF image
  static void readELF(const char* elfImage, ELFSectionVec& sectionVec){
    Elf32_Ehdr* header = (Elf32_Ehdr*)elfImage;
    
    size_t headerSize = sizeof(Elf32_Ehdr);
    size_t numSections = header->e_shnum;
    
    sectionVec.push_back(ELFSection());
    ELFSection& firstSection = sectionVec.back();
    firstSection.name = "";
    
    size_t i = headerSize + 1;
    for(;;){
      sectionVec.push_back(ELFSection());
      ELFSection& section = sectionVec.back();
      section.name = "";
      
      while(elfImage[i] != 0){
	section.name += elfImage[i];
	++i;
      }
      
      ++i;
      
      if(elfImage[i] == 0){
	break;
      }
    }
    
    i = header->e_shoff;
    
    for(size_t j = 0; j < numSections; ++j){
      Elf32_Shdr* sectionHeader = (Elf32_Shdr*)&elfImage[i];
      ELFSection& section = sectionVec[j];
      i += sizeof(Elf32_Shdr);
      section.offset = sectionHeader->sh_offset;
      section.size = sectionHeader->sh_size;
      section.data = elfImage + section.offset;
    }
  }
  
  // assumes that the same number of sections exists in the new image
  // to be created with the same names, in sectionVec input offsets
  // are ignored, but updated based on sizes and data which are expected
  // to be passed in sectionVec
  
  char* updateELF(const char* oldImage,
		  ELFSectionVec& sectionVec,
		  size_t& size){
    
    Elf32_Ehdr* oldHeader = (Elf32_Ehdr*)oldImage;

    size_t numSections = sectionVec.size();
    
    size = sizeof(Elf32_Ehdr);
    for(size_t i = 0; i < numSections; ++i){
      const ELFSection& section = sectionVec[i];
      // .symtab section is aligned to 8 byte boundary
      if(section.name == ".symtab"){
	while(size % 8 != 0){
	  ++size;
	}
      }
      size += section.size;
    }
    
    // start of section headers aligned to a 4-byte boundary
    while(size % 4 != 0){
      ++size;
    }
    
    size_t sectionHeaderSize = sizeof(Elf32_Shdr) * sectionVec.size();
    size += sectionHeaderSize;
    
    char* newImage = (char*)malloc(size);
    memcpy(newImage, oldImage, sizeof(Elf32_Ehdr));
    Elf32_Ehdr* newHeader = (Elf32_Ehdr*)newImage;
    size_t offset = sizeof(Elf32_Ehdr);
    
    for(size_t i = 0; i < numSections; ++i){
      ELFSection& section = sectionVec[i];
      // .symtab section is aligned to 8 byte boundary
      if(section.name == ".symtab"){
	while(offset % 8 != 0){
	  newImage[offset] = 0;
	  ++offset;
	}
      }

      if(section.size > 0){
	memcpy(newImage + offset, section.data, section.size);
	section.offset = offset;
	offset += section.size;
      }
    }
    
    // start of section headers aligned to 4-byte boundary
    while(offset % 4 != 0){
      newImage[offset] = 0;
      ++offset;
    }

    newHeader->e_shoff = offset;
    
    memcpy(newImage + offset, 
	   oldImage + oldHeader->e_shoff, sectionHeaderSize);
    
    for(size_t i = 0; i < numSections; ++i){
      Elf32_Shdr* sectionHeader = (Elf32_Shdr*)(newImage + offset);
      const ELFSection& section = sectionVec[i];
      sectionHeader->sh_offset = section.offset; 
      sectionHeader->sh_size = section.size;
      offset += sizeof(Elf32_Shdr);
    }
    
    for(size_t i = 0; i < 3; ++i){
      newImage[offset++] = 0;
    }
    
    return newImage;
  }
  
} // end namespace

void __sc_init_opencl(){
  __sc_opencl = true;

  cl_uint numPlatforms;

  cl_int ret = clGetPlatformIDs(0, NULL, &numPlatforms);
  assert(ret == CL_SUCCESS);

  assert(numPlatforms > 0 && "No OpenCL platforms available");

  cl_platform_id platform;

  ret = clGetPlatformIDs(1, &platform, NULL);
  assert(ret == CL_SUCCESS && "Error getting OpenCL platform IDs");
  
  cl_uint numDevices;

  ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  assert(ret == CL_SUCCESS && "Error getting # of OpenCL devices");
  assert(numDevices > 0 && "No OpenCL GPU devices");
  
  ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
                       1, &__sc_opencl_device_id, NULL);
  assert(ret == CL_SUCCESS && "Error getting first OpenCL device");
  
  __sc_opencl_context = 
    clCreateContext(NULL, 1, &__sc_opencl_device_id, NULL, NULL, &ret);
  assert(ret == CL_SUCCESS && "Error creating OpenCL context");
  
  __sc_opencl_command_queue = 
    clCreateCommandQueue(__sc_opencl_context, __sc_opencl_device_id,
                         0, &ret);
  assert(ret == CL_SUCCESS && "Error creating OpenCL command queue");
}

extern "C"
void __sc_opencl_build_program(const void* bitcode, size_t size){
  // create a stub kernel so we can generate the AMD binary ELF image
  // and fill it in with the real program LLVM IR bitcode
  static const char* stub = "__kernel void stub(__global int* a){}";

  cl_int ret;

  size_t stubSize = strlen(stub);
  cl_program stubProgram = 
    clCreateProgramWithSource(__sc_opencl_context, 1, 
                              (const char**)&stub, &stubSize, &ret);
  assert(ret == CL_SUCCESS && "Error creating OpenCL binary stub");

  clBuildProgram(stubProgram, 1, &__sc_opencl_device_id, 
                 "-fno-bin-amdil -fno-bin-exe", NULL, NULL);

  size_t stubImageSize;
  size_t numDevices = 1;
  ret = clGetProgramInfo(stubProgram, CL_PROGRAM_BINARY_SIZES, 
                         sizeof(size_t), &stubImageSize, &numDevices);
  assert(ret == CL_SUCCESS && "Error reading OpenCL stub size");

  char* stubImage = (char*)malloc(sizeof(char)*stubImageSize);

  size_t numPrograms = 1;
  ret = 
    clGetProgramInfo(stubProgram, CL_PROGRAM_BINARIES, 
                     stubImageSize, stubImage, &numPrograms);
  assert(ret == CL_SUCCESS && "Error reading OpenCL stub program");

  ELFSectionVec sections;
  readELF(stubImage, sections);
  ELFSection* irSection = 0;
  for(size_t i = 0; i < sections.size(); ++i){
    if(sections[i].name == ".llvmir"){
      irSection = &sections[i];
      break;
    }
  }
  assert(irSection && "Failed to find .llvmir section");

  irSection->data = (const char*)bitcode;
  irSection->size = size;
  
  size_t newSize;
  const unsigned char* newImage = 
    (const unsigned char*)updateELF(stubImage, sections, newSize);
  
  free(stubImage);
  clReleaseProgram(stubProgram);

  cl_int status;
  
  __sc_opencl_program = 
    clCreateProgramWithBinary(__sc_opencl_context,
                              1,
                              &__sc_opencl_device_id,
                              &newSize,
                              &newImage,
                              &status,
                              &ret);

  assert(status == CL_SUCCESS && ret == CL_SUCCESS &&
         "Error creating OpenCL program");
}

extern "C"
void* __sc_get_opencl_device_ptr(const char* meshName,
                                 const char* fieldName){
  
}

extern "C"
void __sc_put_opencl_device_ptr(const char* meshName,
                                const char* fieldName,
                                void* memPtr){

}

