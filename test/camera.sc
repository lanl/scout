
using namespace std;

//SC_TODO: temporary workaround for broken scout vectors in camera
typedef float float3d __attribute__((ext_vector_type(3)));

int main(int argc, char** argv){

  float3d mypos;
  mypos.x = 350.0;
  mypos.y = -100.0;
  mypos.z = 650.0;

  float3d mylookat;
  mylookat.x = 350.0; 
  mylookat.y = 200.0; 
  mylookat.z = 25.0; 

  float3d myup;
  myup.x = -1.0;
  myup.y = 0.0;
  myup.z = 0.0;
  
  camera cam {
    near = 70.0;
    far = 500.0;
    fov = 40.0;
    pos = mypos;
    lookat = mylookat;
    up = myup;
  };
  return 0;
}
