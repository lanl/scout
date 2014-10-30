typedef float float4 __attribute__((ext_vector_type(4)));

extern "C" {
  float4 hsva(float, float, float, float);
  float4 hsv(float, float, float);
}
