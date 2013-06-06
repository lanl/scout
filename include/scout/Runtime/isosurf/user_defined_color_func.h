#ifndef USER_DEFINED_COLOR_FUNC
#define USER_DEFINED_COLOR_FUNC

#include <scout/Runtime/isosurf/piston/piston_math.h>

typedef psfloat4 (^color_func_t)(float scalar, float min_scalar, float max_scalar);

struct user_defined_color_func : public thrust::unary_function<float, psfloat4>
{
  __host__ __device__
    user_defined_color_func(color_func_t color_func, float min_scalar, float max_scalar) 
    : _color_func(color_func), _min_scalar(min_scalar), _max_scalar(max_scalar) {}

  float _min_scalar;
  float _max_scalar;
  color_func_t _color_func;

  __host__ __device__
    psfloat4 operator()(float scalar)  const
    {
      return _color_func(scalar, _min_scalar, _max_scalar);
    }
};

#endif
