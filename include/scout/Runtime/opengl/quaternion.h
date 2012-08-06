/*
 *           -----  The Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 * 
 *-----
 *
 * $Revision$
 * $Date$
 * $Author$
 *
 *-----
 * 
 */

#ifndef SCOUT_QUATERNION_H_
#define SCOUT_QUATERNION_H_

#include "scout/Runtime/opengl/vectors.h"
#include "scout/Runtime/opengl/matrix.h"


namespace scout
{
  template< class T>
  class quaternion
  {
   public:

    quaternion()
        : x(0.0), y(0.0), z(0.0), w(0.0)
    { }

    quaternion(const T v[4])
    {
      _values[0] = v[0];
      _values[1] = v[1];
      _values[2] = v[2];
      _values[3] = v[3];
    }

    quaternion(T q0, T q1, T q2, T q3)
    {
      setValue(q0, q1, q2, q3);
    }

    quaternion(const glvec3<T> &axis, T radians)
    { setValue(axis, radians); }

    quaternion(const glvec3<T> &from, const glvec3<T> &to)
    { setValue(from, to); }

    quaternion(const glvec3<T> &from, const glvec3<T> & from_up,
               const glvec3<T>& to, const glvec3<T>& to_up)
    { setValue(from, from_up, to, to_up); }

    quaternion(const matrix4<T> &m)
    { setValue(m); }
    
    const T * getValue() const
    { return  &_values[0]; }

    void getValue(T &q0, T &q1, T &q2, T &q3) const
    {
      q0 = _values[0];
      q1 = _values[1];
      q2 = _values[2];
      q3 = _values[3];
    }

    quaternion &setValue(T q0, T q1, T q2, T q3)
    {
      _values[0] = q0;
      _values[1] = q1;
      _values[2] = q2;
      _values[3] = q3;
      return *this;
    }

    void getValue(glvec3<T> &axis, T &radians) const
    {
      radians = T(acos(_values[3]) * T(2.0));
      
      if (radians == T(0.0)) {
        axis = glvec3<T>(0.0, 0.0, 1.0);
      } else {
        axis[0] = _values[0];
        axis[1] = _values[1];
        axis[2] = _values[2];
        axis = normalize(axis);
      }
    }

    quaternion & setValue(const T * qp)
    {
      for (int i = 0; i < 4; i++) _values[i] = qp[i];

      return *this;
    }

    quaternion & setValue(const glvec3<T> &axis, T theta)
    {
      T sqnorm = square_norm(axis);

      if (sqnorm == T(0.0))
      {
        // axis too small.
        x = y = z = T(0.0);
        w = T(1.0);
      } 
      else 
      {
        theta *= T(0.5);
        T sin_theta = T(sin(theta));

        if (sqnorm != T(1)) 
          sin_theta /= T(sqrt(sqnorm));
        x = sin_theta * axis[0];
        y = sin_theta * axis[1];
        z = sin_theta * axis[2];
        w = T(cos(theta));
      }
      return *this;
    }

    quaternion & setValue(const glvec3<T> & rotateFrom, const glvec3<T> & rotateTo)
    {
      glvec3<T> p1, p2;
      T alpha;

      p1 = normalize(rotateFrom);
      p2 = normalize(rotateTo);

      alpha = dot(p1, p2);

      if(alpha == T(1.0)) {
        *this = quaternion(); 
        return *this; 
      }

      // ensures that the anti-parallel case leads to a positive dot
      if(alpha == T(-1.0)) {
        glvec3<T> v;

        if(p1[0] != p1[1] || p1[0] != p1[2])
          v = glvec3<T>(p1[1], p1[2], p1[0]);
        else
          v = glvec3<T>(-p1[0], p1[1], p1[2]);

        v -= p1 * dot(p1, v);
        v = normalize(v);

        setValue(v, T(3.1415926));
        return *this;
      }

      p1 = normalize(cross(p1, p2));  
        
      setValue(p1,T(acos(alpha)));

      return *this;
    }

    quaternion & setValue(const glvec3<T> & from_look, const glvec3<T> & from_up,
                          const glvec3<T> & to_look, const glvec3<T> & to_up)
    {
      quaternion r_look = quaternion(from_look, to_look);

      glvec3<T> rotated_from_up(from_up);
      r_look.mult_glvec(rotated_from_up);

      quaternion r_twist = quaternion(rotated_from_up, to_up);

      *this = r_twist;
      *this *= r_look;
      return *this;
    }

    quaternion & setValue(const matrix4<T> &m)
    {
      T tr, s;
      int i, j, k;
      const int nxt[3] = { 1, 2, 0 };

      tr = m(0,0) + m(1,1) + m(2,2);

      if ( tr > T(0) )
      {
        s = T(sqrt( tr + m(3,3) ));
        _values[3] = T ( s * 0.5 );
        s = T(0.5) / s;

        _values[0] = T ( ( m(1,2) - m(2,1) ) * s );
        _values[1] = T ( ( m(2,0) - m(0,2) ) * s );
        _values[2] = T ( ( m(0,1) - m(1,0) ) * s );
      }
      else
      {
        i = 0;
        if ( m(1,1) > m(0,0) )
          i = 1;

        if ( m(2,2) > m(i,i) )
          i = 2;

        j = nxt[i];
        k = nxt[j];

        s = T(sqrt( ( m(i,j) - ( m(j,j) + m(k,k) )) + T(1.0) ));

        _values[i] = T ( s * 0.5 );
        s = T(0.5 / s);

        _values[3] = T ( ( m(j,k) - m(k,j) ) * s );
        _values[j] = T ( ( m(i,j) + m(j,i) ) * s );
        _values[k] = T ( ( m(i,k) + m(k,i) ) * s );
      }

      return *this;
    }

    quaternion & operator *= (const quaternion<T> & qr)
    {
      quaternion ql(*this);

      w = ql.w * qr.w - ql.x * qr.x - ql.y * qr.y - ql.z * qr.z;
      x = ql.w * qr.x + ql.x * qr.w + ql.y * qr.z - ql.z * qr.y;
      y = ql.w * qr.y + ql.y * qr.w + ql.z * qr.x - ql.x * qr.z;
      z = ql.w * qr.z + ql.z * qr.w + ql.x * qr.y - ql.y * qr.x;

      return *this;
    }

    friend quaternion normalize(const quaternion<T> &q)
    {
      quaternion r(q);
      T rnorm = T(1.0) / T(sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z));
        
      r.x *= rnorm;
      r.y *= rnorm;
      r.z *= rnorm;
      r.w *= rnorm;
    }

    friend quaternion<T> conjugate(const quaternion<T> & q)
    {
      quaternion<T> r(q);
      r._values[0] *= T(-1.0);
      r._values[1] *= T(-1.0);
      r._values[2] *= T(-1.0);
      return r;
    }

    friend quaternion<T> inverse(const quaternion<T> & q)
    {
      return conjugate(q);
    }

    //
    // Quaternion multiplication with cartesian glvector
    // v' = q*v*q(star)
    //
    void multVec(const glvec3<T> &src, glvec3<T> &dst) const
    {
      T v_coef = w * w - x * x - y * y - z * z;                     
      T u_coef = T(2.0) * (src[0] * x + src[1] * y + src[2] * z);  
      T c_coef = T(2.0) * w;                                       

      dst.v[0] = v_coef * src.v[0] + u_coef * x + c_coef *
        (y * src.v[2] - z * src.v[1]);
      dst.v[1] = v_coef * src.v[1] + u_coef * y + c_coef *
        (z * src.v[0] - x * src.v[2]);
      dst.v[2] = v_coef * src.v[2] + u_coef * z + c_coef *
        (x * src.v[1] - y * src.v[0]);
    }

    void multVec(glvec3<T> & src_and_dst) const
    { mult_glvec(glvec3<T>(src_and_dst), src_and_dst); }

    void scaleAngle(T scaleFactor)
    {
      glvec3<T> axis;
      T radians;

      get_value(axis, radians);
      radians *= scaleFactor;
      setValue(axis, radians);
    }

    friend quaternion<T> slerp(const quaternion<T> & p,
                               const quaternion<T> & q,
                               T alpha)
    {
      quaternion r;

      T cos_omega = p.x * q.x + p.y * q.y + p.z * q.z + p.w * q.w;
      // if B is on opposite hemisphere from A, use -B instead

      int bflip;
      if ((bflip = (cos_omega < T(0))))
        cos_omega = -cos_omega;

      // complementary interpolation parameter
      T beta = T(1) - alpha;     

      if(cos_omega >= T(1))
        return p;

      T omega = T(acos(cos_omega));
      T one_over_sin_omega = T(1.0) / T(sin(omega));

      beta    = T(sin(omega*beta)  * one_over_sin_omega);
      alpha   = T(sin(omega*alpha) * one_over_sin_omega);

      if (bflip)
        alpha = -alpha;

      r.x = beta * p._values[0]+ alpha * q._values[0];
      r.y = beta * p._values[1]+ alpha * q._values[1];
      r.z = beta * p._values[2]+ alpha * q._values[2];
      r.w = beta * p._values[3]+ alpha * q._values[3];
      return r;
    }

    T & operator [](int i)
    {
      return _values[i];
    }

    const T & operator [](int i) const
    {
      return _values[i];
    }

    
    friend bool operator == (const quaternion<T> & lhs,
                             const quaternion<T> & rhs)
    {
      bool r = true;
      for (int i = 0; i < 4; i++)
        r &= lhs._values[i] == rhs._values[i]; 
      return r;
    }

    friend bool operator != (const quaternion<T> & lhs,
                             const quaternion<T> & rhs)
    {
      bool r = true;
      for (int i = 0; i < 4; i++)
        r &= lhs._values[i] == rhs._values[i]; 
      return r;
    }

    friend quaternion<T> operator * (const quaternion<T> & lhs,
                                     const quaternion<T> & rhs)
    {	
      quaternion r(lhs); 
      r *= rhs; 
      return r; 
    }

    union {
      struct {
        T x;
        T y;
        T z;
        T w;
      };
      T _values[4];
    };
  };

  typedef quaternion<float> quatf;
  typedef quaternion<double> quatd;
}

#endif
