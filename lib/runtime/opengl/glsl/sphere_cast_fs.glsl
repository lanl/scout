/*
 *  PointSphere.frag
 *  PointSpheres
 *
 *  Created by Patrick McCormick on 2/6/10.
 *  Copyright 2010 Los Alamos National Laboratory. All rights reserved.
 *
 *
 *   The goal of this shader is to experiment with converting point sprites 
 *   into shaded spheres for reducing the amount of geometry data that is 
 *   sent over the bus.  
 *
 *   The basic idea is to use the quad produced by the sprite into a set of
 *   rays that are then used for simple ray-sphere interesections.  The 
 *   intersection points are then used to modify the depth of each fragment
 *   to produce a correct z-buffer rendition of the sphere. 
 *     
 */

uniform float near, far;

varying vec3 WSc;
varying vec4 center, surfColor;
varying float WSr;
varying float pointSize;
varying float cameraDepth;

void main()
{
  if (pointSize == 0.0) {
    discard;
  } else {
    vec3  Ro = vec3(gl_PointCoord.x, gl_PointCoord.y, 1.0);  
    vec3  Rd = vec3(0.0, 0.0, -1.0);
    vec3  Sc = vec3(0.5, 0.5, 0.5); 
    const float radius = 0.5;
    const float radius_sq = radius * radius;

    // ray-sphere intersection
    vec3  temp = Ro - Sc;  
    float b = 2.0 * dot(Rd, temp);
    float c = dot(temp, temp) - radius_sq;
    float disc = b * b - 4.0 * c;
  
    if (disc < 0.0) 
      discard;
    
    float t0 = (-b - sqrt(disc)) * 0.5;  // only front-most intersection. 
  
    if (t0 < 0.0)
      discard;

    vec3 Ri = Ro + (t0 * Rd);       // ray intersection point.
    vec3 Sn = normalize((Ri - Sc)); // surface normal.

    // fragment depth
    float depth = center.z + Sn.z * WSr;
    float vp0 = 0.5 * (far + near);
    float vp1 = 1.0 / (far - near);
    gl_FragDepth = 0.5 + (vp0 + (far * near / depth)) * vp1;    

    float intensity;
    float Ka = 0.1;
    float Kd = 0.8;
    float Ks = 0.6;
    
    // lighting calculations
    vec4  L  = normalize(vec4(10.6, 0.7, 10.0, 0.0));
    vec3 specColor = vec3(0.8, 0.7, 0.7);
    float specExp = 200.0;
    
    intensity = Ka;
    intensity += Kd * clamp(dot(L.xyz, Sn), 0.0, 1.0);
    surfColor *= intensity;

    if (pointSize > 4.0) {
      vec4 halfV = normalize(L / 2.0);
      intensity = clamp(dot(halfV.xyz, Sn), 0.0, 1.0);
      intensity = Ks * pow(intensity, specExp);
      surfColor.rgb += specColor * intensity;
    }

    const float LOG2 = 1.442695;
    float z = gl_FragCoord.z / gl_FragCoord.w;
    float fogFactor = exp2(-0.00225 * 0.00225 * z * z * LOG2);
    fogFactor = clamp(fogFactor, 0.65, 1.0);
    vec3 fogColor = surfColor.rgb/1000.0;
    surfColor.rgb = mix(fogColor, surfColor.rgb, fogFactor);
  }
    
  gl_FragColor = surfColor;
}
