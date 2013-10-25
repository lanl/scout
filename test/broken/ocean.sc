/*
 * ###########################################################################
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * 
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 * 
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided 
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 * ########################################################################### 
 * 
 * Notes
 *
 * Simple collocated shallow water code based on ocean5.src from 'old Scout'
 * Jamal Mohd-Yusof 2011
 * ##### 
 */ 
 
#include <stdio.h>

#define PRINT_DIAGS 0

#define N_BODIES 1
#define SOLUBLE 1
#define SHOW_SOLID  1
#define M_W 2000
#define M_H 1000
#define uchar unsigned char

uniform mesh OceanMeshType {
cells:
    float h; 
    float u; 
    float v; 
    float psi;
    float f;
    float wind;
    float h_next; 
    float u_next;
    float v_next;
    float psi_next;
    float mask;
    float dx;
};

void ReadFloat(float* in_buf, float* field_min, float* field_max, const char* file_name, const char* var_name) 
{

    FILE* in_file = fopen(file_name, "rb");
    fread(in_buf, sizeof(float), M_W*M_H, in_file);
    fclose(in_file);

    *field_min = 100000.0;
    *field_max = 0.0;
    for (int i=0;i< M_W*M_H;i++) {
        if (in_buf[i] > *field_max) *field_max = in_buf[i];
        if (in_buf[i] < *field_min) *field_min = in_buf[i];

#if (PRINT_DIAGS)
        printf("%f\n", in_buf[i]);
#endif
    }
    printf ("%s ranges from %e to %e\n", var_name, *field_min, *field_max);

}

int main(int argc, char *argv[])
{
    const int NTIME_STEPS     = 10000;
    const float MAX_TEMP      = 520.0f;


    OceanMeshType ocean_mesh[M_W,M_H];

    int c_x[N_BODIES] = {M_W/2};
    int c_y[N_BODIES] = {M_H/2};
    int r2cyl = M_H;

    // bounds for normalization
    float field_min, field_max;
    float u_min, u_max;
    float v_min, v_max;
    float h_min, h_max;
    float psi_min, psi_max;
    float f_min, f_max;
    float dx_min, dx_max;
    float wind_min, wind_max;
    float mask_min, mask_max;

    // multiplier to get things moving in less slow fashion
    float vel_mult = 1.0;

    // This allows us to read in the old .raw files and replicate earlier simulations
    float* in_buf = (float*) malloc (M_W*M_H*sizeof(float));
    // needed for weird uchar mask data we inherited
    uchar* in_uch = (uchar*) malloc (M_W*M_H*sizeof(uchar));

    FILE * in_file;

    // use single buffer to avoid destroying memory

    // read dx
    ReadFloat(in_buf, &dx_min, &dx_max, "data/xfile.raw", "dx");


    forall cells c in ocean_mesh {
        dx = in_buf[c.position.x + M_W*c.position.y];
    }

    // read u
    ReadFloat(in_buf, &u_min, &u_max, "data/ufile.raw", "u");

    forall cells c in ocean_mesh {
        u = vel_mult*in_buf[c.position.x + M_W*c.position.y];
    }

    // read v
    ReadFloat(in_buf, &v_min, &v_max, "data/vfile.raw", "v");

    forall cells c in ocean_mesh {
        v = vel_mult*in_buf[c.position.x + M_W*c.position.y];
    }

    // read h
    ReadFloat(in_buf, &h_min, &h_max, "data/hfile.raw", "h");

    forall cells c in ocean_mesh {
        h = in_buf[c.position.x + M_W*c.position.y];
    }

    // read wind
    ReadFloat(in_buf, &wind_min, &wind_max, "data/wfile.raw", "wind");

    forall cells c in ocean_mesh {
        wind = in_buf[c.position.x + M_W*c.position.y];
    }

    // read f
    ReadFloat(in_buf, &f_min, &f_max, "data/ffile.raw", "f");

    forall cells c in ocean_mesh {
        f = in_buf[c.position.x + M_W*c.position.y];
    }

    // read psi
    ReadFloat(in_buf, &psi_min, &psi_max, "data/pfile.raw", "psi");

    forall cells c in ocean_mesh {
        psi = in_buf[c.position.x + M_W*c.position.y];
    }

    // set default mask value to 1.0 (fluid)

    forall cells c in ocean_mesh {
        mask = 1.0;
    }
    // now we do BC'c here if we want to
    forall cells c in ocean_mesh {
        /*
        // L, R boundaries
        if (c.position.x == 0 || c.position.x == (ocean_mesh.width-1)) {
        h = MAX_TEMP;
        h_next = MAX_TEMP;
        mask = 0.0;
        }
         */

        // top, bottom boundaries
        if (c.position.y == 0 || c.position.y == (ocean_mesh.height-1)) {
            h = MAX_TEMP;
            h_next = MAX_TEMP;
            mask = 0.0;
        }

        for (int i=0;i<N_BODIES;i++) {
            float r2 = (c.position.x - c_x[i])*(c.position.x - c_x[i]) + (c.position.y - c_y[i])*(c.position.y - c_y[i]);
            if (r2 < r2cyl) 
            {
                if (SOLUBLE) {
                    h = 500 + 100*r2/r2cyl;
                } else  {
                    mask = 0.0; 
                }
                h = 1.0*MAX_TEMP;
                h_next = 1.0*MAX_TEMP;
                psi = 2.0;
            }
        }

    }

    // read mask
    in_file = fopen("data/mask_small.raw", "rb");
    fread(in_uch, sizeof(uchar), M_W*M_H, in_file);
    fclose(in_file);

    field_min = 100000.0;
    field_max = 0.0;
    for (int i=0;i< M_W*M_H;i++) {
        if ((float)in_uch[i] > field_max) field_max = (float)in_uch[i];
        if ((float)in_uch[i] < field_min) field_min = (float)in_uch[i];

#if (PRINT_DIAGS)
        printf("%f\n", in_uch[i]);
#endif
    }
    printf ("mask ranges from %e to %e\n", field_min, field_max);
    mask_max = field_max;
    mask_min = field_min;

    forall cells c in ocean_mesh {
        mask = (255.0-(float)in_uch[c.position.x + M_W*c.position.y])/255.0;
    }

    //const float dx    = 1.0;
    const float dy    = dx_min;  // note that since this is 1/dx, we are setting this to == dx at the equator

    printf("dy (uniform) = %e\n", dy);

    const float dt    = 50.0;
    const float grav  = 0.031;
    const float alpha = 10.0;
    const float tau0  = 0.1/1.0;
    const float fb    = 0.0001;
    const float visc  = 300.0;

    // Time steps loop. 
    for(unsigned int n = 0; n < NTIME_STEPS; ++n) {

        forall cells c in ocean_mesh {
            float hn = cshift(c.h,  0, 1);
            float hs = cshift(c.h,  0,-1);
            float he = cshift(c.h,  1, 0);
            float hw = cshift(c.h, -1, 0);

            float un = cshift(c.u,  0, 1);
            float us = cshift(c.u,  0,-1);
            float ue = cshift(c.u,  1, 0);
            float uw = cshift(c.u, -1, 0);

            float vn = cshift(c.v,  0, 1);
            float vs = cshift(c.v,  0,-1);
            float ve = cshift(c.v,  1, 0);
            float vw = cshift(c.v, -1, 0);

            float pn = cshift(c.psi,  0, 1);
            float ps = cshift(c.psi,  0,-1);
            float pe = cshift(c.psi,  1, 0);
            float pw = cshift(c.psi, -1, 0);

            float ul = 0.25*(cshift(c.u, 1, 1) + cshift(c.u, 1, -1) + cshift(c.u, -1, -1) + cshift(c.u, -1, 1));
            float vl = 0.25*(cshift(c.v, 1, 1) + cshift(c.v, 1, -1) + cshift(c.v, -1, -1) + cshift(c.v, -1, 1));

            /*
               float ul = c.u;
               float vl = c.v;
             */

            psi_next = psi -
                (mask)*
                dt*
                (
                 c.u*(pe-pw)*0.5*c.dx + c.v*(pn-ps)*0.5*dy    // advection
                 + alpha*0.25*(                     // diffusion
                     (pe+pw-2.0*psi)*c.dx*c.dx
                     + (pn+ps-2.0*psi)*dy*dy
                     )
                );

            h_next = h - (mask)*dt*0.5*((he*ue - hw*uw)*c.dx + (hn*vn - hs*vs)*dy);

            float nlu = -c.u*(ue-uw)*0.5*c.dx -c.v*(un-us)*0.5*dy ;

            // u(n+1) = u(n) + dt* [ - u*du/dx - v*du/dy + f*v + f_w - g*dh/dx - R*u + alpha*d2u/dx2 ]


            u_next = u + 
                (mask)*
                dt*(
                        nlu 
                        - grav*(he-hw)*0.5*c.dx
                        + alpha*visc*0.25*(              // viscous terms
                            (ue+uw-2.0*u)*c.dx*c.dx
                            + (un+us-2.0*u)*dy*dy
                            )
                        + tau0*c.wind                     // wind forcing 
                        + c.f*vl                           // coriolis
                        - fb*c.u                          // bottom friction
                   );

            float nlv = -c.u*(ve-vw)*0.5*c.dx -c.v*(vn-vs)*0.5*dy;

            // v(n+1) = v(n) + dt* [ - u*dv/dx - v*dv/dy - f*u - g*dh/dy - R*v + alpha*d2v/dx2 ]

            v_next = v +
                (mask)*
                dt*(
                        nlv 
                        - grav*(hn-hs)*0.5*dy
                        + alpha*visc*0.25*(              // viscous terms
                            (ve+vw-2.0*v)*c.dx*c.dx
                            + (vn+vs-2.0*v)*dy*dy
                            )
                        - c.f*ul                           // coriolis
                        - fb*c.v                          // bottom friction
                   );

        }


        forall cells c in ocean_mesh {
            h = h_next;
            u = u_next;
            v = v_next;
            psi = psi_next;
        }
        
        renderall cells c in ocean_mesh {
            //float norm_h = 0.5*(psi - psi_min)/(psi_max-psi_min);
            //float norm_h = 0.5*(u - u_min)/(u_max-u_min);
            //float norm_h = 0.5*(v - v_min)/(v_max-v_min);
            float norm_h = 0.5*(h - h_min + 10)/(h_max-h_min);
            //float norm_h = 0.5*(f - f_min)/(f_max-f_min);
            //float norm_h = 0.5*(wind - wind_min)/(wind_max-psi_min);

            float hue = 240.0f - 240.0f * norm_h;
#if (SHOW_SOLID)
            color = hsv(hue, 1.0f, mask);
#else
            color = hsv(hue, 1.0f, 1.0f);
#endif
        }
        if (n%10 == 0) {
            printf("%u..", n);
            fflush(stdout);
        }

    }
    printf("\n");

    return 0;
}
