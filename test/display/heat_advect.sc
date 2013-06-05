/*
 * -----  Scout Programming Language -----
 *
 * This file is distributed under an open source license by Los Alamos
 * National Security, LCC.  See the file License.txt (located in the
 * top level of the source distribution) for details.
 *
 *-----
 *
 * Simplistic 2D heat transfer...
 * Modified to include (quasi-)soluble bodies and constant advection term
 * Jamal Mohd-Yusof 10/20/11
 * Modified to include full u,v advection with compressible fluid (no Poisson solve)
 * Jamal Mohd-Yusof 10/21/11
 *
 */

#include <stdio.h>

#define N_BODIES 5
#define SOLUBLE 0
#define SHOW_SOLID 1 // (1-SOLUBLE)
#define M_W 512
#define M_H 512

int main(int argc, char *argv[])
{
    uniform mesh HeatMeshType{
cells:
        float h, P; // enthalpy, pressure, density
        float h_next;
        float u, u_next;
        float v, v_next;
        float rho, rho_next;
        float mask;
    };

    HeatMeshType heat_mesh[M_W,M_H];

    // define simulation parameters
    const int NTIME_STEPS     = 100;
    const float MAX_TEMP      = 100.0;
    const float MAX_U         = 0.001;
    const float domain_width  = 30.0;
    const float domain_height = 15.0;

    const float dx    = domain_width / M_W;
    const float dy    = domain_height / M_H;

    const float R = 0.00000001;
    const float alpha = 0.00005;
    const float mu = alpha;

    float dxdymin = !(dx<dy)?dy:dx;
    const float dt    = 0.1f * (dxdymin*dxdymin)/2.0f/(alpha + mu);

    printf ("dx = %e, dy = %e, alpha = %e, dt = %e\n", dx, dy, alpha, dt);

    int c_x[N_BODIES] = {M_W/4 + 10 , M_W/4, 3*M_W/4 + 10 , 3*M_W/4, M_W/2};
    int c_y[N_BODIES] = {M_H/4, 3*M_H/4 + 10, M_H/4, 3*M_H/4, M_H/2 + 10};
    int r2cyl = M_W/8;
    int i;

    // Initial condition setup
    forall cells c of heat_mesh {
        h = 0.0f;
        h_next = 0.0f;
        u = MAX_U;
        v = 0.0001;
        u_next = MAX_U;
        v_next = 0.0001;
        rho = 1.0;
        rho_next = 1.0;
        mask = 1.0;

        if (c.position.y == 0 || c.position.y == (heat_mesh.height-1)) {
            h = MAX_TEMP;
            h_next = MAX_TEMP;
            mask = 0.0;
            u = 0.0;
            v = 0.0;
            u_next = 0.0;
            v_next = 0.0;
        }

        for (int i=0;i<N_BODIES;i++) {
            float r2 = (c.position.x - c_x[i])*(c.position.x - c_x[i]) + (c.position.y - c_y[i])*(c.position.y - c_y[i]);
            if (r2 < r2cyl)
            {
                if (SOLUBLE) {
                    mask = r2/r2cyl;
                } else  {
                    mask = 0.0;
                }
                h = MAX_TEMP;
                h_next = MAX_TEMP;
                u = 0.0;
                v = 0.0;
                u_next = 0.0;
                v_next = 0.0;
            }
        }
    }

    forall cells c of heat_mesh {
        // simple EOS
        P = R*rho*h;
    }

    // Time steps loop.
    for(unsigned int n = 0; n < NTIME_STEPS; ++n) {

        forall cells c of heat_mesh {
            // compute new pressure (EOS)
            P = R*rho*h;
        }

        forall cells c of heat_mesh {
            // compute new velocities
            // u derivatives
            float dudx = 0.5*(cshift(c.u, 1, 0) - cshift(c.u, -1,  0))/dx;
            float dudy = 0.5*(cshift(c.u, 0, 1) - cshift(c.u,  0, -1))/dy;

            float d2udx2 = cshift(c.u, 1, 0) - 2.0f * c.u + cshift(c.u, -1,  0);
            d2udx2 /= dx * dx;

            float d2udy2 = cshift(c.u, 0, 1) - 2.0f * c.u + cshift(c.u,  0, -1);
            d2udy2 /= dy * dy;

            // v derivatives
            float dvdx = 0.5*(cshift(c.v, 1, 0) - cshift(c.v, -1,  0))/dx;
            float dvdy = 0.5*(cshift(c.v, 0, 1) - cshift(c.v,  0, -1))/dy;

            float d2vdx2 = cshift(c.v, 1, 0) - 2.0f * c.v + cshift(c.v, -1,  0);
            d2vdx2 /= dx * dx;

            float d2vdy2 = cshift(c.v, 0, 1) - 2.0f * c.v + cshift(c.v,  0, -1);
            d2vdy2 /= dy * dy;

            // P derivatives
            float dPdx = 0.5*(cshift(c.P, 1, 0) - cshift(c.P, -1,  0))/dx;
            float dPdy = 0.5*(cshift(c.P, 0, 1) - cshift(c.P,  0, -1))/dy;

            float du = (u*dudx + v*dudy + dPdx - mu*(d2udx2 + d2udy2));
            float dv = (u*dvdx + v*dvdy + dPdy - mu*(d2vdx2 + d2vdy2));

            // compute new enthalpy
            float dhdx = 0.5*(cshift(c.h, 1, 0) - cshift(c.h, -1, 0))/dx;

            float d2hdx2 = cshift(c.h, 1, 0) - 2.0f * c.h + cshift(c.h, -1,  0);
            d2hdx2 /= dx * dx;

            float d2hdy2 = cshift(c.h, 0, 1) - 2.0f * c.h + cshift(c.h,  0, -1);
            d2hdy2 /= dy * dy;

            u_next = u - mask*dt*du;
            v_next = v - mask*dt*dv;

            h_next = mask*dt*(alpha * (d2hdx2 + d2hdy2) - mask*u*dhdx) + c.h;

            rho_next = rho - dt*(dudx + dvdy);

        }

        forall cells c of heat_mesh {
            h = h_next;
            u = u_next;
            v = v_next;
            rho = rho_next;
        }

        renderall cells c of heat_mesh {
            // Temperature
            float norm_h = h / MAX_TEMP;
            // Density
            //float norm_h = rho / 10.0;
            // U
            //float norm_h = 0.5*(u + MAX_U) / MAX_U;
            // V
            // float norm_h = 0.5*(v + MAX_U) / MAX_U;

            float hue = 240.0f - 240.0f * norm_h;
#if (SHOW_SOLID)
            color = hsv(hue, 1.0f, mask);
#else
            color = hsv(hue, 1.0f, 1.0f);
#endif
        }

        if (n%100 == 0) {
            printf("%u..", n);
            fflush(stdout);
        }
    }
    printf("\n");

    return 0;
}
