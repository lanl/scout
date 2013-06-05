// Lennard-Jones force (and energy computation) 
// input is the pos array (including a flag in .w, not used)
// output is the force array (including energy in .w)

// Since each cell is in it's own neighbor-list, basically neighbor_list[i,j,k] = [i-1:i+1, j-i:j+1, k-1:k+1]

uniform mesh MDMeshType[128, 128] {
cells:
    float4 pos[N_MAX_ATOMS];
    float4 vel[N_MAX_ATOMS];
    float4 force[N_MAX_ATOMS];
    float4 old_force[N_MAX_ATOMS];
    float cell_extent[6]; // xmin, xmax, ymin, ymax, zmin, zmax
    int3 shift_flag[N_MAX_ATOMS];
    int neighbor_list[N_MAX_NEIGHBORS];
    int n_atoms;
}


void LJForce (MDMeshType* md_mesh, float s6, float epsilon)
{

    forall cells c in md_mesh {    // loop over all mesh cells
        forall i in [0:c.n_atoms:] {  // loop over all atoms in the cell

            float dx, dy, dz;
            float r2, r6;
            float fx, fy, fz;

            float e, s6, fr;

            // zero out sums
            fx = 0.0;
            fy = 0.0;
            fz = 0.0;

            e = 0.0;

            forall cells n in c.neighbor_list { // loop over all neighbor cells
                forall j in [0:n.n_atoms:] { // loop over all atoms in neighbor cell

                    dx = c.pos.x[i] - n.pos.x[j];
                    dy = c.pos.y[i] - n.pos.y[j];
                    dz = c.pos.z[i] - n.pos.z[j];


                    r2 = (dx*dx + dy*dy + dz*dz);

                    if (r2 > 0.0) {
                        r2 = 1.0/r2;
                        r6 = r2*r2*r2;

                        e += r6*(s6*r6 - 1.0);

                        fr = 4.0*epsilon*s6*r2*r6*(12.0*r6*s6 - 6.0);

                        fx += dx*fr;
                        fy += dy*fr;
                        fz += dz*fr;

                    } else {
                    }

                } // loop over atoms in neighbor cell
            } // loop over neighbor cells

            c.force.x[i] = fx;
            c.force.y[i] = fy;
            c.force.z[i] = fz;

            // since we loop over all particles, each particle contributes 1/2 the pair energy to the total
            // energy is saved in .w of force
            c.force.w[i] = 0.5*e*4.0*epsilon*s6;
        } // loop over atoms in cell
    } // loop over cells
} // end force computation subroutine


// Simple position update test
// input is force and velocity, output is new position
// vel.w contains 1/M (to avoid divide since we need to compute acceleration
void PosUpdate(MDMeshType* md_mesh)
{
    float dx,dy,dz;

    forall cells c in md_mesh {    // loop over all mesh cells
        forall i in [0:c.n_atoms:] {  // loop over all atoms in the cell

            // update the velocity using Verlet formula

            vel.x[i] = vel.x[i] + 0.5*dt*(force.x[i] + old_force.x[i])*vel.w[i];
            vel.y[i] = vel.y[i] + 0.5*dt*(force.y[i] + old_force.y[i])*vel.w[i];
            vel.z[i] = vel.z[i] + 0.5*dt*(force.z[i] + old_force.z[i])*vel.w[i];

            // compute position change using Verlet
            dx = dt*(c.vel.x[i] + 0.5*dt*c.force.x[i]*vel.w[i]);
            dy = dt*(c.vel.y[i] + 0.5*dt*c.force.y[i]*vel.w[i]);
            dz = dt*(c.vel.z[i] + 0.5*dt*c.force.z[i]*vel.w[i]);

            c.pos.x[i] += dx;
            c.pos.y[i] += dy;
            c.pos.z[i] += dz;

            // rotate force values

            old_force[i] = force[i];

            // update shift flags
            int shift_flag[3]= {0, 0, 0};

            if (c.pos.x < c.cell_extent[0]) c.shift_flag.x = -1;
            if (c.pos.x > c.cell_extent[1]) c.shift_flag.x =  1;

            if (c.pos.y < c.cell_extent[2]) c.shift_flag.y = -1;
            if (c.pos.y > c.cell_extent[3]) c.shift_flag.y =  1;

            if (c.pos.z < c.cell_extent[4]) c.shift_flag.z = -1;
            if (c.pos.z > c.cell_extent[5]) c.shift_flag.z =  1;

        }
    }
}

// Gather from neighbor cells
// assumes offset structure exists
void AtomGather(MDMeshType* md_mesh)
{
    forall cells c in md_mesh {    // loop over all mesh cells
        forall cells n in c.neighbor_list { // loop over all neighbor cells
            forall i in [0:n.n_atoms] { // loop over all atoms in neighbor cell
                if (n.offset + n.shift_flag == 0) {
                    // copy data into this cell
                    c.pos[c.n_atoms] = n.pos[i];
                    c.vel[c.n_atoms] = n.vel[i];
                    // no need for force
                    // increment local atom counter
                    c.n_atoms ++;
                }
            }
        }
    }
}

// Naive atom packing scheme
void AtomPack(MDMeshType* md_mesh)
{
    forall cells c in md_mesh {    // loop over all mesh cells
        forall i in [0:c.n_atoms] { // this needs to be in order to work, I think
            if (c.shift_flag[i] != 0) {  // atom has been moved out of cell
                // copy data from last atom into this slot (assumes last atom has zero shift flag)
                c.pos[i] = c.pos[c.n_atoms-1];
                c.vel[i] = c.vel[c.n_atoms-1];
                // zero shift flag
                c.shift_flag[i] = {0, 0, 0};
                // decrement atom counter
                c.n_atoms --;
            }
        }
    }
}
