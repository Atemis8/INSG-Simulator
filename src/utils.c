#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>
#include "../headers/utils.h"
#include "../headers/finite_diff.h"
#include "../headers/mpi_domain.h"

#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h> 

// Make the right choice of makedir on garbadgedows
#ifdef _WIN32
#include <direct.h>
#define mkdir_func(path) _mkdir(path)
#else
#include <errno.h>
#define mkdir_func(path) mkdir(path, 0700)
#endif

void mprintf(const char *fmt, ...) {
    if (mpi_rank() != 0) return;

    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

int initialize_dump(const char* dir_path) {
    // Only rank 0 handles directory operations
    int result = 0;
    if (mpi_rank() == 0) {
        struct stat st = {0};
        if (stat(dir_path, &st) == -1) {
            if (mkdir_func(dir_path) != 0) {
                perror("Failed to create directory");
                result = -1;
            }
        } else {
            DIR* dir = opendir(dir_path);
            if (!dir) { 
                perror("Failed to open directory");
                result = -1;
            } else {
                struct dirent* entry;
                char filepath[PATH_MAX];
                int err = 0;
                
                while ((entry = readdir(dir)) != NULL) {
                    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
                    snprintf(filepath, sizeof(filepath), "%s/%s", dir_path, entry->d_name);
            
                    struct stat path_stat;
                    if (stat(filepath, &path_stat) == 0 && S_ISREG(path_stat.st_mode) && remove(filepath) != 0) {
                        fprintf(stderr, "Failed to remove file %s: %s\n", filepath, strerror(errno));
                        err += 1;
                    }
                }
                
                closedir(dir);
                result = err;
            }
        }
    }
    
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    return result;
}

void save_fieldtxt(ScalarField *f, const char* file) {
    FILE *fp = fopen(file, "w");
    if (fp == NULL) { printf("Error opening file!\n"); return; }

    for(int x = 0; x < f->nx; ++x) {
        for(int y = 0; y < f->ny; ++y) {
            fprintf(fp, "%.12e ", get_scal(f, x, y));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

#ifdef USE_MPI
void save_field_mpiio(ScalarField* f, const char *filename, MPIDomain *domain) {
    MPI_File fh;
    MPI_Status status;
    int gw = domain->ghost_width;
    // Open file for collective write (all ranks must participate)
    int ret = MPI_File_open(MPI_COMM_WORLD, filename, 
                           MPI_MODE_CREATE | MPI_MODE_WRONLY,
                           MPI_INFO_NULL, &fh);
    if (ret != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(ret, error_string, &length);
        fprintf(stderr, "Rank %d: MPI_File_open failed: %s\n", domain->rank, error_string);
        return;
    }
    
    // Only rank 0 writes the NPY header
    MPI_Offset header_size = 0;
    if (domain->rank == 0) {
        char header_data[1024];
        char *ptr = header_data;
        
        // Write magic string
        memcpy(ptr, "\x93NUMPY", 6);
        ptr += 6;
        
        // Write version
        *ptr++ = 1;  // major
        *ptr++ = 0;  // minor
        
        // Prepare header string
        char header_str[512];
        snprintf(header_str, sizeof(header_str),
                "{'descr': '<f8', 'fortran_order': True, 'shape': (%d, %d), }",
                domain->global_ny, domain->global_nx);
        
        int header_len = (int)strlen(header_str);
        int padding = 16 - ((10 + header_len) % 16);
        header_len += padding;
        for (int i = strlen(header_str); i < header_len - 1; ++i)
            header_str[i] = ' ';
        header_str[header_len - 1] = '\n';
        header_str[header_len] = '\0';
        
        // Write header length
        uint16_t hlen = (uint16_t)header_len;
        memcpy(ptr, &hlen, 2);
        ptr += 2;
        
        // Write header string
        memcpy(ptr, header_str, header_len);
        ptr += header_len;
        
        header_size = ptr - header_data;
        
        // Write header to file
        MPI_File_write(fh, header_data, header_size, MPI_BYTE, &status);
    }
    // Broadcast header size to all ranks
    MPI_Bcast(&header_size, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);
    
    // Extract interior data (without ghost cells)
    int local_nx = domain->nx;
    int local_ny = domain->ny;
    double *interior_data = malloc(local_nx * local_ny * sizeof(double));
    
    // Copy interior data (column-major: x varies fastest)
    for (int x = 0; x < local_nx; ++x) {
        for (int y = 0; y < local_ny; ++y) {
            // Source: field with ghost cells at (gw+x, gw+y)
            // Dest: interior array at index x*local_ny + y
            interior_data[x * local_ny + y] = get_scal(f, gw + x, gw + y);
        }
    }
    // Create MPI datatype for this rank's portion of the global array
    // The global array is stored in column-major order (Fortran order)
    // Size of the global array
    int gsizes[2] = {domain->global_ny, domain->global_nx};  // [rows, cols] in column-major
    
    // Size of local array (without ghost cells)
    int lsizes[2] = {local_ny, local_nx};
    
    // Starting position in global array
    int starts[2] = {domain->start_y, domain->start_x};
    
    MPI_Datatype filetype;
    MPI_Type_create_subarray(2, gsizes, lsizes, starts,
                            MPI_ORDER_FORTRAN, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);
    
    // Set the file view (offset by header size)
    MPI_File_set_view(fh, header_size, MPI_DOUBLE, filetype,
                     "native", MPI_INFO_NULL);
    
    // Collective write
    MPI_File_write_all(fh, interior_data, local_nx * local_ny, MPI_DOUBLE, &status);
    // Cleanup
    MPI_Type_free(&filetype);
    free(interior_data);
    MPI_File_close(&fh);
}
#endif

void save_field(ScalarField* f, const char *filename, MPIDomain *domain) {
#ifdef USE_MPI
    if (domain->size > 1) {
        save_field_mpiio(f, filename, domain);
    } else {
#endif
        // Single process mode - use original method
        FILE *fp = fopen(filename, "wb");
        if (fp == NULL) { perror("Failed to open file"); return; }

        fwrite("\x93NUMPY", 1, 6, fp);
        uint8_t major = 1, minor = 0;
        fwrite(&major, 1, 1, fp);
        fwrite(&minor, 1, 1, fp);

        char header[256];
        const char *descr = "<f8";
        snprintf(header, sizeof(header),
                "{'descr': '%s', 'fortran_order': True, 'shape': (%d, %d), }",
                descr, f->ny, f->nx);
        int header_len = (int)strlen(header);
        int padding = 16 - ((10 + header_len) % 16);
        header_len += padding;
        for (int i = 0; i < padding; ++i)
            strcat(header, " ");
        header[header_len - 1] = '\n';
        uint16_t hlen = (uint16_t)header_len;
        fwrite(&hlen, 2, 1, fp);
        fwrite(header, 1, header_len, fp);
        fwrite(f->v, sizeof(double), f->nx * f->ny, fp);

        fclose(fp);
#ifdef USE_MPI
    }
#endif
}

void save_scalar_value(double value, const char *filename) {
    // Only rank 0 saves scalar values (they're the same across all ranks)
    if (mpi_rank() != 0) return;
    
    FILE *fp = fopen(filename, "ab"); 
    if (fp == NULL) {
        perror("Failed to open file");
        return;
    }

    fwrite(&value, sizeof(double), 1, fp);
    fclose(fp);
}

PostProcessor initialize_postprocessor(Simulation *s, const char* dir_path) {
    MACMesh *m = &s->mesh;
    return (PostProcessor) {
        .w = allocate_field(m->P.nx-1, m->P.ny-1),
        .dir = dir_path,
        .m = s
    };
}

void dump_params(SimulationParams *params, PostProcessor *p) {
    // Only rank 0 writes parameters
    if (mpi_rank() != 0) return;
    
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/sim.params", p->dir);
    FILE* file = fopen(filename, "w");
    if(file == NULL) return;

    fprintf(file, "nu=%.12e\n", params->nu);
    fprintf(file, "dt=%.12e\n", params->dt);
    fprintf(file, "dtau=%.12e\n", params->dtau);
    fprintf(file, "h=%.12e\n", params->h);
    fprintf(file, "r=%.12e\n", params->nu * params->dt / (params->h * params->h));
    fprintf(file, "Re=%.12e\n", params->Re);
    fprintf(file, "L=%.12e\n", params->body.L);
    fprintf(file, "H=%.12e\n", params->body.H);
    fprintf(file, "Lfish=%.12e\n", params->body.Lfish);
    fprintf(file, "dump=%d\n", params->dump_period);
    fprintf(file, "mode=%d\n", params->mode);
    fprintf(file, "global_nx=%d\n", params->domain.global_nx);
    fprintf(file, "global_ny=%d\n", params->domain.global_ny);
    fprintf(file, "mpi_dims=%dx%d\n", params->domain.dims[0], params->domain.dims[1]);
    fprintf(file, "mpi_size=%d\n", params->domain.size);
    fclose(file);
}

void dump_mesh(int ep, PostProcessor *p) {
    MPIDomain *domain = &(p->m->params->domain);
    
    char filename[256];
    
    // Save vorticity mask
    snprintf(filename, sizeof(filename), "%s/mask_%05d.npy", p->dir, ep);
    compute_vorticity_mask(&p->m->params->body, &p->w, p->m->t);
    save_field(&(p->w), filename, domain);

    // Save vorticity
    snprintf(filename, sizeof(filename), "%s/w_%05d.npy", p->dir, ep);
    vorticity(&(p->w), &(p->m->mesh));
    save_field(&(p->w), filename, domain);

    // Save u velocity
    snprintf(filename, sizeof(filename), "%s/u_%05d.npy", p->dir, ep);
    save_field(&(p->m->mesh.uv.u), filename, domain);

    // Save v velocity
    snprintf(filename, sizeof(filename), "%s/v_%05d.npy", p->dir, ep);
    save_field(&(p->m->mesh.uv.v), filename, domain);

    // Save pressure
    snprintf(filename, sizeof(filename), "%s/P_%05d.npy", p->dir, ep);
    save_field(&(p->m->mesh.P), filename, domain);
}

void dump_fish_data(FishData *body, PostProcessor *p) {
    // Only rank 0 saves fish data (it's global)
    if (mpi_rank() != 0) return;
    
    char filename[256];

    snprintf(filename, sizeof(filename), "%s/forces_x.bin", p->dir);
    save_scalar_value(body->xforce, filename);

    snprintf(filename, sizeof(filename), "%s/forces_y.bin", p->dir);
    save_scalar_value(body->yforce, filename);

    snprintf(filename, sizeof(filename), "%s/ufish.bin", p->dir);
    save_scalar_value(body->ufish, filename);

    snprintf(filename, sizeof(filename), "%s/vfish.bin", p->dir);
    save_scalar_value(body->vfish, filename);

    snprintf(filename, sizeof(filename), "%s/period.bin", p->dir);
    save_scalar_value(body->cont->period, filename);
}

void dump_mesh_data(SimulationParams *sim, PostProcessor *p) {
    // Compute local maxima
    double local_maxu = absmax_field(&(p->m->mesh.uv.u));
    double local_maxv = absmax_field(&(p->m->mesh.uv.v));
    vorticity(&(p->w), &(p->m->mesh));
    double local_maxw = absmax_field(&(p->w));
    
    // Global reduction to find true maximum
    double global_maxu = local_maxu;
    double global_maxv = local_maxv;
    double global_maxw = local_maxw;
    
#ifdef USE_MPI
    MPI_Allreduce(&local_maxu, &global_maxu, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_maxv, &global_maxv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_maxw, &global_maxw, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif

    // Only rank 0 writes the results
    if (mpi_rank() != 0) return;
    
    char filename[256];
    double Re_mesh_v = (global_maxu + global_maxv) * p->m->mesh.h / sim->nu;
    double Re_mesh_w = global_maxw * p->m->mesh.h * p->m->mesh.h / sim->nu;

    snprintf(filename, sizeof(filename), "%s/Re_mesh_velocity.bin", p->dir);
    save_scalar_value(Re_mesh_v, filename);
    
    snprintf(filename, sizeof(filename), "%s/Re_mesh_vorticity.bin", p->dir);
    save_scalar_value(Re_mesh_w, filename);
}

static int get_latest_episode(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) { perror("opendir failed"); exit(1); }

    int max_ep = -1;
    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        int ep;
        if (sscanf(entry->d_name, "u_%05d.npy", &ep) == 1)
            if (ep > max_ep) max_ep = ep;
    }
    closedir(d);

    if (max_ep == -1) {
        fprintf(stderr, "No episode data found in directory\n");
        exit(1);
    }

    return max_ep;
}

double mod_d(double x, double N) {
    return fmod(fmod(x, N) + N, N);
}