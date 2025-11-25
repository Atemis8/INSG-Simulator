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
    struct stat st = {0};
    if (stat(dir_path, &st) == -1) {
        if (mkdir_func(dir_path) == 0) return 0;
        else { perror("Failed to create directory"); return -1; }
    } else {
        DIR* dir = opendir(dir_path);
        if (!dir) { perror("Failed to open directory"); return -1; }
        
        struct dirent* entry;
        char filepath[PATH_MAX];
        int err = 0;
        
        while ((entry = readdir(dir)) != NULL) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue; // Does not delete folder
            snprintf(filepath, sizeof(filepath), "%s/%s", dir_path, entry->d_name);
    
            struct stat path_stat;
            if (stat(filepath, &path_stat) == 0 && S_ISREG(path_stat.st_mode) && remove(filepath) != 0) {
                fprintf(stderr, "Failed to remove file %s: %s\n", filepath, strerror(errno));
                err += 1;
            }
        }
        
        closedir(dir);
        return err;
    }
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

void save_field(ScalarField* f, const char *filename) {
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
}

ScalarField load_field(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Failed to open file");
        exit(1);
    }

    char magic[6];
    fread(magic, 1, 6, fp);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "Invalid .npy file\n");
        fclose(fp);
        exit(1);
    }

    uint8_t major, minor;
    fread(&major, 1, 1, fp);
    fread(&minor, 1, 1, fp);

    uint16_t header_len;
    fread(&header_len, 2, 1, fp);

    char header[256] = {0};
    fread(header, 1, header_len, fp);

    int ny = 0, nx = 0;
    if (sscanf(header, "{'descr': '<f8', 'fortran_order': True, 'shape': (%d, %d)", &ny, &nx) != 2) {
        fprintf(stderr, "Failed to parse shape from header\n");
        fclose(fp);
        exit(1);
    }

    ScalarField f = allocate_field(nx, ny);
    size_t n = fread(f.v, sizeof(double), nx * ny, fp);
    if (n != (size_t)(nx * ny)) {
        fprintf(stderr, "Failed to read all data\n");
        free(f.v);
        fclose(fp);
        exit(1);
    }

    fclose(fp);
    return f;
}

void save_scalar_value(double value, const char *filename) {
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

SimulationParams load_params(const char *dir) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/sim.params", dir);
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open sim.params");
        exit(1);
    }

    double nu = 0, dt = 0, dtau = 0, h = 0, Re = 0;
    double L = 0, H = 0, Lfish = 0;
    int dump_period = 0, mode = 0, nx = 0, ny = 0;

    char line[128];
    while (fgets(line, sizeof(line), file)) {
        double val;
        int ival;
        if (sscanf(line, "nu=%lf", &val) == 1) nu = val;
        else if (sscanf(line, "dt=%lf", &val) == 1) dt = val;
        else if (sscanf(line, "dtau=%lf", &val) == 1) dtau = val;
        else if (sscanf(line, "h=%lf", &val) == 1) h = val;
        else if (sscanf(line, "Re=%lf", &val) == 1) Re = val;
        else if (sscanf(line, "L=%lf", &val) == 1) L = val;
        else if (sscanf(line, "H=%lf", &val) == 1) H = val;
        else if (sscanf(line, "Lfish=%lf", &val) == 1) Lfish = val;
        else if (sscanf(line, "dump=%d", &ival) == 1) dump_period = ival;
        else if (sscanf(line, "mode=%d", &ival) == 1) mode = ival;
        else if (sscanf(line, "nx=%d", &ival) == 1) nx = ival;
        else if (sscanf(line, "ny=%d", &ival) == 1) ny = ival;
    }

    fclose(file);

    FishData body = initialize_body(L, H, h, dt, Lfish, nx, ny, mode);

    SimulationParams params = {
        .nu = nu,
        .dt = dt,
        .dtau = dtau,
        .h = h,
        .Re = Re,
        .dump_period = dump_period,
        .mode = mode,
        .num_epsiodes = 30000,
        .body = body
    };

    return params;
}

void dump_params(SimulationParams *params, PostProcessor *p) {
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
    fprintf(file, "nx=%d\n", params->body.nx);
    fprintf(file, "ny=%d\n", params->body.ny);
    fclose(file);
}

void dump_mesh(int ep, PostProcessor *p) {

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/mask_%05d.npy", p->dir, ep);
    compute_vorticity_mask(&p->m->params->body, &p->w, p->m->t);
    save_field(&(p->w), filename);

    snprintf(filename, sizeof(filename), "%s/w_%05d.npy", p->dir, ep);
    vorticity(&(p->w), &(p->m->mesh));
    save_field(&(p->w), filename);

    snprintf(filename, sizeof(filename), "%s/u_%05d.npy", p->dir, ep);
    save_field(&(p->m->mesh.uv.u), filename);

    snprintf(filename, sizeof(filename), "%s/v_%05d.npy", p->dir, ep);
    save_field(&(p->m->mesh.uv.v), filename);

    snprintf(filename, sizeof(filename), "%s/P_%05d.npy", p->dir, ep);
    save_field(&(p->m->mesh.P), filename);
}

void dump_fish_data(FishData *body, PostProcessor *p) {
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
    char filename[256];
    double Re_mesh_v = (absmax_field(&(p->m->mesh.uv.u)) + absmax_field(&(p->m->mesh.uv.v))) * p->m->mesh.h / sim->nu;
    vorticity(&(p->w), &(p->m->mesh));
    double Re_mesh_w = absmax_field(&(p->w)) * p->m->mesh.h * p->m->mesh.h / sim->nu;

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

Simulation load_simulation(SimulationParams *params, const char *dir, int *ep_o) {
    Simulation sim = init_simulation(params);

    *ep_o = get_latest_episode(dir);
    sim.t = params->dt * *ep_o;

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/u_%05d.npy", dir, *ep_o);
    sim.mesh.uv.u = load_field(filename);

    snprintf(filename, sizeof(filename), "%s/v_%05d.npy", dir, *ep_o);
    sim.mesh.uv.v = load_field(filename);

    snprintf(filename, sizeof(filename), "%s/P_%05d.npy", dir, *ep_o);
    sim.mesh.P = load_field(filename);

    return sim;
}

double mod_d(double x, double N) {
    return fmod(fmod(x, N) + N, N);
}