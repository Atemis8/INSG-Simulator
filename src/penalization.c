#include "../headers/penalization.h"
#include "../headers/utils.h"
#include "../headers/mesh.h"
#include "../headers/config.h"
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#define xforu(x, h, start_x) ((h) * (double)((start_x) + (x)))
#define yforu(y, h, start_y) ((h) * (double)((start_y) + (y) - 0.5))
#define xforv(x, h, start_x) ((h) * (double)((start_x) + (x) - 0.5))
#define yforv(y, h, start_y) ((h) * (double)((start_y) + (y)))

double get_half_width(FishData* fish, double x) {
    double LFish = fish->Lfish;
    double wh = 0.04 * LFish;
    double xt = 0.95 * LFish;
    double wt = 0.01 * LFish;

    if (0 <= x && x < wh) {
        return (double) sqrt(2 * wh * x - x * x);
    } else if (wh <= x && x < xt) {
        return (double) (wh - (wh - wt) * (x - wh) / (xt - wh) * (x - wh)/(xt - wh));
    } else if (xt <= x && x < LFish) {
        return (double) (wt * ((LFish - x)/(LFish - xt)));
    }
   
    return 0.0;
}

double get_lateral_displacement(FishData* fish, double x, double t) {
    double LFish = fish->Lfish;
    double T = fish->cont->period;
    return (double) LFish / 264. * (1 + (32* x)/LFish) * sin(2 * PI *(x/LFish - t / T));
}

double lateral_displacement_dt(FishData* fish, double x, double t) {
    double LFish = fish->Lfish;
    double T = fish->cont->period;
    return (double) LFish / 264. * (-2 * PI / T) * (1 + (32 * x)/LFish) * cos(2 * PI *(x/LFish - t / T));
}

double compute_fish_surface(FishData* fish) {
    // Only rank 0 computes or all ranks compute the same value
    // Since this doesn't depend on the domain decomposition
    int nx = fish->domain->global_nx;  // Use global size
    double Sfish = 0.0;
    for (int x = 0; x < nx; x++) {
        double xpos = (x - 0.5) * fish->h;
        double w = get_half_width(fish, xpos - fish->xfish);
        Sfish += w;
    }
    Sfish *= 2 * fish->L / (nx - 1);
    return Sfish;
}

void no_control(FishData* dat) {
    dat->cont->period = 1.0;
}

#define KP 0.02
#define KI 0.01
#define MAX_T 5.0
#define MIN_T 1.0
#define DELTA_MAX_UP 0.001
#define DELTA_MAX_DOWN 0.001
#define MAX_I 0.1

void pid_control(FishData *dat) {
    FishController *ctrl = dat->cont;

    // Update moving average
    double old = ctrl->speed_buffer[ctrl->buf_idx];
    ctrl->speed_buffer[ctrl->buf_idx] = dat->ufish;
    ctrl->avg_speed += (dat->ufish - old) / (double) MEM_SIZE;
    ctrl->buf_idx = (ctrl->buf_idx + 1) % MEM_SIZE;

    // Error between desired and average speed
    double error = ctrl->target_speed - ctrl->avg_speed;

    // Allow integration in a way that avoids saturation
    if ((ctrl->period > MIN_T || error > 0) && (ctrl->period < MAX_T || error < 0))
        ctrl->integral += error * dat->dt;

    // Clamp integral to prevent windup
    ctrl->integral = fmin(MAX_I, fmax(-MAX_I, ctrl->integral));

    // Compute output delta
    double delta = KP * error + KI * ctrl->integral;
    // Limit how much we change the period
    double frac = fmin(1.0, fabs(error) / 1e-3);
    double cap_u = DELTA_MAX_UP * frac;
    double cap_d = DELTA_MAX_DOWN * frac;

    delta = fmin(cap_u, fmax(-cap_d, delta));

    if(fabs(error) < 1e-3) delta = 0.0;

    // Update period
    double new_period = ctrl->period + delta;
    double smoothing = 0.95; 
    ctrl->period = smoothing * ctrl->period + (1-smoothing) * new_period;
    ctrl->period = fmin(MAX_T, fmax(ctrl->period, MIN_T));

    mprintf(" E:%.3F, I:%.3f, T:%.3f, del:%.3f ", error, ctrl->integral, ctrl->period, delta);
}

FishController* create_controller(double target, double start, void (*up) (FishData*)) {
    FishController* ctrl = malloc(sizeof(FishController));
    memset(ctrl, 0, sizeof(FishController));
    ctrl->up = up;
    ctrl->period = start;
    ctrl->target_speed = target;
    return ctrl;
}

FishData initialize_body(MPIDomain *domain, double L, double H, double h, double dt, double Lfish, int mode) {
    double x_fish;
    double y_fish;
    switch (mode) {
    case M_PERIODIC:    
        x_fish = 0.1 * L;
        y_fish = 0.5 * H;
        break;
    
    case M_BOUNDARY:
        x_fish = 0.75 * Lfish;
        y_fish = 0.5 * H;
        break;
    }
    FishData data = (FishData) {
        .domain = domain,
        .Lfish = Lfish,
        .xfish = x_fish,
        .yfish = y_fish,
        .L = L,
        .H = H,
        .h = h,
        .dt = dt,
        .ufish = 0.0,
        .vfish = 0.0,
    };
    data.area = compute_fish_surface(&data);
    mprintf("Fish surface : %.3e\n", data.area);

    return data;
}

void compute_y_bounds(FishData* fish, double xpos, double time, double y_data[2]) {
    double dx = xpos - fish->xfish;
    double w = get_half_width(fish, dx);
    double ym = get_lateral_displacement(fish, dx, time);

    y_data[0] = fish->yfish + ym - w;
    y_data[1] = fish->yfish + ym + w;
}

void compute_speed_mask(FishData* fish, VectorField* out, double time) {
    int nx = fish->domain->tnx;
    int ny = fish->domain->tny;
    int gw = fish->domain->ghost_width;
    int start_x = fish->domain->start_x;
    int start_y = fish->domain->start_y;
    double L = fish->L;
    double xfish = fish->xfish;
    double Lfish = fish->Lfish;
    VectorField* mask = &(fish->mask);

    op_vecfield(mask, 0.0, set_scal);
    op_vecfield(out, 0.0, set_scal);
    
    // Loop over interior + ghost cells for u-component
    for (int x = 1; x < nx-2; x++) {
        for(int shiftx = -1; shiftx <= 1; ++shiftx) { 
            double bounds_u[2];
            // Convert local x to global coordinate
            double xpos_u = xforu(x, fish->h, start_x) + shiftx * L;
            compute_y_bounds(fish, xpos_u, time, bounds_u);

            if(xpos_u >= xfish && xpos_u <= xfish + Lfish) {
                for (int y = 1; y < ny - 1; y++) {
                    // Convert local y to global coordinate
                    double ypos_u = yforu(y, fish->h, start_y);
                    if (ypos_u >= bounds_u[0] && ypos_u <= bounds_u[1]) {
                        set_scal(&out->u, x, y, fish->ufish);
                        set_scal(&mask->u, x, y, 1.0);
                    }
                }
            }
        }
    }

    // Loop over interior + ghost cells for v-component
    for (int x = 1; x < nx - 1; x++) {
        for(int shiftx = -1; shiftx <= 1; ++shiftx) { 
            double bounds_v[2];
            // Convert local x to global coordinate
            double xpos_v = xforv(x, fish->h, start_x) + shiftx * L;
            compute_y_bounds(fish, xpos_v, time, bounds_v);
            double dym = lateral_displacement_dt(fish, xpos_v - fish->xfish, time);

            if(xpos_v >= xfish && xpos_v <= xfish + Lfish) {
                for (int y = 1; y < ny-2; y++) {
                    // Convert local y to global coordinate
                    double ypos_v = yforv(y, fish->h, start_y);
                    if (ypos_v >= bounds_v[0] && ypos_v <= bounds_v[1]) {
                        set_scal(&out->v, x, y, dym + fish->vfish);
                        set_scal(&mask->v, x, y, 1.0);
                    }
                }
            }
        }
    }
}

void compute_vorticity_mask(FishData* fish, ScalarField* vort_mask, double time) {
    int nx = fish->domain->tnx;
    int ny = fish->domain->tny;
    int start_x = fish->domain->start_x;
    int start_y = fish->domain->start_y;
    double L = fish->L;
    double h = fish->h;
    double xfish = fish->xfish;
    double Lfish = fish->Lfish;

    op_field(vort_mask, 0.0, set_scal);
    for (int i = 0; i < nx - 1; ++i) {
        for (int j = 0; j < ny - 1; ++j) {
            // Convert to global coordinates
            double x = (start_x + i) * h;
            double y = (start_y + j) * h;

            for (int shiftx = -1; shiftx <= 1; ++shiftx) {
                double xshifted = x + shiftx * L;
                if (xshifted >= xfish && xshifted <= xfish + Lfish) {
                    double bounds[2];
                    compute_y_bounds(fish, xshifted, time, bounds);
                    if (y >= bounds[0] && y <= bounds[1]) {
                        set_scal(vort_mask, i, j, 1.0);
                        break;
                    }
                }
            }
        }
    }
}

void compute_forces(FishData* data, VectorField* integ, int mode) {
    int nx = integ->u.nx;
    int ny = integ->u.ny;
    int gw = data->domain->ghost_width;

    double h = data->h;
    double dt = data->dt;

    // Local computations on interior cells only (excluding ghost cells)
    double local_maxu = 0.0;
    double local_Iu = 0.0;
    for (int x = gw; x < nx - gw - 1; ++x)
        for (int y = gw + 1; y < ny - gw; ++y) {
            double u = (get_scal(&integ->u, x, y) + get_scal(&integ->u, x + 1, y)) / 2.0;
            if(local_maxu < fabs(u)) local_maxu = fabs(u);
            local_Iu += u * h * h;
        }

    double local_maxv = 0.0;
    double local_Iv = 0.0;
    for (int x = gw + 1; x < nx - gw; ++x)
        for (int y = gw; y < ny - gw - 1; ++y) {
            double v = (get_scal(&integ->v, x, y) + get_scal(&integ->v, x, y + 1)) / 2.0;
            if(local_maxv < fabs(v)) local_maxv = fabs(v);
            local_Iv += v * h * h;
        }
    
    // Global reductions across all MPI processes
    double Iu = 0.0;
    double Iv = 0.0;
    
#ifdef USE_MPI
    MPI_Allreduce(&local_Iu, &Iu, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_Iv, &Iv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    Iu = local_Iu;
    Iv = local_Iv;
#endif

    mprintf(" au : %.3e, av : %.3e, ", Iu / data->area, Iv / data->area);
    data->ufish += Iu / data->area;
    if (mode == M_BOUNDARY) data->vfish = 0;
    if (mode == M_PERIODIC){
        data->vfish += Iv / data->area;
        data->xfish = data->xfish + data->ufish * dt;
        data->yfish = data->yfish + data->vfish * dt;
    }

    data->xforce = Iu/dt;
    data->yforce = Iv/dt;

    // check if the fish is two times outside the domain 
    if (data->xfish < -data->L) data->xfish += data->L;
    if (data->xfish > 2 * data->L) data->xfish -= data->L;
    
    mprintf(" xfish : %.2f", data->xfish);
}

void free_body(FishData* fish) {
    free_vecfield(&(fish->mask));
    free((fish->cont));
}