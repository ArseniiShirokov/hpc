#define M_PI 3.14159265358979323846

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "math.h"
#include <omp.h>

struct vector3d{
    double x, y, z;
};

struct state {
    double *prev_state;
    double *cur_state;
    double *next_state;
};

// Equation hyperparameters
struct vector3d L;
double T, a;
int N, K;
int threads_num;

struct state u; 

int setup(int argc, char **argv) {
    if (argc != 8) {
        return 1;
    }
    // Init definition area
    L.x = strtod(argv[1], NULL);
    L.y = strtod(argv[2], NULL);
    L.z = strtod(argv[3], NULL);
    T = strtod(argv[4], NULL);

    // Init temporal resolution
    N = atoi(argv[5]);
    // Init time step count
    K = atoi(argv[6]);

    // coeff a
    a = 1 / M_PI;

    // threads num
    threads_num = atoi(argv[7]);
    return 0;
}

double get_value(double* arr, int i, int j, int k, int dim) {
    return arr[k + dim * j + dim * dim * i];
}

void set_value(double* arr, int i, int j, int k, double val, int dim) {
    arr[k + dim * j + dim * dim * i] = val;
}

double compute_analytical_solution(double x, double y, double z, double t) {
    struct vector3d r_L = {1 / L.x, 1 / L.y, 1 / L.z};
    double a_t = sqrt(r_L.x * r_L.x + r_L.y * r_L.y + 4 * r_L.z * r_L.z);
    return sin(M_PI * x * r_L.x) * sin(M_PI * y * r_L.y) * sin(2 * M_PI * z * r_L.z) * cos(a_t * t + 2 * M_PI);
}

double coumpute_phi(double x, double y, double z) {
    return compute_analytical_solution(x, y, z, 0);
}

int check_is_boundary(int i, int j, int k) {
    return (i == 0) || (j == 0) || (k == 0) || (i == N - 1) || (j == N - 1) || (k == N - 1);
}

double compute_laplace_operator(int i, int j, int k, double h) {
    double *u_n = u.cur_state;
    if (check_is_boundary(i, j, k)){
        exit(1);
    }

    double central_term = 2 * get_value(u_n, i, j, k, N);
    double dx = get_value(u_n, i - 1, j, k, N) - central_term + get_value(u_n, i + 1, j, k, N);
    double dy = get_value(u_n, i, j - 1, k, N) - central_term + get_value(u_n, i, j + 1, k, N);
    double dz = get_value(u_n, i, j, k - 1, N) - central_term + get_value(u_n, i, j, k + 1, N);
    return (dx + dy + dz) / (h * h);
}

double compute_next_u(int i, int j, int k, double h, double tau) {
    if (check_is_boundary(i, j, k)){
        exit(1);
    }

    double laplace_func = compute_laplace_operator(i, j, k, h);
    return (a * a) * (tau * tau) * laplace_func + 2 * get_value(u.cur_state, i, j, k, N) - get_value(u.prev_state, i, j, k, N); 
}

void apply_init_conditions() {
    u.prev_state = (double *)malloc(N * N * N * sizeof(double));
    u.cur_state = (double *)malloc(N * N * N * sizeof(double));
    u.next_state = (double *)malloc(N * N * N * sizeof(double));

    double x, y, z;
    double h_x = L.x / (N - 1);
    double h_y = L.y / (N - 1);
    double h_z = L.z / (N - 1);
    double tau = T / K;
    double val;

    #pragma omp parallel for collapse(3) private(x, y, z, val)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                x = h_x * i;
                y = h_y * j;
                z = h_z * k;

                // u(x, y, z, 0) = phi(x, y, z)
                val = coumpute_phi(x, y, z);
                set_value(u.cur_state, i, j, k, val, N);
            }
        }
    }

    #pragma omp parallel for collapse(3) private(x, y, z, val)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                x = h_x * i;
                y = h_y * j;
                z = h_z * k;

                // t=0: du / dt = 0
                if (!check_is_boundary(i, j, k)) {
                    val = coumpute_phi(x, y, z) + 0.5 * (a * a) * (tau * tau) * compute_laplace_operator(i, j, k, h_x);
                } else {
                    val = compute_analytical_solution(x, y, z, tau);
                }
                set_value(u.next_state, i, j, k, val, N);
            }
        }
    }
}

void compute_next_state() {
    double* tmp;
    tmp = u.prev_state;
    u.prev_state = u.cur_state;
    u.cur_state = u.next_state;
    u.next_state = tmp;

    double h = L.x / (N - 1);
    double tau = T / K;

    #pragma omp parallel for collapse(3)
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            for (int k = 1; k < N - 1; k++) {
                set_value(u.next_state, i, j, k, compute_next_u(i, j, k, h, tau), N);
            }
        }
    }

    // Stationary boundary conditions OX, OY
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            set_value(u.next_state, 0, i, j, 0, N);
            set_value(u.next_state, N - 1, i, j, 0, N);
            set_value(u.next_state, i, 0, j, 0, N);
            set_value(u.next_state, 0, N - 1, j, 0, N);
        }
    }
    // Periodically boundary conditions OZ
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double right_val = get_value(u.next_state, i, j, 1, N);
            double left_val = get_value(u.next_state, i, j, N - 2, N);
            double val = (right_val + left_val) / 2;
            set_value(u.next_state, i, j, 0, val, N);
            set_value(u.next_state, i, j, N - 1, val, N);
        }
    }
}

double compute_err(double *arr, double t) {
    double x, y, z;
    double h = L.x / (N - 1);
    double max_error = 0;

    #pragma omp parallel for collapse(3) private(x, y, z) reduction(max:max_error) 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                x = h * i;
                y = h * j;
                z = h * k;
                double gt_val = compute_analytical_solution(x, y, z, t);
                double val = get_value(arr, i, j, k, N);
                double err = fabs(val - gt_val);
                if (err > max_error) {
                    max_error = err;
                }
            }
        }
    }
    return max_error;
}

void free_all() {
    free(u.cur_state);
    free(u.prev_state);
    free(u.next_state);
}

int main(int argc, char **argv) {
    double begin = omp_get_wtime();

    int err = setup(argc, argv);
    if (err) {
        printf("Incorrect input args\n");
        return 1;
    }
    omp_set_num_threads(threads_num);

    apply_init_conditions();

    double tau = T / K;
    double max_err = compute_err(u.cur_state, 0);
    printf("Max abs err on iter %d: %f\n", 0, max_err);

    max_err = fmax(compute_err(u.next_state, tau), max_err);
    printf("Max abs err on iter %d: %f\n", 1, max_err);

    for (int t = 2; t <= K; ++t) {
        compute_next_state();
        double err = compute_err(u.next_state, tau * t);
        max_err = fmax(max_err, err);
        printf("Max abs err on iter %d: %f\n", t, max_err);
    }

    double end = omp_get_wtime();
    double time_spent = end - begin;
    printf("Max abs err: %1.10f\n", max_err);

    #pragma omp parallel
    {
        #pragma omp single
        printf("num_threads = %d\n", omp_get_num_threads());
    }

    printf("Time: %1.8f\n", time_spent);
    free_all();

    return 0;
}
