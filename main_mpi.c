#define M_PI 3.14159265358979323846
#define DIM_NUM 3
#define SHIFT_NUM 2

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "math.h"
#include <mpi.h>

struct vector3d{
    double x, y, z;
};

struct state {
    double *prev_state;
    double *cur_state;
    double *next_state;
};

struct exchange_buffers {
    double *x_send_buffer;
    double *y_send_buffer;
    double *z_send_buffer;
    double *x_recieve_buffer;
    double *y_recieve_buffer;
    double *z_recieve_buffer;
};

struct procces_info {
    int coords[DIM_NUM];
    int dots_per_process[DIM_NUM];
};

// Equation hyperparameters
struct vector3d L;
double T, a;
int N, K;
int threads_num;

// Mpi vars
MPI_Comm grid;
int dims[DIM_NUM] = {0, 0, 0};
int period[DIM_NUM] = {0, 0, 1};
int rank_count;

struct state u; 
struct exchange_buffers buffs;
struct procces_info info;

int setup(int argc, char **argv) {
    if (argc != 7) {
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
    return 0;
}

void set_value(double* arr, int i, int j, int k, double val, int dim_x, int dim_y, int dim_z) {
    arr[k + dim_z * j + dim_y * dim_z * i] = val;
}

double get_value(double* arr, int i, int j, int k, int dim_x, int dim_y, int dim_z) {
    return arr[k + dim_z * j + dim_y * dim_z * i];
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

double compute_laplace_operator(int i, int j, int k, double h, int dim_x, int dim_y, int dim_z) {
    double *u_n = u.cur_state;

    double central_term = 2 * get_value(u_n, i, j, k, dim_x, dim_y, dim_z);
    double dx = get_value(u_n, i - 1, j, k, dim_x, dim_y, dim_z) - central_term + get_value(u_n, i + 1, j, k, dim_x, dim_y, dim_z);
    double dy = get_value(u_n, i, j - 1, k, dim_x, dim_y, dim_z) - central_term + get_value(u_n, i, j + 1, k, dim_x, dim_y, dim_z);
    double dz = get_value(u_n, i, j, k - 1, dim_x, dim_y, dim_z) - central_term + get_value(u_n, i, j, k + 1, dim_x, dim_y, dim_z);
    return (dx + dy + dz) / (h * h);
}

double compute_next_u(int i, int j, int k, double h, double tau, int dim_x, int dim_y, int dim_z) {
    double laplace_func = compute_laplace_operator(i, j, k, h, dim_x, dim_y, dim_z);
    return (a * a) * (tau * tau) * laplace_func + 2 * get_value(u.cur_state, i, j, k, dim_x, dim_y, dim_z) - get_value(u.prev_state, i, j, k, dim_x, dim_y, dim_z); 
}

// Data exchange functions
int get_buffer_index_send(int k, int l, int dim, int shift) {
    int dim_shift;

    switch (dim) {
    case 0:
        // OX
        if (shift == 1) {
            dim_shift = info.dots_per_process[dim] * (info.dots_per_process[1] + 2) * (info.dots_per_process[2] + 2);
        }
        else {
            dim_shift = (info.dots_per_process[1] + 2) * (info.dots_per_process[2] + 2);
        }
        return dim_shift + (k + 1) * (info.dots_per_process[2] + 2) + l + 1;

    case 1:
        // OY
        if (shift == 1) {
            dim_shift = info.dots_per_process[dim] * (info.dots_per_process[2] + 2);
        }
        else {
            dim_shift = info.dots_per_process[2] + 2;
        }
        return (k + 1) * (info.dots_per_process[1] + 2) * (info.dots_per_process[2] + 2) + dim_shift + l + 1;

    case 2:
        // OZ
        if (shift == 1) {
            dim_shift = info.dots_per_process[dim];
        }
        else {
            dim_shift = 1;
        }
        return (k + 1) * (info.dots_per_process[1] + 2) * (info.dots_per_process[2] + 2) + (l + 1) * (info.dots_per_process[2] + 2) + dim_shift;
    }
}

int get_buffer_index_receive(int k, int l, int dim, int shift) {
    int dim_shift;

    switch (dim) {
    case 0:
        // OX
        if (shift == 1) {
            dim_shift = 0;
        }
        else {
            dim_shift = (info.dots_per_process[dim] + 1) * (info.dots_per_process[1] + 2) * (info.dots_per_process[2] + 2);
        }
        return dim_shift + (k + 1) * (info.dots_per_process[2] + 2) + l + 1;

    case 1:
        // OY
        if (shift == 1) {
            dim_shift = 0;
        }
        else {
            dim_shift = (info.dots_per_process[dim] + 1) * (info.dots_per_process[2] + 2);
        }
        return (k + 1) * (info.dots_per_process[1] + 2) * (info.dots_per_process[2] + 2) + dim_shift + l + 1;

    case 2:
        // OZ
        if (shift == 1) {
            dim_shift = 0;
        }
        else {
            dim_shift = info.dots_per_process[dim] + 1;
        }
        return (k + 1) * (info.dots_per_process[1] + 2) * (info.dots_per_process[2] + 2) + (l + 1) * (info.dots_per_process[2] + 2) + dim_shift;
    }
}

void data_exchange() {
    int source, dest;
    int dims_indexes[DIM_NUM - 1];
    int sources_array[SHIFT_NUM];
    MPI_Status status;
    MPI_Request send_req;
    double *send_buffer = NULL;
    double *recieve_buffer = NULL;

    for (int i = 0; i < DIM_NUM; i++) {
        if (dims[i] == 1) {
            continue;
        }

        if (i == 0) {
            // OX
            dims_indexes[0] = 1;
            dims_indexes[1] = 2;
            send_buffer = buffs.x_send_buffer;
            recieve_buffer = buffs.x_recieve_buffer;
        }
        else if (i == 1) {
            // OY
            dims_indexes[0] = 0;
            dims_indexes[1] = 2;
            send_buffer = buffs.y_send_buffer;
            recieve_buffer = buffs.y_recieve_buffer;
        } else {
            // OZ
            dims_indexes[0] = 0;
            dims_indexes[1] = 1;
            send_buffer = buffs.z_send_buffer;
            recieve_buffer = buffs.z_recieve_buffer;
        }

        MPI_Cart_shift(grid, i, 1, &source, &dest);
        int shifts[SHIFT_NUM] = {1, -1};

        for (int j = 0; j < SHIFT_NUM; j++) {
            if ((j == 0 && info.coords[i] != (dims[i] - 1)) || (j == 1 && info.coords[i] != 0)) {
                #pragma omp parallel for collapse(2)
                for (int k = 0; k < info.dots_per_process[dims_indexes[0]]; k++) {
                    for (int l = 0; l < info.dots_per_process[dims_indexes[1]]; l++)
                    {
                        send_buffer[k * info.dots_per_process[dims_indexes[1]] + l] = u.cur_state[get_buffer_index_send(k, l, i, shifts[j])];
                    }
                }

                MPI_Sendrecv(send_buffer, info.dots_per_process[dims_indexes[0]] * info.dots_per_process[dims_indexes[1]], MPI_DOUBLE, j == 0 ? dest : source, 0,
                             recieve_buffer, info.dots_per_process[dims_indexes[0]] * info.dots_per_process[dims_indexes[1]], MPI_DOUBLE, j == 0 ? dest : source, 0,
                             grid, &status);

                #pragma omp parallel for collapse(2)
                for (int k = 0; k < info.dots_per_process[dims_indexes[0]]; k++) {
                    for (int l = 0; l < info.dots_per_process[dims_indexes[1]]; l++) {
                        u.cur_state[get_buffer_index_receive(k, l, i, -shifts[j])] = recieve_buffer[k * info.dots_per_process[dims_indexes[1]] + l];
                    }
                }
            }
        }
    }
}

double *period_data_exchange(int shift, int z_index) {
    int source, dest;
    MPI_Status status;
    int size_x = info.dots_per_process[0] + 2;
    int size_y = info.dots_per_process[1] + 2;
    int size_z = info.dots_per_process[2] + 2;

    MPI_Cart_shift(grid, 2, shift, &source, &dest);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < info.dots_per_process[0]; i++)
    {
        for (int j = 0; j < info.dots_per_process[1]; j++)
        {
            buffs.z_send_buffer[i * info.dots_per_process[1] + j] = get_value(u.next_state, i + 1, j + 1, z_index, size_x, size_y, size_z);
        }
    }
    MPI_Sendrecv(buffs.z_send_buffer, info.dots_per_process[0] * info.dots_per_process[1], MPI_DOUBLE, dest, 0,
                 buffs.z_recieve_buffer, info.dots_per_process[0] * info.dots_per_process[1], MPI_DOUBLE, dest, 0,
                 grid, &status);

    return buffs.z_recieve_buffer;
}
// End data exhange block

void apply_init_conditions() {
    int size_x = info.dots_per_process[0] + 2;
    int size_y = info.dots_per_process[1] + 2;
    int size_z = info.dots_per_process[2] + 2;
    int overall_size = size_x * size_y * size_z;

    u.prev_state = (double *)malloc(overall_size * sizeof(double));
    u.cur_state = (double *)malloc(overall_size * sizeof(double));
    u.next_state = (double *)malloc(overall_size * sizeof(double));

    buffs.x_send_buffer = (double *)malloc(size_y * size_z * sizeof(double));
    buffs.y_send_buffer = (double *)malloc(size_x * size_z * sizeof(double));
    buffs.z_send_buffer = (double *)malloc(size_y * size_x * sizeof(double));

    buffs.x_recieve_buffer = (double *)malloc(size_y * size_z * sizeof(double));
    buffs.y_recieve_buffer = (double *)malloc(size_x * size_z * sizeof(double));
    buffs.z_recieve_buffer = (double *)malloc(size_y * size_x * sizeof(double));

    double x, y, z;
    double h_x = L.x / (N - 1);
    double h_y = L.y / (N - 1);
    double h_z = L.z / (N - 1);
    double tau = T / K;
    double val;

    #pragma omp parallel for collapse(3) private(x, y, z, val)
    for (int i = 0; i < info.dots_per_process[0]; i++) {
        for (int j = 0; j < info.dots_per_process[1]; j++) {
            for (int k = 0; k < info.dots_per_process[2]; k++) {
                int global_i = info.coords[0] * info.dots_per_process[0] + i;
                int global_j = info.coords[1] * info.dots_per_process[1] + j;
                int global_k = info.coords[2] * info.dots_per_process[2] + k;

                x = h_x * global_i;
                y = h_y * global_j;
                z = h_z * global_k;

                // u(x, y, z, 0) = phi(x, y, z)
                val = coumpute_phi(x, y, z);
                set_value(u.cur_state, i + 1, j + 1, k + 1, val, size_x, size_y, size_z);
            }
        }
    }

    data_exchange();
    #pragma omp parallel for collapse(3) private(x, y, z, val)
    for (int i = 0; i < info.dots_per_process[0]; i++) {
        for (int j = 0; j < info.dots_per_process[1]; j++) {
            for (int k = 0; k < info.dots_per_process[2]; k++) {
                int global_i = info.coords[0] * info.dots_per_process[0] + i;
                int global_j = info.coords[1] * info.dots_per_process[1] + j;
                int global_k = info.coords[2] * info.dots_per_process[2] + k;

                x = h_x * global_i;
                y = h_y * global_j;
                z = h_z * global_k;

                // t=0: du / dt = 0
                if (!check_is_boundary(global_i, global_j, global_k)) {
                    val = coumpute_phi(x, y, z) + 0.5 * (a * a) * (tau * tau) * compute_laplace_operator(i + 1, j + 1, k + 1, h_x, size_x, size_y, size_z);
                } else {
                    val = compute_analytical_solution(x, y, z, tau);
                }
                set_value(u.next_state, i + 1, j + 1, k + 1, val, size_x, size_y, size_z);
            }
        }
    }
}

void free_all() {
    free(u.cur_state);
    free(u.prev_state);
    free(u.next_state);
    free(buffs.x_recieve_buffer);
    free(buffs.x_send_buffer);
    free(buffs.y_recieve_buffer);
    free(buffs.y_send_buffer);
    free(buffs.z_recieve_buffer);
    free(buffs.z_send_buffer);
}

double compute_err(double *arr, double t) {
    int size_x = info.dots_per_process[0] + 2;
    int size_y = info.dots_per_process[1] + 2;
    int size_z = info.dots_per_process[2] + 2;

    double x, y, z;
    double h = L.x / (N - 1);
    double max_error = 0;

    #pragma omp parallel for collapse(3) private(x, y, z) reduction(max:max_error) 
    for (int i = 0; i < info.dots_per_process[0]; i++) {
        for (int j = 0; j < info.dots_per_process[1]; j++) {
            for (int k = 0; k < info.dots_per_process[2]; k++) {
                int global_i = info.coords[0] * info.dots_per_process[0] + i;
                int global_j = info.coords[1] * info.dots_per_process[1] + j;
                int global_k = info.coords[2] * info.dots_per_process[2] + k;

                x = h * global_i;
                y = h * global_j;
                z = h * global_k;
                double gt_val = compute_analytical_solution(x, y, z, t);
                double val = get_value(arr, i + 1, j + 1, k + 1, size_x, size_y, size_z);
                double err = fabs(val - gt_val);
                if (err > max_error) {
                    max_error = err;
                }
            }
        }
    }
    return max_error;
}

void compute_next_state() {
    int size_x = info.dots_per_process[0] + 2;
    int size_y = info.dots_per_process[1] + 2;
    int size_z = info.dots_per_process[2] + 2;


    double* tmp;
    tmp = u.prev_state;
    u.prev_state = u.cur_state;
    u.cur_state = u.next_state;
    u.next_state = tmp;

    data_exchange();

    double h = L.x / (N - 1);
    double tau = T / K;

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < info.dots_per_process[0]; i++) {
        for (int j = 0; j < info.dots_per_process[1]; j++) {
            for (int k = 0; k < info.dots_per_process[2]; k++) {
                int global_i = info.coords[0] * info.dots_per_process[0] + i;
                int global_j = info.coords[1] * info.dots_per_process[1] + j;
                int global_k = info.coords[2] * info.dots_per_process[2] + k;
                if (!check_is_boundary(global_i, global_j, global_k)) {
                    set_value(u.next_state, i + 1, j + 1, k + 1, compute_next_u(i + 1, j + 1, k + 1, h, tau, size_x, size_y, size_z), size_x, size_y, size_z);
                } else {
                    // Stationary boundary conditions OX, OY
                    set_value(u.next_state, i + 1, j + 1, k + 1, 0.0, size_x, size_y, size_z);
                }
            }
        }
    }

    // Periodically boundary conditions OZ
    if (info.coords[2] == 0 || info.coords[2] == dims[2] - 1) {
        int shift = info.coords[2] == 0 ? -1 : 1;
        int z_index = info.coords[2] == 0 ? 2 : info.dots_per_process[2] - 1;

        double *recieve_buffer = period_data_exchange(shift, z_index);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < info.dots_per_process[0]; i++) {
            for (int j = 0; j < info.dots_per_process[1]; j++) {
                double value = recieve_buffer[i * info.dots_per_process[1] + j];
                value += get_value(u.next_state, i + 1, j + 1, z_index, size_x, size_y, size_z);
                set_value(u.next_state, i + 1, j + 1, z_index, value / 2, size_x, size_y, size_z);
            }
        }
    }
}

int main(int argc, char **argv) {
    int err = setup(argc, argv);
    if (err) {
        printf("Incorrect input args\n");
        return 1;
    }

    double tau = T / K;
    double max_err;

    // Init Mpi
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    double begin = MPI_Wtime();

    // Create Proccess grid
    MPI_Dims_create(rank_count, DIM_NUM, dims);
    MPI_Cart_create(MPI_COMM_WORLD, DIM_NUM, dims, period, 1, &grid);
    MPI_Cart_coords(grid, rank, rank_count, info.coords);

    for (int i = 0; i < DIM_NUM; i++) {
        info.dots_per_process[i] = (int)(N / dims[i]);
    }

    apply_init_conditions();

    double error = compute_err(u.cur_state, 0);
    MPI_Reduce(&error, &max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
         printf("Max abs err on iter %d: %f\n", 0, max_err);
    }

    error = compute_err(u.next_state, tau);
    MPI_Reduce(&error, &max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
         printf("Max abs err on iter %d: %f\n", 1, max_err);
    }

    for (int t = 2; t <= K; ++t) {
        compute_next_state();
        double local_err = compute_err(u.next_state, tau * t);
        error = fmax(error, local_err);
        MPI_Reduce(&error, &max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Max abs err on iter %d: %f\n", t, max_err);
        }
    }

    double end = MPI_Wtime();;
    double time_spent = end - begin;
    // printf("Max abs err: %1.10f\n", max_err);

    if (rank == 0) {
        printf("Time: %1.8f\n", time_spent);
    }
    free_all();
    MPI_Finalize();
    return 0;
}
