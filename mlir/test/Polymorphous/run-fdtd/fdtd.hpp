void fdtd(int tmax, int nx, int ny,
                           DATA_TYPE POLYBENCH_2D(ex, NX, NY, nx, ny),
                           DATA_TYPE POLYBENCH_2D(ey, NX, NY, nx, ny),
                           DATA_TYPE POLYBENCH_2D(hz, NX, NY, nx, ny),
                           DATA_TYPE POLYBENCH_1D(_fict_, TMAX, tmax));