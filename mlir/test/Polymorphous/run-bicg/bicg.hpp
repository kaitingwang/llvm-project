#include <polybench.h>
#include "bicg.h"
 
void bicg(int m, int n, DATA_TYPE POLYBENCH_2D(A, N, M, n, m),
                        DATA_TYPE POLYBENCH_1D(s, M, m),
                        DATA_TYPE POLYBENCH_1D(q, N, n),
                        DATA_TYPE POLYBENCH_1D(p, M, m),
                        DATA_TYPE POLYBENCH_1D(r, N, n));