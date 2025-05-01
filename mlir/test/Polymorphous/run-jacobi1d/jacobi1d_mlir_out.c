#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <polybench.h>
#include "jacobi1d.h"
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define max(x,y)    ((x) > (y) ? (x) : (y))
#define min(x,y)    ((x) < (y) ? (x) : (y))
#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0) 
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2) 


#define S1(i0, i1, i2, i3) arg4[i3] = (((arg3[(i3 + -1)] + arg3[i3]) + arg3[(i3 + 1)]) * 0.333330f);
#define S2(i0, i1, i2, i3) arg3[i3] = (((arg4[(i3 + -1)] + arg4[i3]) + arg4[(i3 + 1)]) * 0.333330f);


void jacobi1d(int arg1, int arg2, DATA_TYPE POLYBENCH_1D(arg3, N, arg2), DATA_TYPE POLYBENCH_1D(arg4, N, arg2)) {
int lbp, ubp, lbv, ubv;
int k0, k1, k2, k3;
if ((arg1 >= 1) && (arg2 >= 3)) {
  for (k0=0;k0<=floord(3*arg1+arg2-5,32);k0++) {
    lbp=max(ceild(2*k0,3),ceild(32*k0-arg1+1,32));
    ubp=min(min(floord(2*arg1+arg2-4,32),floord(64*k0+arg2+60,96)),k0);
#pragma omp parallel for private(k1, k2, k3, ubv, lbv)
    for (k1=lbp;k1<=ubp;k1++) {
      for (k2=max(ceild(32*k1-arg2+2,2),32*k0-32*k1);k2<=min(min(arg1-1,16*k1+15),32*k0-32*k1+31);k2++) {
        if (k1 >= ceild(k2+1,16)) {
          S1((k0-k1),k1,k2,(32*k1-2*k2));
        }
        if (k1 <= floord(k2,16)) {
          S1((k0-k1),k1,k2,1);
        }
        for (k3=max(32*k1+1,2*k2+2);k3<=(min(32*k1+31,2*k2+arg2-2))-7;k3+=8) {
          S1((k0-k1),k1,k2,(-2*k2+k3));
          S2((k0-k1),k1,k2,(-2*k2+k3-1));
          S1((k0-k1),k1,k2,(-2*k2+(k3+1)));
          S2((k0-k1),k1,k2,(-2*k2+(k3+1)-1));
          S1((k0-k1),k1,k2,(-2*k2+(k3+2)));
          S2((k0-k1),k1,k2,(-2*k2+(k3+2)-1));
          S1((k0-k1),k1,k2,(-2*k2+(k3+3)));
          S2((k0-k1),k1,k2,(-2*k2+(k3+3)-1));
          S1((k0-k1),k1,k2,(-2*k2+(k3+4)));
          S2((k0-k1),k1,k2,(-2*k2+(k3+4)-1));
          S1((k0-k1),k1,k2,(-2*k2+(k3+5)));
          S2((k0-k1),k1,k2,(-2*k2+(k3+5)-1));
          S1((k0-k1),k1,k2,(-2*k2+(k3+6)));
          S2((k0-k1),k1,k2,(-2*k2+(k3+6)-1));
          S1((k0-k1),k1,k2,(-2*k2+(k3+7)));
          S2((k0-k1),k1,k2,(-2*k2+(k3+7)-1));
        }
        for (;k3<=min(32*k1+31,2*k2+arg2-2);k3++) {
          S1((k0-k1),k1,k2,(-2*k2+k3));
          S2((k0-k1),k1,k2,(-2*k2+k3-1));
        }
        if (k1 <= floord(2*k2+arg2-33,32)) {
          S2((k0-k1),k1,k2,(32*k1-2*k2+31));
        }
        if (k1 >= ceild(2*k2+arg2-32,32)) {
          S2((k0-k1),k1,k2,(arg2-2));
        }
      }
    }
  }
}

}
