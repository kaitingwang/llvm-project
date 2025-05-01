#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <polybench.h>
#include "lu.h"
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define max(x,y)    ((x) > (y) ? (x) : (y))
#define min(x,y)    ((x) < (y) ? (x) : (y))
#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0) 
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2) 


#define S1(i0, i1, i2, i3, i4, i5) arg2[i3][i4] = (arg2[i3][i4] - (arg2[i3][i5] * arg2[i5][i4]));
#define S2(i0, i1, i2, i3, i4) arg2[i3][i4] = (arg2[i3][i4] / arg2[i4][i4]);
#define S3(i0, i1, i2, i3, i4, i5) arg2[i3][i4] = (arg2[i3][i4] - (arg2[i3][i5] * arg2[i5][i4]));


void lu(int arg1, DATA_TYPE POLYBENCH_2D(arg2,N,N,arg1,arg1)) {
int lbp, ubp, lbv, ubv;
int k0, k1, k2, k3, k4, k5;
if (arg1 >= 2) {
  for (k0=0;k0<=floord(arg1-1,16);k0++) {
    lbp=max(0,ceild(32*k0-arg1+1,32));
    ubp=min(floord(arg1-1,32),k0);
#pragma omp parallel for private(k1, k2, k3, k4, k5, ubv, lbv)
    for (k1=lbp;k1<=ubp;k1++) {
      for (k2=0;k2<=min(min(floord(arg1-2,32),k1),k0-k1);k2++) {
        if (k1 == k2) {
          for (k3=max(32*k0-32*k1,32*k1+32);k3<=min(arg1-1,32*k0-32*k1+31);k3++) {
            for (k4=32*k1;k4<=32*k1+30;k4++) {
              S2((k0-k1),k1,k1,k3,k4);
              for (k5=k4+1;k5<=32*k1+31;k5++) {
                S1((k0-k1),k1,k1,k3,k5,k4);
              }
            }
            S2((k0-k1),k1,k1,k3,(32*k1+31));
          }
        }
        if (k1 >= k2+1) {
          for (k3=max(32*k0-32*k1,32*k1+32);k3<=min(arg1-1,32*k0-32*k1+31);k3++) {
            for (k4=32*k2;k4<=32*k2+31;k4++) {
              for (k5=32*k1;k5<=32*k1+31;k5++) {
                S1((k0-k1),k1,k2,k3,k5,k4);
              }
            }
          }
        }
        if ((k0 == 2*k1) && (k0 == 2*k2)) {
          if (k0%2 == 0) {
            S2((k0/2),(k0/2),(k0/2),(16*k0+1),16*k0);
          }
          for (k5=16*k0+1;k5<=min(arg1-1,16*k0+31);k5++) {
            if (k0%2 == 0) {
              S3((k0/2),(k0/2),(k0/2),(16*k0+1),k5,16*k0);
            }
          }
        }
        for (k3=max(32*k0-32*k1,32*k2+1);k3<=min(32*k1,32*k0-32*k1+31);k3++) {
          for (k4=32*k2;k4<=min(32*k2+31,k3-1);k4++) {
            for (k5=32*k1;k5<=min(arg1-1,32*k1+31);k5++) {
              S3((k0-k1),k1,k2,k3,k5,k4);
            }
          }
        }
        if ((k0 == 2*k1) && (k0 == 2*k2)) {
          for (k3=16*k0+2;k3<=min(arg1-1,16*k0+31);k3++) {
            for (k4=16*k0;k4<=k3-2;k4++) {
              if (k0%2 == 0) {
                S2((k0/2),(k0/2),(k0/2),k3,k4);
              }
              for (k5=k4+1;k5<=k3-1;k5++) {
                if (k0%2 == 0) {
                  S1((k0/2),(k0/2),(k0/2),k3,k5,k4);
                }
              }
              for (k5=k3;k5<=min(arg1-1,16*k0+31);k5++) {
                if (k0%2 == 0) {
                  S3((k0/2),(k0/2),(k0/2),k3,k5,k4);
                }
              }
            }
            if (k0%2 == 0) {
              S2((k0/2),(k0/2),(k0/2),k3,(k3-1));
            }
            for (k5=k3;k5<=min(arg1-1,16*k0+31);k5++) {
              if (k0%2 == 0) {
                S3((k0/2),(k0/2),(k0/2),k3,k5,(k3-1));
              }
            }
          }
        }
        if ((k0 == 2*k1) && (k0 >= 2*k2+2)) {
          for (k3=16*k0+1;k3<=min(arg1-1,16*k0+31);k3++) {
            for (k4=32*k2;k4<=32*k2+31;k4++) {
              for (k5=16*k0;k5<=k3-1;k5++) {
                if (k0%2 == 0) {
                  S1((k0/2),(k0/2),k2,k3,k5,k4);
                }
              }
              for (k5=k3;k5<=min(arg1-1,16*k0+31);k5++) {
                if (k0%2 == 0) {
                  S3((k0/2),(k0/2),k2,k3,k5,k4);
                }
              }
            }
          }
        }
      }
    }
  }
}

}
