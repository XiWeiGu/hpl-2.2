/*
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    guxiwei <guxiwei-hf@loongson.cn>
 *
 */

#include "hpl.h"

void g_cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
		   const int M, const int N, const int K, const double alpha, const double *A, const int lda,
		   const double *B, const int ldb, const double beta, double *C, const int ldc) {
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DGEMM );
#endif
	cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DGEMM );
#endif
}

void g_cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo,
		   const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M,
		   const int N, const double alpha, const double *A, const int lda,
		   double *B, const int ldb) {
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DTRSM );
#endif
	cblas_dtrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DTRSM );
#endif
}

void g_cblas_dgemv(const enum CBLAS_ORDER order,  const enum CBLAS_TRANSPOSE trans,  const int m,
		   const int n, const double alpha, const double  *a, const int lda,
		   const double  *x, const int incx,  const double beta,  double  *y, const int incy) {
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DGEMV );
#endif
	cblas_dgemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DGEMV );
#endif
}

void g_cblas_dger (const enum CBLAS_ORDER order, const int M, const int N, const double  alpha,
		   const double *X, const int incX, const double *Y, const int incY, double *A,
		   const int lda){
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DGER );
#endif
	cblas_dger(order, M, N, alpha, X, incX, Y, incY, A, lda);
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DGER );
#endif
}

void g_cblas_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
		   const enum CBLAS_DIAG Diag, const int N, const double *A, const int lda,
		   double *X, const int incX) {
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DTRSV );
#endif
	cblas_dtrsv(order, Uplo, TransA, Diag, N, A, lda, X, incX);
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DTRSV );
#endif
}

void g_cblas_dswap(const int n, double *x, const int incx, double *y, const int incy) {
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DSWAP );
#endif
	cblas_dswap(n, x, incx, y, incy);
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DSWAP );
#endif
}

void g_cblas_dcopy(const int n, const double *x, const int incx, double *y, const int incy) {
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DCOPY );
#endif
	cblas_dcopy(n, x, incx, y, incy);
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DCOPY );
#endif
}

void g_cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy) {
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DAXPY );
#endif
	cblas_daxpy(n, alpha, x, incx, y, incy);
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DAXPY );
#endif
}

void g_cblas_dscal(const int N, const double alpha, double *X, const int incX) {
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DSCAL );
#endif
	cblas_dscal(N, alpha, X, incX);
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_DSCAL );
#endif
}

CBLAS_INDEX g_cblas_idamax(const int n, const double *x, const int incx) {
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_IDAMAX );
#endif
	CBLAS_INDEX index = cblas_idamax(n, x, incx);
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_IDAMAX );
#endif
   return index;
}
