/*
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 * -- Copyright notice and Licensing terms:
 *
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:
 *
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:
 * This  product  includes  software  developed  at  the  University  of
 * Tennessee, Knoxville, Innovative Computing Laboratory.
 *
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.
 *
 * -- Disclaimer:
 *
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ---------------------------------------------------------------------
 */
/*
 * Include files
 */
#include "hpl.h"

#include "HPL_pdblas.h"

#ifdef STDC_HEADERS
void HPL_pdtest
(
   HPL_T_test *                     TEST,
   HPL_T_grid *                     GRID,
   HPL_T_palg *                     ALGO,
   const int                        N,
   const int                        NB
)
#else
void HPL_pdtest
( TEST, GRID, ALGO, N, NB )
   HPL_T_test *                     TEST;
   HPL_T_grid *                     GRID;
   HPL_T_palg *                     ALGO;
   const int                        N;
   const int                        NB;
#endif
{
/*
 * Purpose
 * =======
 *
 * HPL_pdtest performs  one  test  given a set of parameters such as the
 * process grid, the  problem size, the distribution blocking factor ...
 * This function generates  the data, calls  and times the linear system
 * solver,  checks  the  accuracy  of the  obtained vector solution  and
 * writes this information to the file pointed to by TEST->outfp.
 * HPL_pdtest在给定一组参数（如进程网格、问题大小、分布阻塞因子等）的情况下执行一个测试。
 * 该函数生成数据，调用并计时线性系统求解器，检查获得的向量解的准确性，并将此信息写入TEST->outfp指向的文件。
 *
 * Arguments
 * =========
 *
 * TEST    (global input)                HPL_T_test *
 *         On entry,  TEST  points  to a testing data structure:  outfp
 *         specifies the output file where the results will be printed.
 *         It is only defined and used by the process  0  of the  grid.
 *         thrsh  specifies  the  threshhold value  for the test ratio.
 *         Concretely, a test is declared "PASSED"  if and only if the
 *         following inequality is satisfied:
 *         ||Ax-b||_oo / ( epsil *
 *                         ( || x ||_oo * || A ||_oo + || b ||_oo ) *
 *                          N )  < thrsh.
 *         epsil  is the  relative machine precision of the distributed
 *         computer. Finally the test counters, kfail, kpass, kskip and
 *         ktest are updated as follows:  if the test passes,  kpass is
 *         incremented by one;  if the test fails, kfail is incremented
 *         by one; if the test is skipped, kskip is incremented by one.
 *         ktest is left unchanged.
 * TEST（全局输入）               HPL_T_test *
 *         在输入时，TEST指向一个测试数据结构：outfp指定将打印结果的输出文件。
 *         它仅由网格的进程0定义和使用。thrsh指定测试比率的阈值。
 *         具体而言，只有当满足以下不等式时，测试才被声明为“通过”：
 *         ||Ax-b||_oo / ( epsil * ( || x ||_oo * || A ||_oo + || b ||_oo ) * N )  < thrsh。
 *         epsil是分布式计算的相对机器精度。最后，测试计数器kfail、kpass、kskip和ktest按以下方式更新：
 *         如果测试通过，则kpass增加一；如果测试失败，则kfail增加一；如果测试被跳过，则kskip增加一。
 *         ktest保持不变。
 *
 * GRID    (local input)                 HPL_T_grid *
 *         On entry,  GRID  points  to the data structure containing the
 *         process grid information.
 * GRID（局部输入）                 HPL_T_grid *
 *         在输入时，GRID指向包含进程网格信息的数据结构。
 *
 * ALGO    (global input)                HPL_T_palg *
 *         On entry,  ALGO  points to  the data structure containing the
 *         algorithmic parameters to be used for this test.
 * ALGO（全局输入）                HPL_T_palg *
 *         在输入时，ALGO指向包含用于此测试的算法参数的数据结构。
 *
 * N       (global input)                const int
 *         On entry,  N specifies the order of the coefficient matrix A.
 *         N must be at least zero.
 * N（全局输入）                const int
 *         在输入时，N指定系数矩阵A的阶数。N必须至少为零。
 *
 * NB      (global input)                const int
 *         On entry,  NB specifies the blocking factor used to partition
 *         and distribute the matrix A. NB must be larger than one.
 * NB（全局输入）                const int
 *         在输入时，NB指定用于划分和分发矩阵A的阻塞因子。NB必须大于一
 *
 * ---------------------------------------------------------------------
 */
/*
 * .. Local Variables ..
 */
#ifdef HPL_DETAILED_TIMING
   double                     HPL_w[HPL_TIMING_N];
#endif
   HPL_T_pmat                 mat;
   double                     wtime[1];
   int                        info[3];
   double                     Anorm1, AnormI, Gflops, Xnorm1, XnormI,
                              BnormI, resid0, resid1;
   double                     * Bptr;
   void                       * vptr = NULL;
   static int                 first=1;
   int                        ii, ip2, mycol, myrow, npcol, nprow, nq;
   char                       ctop, cpfact, crfact;
   time_t                     current_time_start, current_time_end;
/* ..
 * .. Executable Statements ..
 */
   (void) HPL_grid_info( GRID, &nprow, &npcol, &myrow, &mycol );

   mat.n  = N; mat.nb = NB; mat.info = 0;
   mat.mp = HPL_numroc( N, NB, NB, myrow, 0, nprow ); /* 本地行数目 */
   nq     = HPL_numroc( N, NB, NB, mycol, 0, npcol );
   mat.nq = nq + 1; /* 本地列数目 */
/*
 * Allocate matrix, right-hand-side, and vector solution x. [ A | b ] is
 * N by N+1.  One column is added in every process column for the solve.
 * The  result  however  is stored in a 1 x N vector replicated in every
 * process row. In every process, A is lda * (nq+1), x is 1 * nq and the
 * workspace is mp.
 *
 * Ensure that lda is a multiple of ALIGN and not a power of 2
 * 分配矩阵、右侧向量和向量解x。 [ A | b ] 是N乘以N+1。 在每个进程列中为求解添加了一列。
 * 然而，结果存储在在每个进程行中复制的1乘以N向量中。在每个进程中，A是lda * (nq+1)，x是1 * nq，
 * 并且工作空间是mp。
 *
 * 确保lda是ALIGN的倍数，而且不是2的幂。
 */
   mat.ld = ( ( Mmax( 1, mat.mp ) - 1 ) / ALGO->align ) * ALGO->align; /* 本地主维度 */
   do
   {
      ii = ( mat.ld += ALGO->align ); ip2 = 1;
      while( ii > 1 ) { ii >>= 1; ip2 <<= 1; }
   }
   while( mat.ld == ip2 ); /* 保证ld是ALIGN的倍数，并且不是2的幂的最小值 */
/*
 * Allocate dynamic memory
 */
   vptr = (void*)malloc( ( (size_t)(ALGO->align) +
                           (size_t)(mat.ld+1) * (size_t)(mat.nq) ) *
                         sizeof(double) );
   info[0] = (vptr == NULL); info[1] = myrow; info[2] = mycol;
   (void) HPL_all_reduce( (void *)(info), 3, HPL_INT, HPL_max,
                          GRID->all_comm );
   if( info[0] != 0 )
   {
      if( ( myrow == 0 ) && ( mycol == 0 ) )
         HPL_pwarn( TEST->outfp, __LINE__, "HPL_pdtest",
                    "[%d,%d] %s", info[1], info[2],
                    "Memory allocation failed for A, x and b. Skip." );
      (TEST->kskip)++;
      return;
   }
/*
 * generate matrix and right-hand-side, [ A | b ] which is N by N+1.
 * 生成矩阵和右侧向量 [ A | b ]，其大小为 N 乘以 N+1
 */
   mat.A  = (double *)HPL_PTR( vptr,
                               ((size_t)(ALGO->align) * sizeof(double) ) );
   mat.X  = Mptr( mat.A, 0, mat.nq, mat.ld );
   HPL_pdmatgen( GRID, N, N+1, NB, mat.A, mat.ld, HPL_ISEED ); /* 初始化A和b，都是以列存储，b在最后一列 */
#ifdef HPL_CALL_VSIPL
   mat.block = vsip_blockbind_d( (vsip_scalar_d *)(mat.A),
                                 (vsip_length)(mat.ld * mat.nq),
                                 VSIP_MEM_NONE );
#endif
/*
 * Solve linear system
 */
   HPL_ptimer_boot(); (void) HPL_barrier( GRID->all_comm );
   time( &current_time_start );
   HPL_ptimer( 0 );
   HPL_pdgesv( GRID, ALGO, &mat );
   HPL_ptimer( 0 );
   time( &current_time_end );
#ifdef HPL_CALL_VSIPL
   (void) vsip_blockrelease_d( mat.block, VSIP_TRUE );
   vsip_blockdestroy_d( mat.block );
#endif
/*
 * Gather max of all CPU and WALL clock timings and print timing results
 */
   HPL_ptimer_combine( GRID->all_comm, HPL_AMAX_PTIME, HPL_WALL_PTIME,
                       1, 0, wtime );

   if( ( myrow == 0 ) && ( mycol == 0 ) )
   {
      if( first )
      {
         HPL_fprintf( TEST->outfp, "%s%s\n",
                      "========================================",
                      "========================================" );
         HPL_fprintf( TEST->outfp, "%s%s\n",
                      "T/V                N    NB     P     Q",
                      "               Time                 Gflops" );
         HPL_fprintf( TEST->outfp, "%s%s\n",
                      "----------------------------------------",
                      "----------------------------------------" );
         if( TEST->thrsh <= HPL_rzero ) first = 0;
      }
/*
 * 2/3 N^3 - 1/2 N^2 flops for LU factorization + 2 N^2 flops for solve.
 * Print WALL time
 */
      Gflops = ( ( (double)(N) /   1.0e+9 ) *
                 ( (double)(N) / wtime[0] ) ) *
                 ( ( 2.0 / 3.0 ) * (double)(N) + ( 3.0 / 2.0 ) );

      cpfact = ( ( (HPL_T_FACT)(ALGO->pfact) ==
                   (HPL_T_FACT)(HPL_LEFT_LOOKING) ) ?  (char)('L') :
                 ( ( (HPL_T_FACT)(ALGO->pfact) == (HPL_T_FACT)(HPL_CROUT) ) ?
                   (char)('C') : (char)('R') ) );
      crfact = ( ( (HPL_T_FACT)(ALGO->rfact) ==
                   (HPL_T_FACT)(HPL_LEFT_LOOKING) ) ?  (char)('L') :
                 ( ( (HPL_T_FACT)(ALGO->rfact) == (HPL_T_FACT)(HPL_CROUT) ) ?
                   (char)('C') : (char)('R') ) );

      if(      ALGO->btopo == HPL_1RING   ) ctop = '0';
      else if( ALGO->btopo == HPL_1RING_M ) ctop = '1';
      else if( ALGO->btopo == HPL_2RING   ) ctop = '2';
      else if( ALGO->btopo == HPL_2RING_M ) ctop = '3';
      else if( ALGO->btopo == HPL_BLONG   ) ctop = '4';
      else /* if( ALGO->btopo == HPL_BLONG_M ) */ ctop = '5';

      if( wtime[0] > HPL_rzero ) {
         HPL_fprintf( TEST->outfp,
             "W%c%1d%c%c%1d%c%1d%12d %5d %5d %5d %18.2f     %18.3e\n",
             ( GRID->order == HPL_ROW_MAJOR ? 'R' : 'C' ),
             ALGO->depth, ctop, crfact, ALGO->nbdiv, cpfact, ALGO->nbmin,
             N, NB, nprow, npcol, wtime[0], Gflops );
         HPL_fprintf( TEST->outfp,
             "HPL_pdgesv() start time %s\n", ctime( &current_time_start ) );
         HPL_fprintf( TEST->outfp,
             "HPL_pdgesv() end time   %s\n", ctime( &current_time_end ) );
      }
   }
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer_combine( GRID->all_comm, HPL_AMAX_PTIME, HPL_WALL_PTIME,
                       HPL_TIMING_N, HPL_TIMING_BEG, HPL_w );
   if( ( myrow == 0 ) && ( mycol == 0 ) )
   {
      HPL_fprintf( TEST->outfp, "%s%s\n",
                   "--VVV--VVV--VVV--VVV--VVV--VVV--VVV--V",
                   "VV--VVV--VVV--VVV--VVV--VVV--VVV--VVV-" );
/*
 * Recursive panel factorization
 */
      if( HPL_w[HPL_TIMING_RPFACT-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Max aggregated wall time rfact . . . : %18.2f\n",
                      HPL_w[HPL_TIMING_RPFACT-HPL_TIMING_BEG] );
/*
 * Panel factorization
 */
      if( HPL_w[HPL_TIMING_PFACT-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "+ Max aggregated wall time pfact . . : %18.2f\n",
                      HPL_w[HPL_TIMING_PFACT-HPL_TIMING_BEG] );
/*
 * Panel factorization (swap)
 */
      if( HPL_w[HPL_TIMING_MXSWP-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "+ Max aggregated wall time mxswp . . : %18.2f\n",
                      HPL_w[HPL_TIMING_MXSWP-HPL_TIMING_BEG] );
/*
 * Update
 */
      if( HPL_w[HPL_TIMING_UPDATE-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Max aggregated wall time update  . . : %18.2f\n",
                      HPL_w[HPL_TIMING_UPDATE-HPL_TIMING_BEG] );
/*
 * Update (swap)
 */
      if( HPL_w[HPL_TIMING_LASWP-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "+ Max aggregated wall time laswp . . : %18.2f\n",
                      HPL_w[HPL_TIMING_LASWP-HPL_TIMING_BEG] );
/*
 * Upper triangular system solve
 */
      if( HPL_w[HPL_TIMING_PTRSV-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Max aggregated wall time up tr sv  . : %18.2f\n",
                      HPL_w[HPL_TIMING_PTRSV-HPL_TIMING_BEG] );

      if( TEST->thrsh <= HPL_rzero )
         HPL_fprintf( TEST->outfp, "%s%s\n",
                      "========================================",
                      "========================================" );
/*
 * Dgemm
 */
      if( HPL_w[HPL_TIMING_DGEMM-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Dgemm  . : %18.2f\n",
                      HPL_w[HPL_TIMING_DGEMM-HPL_TIMING_BEG] );
/*
 * Dtrsm
 */
      if( HPL_w[HPL_TIMING_DTRSM-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Dtrsm  . : %18.2f\n",
                      HPL_w[HPL_TIMING_DTRSM-HPL_TIMING_BEG] );
/*
 * Dgemv
 */
      if( HPL_w[HPL_TIMING_DGEMV-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Dgemv  . : %18.2f\n",
                      HPL_w[HPL_TIMING_DGEMV-HPL_TIMING_BEG] );
/*
 * Dtrsv
 */
      if( HPL_w[HPL_TIMING_DTRSV-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Dtrsv  . : %18.2f\n",
                      HPL_w[HPL_TIMING_DTRSV-HPL_TIMING_BEG] );
/*
 * Dger
 */
      if( HPL_w[HPL_TIMING_DGER-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Dger  . : %18.2f\n",
                      HPL_w[HPL_TIMING_DGER-HPL_TIMING_BEG] );
/*
 * Dswap
 */
      if( HPL_w[HPL_TIMING_DSWAP-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Dswap  . : %18.2f\n",
                      HPL_w[HPL_TIMING_DSWAP-HPL_TIMING_BEG] );
/*
 * Dcopy
 */
      if( HPL_w[HPL_TIMING_DCOPY-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Dcopy  . : %18.2f\n",
                      HPL_w[HPL_TIMING_DCOPY-HPL_TIMING_BEG] );
/*
 * Daxpy
 */
      if( HPL_w[HPL_TIMING_DAXPY-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Daxpy  . : %18.2f\n",
                      HPL_w[HPL_TIMING_DAXPY-HPL_TIMING_BEG] );
/*
 * Dscal
 */
      if( HPL_w[HPL_TIMING_DSCAL-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Dscal  . : %18.2f\n",
                      HPL_w[HPL_TIMING_DSCAL-HPL_TIMING_BEG] );
/*
 * Idamax
 */
      if( HPL_w[HPL_TIMING_IDAMAX-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Idamax  . : %18.2f\n",
                      HPL_w[HPL_TIMING_IDAMAX-HPL_TIMING_BEG] );
   }
#endif
/*
 * Quick return, if I am not interested in checking the computations
 */
   if( TEST->thrsh <= HPL_rzero )
   { (TEST->kpass)++; if( vptr ) free( vptr ); return; }
/*
 * Check info returned by solve
 */
   if( mat.info != 0 )
   {
      if( ( myrow == 0 ) && ( mycol == 0 ) )
         HPL_pwarn( TEST->outfp, __LINE__, "HPL_pdtest", "%s %d, %s",
                    "Error code returned by solve is", mat.info, "skip" );
      (TEST->kskip)++;
      if( vptr ) free( vptr ); return;
   }
/*
 * Check computation, re-generate [ A | b ], compute norm 1 and inf of A and x,
 * and norm inf of b - A x. Display residual checks.
 */
   HPL_pdmatgen( GRID, N, N+1, NB, mat.A, mat.ld, HPL_ISEED );
   Anorm1 = HPL_pdlange( GRID, HPL_NORM_1, N, N, NB, mat.A, mat.ld );
   AnormI = HPL_pdlange( GRID, HPL_NORM_I, N, N, NB, mat.A, mat.ld );
/*
 * Because x is distributed in process rows, switch the norms
 */
   XnormI = HPL_pdlange( GRID, HPL_NORM_1, 1, N, NB, mat.X, 1 );
   Xnorm1 = HPL_pdlange( GRID, HPL_NORM_I, 1, N, NB, mat.X, 1 );
/*
 * If I am in the col that owns b, (1) compute local BnormI, (2) all_reduce to
 * find the max (in the col). Then (3) broadcast along the rows so that every
 * process has BnormI. Note that since we use a uniform distribution in [-0.5,0.5]
 * for the entries of B, it is very likely that BnormI (<=,~) 0.5.
 */
   Bptr = Mptr( mat.A, 0, nq, mat.ld );
   if( mycol == HPL_indxg2p( N, NB, NB, 0, npcol ) ){
      if( mat.mp > 0 )
      {
         BnormI = Bptr[HPL_idamax( mat.mp, Bptr, 1 )]; BnormI = Mabs( BnormI );
      }
      else
      {
         BnormI = HPL_rzero;
      }
      (void) HPL_all_reduce( (void *)(&BnormI), 1, HPL_DOUBLE, HPL_max,
                             GRID->col_comm );
   }
   (void) HPL_broadcast( (void *)(&BnormI), 1, HPL_DOUBLE,
                          HPL_indxg2p( N, NB, NB, 0, npcol ),
                          GRID->row_comm );
/*
 * If I own b, compute ( b - A x ) and ( - A x ) otherwise
 */
   if( mycol == HPL_indxg2p( N, NB, NB, 0, npcol ) )
   {
      HPL_dgemv( HplColumnMajor, HplNoTrans, mat.mp, nq, -HPL_rone,
                 mat.A, mat.ld, mat.X, 1, HPL_rone, Bptr, 1 );
   }
   else if( nq > 0 )
   {
      HPL_dgemv( HplColumnMajor, HplNoTrans, mat.mp, nq, -HPL_rone,
                 mat.A, mat.ld, mat.X, 1, HPL_rzero, Bptr, 1 );
   }
   else { for( ii = 0; ii < mat.mp; ii++ ) Bptr[ii] = HPL_rzero; }
/*
 * Reduce the distributed residual in process column 0
 */
   if( mat.mp > 0 )
      (void) HPL_reduce( Bptr, mat.mp, HPL_DOUBLE, HPL_sum, 0,
                         GRID->row_comm );
/*
 * Compute || b - A x ||_oo
 */
   resid0 = HPL_pdlange( GRID, HPL_NORM_I, N, 1, NB, Bptr, mat.ld );
/*
 * Computes and displays norms, residuals ...
 */
   if( N <= 0 )
   {
      resid1 = HPL_rzero;
   }
   else
   {
      resid1 = resid0 / ( TEST->epsil * ( AnormI * XnormI + BnormI ) * (double)(N) );
   }

   if( resid1 < TEST->thrsh ) (TEST->kpass)++;
   else                       (TEST->kfail)++;

   if( ( myrow == 0 ) && ( mycol == 0 ) )
   {
      HPL_fprintf( TEST->outfp, "%s%s\n",
                   "----------------------------------------",
                   "----------------------------------------" );
      HPL_fprintf( TEST->outfp, "%s%16.7f%s%s\n",
         "||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)= ", resid1,
         " ...... ", ( resid1 < TEST->thrsh ? "PASSED" : "FAILED" ) );

      if( resid1 >= TEST->thrsh )
      {
         HPL_fprintf( TEST->outfp, "%s%18.6f\n",
         "||Ax-b||_oo  . . . . . . . . . . . . . . . . . = ", resid0 );
         HPL_fprintf( TEST->outfp, "%s%18.6f\n",
         "||A||_oo . . . . . . . . . . . . . . . . . . . = ", AnormI );
         HPL_fprintf( TEST->outfp, "%s%18.6f\n",
         "||A||_1  . . . . . . . . . . . . . . . . . . . = ", Anorm1 );
         HPL_fprintf( TEST->outfp, "%s%18.6f\n",
         "||x||_oo . . . . . . . . . . . . . . . . . . . = ", XnormI );
         HPL_fprintf( TEST->outfp, "%s%18.6f\n",
         "||x||_1  . . . . . . . . . . . . . . . . . . . = ", Xnorm1 );
         HPL_fprintf( TEST->outfp, "%s%18.6f\n",
         "||b||_oo . . . . . . . . . . . . . . . . . . . = ", BnormI );
      }
   }
   if( vptr ) free( vptr );
/*
 * End of HPL_pdtest
 */
}
