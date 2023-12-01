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

#ifdef STDC_HEADERS
void HPL_pdfact
(
   HPL_T_panel *                    PANEL
)
#else
void HPL_pdfact
( PANEL )
   HPL_T_panel *                    PANEL;
#endif
{
/*
 * Purpose
 * =======
 *
 * HPL_pdfact recursively factorizes a  1-dimensional  panel of columns.
 * The  RPFACT  function pointer specifies the recursive algorithm to be
 * used, either Crout, Left- or Right looking.  NBMIN allows to vary the
 * recursive stopping criterium in terms of the number of columns in the
 * panel, and  NDIV allows to specify the number of subpanels each panel
 * should be divided into. Usuallly a value of 2 will be chosen. Finally
 * PFACT is a function pointer specifying the non-recursive algorithm to
 * to be used on at most NBMIN columns. One can also choose here between
 * Crout, Left- or Right looking.  Empirical tests seem to indicate that
 * values of 4 or 8 for NBMIN give the best results.
 * HPL_pdfact 递归地因式分解一维列面板。
 * RPFACT 函数指针指定要使用的递归算法，可以是 Crout、Left-looking 或 Right-looking。
 * NBMIN 允许根据面板中的列数变化递归停止准则，NDIV 允许指定每个面板应划分为的子面板数。
 * 通常会选择值为 2。
 * 最后，PFACT 是一个函数指针，指定用于在最多 NBMIN 列上使用的非递归算法，可以在 Crout、Left-looking 或 Right-looking 之间进行选择。
 * 经验测试似乎表明，对于 NBMIN 的值为 4 或 8 会产生最佳结果。
 *
 * Bi-directional  exchange  is  used  to  perform  the  swap::broadcast
 * operations  at once  for one column in the panel.  This  results in a
 * lower number of slightly larger  messages than usual.  On P processes
 * and assuming bi-directional links,  the running time of this function
 * can be approximated by (when N is equal to N0):
 * 使用双向交换一次性执行面板中一列的 swap::broadcast 操作。这会导致比通常情况下稍大的消息数量更少。
 * 在 P 个进程上，假设具有双向链接，该函数的运行时间可以近似为（当 N 等于 N0 时）：
 *
 *    N0 * log_2( P ) * ( lat + ( 2*N0 + 4 ) / bdwth ) +
 *    N0^2 * ( M - N0/3 ) * gam2-3
 *
 * where M is the local number of rows of  the panel, lat and bdwth  are
 * the latency and bandwidth of the network for  double  precision  real
 * words, and  gam2-3  is  an estimate of the  Level 2 and Level 3  BLAS
 * rate of execution. The  recursive  algorithm  allows indeed to almost
 * achieve  Level 3 BLAS  performance  in the panel factorization.  On a
 * large  number of modern machines,  this  operation is however latency
 * bound,  meaning  that its cost can  be estimated  by only the latency
 * portion N0 * log_2(P) * lat.  Mono-directional links will double this
 * communication cost.
 * 其中 M 是面板的本地行数，lat 和 bdwth 是网络的延迟和带宽，用于双精度实数。
 * gam2-3 是 Level 2 和 Level 3 BLAS 执行速度的估计。递归算法确实允许在面板因子化中几乎达到 Level 3 BLAS 的性能。
 * 在许多现代计算机上，这个操作实际上是延迟受限的，这意味着其成本只能通过延迟部分 N0 * log_2(P) * lat 进行估算。
 * 单向链接将使通信成本加倍。
 *
 * Arguments
 * =========
 *
 * PANEL   (local input/output)          HPL_T_panel *
 *         On entry,  PANEL  points to the data structure containing the
 *         panel information.
 *
 * ---------------------------------------------------------------------
 */
/*
 * .. Local Variables ..
 */
   void                       * vptr = NULL;
   int                        align, jb;
/* ..
 * .. Executable Statements ..
 */
   jb = PANEL->jb; PANEL->n -= jb; PANEL->ja += jb;

   if( ( PANEL->grid->mycol != PANEL->pcol ) || ( jb <= 0 ) ) return;
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_RPFACT );
#endif
   align = PANEL->algo->align;
   vptr  = (void *)malloc( ( (size_t)(align) +
              (size_t)(((4+((unsigned int)(jb) << 1)) << 1) )) *
              sizeof(double) );
   if( vptr == NULL )
   { HPL_pabort( __LINE__, "HPL_pdfact", "Memory allocation failed" ); }
/*
 * Factor the panel - Update the panel pointers
 */
   PANEL->algo->rffun( PANEL, PANEL->mp, jb, 0, (double *)HPL_PTR( vptr,
                       ((size_t)(align) * sizeof(double) ) ) );
   if( vptr ) free( vptr );

   PANEL->A   = Mptr( PANEL->A, 0, jb, PANEL->lda );
   PANEL->nq -= jb; PANEL->jj += jb;
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_RPFACT );
#endif
/*
 * End of HPL_pdfact
 */
}
