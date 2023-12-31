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
void HPL_pdpanel_new
(
   HPL_T_grid *                     GRID,
   HPL_T_palg *                     ALGO,
   const int                        M,
   const int                        N,
   const int                        JB,
   HPL_T_pmat *                     A,
   const int                        IA,
   const int                        JA,
   const int                        TAG,
   HPL_T_panel * *                  PANEL
)
#else
void HPL_pdpanel_new
( GRID, ALGO, M, N, JB, A, IA, JA, TAG, PANEL )
   HPL_T_grid *                     GRID;
   HPL_T_palg *                     ALGO;
   const int                        M;
   const int                        N;
   const int                        JB;
   HPL_T_pmat *                     A;
   const int                        IA;
   const int                        JA;
   const int                        TAG;
   HPL_T_panel * *                  PANEL;
#endif
{
/*
 * Purpose
 * =======
 *
 * HPL_pdpanel_new creates and initializes a panel data structure.
 * HPL_pdpanel_new 创建并初始化一个面板数据结构。
 *
 *
 * Arguments
 * =========
 *
 * GRID    (local input)                 HPL_T_grid *
 *         On entry,  GRID  points  to the data structure containing the
 *         process grid information.
 *
 * ALGO    (global input)                HPL_T_palg *
 *         On entry,  ALGO  points to  the data structure containing the
 *         algorithmic parameters.
 *
 * M       (local input)                 const int
 *         On entry, M specifies the global number of rows of the panel.
 *         M must be at least zero.
 * M（局部输入）                const int
 *         在输入时，M指定面板的全局行数。M必须至少为零。
 *
 * N       (local input)                 const int
 *         On entry,  N  specifies  the  global number of columns of the
 *         panel and trailing submatrix. N must be at least zero.
 * N（局部输入）                const int
 *         在输入时，N指定面板和尾部子矩阵的全局列数。N必须至少为零。
 *
 * JB      (global input)                const int
 *         On entry, JB specifies is the number of columns of the panel.
 *         JB must be at least zero.
 * JB（全局输入）                const int
 *         在输入时，JB指定面板的列数。JB必须至少为零。
 *
 * A       (local input/output)          HPL_T_pmat *
 *         On entry, A points to the data structure containing the local
 *         array information.
 * A（局部输入/输出）          HPL_T_pmat *
 *         在输入时，A指向包含本地数组信息的数据结构。
 *
 * IA      (global input)                const int
 *         On entry,  IA  is  the global row index identifying the panel
 *         and trailing submatrix. IA must be at least zero.
 * IA（全局输入）                const int
 *         在输入时，IA是标识面板和尾部子矩阵的全局行索引。IA必须至少为零。
 *
 * JA      (global input)                const int
 *         On entry, JA is the global column index identifying the panel
 *         and trailing submatrix. JA must be at least zero.
 * JA（全局输入）                const int
 *         在输入时，JA是标识面板和尾部子矩阵的全局列索引。JA必须至少为零。
 *
 * TAG     (global input)                const int
 *         On entry, TAG is the row broadcast message id.
 * TAG（全局输入）                const int
 *         在输入时，TAG是行广播消息的标识符。
 *
 * PANEL   (local input/output)          HPL_T_panel * *
 *         On entry,  PANEL  points  to  the  address  of the panel data
 *         structure to create and initialize.
 * PANEL（局部输入/输出）          HPL_T_panel * *
 *         在输入时，PANEL指向要创建和初始化的面板数据结构的地址。
 *
 * ---------------------------------------------------------------------
 */
/*
 * .. Local Variables ..
 */
   HPL_T_panel                * p = NULL;
/* ..
 * .. Executable Statements ..
 */
/*
 * Allocate the panel structure - Check for enough memory
 */
   if( !( p = (HPL_T_panel *)malloc( sizeof( HPL_T_panel ) ) ) )
   {
      HPL_pabort( __LINE__, "HPL_pdpanel_new", "Memory allocation failed" );
   }

   HPL_pdpanel_init( GRID, ALGO, M, N, JB, A, IA, JA, TAG, p );
   *PANEL = p;
/*
 * End of HPL_pdpanel_new
 */
}
