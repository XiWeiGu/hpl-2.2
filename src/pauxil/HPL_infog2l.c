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
void HPL_infog2l
(
   int                              I,
   int                              J,
   const int                        IMB,
   const int                        MB,
   const int                        INB,
   const int                        NB,
   const int                        RSRC,
   const int                        CSRC,
   const int                        MYROW,
   const int                        MYCOL,
   const int                        NPROW,
   const int                        NPCOL,
   int *                            II,
   int *                            JJ,
   int *                            PROW,
   int *                            PCOL
)
#else
void HPL_infog2l
( I, J, IMB, MB, INB, NB, RSRC, CSRC, MYROW, MYCOL, NPROW, NPCOL, II, JJ, PROW, PCOL )
   int                              I;
   int                              J;
   const int                        IMB;
   const int                        MB;
   const int                        INB;
   const int                        NB;
   const int                        RSRC;
   const int                        CSRC;
   const int                        MYROW;
   const int                        MYCOL;
   const int                        NPROW;
   const int                        NPCOL;
   int *                            II;
   int *                            JJ;
   int *                            PROW;
   int *                            PCOL;
#endif
{
/*
 * Purpose
 * =======
 *
 * HPL_infog2l computes the starting local index II, JJ corresponding to
 * the submatrix starting globally at the entry pointed by  I,  J.  This
 * routine returns the coordinates in the grid of the process owning the
 * matrix entry of global indexes I, J, namely PROW and PCOL.
 * HPL_infog2l计算与从指针I、J指向的全局子矩阵对应的起始本地索引II、JJ。该例程返回拥有
 * 全局索引I、J的矩阵条目的进程在网格中的坐标，即PROW和PCOL。
 *
 * Arguments
 * =========
 *
 * I       (global input)                int
 *         On entry,  I  specifies  the  global  row index of the matrix
 *         entry. I must be at least zero.
 * I（全局输入）                int
 *         在输入时，I指定矩阵条目的全局行索引。I必须至少为零。
 *
 * J       (global input)                int
 *         On entry,  J  specifies the global column index of the matrix
 *         entry. J must be at least zero.
 * J（全局输入）                int
 *         在输入时，J指定矩阵条目的全局列索引。J必须至少为零。
 *
 * IMB     (global input)                const int
 *         On entry,  IMB  specifies  the size of the first row block of
 *         the global matrix. IMB must be at least one.
 * IMB（全局输入）                const int
 *         在输入时，IMB指定全局矩阵的第一行块的大小。IMB必须至少为一。
 *
 * MB      (global input)                const int
 *         On entry,  MB specifies the blocking factor used to partition
 *         and  distribute the rows of the matrix A.  MB  must be larger
 *         than one.
 * MB（全局输入）                const int
 *         在输入时，MB指定用于划分和分发矩阵A的行的阻塞因子。MB必须大于一。
 *
 * INB     (global input)                const int
 *         On entry, INB specifies the size of the first column block of
 *         the global matrix. INB must be at least one.
 * INB（全局输入）                const int
 *         在输入时，INB指定全局矩阵的第一列块的大小。INB必须至少为一。
 *
 * NB      (global input)                const int
 *         On entry,  NB specifies the blocking factor used to partition
 *         and distribute the columns of the matrix A. NB must be larger
 *         than one.
 * NB（全局输入）                const int
 *         在输入时，NB指定用于划分和分发矩阵A的列的阻塞因子。NB必须大于一。
 *
 * RSRC    (global input)                const int
 *         On entry,  RSRC  specifies  the row coordinate of the process
 *         that possesses the row  I.  RSRC  must  be at least zero  and
 *         strictly less than NPROW.
 * RSRC（全局输入）                const int
 *         在输入时，RSRC指定拥有行I的进程的行坐标。RSRC必须至少为零，且严格小于NPROW。
 *
 * CSRC    (global input)                const int
 *         On entry, CSRC specifies the column coordinate of the process
 *         that possesses the column J. CSRC  must be at least zero  and
 *         strictly less than NPCOL.
 * CSRC（全局输入）                const int
 *         在输入时，CSRC指定拥有列J的进程的列坐标。CSRC必须至少为零，且严格小于NPCOL。
 *
 * MYROW   (local input)                 const int
 *         On entry, MYROW  specifies my  row process  coordinate in the
 *         grid. MYROW is greater than or equal  to zero  and  less than
 *         NPROW.
 * MYROW（局部输入）                const int
 *         在输入时，MYROW指定了我的行进程在网格中的坐标。MYROW大于或等于零，且小于NPROW。
 *
 * MYCOL   (local input)                 const int
 *         On entry, MYCOL specifies my column process coordinate in the
 *         grid. MYCOL is greater than or equal  to zero  and  less than
 *         NPCOL.
 * MYCOL（局部输入）                const int
 *         在输入时，MYCOL指定了我的列进程在网格中的坐标。MYCOL大于或等于零，且小于NPCOL。
 *
 * NPROW   (global input)                const int
 *         On entry,  NPROW  specifies the number of process rows in the
 *         grid. NPROW is at least one.
 * NPROW（全局输入）                const int
 *         在输入时，NPROW指定网格中的进程行数。NPROW至少为一。
 *
 * NPCOL   (global input)                const int
 *         On entry,  NPCOL  specifies  the number of process columns in
 *         the grid. NPCOL is at least one.
 * NPCOL（全局输入）                const int
 *         在输入时，NPCOL指定网格中的进程列数。NPCOL至少为一。
 *
 * II      (local output)                int *
 *         On exit, II  specifies the  local  starting  row index of the
 *         submatrix. On exit, II is at least 0.
 * II（局部输出）                int *
 *         在退出时，II指定子矩阵的本地起始行索引。在退出时，II至少为0。
 *
 * JJ      (local output)                int *
 *         On exit, JJ  specifies the local starting column index of the
 *         submatrix. On exit, JJ is at least 0.
 * JJ（局部输出）                int *
 *         在退出时，JJ指定子矩阵的本地起始列索引。在退出时，JJ至少为0。
 *
 * PROW    (global output)               int *
 *         On exit, PROW is the row coordinate of the process owning the
 *         entry specified by the global index I.  PROW is at least zero
 *         and less than NPROW.
 * PROW（全局输出）               int *
 *         在退出时，PROW是拥有由全局索引I指定的条目的进程的行坐标。PROW至少为零，且小于NPROW。
 *
 * PCOL    (global output)               int *
 *         On exit, PCOL  is the column coordinate of the process owning
 *         the entry specified by the global index J.  PCOL  is at least
 *         zero and less than NPCOL.
 * PCOL（全局输出）               int *
 *         在退出时，PCOL是拥有由全局索引J指定的条目的进程的列坐标。PCOL至少为零，且小于NPCOL。
 *
 * ---------------------------------------------------------------------
 */
/*
 * .. Local Variables ..
 */
   int            ilocblk, imb, inb, mb, mydist, nb, nblocks, csrc, rsrc;
/* ..
 * .. Executable Statements ..
 */
   imb   = IMB;
   *PROW = RSRC;

   if( ( *PROW == -1 ) || ( NPROW == 1 ) )
   {
/*
 * The data is not distributed,  or there is just one process row in the
 * grid.
 */
     *II = I;
   }
   else if( I < imb )
   {
/*
 * I refers to an entry in the first block of rows
 */
     *II = ( MYROW == *PROW ? I : 0 );
   }
   else
   {
      mb   = MB;
      rsrc = *PROW;
/*
 * The discussion goes as follows:  compute  my distance from the source
 * process so that  within  this process coordinate system,  the  source
 * process   is  the  process  such  that  mydist = 0,  or  equivalently
 * MYROW == rsrc.
 *
 * Find  out  the global coordinate of the block I belongs to (nblocks),
 * as well as the minimum local number of blocks that every process has.
 *
 * when mydist < nblocks-ilocblk*NPROCS,  I own ilocblk + 1 full blocks,
 * when mydist > nblocks-ilocblk*NPROCS,  I own ilocblk     full blocks,
 * when mydist = nblocks-ilocblk*NPROCS,  I own ilocblk     full blocks
 * but not I, or I own ilocblk + 1 blocks and the entry I refers to.
 */
      if( MYROW == rsrc )
      {
/*
 * I refers  to an entry  that is not in the first block, find out which
 * process has it.
 */
         nblocks = ( I - imb ) / mb + 1;
         *PROW  += nblocks;
         *PROW  -= ( *PROW / NPROW ) * NPROW;
/*
 * Since  mydist = 0  and nblocks - ilocblk * NPROW >= 0, there are only
 * three possible cases:
 *
 *   1) When 0 = mydist = nblocks - ilocblk * NPROW = 0 and I do not own
 *      I, in which case II = IMB + ( ilocblk - 1 ) * MB. Note that this
 *      case  cannot  happen  when  ilocblk is zero, since nblocks is at
 *      least one.
 *
 *   2) When  0 = mydist = nblocks - ilocblk * NPROW = 0 and I own I, in
 *      which  case  I  and  II  can  respectively  be  written as IMB +
 *      (nblocks-1)*NB + IL  and  IMB + (ilocblk-1) * MB + IL.  That  is
 *      II = I + (ilocblk-nblocks)*MB. Note that this case cannot happen
 *      when ilocblk is zero, since nblocks is at least one.
 *
 *   3) mydist = 0 < nblocks - ilocblk * NPROW,  the source process owns
 *      ilocblk+1 full blocks,  and  therefore  II = IMB + ilocblk * MB.
 *      Note that when ilocblk is zero, II is just IMB.
 */
         if( nblocks < NPROW )
         {
            *II = imb;
         }
         else
         {
            ilocblk = nblocks / NPROW;
            if( ilocblk * NPROW >= nblocks )
            {
               *II = ( ( MYROW == *PROW ) ?
                       I   + ( ilocblk - nblocks ) * mb :
                       imb + ( ilocblk - 1       ) * mb );
            }
            else
            {
               *II =  imb + ilocblk * mb;
            }
         }
      }
      else
      {
/*
 * I refers  to  an entry that is not in the first block, find out which
 * process has it.
 */
         nblocks = ( I -= imb ) / mb + 1;
         *PROW  += nblocks;
         *PROW  -= ( *PROW / NPROW ) * NPROW;
/*
 * Compute  my distance from the source process so that within this pro-
 * cess coordinate system,  the  source process is the process such that
 * mydist=0.
 */
         if( ( mydist  = MYROW - rsrc ) < 0 ) mydist += NPROW;
/*
 * When mydist <  nblocks - ilocblk * NPROW, I own ilocblk+1 full blocks
 * of size MB since I am not the source process, i.e. II=(ilocblk+1)*MB.
 * When mydist>=nblocks-ilocblk*NPROW and I do not own I,  I own ilocblk
 * full blocks of size MB, i.e. II = ilocblk*MB, otherwise I own ilocblk
 * blocks and I,  in which case I can be written as IMB + (nblocks-1)*MB
 * + IL and II = ilocblk*MB + IL = I - IMB + (ilocblk - nblocks + 1)*MB.
 */
         if( nblocks < NPROW )
         {
            mydist -= nblocks;
            *II     = ( ( mydist < 0 ) ? mb :
                        ( ( MYROW == *PROW ) ?
                          I + ( 1 - nblocks ) * mb : 0 ) );
         }
         else
         {
            ilocblk = nblocks / NPROW;
            mydist -= nblocks - ilocblk * NPROW;
            *II     = ( ( mydist < 0 ) ? ( ilocblk + 1 ) * mb :
                        ( ( MYROW == *PROW ) ?
                          ( ilocblk - nblocks + 1 ) * mb + I :
                          ilocblk * mb ) );
         }
      }
   }
/*
 * Idem for the columns
 * 同理适用于列
 */
   inb   = INB;
   *PCOL = CSRC;

   if( ( *PCOL == -1 ) || ( NPCOL == 1 ) )
   {
      *JJ = J;
   }
   else if( J < inb )
   {
      *JJ = ( MYCOL == *PCOL ? J : 0 );
   }
   else
   {
      nb   = NB;
      csrc = *PCOL;

      if( MYCOL == csrc )
      {
         nblocks = ( J - inb ) / nb + 1;
         *PCOL  += nblocks;
         *PCOL  -= ( *PCOL / NPCOL ) * NPCOL;

         if( nblocks < NPCOL )
         {
            *JJ = inb;
         }
         else
         {
            ilocblk = nblocks / NPCOL;
            if( ilocblk * NPCOL >= nblocks )
            {
               *JJ = ( ( MYCOL == *PCOL ) ?
                       J   + ( ilocblk - nblocks ) * nb :
                       inb + ( ilocblk - 1       ) * nb );
            }
            else
            {
               *JJ = inb + ilocblk * nb;
            }
         }
      }
      else
      {
         nblocks = ( J -= inb ) / nb + 1;
         *PCOL  += nblocks;
         *PCOL  -= ( *PCOL / NPCOL ) * NPCOL;

         if( ( mydist = MYCOL - csrc ) < 0 ) mydist += NPCOL;

         if( nblocks < NPCOL )
         {
            mydist -= nblocks;
            *JJ     = ( ( mydist < 0 ) ? nb : ( ( MYCOL == *PCOL ) ?
                        J + ( 1 - nblocks )*nb : 0 ) );
         }
         else
         {
            ilocblk = nblocks / NPCOL;
            mydist -= nblocks - ilocblk * NPCOL;
            *JJ     = ( ( mydist < 0 ) ? ( ilocblk + 1 ) * nb :
                        ( ( MYCOL == *PCOL ) ?
                          ( ilocblk - nblocks + 1 ) * nb + J :
                          ilocblk * nb ) );
         }
      }
   }
/*
 * End of HPL_infog2l
 */
}
