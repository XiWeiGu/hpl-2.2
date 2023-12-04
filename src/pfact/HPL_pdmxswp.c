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
void HPL_pdmxswp
(
   HPL_T_panel *                    PANEL,
   const int                        M,
   const int                        II,
   const int                        JJ,
   double *                         WORK
)
#else
void HPL_pdmxswp
( PANEL, M, II, JJ, WORK )
   HPL_T_panel *                    PANEL;
   const int                        M;
   const int                        II;
   const int                        JJ;
   double *                         WORK;
#endif
{
/*
 * Purpose
 * =======
 *
 * HPL_pdmxswp swaps  and  broadcasts  the  absolute value max row using
 * bi-directional exchange.  The buffer is partially set by HPL_dlocmax.
 *
 * Bi-directional  exchange  is  used  to  perform  the  swap::broadcast
 * operations  at once  for one column in the panel.  This  results in a
 * lower number of slightly larger  messages than usual.  On P processes
 * and assuming bi-directional links,  the running time of this function
 * can be approximated by
 *
 *    log_2( P ) * ( lat + ( 2 * N0 + 4 ) / bdwth )
 * HPL_pdmxswp 通过双向交换交换并广播绝对值最大行。缓冲区部分由 HPL_dlocmax 设置。
 *
 * 双向交换用于为面板中的一列一次执行交换::广播操作。这导致比通常更大的消息数量稍小。
 * 在 P 个进程和假设双向链接的情况下，该函数的运行时间可以近似为
 *
 *    log_2( P ) * ( lat + ( 2 * N0 + 4 ) / bdwth )
 *
 * where  lat and bdwth are the latency and bandwidth of the network for
 * double precision real elements.  Communication  only  occurs  in  one
 * process  column. Mono-directional links  will cause the communication
 * cost to double.
 * 这里，lat和bdwth分别是双精度实数元素的网络延迟和带宽。通信仅在一个进程列中发生。
 * 单向链接将导致通信成本加倍。
 *
 * Arguments
 * =========
 *
 * PANEL   (local input/output)          HPL_T_panel *
 *         On entry,  PANEL  points to the data structure containing the
 *         panel information.
 *
 * M       (local input)                 const int
 *         On entry,  M specifies the local number of rows of the matrix
 *         column on which this function operates.
 *
 * II      (local input)                 const int
 *         On entry, II  specifies the row offset where the column to be
 *         operated on starts with respect to the panel.
 * II（本地输入）               const int
 * 在进入时，II指定相对于面板而言要进行操作的列的起始行偏移。
 *
 * JJ      (local input)                 const int
 *         On entry, JJ  specifies the column offset where the column to
 *         be operated on starts with respect to the panel.
 * JJ（本地输入）               const int
 * 在进入时，JJ指定相对于面板而言要进行操作的列的起始列偏移。
 *
 * WORK    (local workspace)             double *
 *         On entry, WORK  is a workarray of size at least 2 * (4+2*N0).
 *         It  is assumed that  HPL_dlocmax  was called  prior  to  this
 *         routine to  initialize  the first four entries of this array.
 *         On exit, the  N0  length max row is stored in WORK[4:4+N0-1];
 *         Note that this is also the  JJth  row  (or column) of L1. The
 *         remaining part is used as a temporary array.
 * WORK（本地工作空间）       double *
 * 在进入时，WORK是大小至少为2 *（4+2*N0）的工作数组。
 * 假设在调用此例程之前已调用了HPL_dlocmax以初始化此数组的前四个条目。
 * 在退出时，长度为N0的最大行存储在WORK[4:4+N0-1]中；
 * 请注意，这也是L1的第JJ行（或列）。其余部分用作临时数组。
 *
 * ---------------------------------------------------------------------
 */
/*
 * .. Local Variables ..
 */
   double                     gmax, tmp1;
   double                     * A0, * Wmx, * Wwork;
   HPL_T_grid                 * grid;
   MPI_Comm                   comm;
   unsigned int               hdim, ip2, ip2_, ipow, k, mask;
   int                        Np2, cnt_, cnt0, i, icurrow, lda, mydist,
                              mydis_, myrow, n0, nprow, partner, rcnt,
                              root, scnt, size_;
/* ..
 * .. Executable Statements ..
 */
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_MXSWP );
#endif
   grid = PANEL->grid; myrow = grid->myrow; nprow = grid->nprow;
/*
 * ip2  : the smallest power of two less than or equal to nprow;
 * hdim : dimension of the hypercube made of those ip2 processes;
 * Np2  : logical flag indicating whether or not nprow is a power of 2;
 * ip2  ：小于或等于nprow的最小的二的幂；
 * hdim ：由这些ip2进程组成的超立方体的维数；
 * Np2  ：逻辑标志，指示nprow是否是2的幂。
 */
   comm    = grid->col_comm; ip2 = (unsigned int)(grid->row_ip2);
   hdim    = (unsigned int)(grid->row_hdim);     n0  = PANEL->jb;
   icurrow = PANEL->prow; Np2 = (int)( ( size_ = nprow - ip2 ) != 0 );
   mydist  = MModSub( myrow, icurrow, nprow );
/*
 * Set up pointers in workspace:  WORK and Wwork  point to the beginning
 * of the buffers of size 4 + 2*N0 to be combined. Wmx points to the row
 * owning the local (before combine) and global (after combine) absolute
 * value max. A0 points to the copy of the current row of the matrix.
 * 在工作空间中设置指针：WORK和Wwork指向要组合的大小为4 + 2*N0的缓冲区的开头。
 * Wmx指向拥有本地（合并前）和全局（合并后）绝对值最大值的行。
 * A0指向矩阵当前行的副本。
 */
   cnt0  = ( cnt_ = n0 + 4 ) + n0; A0 = ( Wmx = WORK + 4 ) + n0;
   Wwork = WORK + cnt0;
/*
 * Wmx[0:N0-1] := A[ilindx,0:N0-1] where ilindx is  (int)(WORK[1])  (row
 * with max in current column). If I am the current process row, pack in
 * addition the current row of A in A0[0:N0-1].  If I do not own any row
 * of A, then zero out Wmx[0:N0-1].
 * 如果ilindx是(WORK[1])（当前列中具有最大值的行），则Wmx[0:N0-1] := A[ilindx,0:N0-1]。
 * 如果我是当前的进程行，还将A的当前行打包到A0[0:N0-1]中。如果我不拥有A的任何行，则将Wmx[0:N0-1]清零。
 */
   if( M > 0 )
   {
      lda = PANEL->lda;
      HPL_dcopy( n0, Mptr( PANEL->A, II+(int)(WORK[1]), 0, lda ), lda,
                 Wmx, 1 ); /* 将主元所在的列拷贝至Wmx中，由于数据是按列存储，拷贝过程为单个元素逐一拷贝 */
      if( myrow == icurrow )
      { HPL_dcopy( n0, Mptr( PANEL->A, II, 0, lda ), lda, A0, 1 ); } /* 将第一行元素拷贝至A0中，由于数据是按列存储，拷贝过程为单个元素逐一拷贝 */
   }
   else { for( i = 0; i < n0; i++ ) Wmx[i] = HPL_rzero; }
/*
 * Combine the results (bi-directional exchange):  the process coordina-
 * tes are relative to icurrow,  this allows to reduce the communication
 * volume when nprow is not a power of 2.
 *
 * When nprow is not a power of 2:  proc[i-ip2] receives local data from
 * proc[i]  for all i in [ip2..nprow).  In addition,  proc[0]  (icurrow)
 * sends to proc[ip2] the current row of A  for later broadcast in procs
 * [ip2..nprow).
 */
   if( ( Np2 != 0 ) &&
       ( ( partner = (int)((unsigned int)(mydist) ^ ip2 ) ) < nprow ) )
   {
      if( ( mydist & ip2 ) != 0 )
      {
         if( mydist == (int)(ip2) )
            (void) HPL_sdrv( WORK, cnt_, MSGID_BEGIN_PFACT, A0, n0,
                             MSGID_BEGIN_PFACT, MModAdd( partner,
                             icurrow, nprow ), comm );
         else
            (void) HPL_send( WORK, cnt_, MModAdd( partner, icurrow,
                             nprow ), MSGID_BEGIN_PFACT, comm );
      }
      else
      {
         if( mydist == 0 )
            (void) HPL_sdrv( A0, n0, MSGID_BEGIN_PFACT, Wwork, cnt_,
                             MSGID_BEGIN_PFACT, MModAdd( partner,
                             icurrow, nprow ), comm );
         else
            (void) HPL_recv( Wwork, cnt_, MModAdd( partner, icurrow,
                             nprow ), MSGID_BEGIN_PFACT, comm );

         tmp1 = Mabs( Wwork[0] ); gmax = Mabs( WORK[0] );
         if( ( tmp1 > gmax ) ||
             ( ( tmp1 == gmax ) && ( Wwork[3] < WORK[3] ) ) )
         { HPL_dcopy( cnt_, Wwork, 1, WORK, 1 ); }
      }
   }

   if( mydist < (int)(ip2) )
   {
/*
 * power of 2 part of the processes collection: processes  [0..ip2)  are
 * combining (binary exchange); proc[0] has two rows to send, but one to
 * receive.  At every step  k  in [0..hdim) of the algorithm,  a process
 * pair exchanging 2 rows is such that  myrow >> k+1 is 0.  Among  those
 * processes the ones  that are sending one more row than  what they are
 * receiving are such that myrow >> k is equal to 0.
 * power of 2部分的进程集合：进程[0..ip2)正在进行组合（二进制交换）；proc[0]有两行要发送，但只接收一行。
 * 在算法的每一步k中[0..hdim)，交换2行的进程对满足myrow >> k+1是0。
 * 在这些进程中，发送比接收多一行的进程是使myrow >> k等于0的进程。
 */
      k = 0; ipow = 1;

      while( k < hdim )
      {
         if( ( (unsigned int)(mydist) >> ( k + 1 ) ) == 0 )
         {
            if( ( (unsigned int)(mydist) >> k ) == 0 )
            { scnt = cnt0; rcnt = cnt_; }
            else
            { scnt = cnt_; rcnt = cnt0; }
         }
         else { scnt = rcnt = cnt_; }

         partner = (int)( (unsigned int)(mydist) ^ ipow );
         (void) HPL_sdrv( WORK, scnt, MSGID_BEGIN_PFACT, Wwork, rcnt,
                          MSGID_BEGIN_PFACT, MModAdd( partner, icurrow,
                          nprow ), comm );

         tmp1 = Mabs( Wwork[0] ); gmax = Mabs( WORK[0] );
         if( ( tmp1 > gmax ) ||
             ( ( tmp1 == gmax ) && ( Wwork[3] < WORK[3] ) ) )
         {
            HPL_dcopy( ( rcnt == cnt0 ? cnt0 : cnt_ ), Wwork, 1,
                       WORK, 1 );
         }
         else if( rcnt == cnt0 )
         { HPL_dcopy( n0, Wwork+cnt_, 1, A0, 1 ); }

         ipow <<= 1; k++;
      }
   }
   else if( size_ > 1 )
   {
/*
 * proc[ip2] broadcast current row of A to procs [ip2+1..nprow).
 */
      k = (unsigned int)(size_) - 1; ip2_ = mask = 1;
      while( k > 1 ) { k >>= 1; ip2_ <<= 1; mask <<= 1; mask++; }

      root   = MModAdd( icurrow, (int)(ip2), nprow );
      mydis_ = MModSub( myrow,   root,       nprow );

      do
      {
         mask ^= ip2_;
         if( ( mydis_ & mask ) == 0 )
         {
            partner = (int)(mydis_ ^ ip2_);
            if( ( mydis_ & ip2_ ) != 0 )
            {
               (void) HPL_recv( A0, n0, MModAdd( root, partner,
                                nprow ), MSGID_BEGIN_PFACT, comm );
            }
            else if( partner < size_ )
            {
               (void) HPL_send( A0, n0, MModAdd( root, partner,
                                nprow ), MSGID_BEGIN_PFACT, comm );
            }
         }
         ip2_ >>= 1;
      } while( ip2_ > 0 );
   }
/*
 * If nprow is not a power of 2,  for all i in [ip2..nprow), proc[i-ip2]
 * sends the pivot row to proc[i]  along  with the first four entries of
 * the WORK array.
 */
   if( ( Np2 != 0 ) &&
       ( ( partner = (int)((unsigned int)(mydist) ^ ip2 ) ) < nprow ) )
   {
      if( ( mydist & ip2 ) != 0 )
      {
         (void) HPL_recv( WORK, cnt_, MModAdd( partner, icurrow,
                          nprow ), MSGID_BEGIN_PFACT, comm );
      }
      else
      {
         (void) HPL_send( WORK, cnt_, MModAdd( partner, icurrow,
                          nprow ), MSGID_BEGIN_PFACT, comm );
      }
   }
/*
 * Save the global pivot index in pivot array
 * 将全局主元索引保存在主元数组中
 */
   (PANEL->DPIV)[JJ] = WORK[2];
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_MXSWP );
#endif
/*
 * End of HPL_pdmxswp
 */
}
