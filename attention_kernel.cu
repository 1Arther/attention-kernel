#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
/*
Q:       [BH, S, D]
K:       [BH, S, D]
V:       [BH, S, D]
scores:  [BH, S, S]
probs:   [BH, S, S]
O:       [BH, S, D]
*/
__inline__ __device__ float reduce_sum(float val){
    for(int offset=warpSize/2;offset>0;offset>>=1){
        val+=__shfl_down_sync(0xffffffff,val,offset);
    }
    return val;
}

__inline__ __device__ float  reduce_max(float val){
    for(int offset=warpSize/2;offset>0;offset>>=1){
        val=fmaxf(__shfl_down_sync(0xffffffff,val,offset),val);
    }
    return val;
}

//一维grid一维block
__global__ void qk_matmul_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ scores,
    int BH,
    int S,
    int D
) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int total=BH*S*S;
    if(idx>=total) return;
    int k_pos=idx%S;
    int q_pos=idx/S%S;
    int bh=idx/(S*S);
    const float* q_ptr=Q+bh*(S*D)+q_pos*D;
    const float* k_ptr=K+bh*(S*D)+k_pos*D;  // 经过转置才能直接相乘
    float sum=0.0f;
    for(int i=0;i<D;i++){
        sum+=(q_ptr[i]*k_ptr[i]);
    }
    scores[idx]=sum;
}

//三维grid二维block
template<int BLOCK_Q, int BLOCK_K, int BLOCK_D>
__global__ void qk_matmul_tiled_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ scores,
    int BH,
    int S,
    int D
) {
    int ty=threadIdx.y;
    int tx=threadIdx.x;
    int row=blockIdx.y;
    int col=blockIdx.x;

    int q_pos=row*BLOCK_Q+ty;
    int k_pos=col*BLOCK_K+tx;
    int bh=blockIdx.z;

    int idx=bh*(S*S)+q_pos*S+k_pos;

    const float* q_ptr=Q+bh*S*D;
    const float* k_ptr=K+bh*S*D;
    __shared__ float q_smem[BLOCK_Q][BLOCK_D];
    __shared__ float k_smem[BLOCK_D][BLOCK_K];

    float sum=0.0f;
    for(int d=0;d<D;d+=BLOCK_D){
        int q_d=d+tx;
        if(q_pos<S && q_d<D){
            q_smem[ty][tx]=q_ptr[q_pos*D+q_d];
        }else{
            q_smem[ty][tx]=0.0f;
        }
        int k_d=d+ty;
        if(k_pos<S && k_d<D){
            k_smem[ty][tx]=k_ptr[k_pos*D+k_d];
        }else{
            k_smem[ty][tx]=0.0f;
        }
        __syncthreads();
        for(int i=0;i<BLOCK_D;i++){
            sum+=q_smem[ty][i]*k_smem[i][tx];
        }
        __syncthreads();
    }
    if(q_pos<S && k_pos<S){
        scores[idx]=sum;
    }   
}

//一维grid一维block
__global__ void scaled_causal_softmax_warp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N,
    int query_len,
    float scale
){
    int tid=threadIdx.x;
    int row=blockIdx.x;
    if(row>=M) return;
    const float* row_input =input+row*N;
    float* row_output=output+row*N;
    int warpId=tid/warpSize;
    int laneId=tid%warpSize;
    extern __shared__ float smem[];
    int q=row % query_len;
    float max=-FLT_MAX;
    for(int i=tid;i<N;i+=blockDim.x){
        bool valid= i<=q;
        if(valid){
            max=fmaxf(max,row_input[i]*scale);
        }
    }
    max=reduce_max(max);
    if(laneId==0){
        smem[warpId]=max;
    }
    __syncthreads();
    if(warpId==0){
        int num=(blockDim.x+warpSize-1)/warpSize;
        max=(laneId<num)?smem[laneId]:-FLT_MAX;
        max=reduce_max(max);
    }
    if(tid==0){
        smem[0]=max;
    }
    __syncthreads();
    max=smem[0];
    float sum=0.0f;
    for(int i=tid;i<N;i+=blockDim.x){
        bool valid= i<=q;
        if(!valid){
            row_output[i]=0;
        }else{
            float e=expf(row_input[i]*scale-max);
            row_output[i]=e;
            sum+=e;
        }
    }
    sum=reduce_sum(sum);
    if(laneId==0){
        smem[warpId]=sum;
    }
    __syncthreads();
    if(warpId==0){
        int num=(blockDim.x+warpSize-1)/warpSize;
        sum=(laneId<num)?smem[laneId]:0.0f;
        sum=reduce_sum(sum);
    }
    if(tid==0){
        smem[0]=sum;
    }
    __syncthreads();
    sum=smem[0];
    float inv_sum=sum>0.0f?1.0f/sum:0.0f;
    for(int i=tid;i<N;i+=blockDim.x){
        row_output[i]=row_output[i]*inv_sum;
    }
}

//一维grid一维block
__global__ void pv_matmul_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ V,
    float* __restrict__ O,
    int BH,
    int S,
    int D
) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int total=BH*S*D;
    if(idx>=total) return;
    int v_pos=idx%D;
    int probs_pos=idx/D%S;
    int bh=idx/(S*D);
    const float* probs_ptr=probs+bh*(S*S)+probs_pos*S;
    const float* v_ptr=V+bh*(S*D)+v_pos;//非转置
    float sum=0.0f;
    for(int i=0;i<S;i++){
        sum+=probs_ptr[i]*v_ptr[i*D];
    }
    O[idx]=sum;
}

//三维grid二维block
template<int BLOCK_P,int BLOCK_V,int BLOCK_S>
__global__ void pv_matmul_tiled_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ V,
    float* __restrict__ O,
    int BH,
    int S,
    int D
) {
   int tx=threadIdx.x;
   int ty=threadIdx.y;
   int col=blockIdx.x;
   int row=blockIdx.y;
   int bh=blockIdx.z;

   int probs_pos=row*BLOCK_P+ty;
   int v_pos=col*BLOCK_V+tx;

   int idx=bh*(S*D)+probs_pos*D+v_pos;

   const float* probs_ptr=probs+bh*(S*S);
   const float* v_ptr=V+bh*(S*D);

   __shared__ float probs_smem[BLOCK_P][BLOCK_S];
   __shared__ float v_smem[BLOCK_S][BLOCK_V];
   float sum=0.0f;
   for(int r=0;r<S;r+=BLOCK_S){
        int probs_s=r+tx;
        if(probs_pos<S && probs_s<S){
            probs_smem[ty][tx]=probs_ptr[probs_pos*S+probs_s];
        }else{
            probs_smem[ty][tx]=0.0f;
        }
        int v_d=r+ty;
        if(v_pos<D && v_d<S){
            v_smem[ty][tx]=v_ptr[v_d*D+v_pos];
        }else{
            v_smem[ty][tx]=0.0f;
        }
        __syncthreads();
        for(int k=0;k<BLOCK_S;k++){
            sum+=probs_smem[ty][k]*v_smem[k][tx];
        }
        __syncthreads();
   }
   if(probs_pos<S && v_pos<D){
        O[idx]=sum;
   }
}

//一维[BH,S,D] -> [BH*S,D] 一维grid一维block
__global__ void fused_causal_attention_row_kernel(
    const float* __restrict__ Q,  //query
    const float* __restrict__ K,  //key
    const float* __restrict__ V,  //value
    float* __restrict__ O,
    int BH,
    int S,
    int D,
    float scale
) {
    int tid=threadIdx.x;
    int row=blockIdx.x;
    if (row >= BH * S) return;
    int q_pos=row%S;
    int bh=row/S;

    //全是[BH,S,D]
    const float* q_ptr=Q+bh*S*D+q_pos*D;  //Q的某一行起点
    const float* k_base=K+bh*S*D;   //K的某一个bh矩阵起点
    const float* v_base=V+bh*S*D;   //V的某一个bh矩阵起点
    float* o_ptr=O+bh*S*D+q_pos*D;  //O的某一行起点

    extern __shared__ float score_smem[];

    float local_max=-FLT_MAX;
    //QKT [BH,S,D]*[BH,S,D]T=[BH,S,S]
    for(int k=tid;k<=q_pos;k+=blockDim.x){  //一个block负责一行，一个thread负责若干个元素
        const float* k_ptr=k_base+k*D;//k的某一行起点
        float dot=0.0f;
        for(int d=0;d<D;d++){
            dot+=q_ptr[d]*k_ptr[d];
        }

        float score=dot*scale;

        score_smem[k]=score;

        local_max=fmaxf(score,local_max);
    }
    __syncthreads();

    int warpNum=(blockDim.x+31)/32;

    int warpId=tid/warpSize;
    int laneId=tid%warpSize;
    __shared__ float max_smem[32];
    local_max=reduce_max(local_max);
    if(laneId==0){
        max_smem[warpId]=local_max;
    }
    __syncthreads();
    if(warpId==0){
        local_max=(laneId<warpNum)?max_smem[laneId]:-FLT_MAX;
        local_max=reduce_max(local_max);
    }
    if(tid==0){
        max_smem[0]=local_max;
    }
    __syncthreads();
    local_max=max_smem[0];

    float local_sum=0.0f;
    for(int k=tid;k<=q_pos;k+=blockDim.x){
        float e=score_smem[k];
        e=expf(e-local_max);
        local_sum+=e;
        score_smem[k]=e;
    }
    local_sum=reduce_sum(local_sum);
    __shared__ float sum_smem[32];
    if(laneId==0){
        sum_smem[warpId]=local_sum;
    }
    __syncthreads();
    if(warpId==0){
        local_sum=laneId<warpNum?sum_smem[laneId]:0.0f;
        local_sum=reduce_sum(local_sum);
    }
    if(tid==0){
        sum_smem[0]=local_sum;
    }
    __syncthreads();
    local_sum=sum_smem[0];
    float inv_sum=local_sum>0.0f?1/local_sum:0.0f;
    for(int k=tid;k<=q_pos;k+=blockDim.x){
        score_smem[k]=score_smem[k]*inv_sum;
    }
    __syncthreads();
    //pv [BH,S,S]*[BH,S,D]->[BH,S,D]
    for(int d=tid;d<D;d+=blockDim.x){
        float acc=0.0f;
        for(int k=0;k<=q_pos;k++){
            acc+=(score_smem[k]*v_base[k*D+d]);
        }
        o_ptr[d]=acc;
    }
}

//二维grid一维block
//一个BLOCK负责BLOCK_M个query，每轮处理BLOCK_N个key和value
template<int BLOCK_M, int BLOCK_N, int MAX_D>
__global__ void flash_attention_v1_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int BH,
    int S,
    int D,
    float scale
) {
    int tid = threadIdx.x;

    int q_block = blockIdx.x;
    int bh = blockIdx.y;

    int q_start = q_block * BLOCK_M;

    //行起始
    const float* q_base = Q + bh * S * D;
    const float* k_base = K + bh * S * D;
    const float* v_base = V + bh * S * D;
    float* o_base = O + bh * S * D;

    __shared__ float q_smem[BLOCK_M][MAX_D];
    __shared__ float k_smem[BLOCK_N][MAX_D];
    __shared__ float v_smem[BLOCK_N][MAX_D];

    __shared__ float score_smem[BLOCK_M][BLOCK_N];

    __shared__ float acc_smem[BLOCK_M][MAX_D];

    __shared__ float m_smem[BLOCK_M];
    __shared__ float l_smem[BLOCK_M];//softmax分母

    __shared__ float m_new_smem[BLOCK_M];
    __shared__ float l_new_smem[BLOCK_M];

    // ============================================================
    // 1. Load Q tile and initialize acc/m/l
    // ============================================================

    for (int idx = tid; idx < BLOCK_M * D; idx += blockDim.x) {
        int qi = idx / D;
        int d = idx % D;

        int q = q_start + qi;

        if (q < S) {
            q_smem[qi][d] = q_base[q * D + d];
        } else {
            q_smem[qi][d] = 0.0f;
        }

        acc_smem[qi][d] = 0.0f;
    }

    if (tid < BLOCK_M) {
        m_smem[tid] = -FLT_MAX;
        l_smem[tid] = 0.0f;
        m_new_smem[tid] = -FLT_MAX;
        l_new_smem[tid] = 0.0f;
    }

    __syncthreads();

    // ============================================================
    // 2. Loop over K/V tiles
    // ============================================================

    for (int k_start = 0; k_start < S; k_start += BLOCK_N) {
        // ------------------------------------------------------------
        // Load K/V tile: [BLOCK_N, D]
        // ------------------------------------------------------------

        for (int idx = tid; idx < BLOCK_N * D; idx += blockDim.x) {
            int kj = idx / D;
            int d = idx % D;

            int k = k_start + kj;

            if (k < S) {
                k_smem[kj][d] = k_base[k * D + d];
                v_smem[kj][d] = v_base[k * D + d];
            } else {
                k_smem[kj][d] = 0.0f;
                v_smem[kj][d] = 0.0f;
            }
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Compute score tile: [BLOCK_M, BLOCK_N]
        //
        // score[qi, kj] = Q[q] · K[k] * scale
        // causal valid iff k <= q
        // ------------------------------------------------------------

        for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
            int qi = idx / BLOCK_N;
            int kj = idx % BLOCK_N;

            //仅仅用来判断是否越界
            int q = q_start + qi;
            int k = k_start + kj;

            float score = -FLT_MAX;

            if (q < S && k < S && k <= q) {
                float dot = 0.0f;

                for (int d = 0; d < D; ++d) {
                    dot += q_smem[qi][d] * k_smem[kj][d];
                }

                score = dot * scale;
            }

            score_smem[qi][kj] = score;
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Online softmax update: compute m_new and l_new
        //
        // m_new = max(m_old, max(score_tile))
        // l_new = exp(m_old - m_new) * l_old
        //       + sum(exp(score_tile - m_new))
        // ------------------------------------------------------------

        if (tid < BLOCK_M) {
            int qi = tid;
            int q = q_start + qi;

            float m_old = m_smem[qi];
            float l_old = l_smem[qi];

            float tile_max = -FLT_MAX;

            if (q < S) {
                for (int kj = 0; kj < BLOCK_N; ++kj) {
                    tile_max = fmaxf(tile_max, score_smem[qi][kj]);
                }
            }

            float m_new = fmaxf(m_old, tile_max);

            float alpha = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);

            float tile_sum = 0.0f;

            if (q < S) {//这一批已经得到最大值，就不需要缩放
                for (int kj = 0; kj < BLOCK_N; ++kj) {
                    float s = score_smem[qi][kj];

                    // invalid score is -FLT_MAX, contribution ~= 0
                    tile_sum += expf(s - m_new);
                }
            }

            float l_new = alpha * l_old + tile_sum;

            m_new_smem[qi] = m_new;
            l_new_smem[qi] = l_new;
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Update acc:
        //
        // acc_new =
        //     exp(m_old - m_new) * acc_old
        //   + exp(score_tile - m_new) @ V_tile
        // ------------------------------------------------------------

        for (int idx = tid; idx < BLOCK_M * D; idx += blockDim.x) {
            int qi = idx / D;
            int d = idx % D;

            int q = q_start + qi;

            if (q < S) {
                float m_old = m_smem[qi];
                float m_new = m_new_smem[qi];

                float alpha = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);

                float acc = acc_smem[qi][d] * alpha;//缩放QKT

                for (int kj = 0; kj < BLOCK_N; ++kj) {  //中间维度
                    float s = score_smem[qi][kj];
                    float p = expf(s - m_new);

                    acc += p * v_smem[kj][d];
                }

                acc_smem[qi][d] = acc;
            }
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Commit m/l for next K/V tile
        // ------------------------------------------------------------

        if (tid < BLOCK_M) {
            m_smem[tid] = m_new_smem[tid];
            l_smem[tid] = l_new_smem[tid];
        }

        __syncthreads();
    }

    // ============================================================
    // 3. Write output: O = acc / l
    // ============================================================

    for (int idx = tid; idx < BLOCK_M * D; idx += blockDim.x) {
        int qi = idx / D;
        int d = idx % D;

        int q = q_start + qi;

        if (q < S) {
            float denom = l_smem[qi];

            float out = denom > 0.0f ? acc_smem[qi][d] / denom : 0.0f;

            o_base[q * D + d] = out;
        }
    }
}

//二维grid一维block
//一个BLOCK负责BLOCK_M个query，每轮处理BLOCK_N个key和value
template<int BLOCK_M, int BLOCK_N, int MAX_D>
__global__ void flash_attention_causal_tile_skipping_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int BH,
    int S,
    int D,
    float scale
) {
    int tid = threadIdx.x;

    int q_block = blockIdx.x;
    int bh = blockIdx.y;

    int q_start = q_block * BLOCK_M;

    int q_end = q_start + BLOCK_M - 1;
    if (q_end >= S) {
        q_end = S - 1;
    }

    //行起始
    const float* q_base = Q + bh * S * D;
    const float* k_base = K + bh * S * D;
    const float* v_base = V + bh * S * D;
    float* o_base = O + bh * S * D;

    __shared__ float q_smem[BLOCK_M][MAX_D];
    __shared__ float k_smem[BLOCK_N][MAX_D];
    __shared__ float v_smem[BLOCK_N][MAX_D];

    __shared__ float score_smem[BLOCK_M][BLOCK_N];

    __shared__ float acc_smem[BLOCK_M][MAX_D];

    __shared__ float m_smem[BLOCK_M];
    __shared__ float l_smem[BLOCK_M];//softmax分母

    __shared__ float m_new_smem[BLOCK_M];
    __shared__ float l_new_smem[BLOCK_M];

    // ============================================================
    // 1. Load Q tile and initialize acc/m/l
    // ============================================================

    for (int idx = tid; idx < BLOCK_M * D; idx += blockDim.x) {
        int qi = idx / D;
        int d = idx % D;

        int q = q_start + qi;

        if (q < S) {
            q_smem[qi][d] = q_base[q * D + d];
        } else {
            q_smem[qi][d] = 0.0f;
        }

        acc_smem[qi][d] = 0.0f;
    }

    if (tid < BLOCK_M) {
        m_smem[tid] = -FLT_MAX;
        l_smem[tid] = 0.0f;
        m_new_smem[tid] = -FLT_MAX;
        l_new_smem[tid] = 0.0f;
    }

    __syncthreads();

    // ============================================================
    // 2. Loop over K/V tiles
    // ============================================================

    for (int k_start = 0; k_start < S; k_start += BLOCK_N) {
        // ------------------------------------------------------------
        // Load K/V tile: [BLOCK_N, D]
        // ------------------------------------------------------------
        //不可以看未来token
        if (k_start > q_end) {
            break;
        }

        for (int idx = tid; idx < BLOCK_N * D; idx += blockDim.x) {
            int kj = idx / D;
            int d = idx % D;

            int k = k_start + kj;

            if (k < S) {
                k_smem[kj][d] = k_base[k * D + d];
                v_smem[kj][d] = v_base[k * D + d];
            } else {
                k_smem[kj][d] = 0.0f;
                v_smem[kj][d] = 0.0f;
            }
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Compute score tile: [BLOCK_M, BLOCK_N]
        //
        // score[qi, kj] = Q[q] · K[k] * scale
        // causal valid iff k <= q
        // ------------------------------------------------------------

        for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
            int qi = idx / BLOCK_N;
            int kj = idx % BLOCK_N;

            //仅仅用来判断是否越界
            int q = q_start + qi;
            int k = k_start + kj;

            float score = -FLT_MAX;

            if (q < S && k < S && k <= q) {
                float dot = 0.0f;

                for (int d = 0; d < D; ++d) {
                    dot += q_smem[qi][d] * k_smem[kj][d];
                }

                score = dot * scale;
            }

            score_smem[qi][kj] = score;
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Online softmax update: compute m_new and l_new
        //
        // m_new = max(m_old, max(score_tile))
        // l_new = exp(m_old - m_new) * l_old
        //       + sum(exp(score_tile - m_new))
        // ------------------------------------------------------------

        if (tid < BLOCK_M) {
            int qi = tid;
            int q = q_start + qi;

            float m_old = m_smem[qi];
            float l_old = l_smem[qi];

            float tile_max = -FLT_MAX;

            if (q < S) {
                for (int kj = 0; kj < BLOCK_N; ++kj) {
                    tile_max = fmaxf(tile_max, score_smem[qi][kj]);
                }
            }

            float m_new = fmaxf(m_old, tile_max);

            float alpha = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);

            float tile_sum = 0.0f;

            if (q < S) {//这一批已经得到最大值，就不需要缩放
                for (int kj = 0; kj < BLOCK_N; ++kj) {
                    float s = score_smem[qi][kj];

                    // invalid score is -FLT_MAX, contribution ~= 0
                    tile_sum += expf(s - m_new);
                }
            }

            float l_new = alpha * l_old + tile_sum;

            m_new_smem[qi] = m_new;
            l_new_smem[qi] = l_new;
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Update acc:
        //
        // acc_new =
        //     exp(m_old - m_new) * acc_old
        //   + exp(score_tile - m_new) @ V_tile
        // ------------------------------------------------------------

        for (int idx = tid; idx < BLOCK_M * D; idx += blockDim.x) {
            int qi = idx / D;
            int d = idx % D;

            int q = q_start + qi;

            if (q < S) {
                float m_old = m_smem[qi];
                float m_new = m_new_smem[qi];

                float alpha = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);

                float acc = acc_smem[qi][d] * alpha;//缩放QKT

                for (int kj = 0; kj < BLOCK_N; ++kj) {  //中间维度
                    float s = score_smem[qi][kj];
                    float p = expf(s - m_new);

                    acc += p * v_smem[kj][d];
                }

                acc_smem[qi][d] = acc;
            }
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Commit m/l for next K/V tile
        // ------------------------------------------------------------

        if (tid < BLOCK_M) {
            m_smem[tid] = m_new_smem[tid];
            l_smem[tid] = l_new_smem[tid];
        }

        __syncthreads();
    }

    // ============================================================
    // 3. Write output: O = acc / l
    // ============================================================

    for (int idx = tid; idx < BLOCK_M * D; idx += blockDim.x) {
        int qi = idx / D;
        int d = idx % D;

        int q = q_start + qi;

        if (q < S) {
            float denom = l_smem[qi];

            float out = denom > 0.0f ? acc_smem[qi][d] / denom : 0.0f;

            o_base[q * D + d] = out;
        }
    }
}

//用寄存器替代共享内存
template<int BLOCK_M, int BLOCK_N, int MAX_D>
__global__ void flash_attention_causal_tile_skipping_regacc_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int BH,
    int S,
    int D,
    float scale
) {
    int tid = threadIdx.x;

    int q_block = blockIdx.x;
    int bh = blockIdx.y;

    int q_start = q_block * BLOCK_M;

    int q_end = q_start + BLOCK_M - 1;
    if (q_end >= S) {
        q_end = S - 1;
    }

    //行起始
    const float* q_base = Q + bh * S * D;
    const float* k_base = K + bh * S * D;
    const float* v_base = V + bh * S * D;
    float* o_base = O + bh * S * D;

    __shared__ float q_smem[BLOCK_M][MAX_D];
    __shared__ float k_smem[BLOCK_N][MAX_D];
    __shared__ float v_smem[BLOCK_N][MAX_D];
    
    __shared__ float score_smem[BLOCK_M][BLOCK_N];

    //用寄存器替代共享内存，省去了acc_smem
    constexpr int THREADS = 128;
    constexpr int ITEMS_PER_THREAD =
        (BLOCK_M * MAX_D + THREADS - 1) / THREADS;

    float acc_reg[ITEMS_PER_THREAD];

    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        acc_reg[item] = 0.0f;
    }

    __shared__ float m_smem[BLOCK_M];
    __shared__ float l_smem[BLOCK_M];//softmax分母

    __shared__ float m_new_smem[BLOCK_M];
    __shared__ float l_new_smem[BLOCK_M];

    // ============================================================
    // 1. Load Q tile and initialize acc/m/l
    // ============================================================

    for (int idx = tid; idx < BLOCK_M * D; idx += blockDim.x) {
        int qi = idx / D;
        int d = idx % D;

        int q = q_start + qi;

        if (q < S) {
            q_smem[qi][d] = q_base[q * D + d];
        } else {
            q_smem[qi][d] = 0.0f;
        }

        //acc_smem[qi][d] = 0.0f;
    }

    if (tid < BLOCK_M) {
        m_smem[tid] = -FLT_MAX;
        l_smem[tid] = 0.0f;
        m_new_smem[tid] = -FLT_MAX;
        l_new_smem[tid] = 0.0f;
    }

    __syncthreads();

    // ============================================================
    // 2. Loop over K/V tiles
    // ============================================================

    for (int k_start = 0; k_start < S; k_start += BLOCK_N) {
        // ------------------------------------------------------------
        // Load K/V tile: [BLOCK_N, D]
        // ------------------------------------------------------------
        //不可以看未来token
        if (k_start > q_end) {
            break;
        }

        for (int idx = tid; idx < BLOCK_N * D; idx += blockDim.x) {
            int kj = idx / D;
            int d = idx % D;

            int k = k_start + kj;

            if (k < S) {
                k_smem[kj][d] = k_base[k * D + d];
                v_smem[kj][d] = v_base[k * D + d];
            } else {
                k_smem[kj][d] = 0.0f;
                v_smem[kj][d] = 0.0f;
            }
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Compute score tile: [BLOCK_M, BLOCK_N]
        //
        // score[qi, kj] = Q[q] · K[k] * scale
        // causal valid iff k <= q
        // ------------------------------------------------------------

        for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
            int qi = idx / BLOCK_N;
            int kj = idx % BLOCK_N;

            //仅仅用来判断是否越界
            int q = q_start + qi;
            int k = k_start + kj;

            float score = -FLT_MAX;

            if (q < S && k < S && k <= q) {
                float dot = 0.0f;

                for (int d = 0; d < D; ++d) {
                    dot += q_smem[qi][d] * k_smem[kj][d];
                }

                score = dot * scale;
            }

            score_smem[qi][kj] = score;
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Online softmax update: compute m_new and l_new
        //
        // m_new = max(m_old, max(score_tile))
        // l_new = exp(m_old - m_new) * l_old
        //       + sum(exp(score_tile - m_new))
        // ------------------------------------------------------------

        if (tid < BLOCK_M) {
            int qi = tid;
            int q = q_start + qi;

            float m_old = m_smem[qi];
            float l_old = l_smem[qi];

            float tile_max = -FLT_MAX;

            if (q < S) {
                for (int kj = 0; kj < BLOCK_N; ++kj) {
                    tile_max = fmaxf(tile_max, score_smem[qi][kj]);
                }
            }

            float m_new = fmaxf(m_old, tile_max);

            float alpha = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);

            float tile_sum = 0.0f;

            if (q < S) {//这一批已经得到最大值，就不需要缩放
                for (int kj = 0; kj < BLOCK_N; ++kj) {
                    float s = score_smem[qi][kj];

                    // invalid score is -FLT_MAX, contribution ~= 0
                    tile_sum += expf(s - m_new);
                }
            }

            float l_new = alpha * l_old + tile_sum;

            m_new_smem[qi] = m_new;
            l_new_smem[qi] = l_new;
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Update acc:
        //
        // acc_new =
        //     exp(m_old - m_new) * acc_old
        //   + exp(score_tile - m_new) @ V_tile
        // ------------------------------------------------------------

        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            int idx = tid + item * THREADS;

            if (idx < BLOCK_M * D) {
                int qi = idx / D;
                int d = idx % D;

                int q = q_start + qi;

                if (q < S) {
                    float m_old = m_smem[qi];
                    float m_new = m_new_smem[qi];

                    float alpha = (m_old == -FLT_MAX)
                                    ? 0.0f
                                    : expf(m_old - m_new);

                    float acc = acc_reg[item] * alpha;

                    for (int kj = 0; kj < BLOCK_N; ++kj) {
                        float s = score_smem[qi][kj];

                        if (s != -FLT_MAX) {
                            float p = expf(s - m_new);
                            acc += p * v_smem[kj][d];
                        }
                    }

                    acc_reg[item] = acc;
                }
            }
        }
        __syncthreads();

        // ------------------------------------------------------------
        // Commit m/l for next K/V tile
        // ------------------------------------------------------------

        if (tid < BLOCK_M) {
            m_smem[tid] = m_new_smem[tid];
            l_smem[tid] = l_new_smem[tid];
        }

        __syncthreads();
    }

    // ============================================================
    // 3. Write output: O = acc / l
    // ============================================================

    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int idx = tid + item * THREADS;

        if (idx < BLOCK_M * D) {
            int qi = idx / D;
            int d = idx % D;

            int q = q_start + qi;

            if (q < S) {
                float denom = l_smem[qi];

                o_base[q * D + d] =
                    denom > 0.0f ? acc_reg[item] / denom : 0.0f;
            }
        }
    }
}

//float4向量化加载
template<int BLOCK_M, int BLOCK_N, int MAX_D>
__global__ void flash_attention_causal_tile_skipping_regacc_vec4_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int BH,
    int S,
    int D,
    float scale
) {
    int tid = threadIdx.x;

    int q_block = blockIdx.x;
    int bh = blockIdx.y;

    int q_start = q_block * BLOCK_M;

    int q_end = q_start + BLOCK_M - 1;
    if (q_end >= S) {
        q_end = S - 1;
    }

    //行起始
    const float* q_base = Q + bh * S * D;
    const float* k_base = K + bh * S * D;
    const float* v_base = V + bh * S * D;
    float* o_base = O + bh * S * D;

    __shared__ float q_smem[BLOCK_M][MAX_D];
    __shared__ float k_smem[BLOCK_N][MAX_D];
    __shared__ float v_smem[BLOCK_N][MAX_D];
    
    __shared__ float score_smem[BLOCK_M][BLOCK_N];

    //用寄存器替代共享内存，省去了acc_smem
    constexpr int THREADS = 128;
    constexpr int ITEMS_PER_THREAD =
        (BLOCK_M * MAX_D + THREADS - 1) / THREADS;

    float acc_reg[ITEMS_PER_THREAD];

    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        acc_reg[item] = 0.0f;
    }

    __shared__ float m_smem[BLOCK_M];
    __shared__ float l_smem[BLOCK_M];//softmax分母

    __shared__ float m_new_smem[BLOCK_M];
    __shared__ float l_new_smem[BLOCK_M];

    // ============================================================
    // 1. Load Q tile and initialize acc/m/l
    // ============================================================

    if ((D & 3) == 0) {  //向量化读取
        int D4 = D / 4;

        for (int idx4 = tid; idx4 < BLOCK_M * D4; idx4 += blockDim.x) {
            int qi = idx4 / D4;
            int d4 = idx4 % D4;
            int d = d4 * 4;

            int q = q_start + qi;

            float4 q_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            if (q < S) {
                const float* q_gmem = q_base + q * D + d;
                q_vec = *reinterpret_cast<const float4*>(q_gmem);
            }

            q_smem[qi][d + 0] = q_vec.x;
            q_smem[qi][d + 1] = q_vec.y;
            q_smem[qi][d + 2] = q_vec.z;
            q_smem[qi][d + 3] = q_vec.w;
        }
    } else {
        for (int idx = tid; idx < BLOCK_M * D; idx += blockDim.x) {
            int qi = idx / D;
            int d = idx % D;

            int q = q_start + qi;

            if (q < S) {
                q_smem[qi][d] = q_base[q * D + d];
            } else {
                q_smem[qi][d] = 0.0f;
            }
        }
    }

    if (tid < BLOCK_M) {
        m_smem[tid] = -FLT_MAX;
        l_smem[tid] = 0.0f;
        m_new_smem[tid] = -FLT_MAX;
        l_new_smem[tid] = 0.0f;
    }

    __syncthreads();
    // ============================================================
    // 2. Loop over K/V tiles
    // ============================================================

    for (int k_start = 0; k_start < S; k_start += BLOCK_N) {
        // ------------------------------------------------------------
        // Load K/V tile: [BLOCK_N, D]
        // ------------------------------------------------------------
        //不可以看未来token
        if (k_start > q_end) {
            break;
        }

        if ((D & 3) == 0) {  //向量化读取
            int D4 = D / 4;

            for (int idx4 = tid; idx4 < BLOCK_N * D4; idx4 += blockDim.x) {
                int kj = idx4 / D4;
                int d4 = idx4 % D4;
                int d = d4 * 4;

                int k = k_start + kj;

                float4 k_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 v_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                if (k < S) {
                    const float* k_gmem = k_base + k * D + d;
                    const float* v_gmem = v_base + k * D + d;

                    k_vec = *reinterpret_cast<const float4*>(k_gmem);
                    v_vec = *reinterpret_cast<const float4*>(v_gmem);
                }

                k_smem[kj][d + 0] = k_vec.x;
                k_smem[kj][d + 1] = k_vec.y;
                k_smem[kj][d + 2] = k_vec.z;
                k_smem[kj][d + 3] = k_vec.w;

                v_smem[kj][d + 0] = v_vec.x;
                v_smem[kj][d + 1] = v_vec.y;
                v_smem[kj][d + 2] = v_vec.z;
                v_smem[kj][d + 3] = v_vec.w;
            }
        } else {
            for (int idx = tid; idx < BLOCK_N * D; idx += blockDim.x) {
                int kj = idx / D;
                int d = idx % D;

                int k = k_start + kj;

                if (k < S) {
                    k_smem[kj][d] = k_base[k * D + d];
                    v_smem[kj][d] = v_base[k * D + d];
                } else {
                    k_smem[kj][d] = 0.0f;
                    v_smem[kj][d] = 0.0f;
                }
            }
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Compute score tile: [BLOCK_M, BLOCK_N]
        //
        // score[qi, kj] = Q[q] · K[k] * scale
        // causal valid iff k <= q
        // ------------------------------------------------------------

        for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
            int qi = idx / BLOCK_N;
            int kj = idx % BLOCK_N;

            //仅仅用来判断是否越界
            int q = q_start + qi;
            int k = k_start + kj;

            float score = -FLT_MAX;

            if (q < S && k < S && k <= q) {
                float dot = 0.0f;

                for (int d = 0; d < D; ++d) {
                    dot += q_smem[qi][d] * k_smem[kj][d];
                }

                score = dot * scale;
            }

            score_smem[qi][kj] = score;
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Online softmax update: compute m_new and l_new
        //
        // m_new = max(m_old, max(score_tile))
        // l_new = exp(m_old - m_new) * l_old
        //       + sum(exp(score_tile - m_new))
        // ------------------------------------------------------------

        if (tid < BLOCK_M) {
            int qi = tid;
            int q = q_start + qi;

            float m_old = m_smem[qi];
            float l_old = l_smem[qi];

            float tile_max = -FLT_MAX;

            if (q < S) {
                for (int kj = 0; kj < BLOCK_N; ++kj) {
                    tile_max = fmaxf(tile_max, score_smem[qi][kj]);
                }
            }

            float m_new = fmaxf(m_old, tile_max);

            float alpha = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);

            float tile_sum = 0.0f;

            if (q < S) {//这一批已经得到最大值，就不需要缩放
                for (int kj = 0; kj < BLOCK_N; ++kj) {
                    float s = score_smem[qi][kj];

                    // invalid score is -FLT_MAX, contribution ~= 0
                    tile_sum += expf(s - m_new);
                }
            }

            float l_new = alpha * l_old + tile_sum;

            m_new_smem[qi] = m_new;
            l_new_smem[qi] = l_new;
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Update acc:
        //
        // acc_new =
        //     exp(m_old - m_new) * acc_old
        //   + exp(score_tile - m_new) @ V_tile
        // ------------------------------------------------------------

        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            int idx = tid + item * THREADS;

            if (idx < BLOCK_M * D) {
                int qi = idx / D;
                int d = idx % D;

                int q = q_start + qi;

                if (q < S) {
                    float m_old = m_smem[qi];
                    float m_new = m_new_smem[qi];

                    float alpha = (m_old == -FLT_MAX)
                                    ? 0.0f
                                    : expf(m_old - m_new);

                    float acc = acc_reg[item] * alpha;

                    for (int kj = 0; kj < BLOCK_N; ++kj) {
                        float s = score_smem[qi][kj];

                        if (s != -FLT_MAX) {
                            float p = expf(s - m_new);
                            acc += p * v_smem[kj][d];
                        }
                    }

                    acc_reg[item] = acc;
                }
            }
        }
        __syncthreads();

        // ------------------------------------------------------------
        // Commit m/l for next K/V tile
        // ------------------------------------------------------------

        if (tid < BLOCK_M) {
            m_smem[tid] = m_new_smem[tid];
            l_smem[tid] = l_new_smem[tid];
        }

        __syncthreads();
    }

    // ============================================================
    // 3. Write output: O = acc / l
    // ============================================================

    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int idx = tid + item * THREADS;

        if (idx < BLOCK_M * D) {
            int qi = idx / D;
            int d = idx % D;

            int q = q_start + qi;

            if (q < S) {
                float denom = l_smem[qi];

                o_base[q * D + d] =
                    denom > 0.0f ? acc_reg[item] / denom : 0.0f;
            }
        }
    }
}


void launch_qk_matmul(
    const float* d_Q,
    const float* d_K,
    float* d_scores,
    int BH,
    int S,
    int D
) {
    int total = BH * S * S;
    int block = 256;
    int grid = (total + block - 1) / block;

    qk_matmul_kernel<<<grid, block>>>(
        d_Q,
        d_K,
        d_scores,
        BH,
        S,
        D
    );
}

void launch_pv_matmul(
    const float* d_probs,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    int total = BH * S * D;
    int block = 256;
    int grid = (total + block - 1) / block;

    pv_matmul_kernel<<<grid, block>>>(
        d_probs,
        d_V,
        d_O,
        BH,
        S,
        D
    );
}

void launch_scaled_causal_softmax(
    const float* d_input,
    float* d_output,
    int M,
    int N,
    int query_len,
    float scale,
    int block_size
) {
    dim3 grid(M);

    int num_warps = (block_size + 31) / 32;
    size_t smem = num_warps * sizeof(float);

    scaled_causal_softmax_warp_kernel<<<grid, block_size, smem>>>(
        d_input,
        d_output,
        M,
        N,
        query_len,
        scale
    );
}

void launch_attention_forward(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_scores,
    float* d_probs,
    float* d_O,
    int BH,
    int S,
    int D
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // 1. scores = Q @ K^T
    launch_qk_matmul(
        d_Q,
        d_K,
        d_scores,
        BH,
        S,
        D
    );

    // 2. probs = causal_softmax(scores * scale)
    launch_scaled_causal_softmax(
        d_scores,
        d_probs,
        BH * S,
        S,
        S,
        scale,
        256
    );

    // 3. O = probs @ V
    launch_pv_matmul(
        d_probs,
        d_V,
        d_O,
        BH,
        S,
        D
    );
}

void attention_cpu_reference(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::vector<float>& O,
    int BH,
    int S,
    int D
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> scores(static_cast<size_t>(BH) * S * S);
    std::vector<float> probs(static_cast<size_t>(BH) * S * S);

    // 1. scores = Q @ K^T
    for (int bh = 0; bh < BH; ++bh) {
        for (int q = 0; q < S; ++q) {
            for (int k = 0; k < S; ++k) {
                float sum = 0.0f;

                for (int d = 0; d < D; ++d) {
                    float qv = Q[bh * S * D + q * D + d];
                    float kv = K[bh * S * D + k * D + d];
                    sum += qv * kv;
                }

                scores[bh * S * S + q * S + k] = sum;
            }
        }
    }

    // 2. causal softmax
    for (int bh = 0; bh < BH; ++bh) {
        for (int q = 0; q < S; ++q) {
            float max_val = -FLT_MAX;

            for (int k = 0; k < S; ++k) {
                if (k <= q) {
                    float v = scores[bh * S * S + q * S + k] * scale;
                    max_val = std::max(max_val, v);
                }
            }

            float sum_val = 0.0f;

            for (int k = 0; k < S; ++k) {
                float e = 0.0f;

                if (k <= q) {
                    float v = scores[bh * S * S + q * S + k] * scale;
                    e = std::exp(v - max_val);
                    sum_val += e;
                }

                probs[bh * S * S + q * S + k] = e;
            }

            float inv_sum = 1.0f / sum_val;

            for (int k = 0; k < S; ++k) {
                probs[bh * S * S + q * S + k] *= inv_sum;
            }
        }
    }

    // 3. O = probs @ V
    for (int bh = 0; bh < BH; ++bh) {
        for (int q = 0; q < S; ++q) {
            for (int d = 0; d < D; ++d) {
                float sum = 0.0f;

                for (int k = 0; k < S; ++k) {
                    float p = probs[bh * S * S + q * S + k];
                    float vv = V[bh * S * D + k * D + d];
                    sum += p * vv;
                }

                O[bh * S * D + q * D + d] = sum;
            }
        }
    }
}

void launch_qk_matmul_tiled(
    const float* d_Q,
    const float* d_K,
    float* d_scores,
    int BH,
    int S,
    int D
) {
    constexpr int BLOCK_Q = 16;
    constexpr int BLOCK_K = 16;
    constexpr int BLOCK_D = 16;

    dim3 block(BLOCK_K, BLOCK_Q);

    dim3 grid(
        (S + BLOCK_K - 1) / BLOCK_K,
        (S + BLOCK_Q - 1) / BLOCK_Q,
        BH
    );

    qk_matmul_tiled_kernel<BLOCK_Q, BLOCK_K, BLOCK_D>
        <<<grid, block>>>(
            d_Q,
            d_K,
            d_scores,
            BH,
            S,
            D
        );
}


void launch_pv_matmul_tiled(
    const float* d_probs,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    constexpr int BLOCK_P = 16;
    constexpr int BLOCK_V = 16;
    constexpr int BLOCK_S = 16;

    dim3 block(BLOCK_V, BLOCK_P);

    dim3 grid(
        (D + BLOCK_V - 1) / BLOCK_V,
        (S + BLOCK_P - 1) / BLOCK_P,
        BH
    );

    pv_matmul_tiled_kernel<BLOCK_P, BLOCK_V, BLOCK_S>
        <<<grid, block>>>(
            d_probs,
            d_V,
            d_O,
            BH,
            S,
            D
        );
}

void launch_attention_forward_tiled(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_scores,
    float* d_probs,
    float* d_O,
    int BH,
    int S,
    int D
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // 1. scores = Q @ K^T, tiled version
    launch_qk_matmul_tiled(
        d_Q,
        d_K,
        d_scores,
        BH,
        S,
        D
    );

    // 2. probs = causal_softmax(scores * scale)
    launch_scaled_causal_softmax(
        d_scores,
        d_probs,
        BH * S,
        S,
        S,
        scale,
        256
    );

    // 3. O = probs @ V, tiled version
    launch_pv_matmul_tiled(
        d_probs,
        d_V,
        d_O,
        BH,
        S,
        D
    );
}

void launch_attention_forward_fused_row(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    int block_size = 256;
    dim3 grid(BH * S);

    size_t smem = static_cast<size_t>(S) * sizeof(float);

    fused_causal_attention_row_kernel<<<grid, block_size, smem>>>(
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D,
        scale
    );
}

void launch_flash_attention_v1(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    constexpr int BLOCK_M = 4;
    constexpr int BLOCK_N = 32;
    constexpr int MAX_D = 128;

    if (D > MAX_D) {
        fprintf(stderr, "flash_attention_v1 only supports D <= %d, got D = %d\n", MAX_D, D);
        return;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    dim3 grid(
        (S + BLOCK_M - 1) / BLOCK_M,
        BH
    );

    dim3 block(128);

    flash_attention_v1_kernel<BLOCK_M, BLOCK_N, MAX_D>
        <<<grid, block>>>(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D,
            scale
        );
}

void launch_flash_attention_skip_bm4_smem(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    constexpr int BLOCK_M = 4;
    constexpr int BLOCK_N = 32;
    constexpr int MAX_D = 128;

    if (D > MAX_D) {
        fprintf(stderr, "flash_attention_skip_bm4 only supports D <= %d, got D = %d\n", MAX_D, D);
        return;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, BH);
    dim3 block(128);

    flash_attention_causal_tile_skipping_kernel<BLOCK_M, BLOCK_N, MAX_D>
        <<<grid, block>>>(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D,
            scale
        );
}

void launch_flash_attention_skip_bm8_smem(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    constexpr int BLOCK_M = 8;
    constexpr int BLOCK_N = 32;
    constexpr int MAX_D = 128;

    if (D > MAX_D) {
        fprintf(stderr, "flash_attention_skip_bm8 only supports D <= %d, got D = %d\n", MAX_D, D);
        return;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, BH);
    dim3 block(128);

    flash_attention_causal_tile_skipping_kernel<BLOCK_M, BLOCK_N, MAX_D>
        <<<grid, block>>>(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D,
            scale
        );
}


void launch_flash_attention_skip_bm16_smem(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 32;
    constexpr int MAX_D = 64;

    if (D > MAX_D) {
        fprintf(stderr, "flash_attention_skip_bm16 only supports D <= %d, got D = %d\n", MAX_D, D);
        return;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, BH);
    dim3 block(128);

    flash_attention_causal_tile_skipping_kernel<BLOCK_M, BLOCK_N, MAX_D>
        <<<grid, block>>>(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D,
            scale
        );
}


void launch_flash_attention_causal_tile_skipping_smem(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    if (S >= 512 && D <= 64) {
        launch_flash_attention_skip_bm16_smem(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D
        );
    } else if (S >= 256) {
        launch_flash_attention_skip_bm8_smem(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D
        );
    } else {
        launch_flash_attention_skip_bm4_smem(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D
        );
    }
}
void launch_flash_attention_skip_bm4_regacc(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    constexpr int BLOCK_M = 4;
    constexpr int BLOCK_N = 32;
    constexpr int MAX_D = 128;

    if (D > MAX_D) {
        fprintf(stderr, "flash_attention_skip_bm4_regacc only supports D <= %d, got D = %d\n", MAX_D, D);
        return;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, BH);
    dim3 block(128);

    flash_attention_causal_tile_skipping_regacc_kernel<BLOCK_M, BLOCK_N, MAX_D>
        <<<grid, block>>>(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D,
            scale
        );
}
void launch_flash_attention_skip_bm8_regacc(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    constexpr int BLOCK_M = 8;
    constexpr int BLOCK_N = 32;
    constexpr int MAX_D = 128;

    if (D > MAX_D) {
        fprintf(stderr, "flash_attention_skip_bm8_regacc only supports D <= %d, got D = %d\n", MAX_D, D);
        return;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, BH);
    dim3 block(128);

    flash_attention_causal_tile_skipping_regacc_kernel<BLOCK_M, BLOCK_N, MAX_D>
        <<<grid, block>>>(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D,
            scale
        );
}

void launch_flash_attention_skip_bm16_regacc(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 32;
    constexpr int MAX_D = 64;

    if (D > MAX_D) {
        fprintf(stderr, "flash_attention_skip_bm16_regacc only supports D <= %d, got D = %d\n", MAX_D, D);
        return;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, BH);
    dim3 block(128);

    flash_attention_causal_tile_skipping_regacc_kernel<BLOCK_M, BLOCK_N, MAX_D>
        <<<grid, block>>>(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D,
            scale
        );
}
void launch_flash_attention_causal_tile_skipping_regacc(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    if (S >= 512 && D <= 64) {
        launch_flash_attention_skip_bm16_regacc(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D
        );
    } else if (S >= 256) {
        launch_flash_attention_skip_bm8_regacc(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D
        );
    } else {
        launch_flash_attention_skip_bm4_regacc(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D
        );
    }
}

void launch_flash_attention_causal_tile_skipping(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    launch_flash_attention_causal_tile_skipping_vec4(
        d_Q,
        d_K,
        d_V,
        d_O,
        BH,
        S,
        D
    );
}

void launch_flash_attention_skip_bm4_regacc_vec4(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    constexpr int BLOCK_M = 4;
    constexpr int BLOCK_N = 32;
    constexpr int MAX_D = 128;

    if (D > MAX_D) {
        fprintf(stderr, "flash_attention_skip_bm4_regacc_vec4 only supports D <= %d, got D = %d\n", MAX_D, D);
        return;
    }

    if ((D & 3) != 0) {
        launch_flash_attention_skip_bm4_regacc(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D
        );
        return;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, BH);
    dim3 block(128);

    flash_attention_causal_tile_skipping_regacc_vec4_kernel<BLOCK_M, BLOCK_N, MAX_D>
        <<<grid, block>>>(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D,
            scale
        );
}

void launch_flash_attention_skip_bm8_regacc_vec4(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    constexpr int BLOCK_M = 8;
    constexpr int BLOCK_N = 32;
    constexpr int MAX_D = 128;

    if (D > MAX_D) {
        fprintf(stderr, "flash_attention_skip_bm8_regacc_vec4 only supports D <= %d, got D = %d\n", MAX_D, D);
        return;
    }

    if ((D & 3) != 0) {
        launch_flash_attention_skip_bm8_regacc(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D
        );
        return;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, BH);
    dim3 block(128);

    flash_attention_causal_tile_skipping_regacc_vec4_kernel<BLOCK_M, BLOCK_N, MAX_D>
        <<<grid, block>>>(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D,
            scale
        );
}

void launch_flash_attention_skip_bm16_regacc_vec4(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 32;
    constexpr int MAX_D = 64;

    if (D > MAX_D) {
        fprintf(stderr, "flash_attention_skip_bm16_regacc_vec4 only supports D <= %d, got D = %d\n", MAX_D, D);
        return;
    }

    if ((D & 3) != 0) {
        launch_flash_attention_skip_bm16_regacc(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D
        );
        return;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    dim3 grid((S + BLOCK_M - 1) / BLOCK_M, BH);
    dim3 block(128);

    flash_attention_causal_tile_skipping_regacc_vec4_kernel<BLOCK_M, BLOCK_N, MAX_D>
        <<<grid, block>>>(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D,
            scale
        );
}

void launch_flash_attention_causal_tile_skipping_vec4(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int BH,
    int S,
    int D
) {
    if (S >= 512 && D <= 64) {
        launch_flash_attention_skip_bm16_regacc_vec4(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D
        );
    } else if (S >= 256) {
        launch_flash_attention_skip_bm8_regacc_vec4(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D
        );
    } else {
        launch_flash_attention_skip_bm4_regacc_vec4(
            d_Q,
            d_K,
            d_V,
            d_O,
            BH,
            S,
            D
        );
    }
}