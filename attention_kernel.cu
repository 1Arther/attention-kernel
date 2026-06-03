#include <cuda_runtime.h>

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
    const float* k_ptr=K+bh*(D*S)+k_pos*D;  // 经过转置才能直接相乘
    float sum=0.0f;
    for(int i=0;i<D;i++){
        sum+=(q_ptr[i]*k_ptr[i]);
    }
    scores[idx]=sum;
}

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
//一维
__global__ void fused_causal_attention_row_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
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

    const float* q_ptr=Q+bh*S*D+q_pos*D;
    const float* k_base=K+bh*S*D;
    const float* v_base=V+bh*S*D;
    float* o_ptr=O+bh*S*D+q_pos*D;

    extern __shared__ float score_smem[];

    float local_max=-FLT_MAX;
    for(int k=tid;k<=q_pos;k+=blockDim.x){
        const float* k_ptr=k_base+k*D;
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
        //PV
    }
    __syncthreads();
    //pv
    for(int d=tid;d<D;d+=blockDim.x){
        float acc=0.0f;
        for(int k=0;k<=q_pos;k++){
            acc+=(score_smem[k]*v_base[k*D+d]);
        }
        o_ptr[d]=acc;
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