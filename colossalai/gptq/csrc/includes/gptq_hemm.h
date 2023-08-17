


#define LDMATRIX_X1(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

#if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CA_64(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG_64(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CA_NC(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG_NC(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#else
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#define CP_ASYNC_CA_64(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG_64(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#endif

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_ROW_WARPS (BLOCK_COLS / WARP_COLS)  // 2  BLOCK_COLS / WARP_COLS
#define BLOCK_COL_WARPS (BLOCK_ROWS / WARP_ROWS) // 4 BLOCK_ROWS / WARP_ROWS

#define BLOCK_ROW_TILES (BLOCK_COLS / MMA_N)  //  16 BLOCK_COLS / MMA_N
#define BLOCK_COL_TILES (BLOCK_ROWS / MMA_M)  // 16 BLOCK_ROWS / MMA_M

#define WARP_ROW_TILES  (WARP_COLS / MMA_N)  // 8 WARP_COLS / MMA_N
#define WARP_COL_TILES (WARP_ROWS / MMA_M)  // 4 WARP_ROWS / MMA_M

#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_ROW_WARPS * BLOCK_COL_WARPS)  // 8 BLOCK_ROW_WARPS * BLOCK_COL_WARPS
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)  // 256 WARP_SIZE * WARPS_PER_BLOCK

#define CHUNK_K (32 / MMA_K)  // 2 32 / MMA_K

#define THREAD_COPY_BYTES 16

#define CHUNK_LINE_BYTES 64          // CHUNK_K * MMA_K * sizeof(half)
#define CHUNK_COPY_LINES_PER_WARP 8  // WARP_SIZE * THREAD_COPY_BYTES / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES 4      // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP

#define AB_SHMEM_STRIDE 32  // CHUNK_K * MMA_K

#define C_SHMEM_STRIDE (BLOCK_COLS)  // 128 BLOCK_COLS
#define C_SHMEM_OFFSET (WARP_COLS)   // 64 WARP_COLS

#define BLOCK_STRIDE 16

#define SHMEM_BANK_ROWS 2  // 32 * 4 / (AB_SHMEM_STRIDE * sizeof(half))

#define PERMUTED_OFFSET 8
#define PERMUTED_COLS 4

#define K_STAGE 4

#define NW_PER_UINT64 16 // for 4bits 
// #define B_COL_STEP_PER_LANE 2
#define B_LANES_PER_CHUNK_K ((CHUNK_K * MMA_K) / NW_PER_UINT64)
#define B_COL_STEP_PER_WARP (WARP_SIZE / B_LANES_PER_CHUNK_K) // 16
inline __device__ __host__ static size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

inline __device__ float relu(const float x) { return x < 0 ? 0 : x; }
inline __device__ float gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param  = 0.044715;
    return x * 0.5f * (1.0f + tanhf(sqrt_param * (x + mul_param * x * x * x)));
}
inline __device__ float silu(const float x)
{
    return x  / (1 + expf(-x));
}

template <typename T, typename TW, int W_BITS=4>
__global__ static void gptq_bgemm_final(const T *__restrict__  A,
                            TW* weight,
                            T* weight_scales,
                            TW* weight_zeros,
                            int32_t* idx,
                            T* bias,
                            T* residual,
                            T*__restrict__ C,
                            size_t M, 
                            size_t N, 
                            size_t K,
                            size_t group_size,
                            int32_t act_type,
                            bool add_bias,
                            bool add_residual,
                            bool qkv_fused
                            )
{

    const size_t M_tiles = div_ceil(M, MMA_M);
    const size_t N_tiles = div_ceil(N, MMA_N);
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    const size_t grid_dim_z = div_ceil(N, BLOCK_COLS * BLOCK_STRIDE);
    const size_t qkv_offset = blockIdx.z / grid_dim_z; 
    const size_t block_id_z = blockIdx.z % grid_dim_z;

    const size_t block_tile_i = 
        (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * BLOCK_COL_TILES) : (blockIdx.y * BLOCK_COL_TILES);
    const size_t block_tile_j = (block_id_z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;

    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }
    const size_t n_weights_per_int = sizeof(TW) * 8 / W_BITS;
    const int32_t B_MASK = (1 << W_BITS) - 1;

    extern __shared__ half shmem[][AB_SHMEM_STRIDE];

    const size_t shm_a_size = K_STAGE * BLOCK_ROWS * CHUNK_K * MMA_K * sizeof(half);
    const size_t shm_weight_size = K_STAGE * BLOCK_COLS * CHUNK_K * MMA_K / n_weights_per_int * sizeof(TW);
    const size_t shm_scales_size =  K_STAGE * ((CHUNK_K * MMA_K + group_size - 1)/group_size) * BLOCK_COLS * sizeof(half);
    const size_t shm_idx_size = K_STAGE * CHUNK_K * MMA_K * sizeof(int32_t);
    
    TW *shm_qweight_ptr = (TW*)(((void*)&shmem[0][0]) + shm_a_size);
    half *shm_scales_ptr = (half*)(((void*)shm_qweight_ptr) + shm_weight_size);
    int32_t *shm_idx_ptr = (int32_t*)(((void*)shm_scales_ptr) + shm_scales_size);
    TW *shm_zeros_ptr = (TW*)(((void*)shm_idx_ptr) + shm_idx_size);

 #define SHM_WEIGHT(i, j) (shm_qweight_ptr[(i) * BLOCK_COLS + (j)])
 #define SHM_SCALES(i, j) (shm_scales_ptr[(i) * BLOCK_COLS + (j)])
 #define SHM_IDX(i, j) (shm_idx_ptr[(i) * CHUNK_K * MMA_K + (j)])
 #define SHM_ZEROS(i, j) (shm_zeros_ptr[(i) * BLOCK_COLS/n_weights_per_int + (j)])
 #define DEBUG_ADDR(i, j, s) {} //if ((uint64_t) i > (uint64_t)j) { printf("wrong addrs: %s\n", s);}

    TW* zeros_ptr = weight_zeros + block_tile_j * MMA_N / n_weights_per_int;
    size_t lanes_per_zk_row = BLOCK_COLS * W_BITS  / 8 / THREAD_COPY_BYTES;
    

    for(size_t i = 0; i < K / group_size; i += THREADS_PER_BLOCK / lanes_per_zk_row)
    {
        if(i + threadIdx.x / lanes_per_zk_row  < K / group_size)
        {
            *(((int4*)&SHM_ZEROS(i + threadIdx.x / lanes_per_zk_row , 0)) + (lane_id % lanes_per_zk_row)) = 
                *(((int4*)(zeros_ptr + (i + threadIdx.x / lanes_per_zk_row) * N / n_weights_per_int)) + (lane_id % lanes_per_zk_row));
        }
    }
    __syncthreads();

    const size_t shm_weight_off = 0;
    const size_t shm_scales_off = 0; 
    const size_t shm_idx_off = 0;
    const size_t shm_zeros_off = 0;

    const size_t THREAD_COPY_WEIGHT_BYTES = 8;
    const size_t THREAD_COPY_SCALE_BYTES = 4;

    TW* qweight_ptr = weight + block_tile_j * MMA_N;
    const size_t w_lanes_per_col = AB_SHMEM_STRIDE / n_weights_per_int;
    const size_t w_cols_per_warp = BLOCK_COLS / WARPS_PER_BLOCK;
    const size_t w_cols_per_lane = (THREAD_COPY_WEIGHT_BYTES / sizeof(TW));
    const size_t w_cols_per_iter = WARP_SIZE / w_lanes_per_col * w_cols_per_lane;
    const size_t weight_iters = w_cols_per_warp / w_cols_per_iter;


    T* scales_ptr = weight_scales + block_tile_j * MMA_N;
    const size_t s_lanes_per_col = (AB_SHMEM_STRIDE + group_size - 1) / group_size; 
    const size_t s_cols_per_lane = (THREAD_COPY_SCALE_BYTES / sizeof(T));
    const size_t s_cols_per_iter = WARP_SIZE / s_lanes_per_col * s_cols_per_lane;
    const size_t warps_load_s = BLOCK_COLS / s_cols_per_iter;

    const size_t W_shmem_stage_off = CHUNK_K * MMA_K / n_weights_per_int;
    // const size_t Z_shmem_stage_off = BLOCK_COLS / n_weights_per_int;
    const size_t Z_shmem_stage_off = 0;
    const size_t S_shmem_stage_off = (CHUNK_K * MMA_K + group_size - 1)/ group_size;

    size_t W_shmem_store_off = 0;
    size_t S_shmem_store_off = 0;
    size_t Z_shmem_store_off = 0;

    size_t W_shmem_load_off = 0;
    size_t S_shmem_load_off = 0;
    size_t Z_shmem_load_off = 0;

    size_t W_shmem_idx = 0;
    size_t Z_shmem_idx = 0;
    size_t S_shmem_idx = 0;

    int2* W_lane_ptr = nullptr;
    int* S_lane_ptr = nullptr;



    half *shmem_warp_tile_row_ptr = &shmem[0][0] + (warp_id / BLOCK_ROW_WARPS) * C_SHMEM_STRIDE * WARP_ROWS;
    const half *shmem_warp_stream_ptr = &shmem[0][0] + warp_id * MMA_M * 2 * C_SHMEM_STRIDE;

    const size_t gmem_idx = (block_tile_i * MMA_M + warp_id * MMA_M * 2) * N + block_tile_j * MMA_N;
    const half *src_gmem_warp_stream_ptr = &C[qkv_offset * M * N + gmem_idx];
    const T *residual_ptr = &residual[gmem_idx];
    const T *bias_ptr = &bias[qkv_offset * N + block_tile_j * MMA_N];

    uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2];

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) { 
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

    const half *A_warp_ptr = &A[block_tile_i * MMA_M * K] + BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id;
    const size_t A_shmem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

    const size_t A_shmem_stage_off = BLOCK_ROWS;

    size_t shmem_store_idx = 0;
    size_t shmem_load_idx = 0;

    size_t A_shmem_store_off = 0;
    size_t A_shmem_load_off = 0;

    size_t A_shmem_idx = 0;
    int4 *A_lane_ptr = nullptr;

    A_shmem_idx = A_shmem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    A_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_shmem_iters; ++i) {
        if((uint64_t)A_lane_ptr < (uint64_t)&A[M*K-1])
        {
            uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(&shmem[A_shmem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                        (A_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
                                        CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;

            CP_ASYNC_CG(A_shmem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }
    }

    W_shmem_store_off = 0;
    S_shmem_store_off = 0;

    W_shmem_idx = shm_weight_off + W_shmem_store_off + lane_id % w_lanes_per_col;
    W_lane_ptr = (int2 *)(qweight_ptr + lane_id % w_lanes_per_col * N + warp_id * w_cols_per_warp + 
                        lane_id / w_lanes_per_col * w_cols_per_lane);

#pragma unroll
    for (size_t i = 0; i < weight_iters; ++i) {
        DEBUG_ADDR(W_lane_ptr, &weight[N * K / n_weights_per_int - 1], "Weight");

        uint32_t W_shmem_lane_addr = __cvta_generic_to_shared(&SHM_WEIGHT(W_shmem_idx, 
                warp_id * w_cols_per_warp + i * w_cols_per_iter)) + THREAD_COPY_WEIGHT_BYTES * (lane_id / w_lanes_per_col);

        CP_ASYNC_CA(W_shmem_lane_addr, W_lane_ptr, THREAD_COPY_WEIGHT_BYTES);
        W_lane_ptr = (int2 *)((TW *)W_lane_ptr + w_cols_per_iter);
    }

    if(warp_id < warps_load_s)
    {
        S_shmem_idx = shm_scales_off + S_shmem_store_off + lane_id % s_lanes_per_col;
        S_lane_ptr = (int *)(scales_ptr + warp_id * s_cols_per_iter + (lane_id % s_lanes_per_col) / group_size * N 
                            + (lane_id / s_lanes_per_col) * s_cols_per_lane);

        DEBUG_ADDR(S_lane_ptr, scales_ptr + N * K / group_size, "Scales");

        uint32_t S_shmem_lane_addr = __cvta_generic_to_shared(&SHM_SCALES(S_shmem_idx,
                warp_id * s_cols_per_iter)) + (lane_id / s_lanes_per_col) * THREAD_COPY_SCALE_BYTES;

        CP_ASYNC_CA(S_shmem_lane_addr, S_lane_ptr, THREAD_COPY_SCALE_BYTES);
    }

    CP_ASYNC_COMMIT_GROUP();


    shmem_store_idx = (shmem_store_idx + 1) % K_STAGE;
    A_shmem_store_off = shmem_store_idx * A_shmem_stage_off;
    A_shmem_idx = A_shmem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    A_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_shmem_iters; ++i) {
        if((uint64_t)A_lane_ptr < (uint64_t)&A[M*K-1])
        {
            uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(&shmem[A_shmem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                        (A_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
                                        CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;

            CP_ASYNC_CG(A_shmem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }
    }

    W_shmem_store_off = shmem_store_idx * W_shmem_stage_off;
    S_shmem_store_off = shmem_store_idx * S_shmem_stage_off;
    Z_shmem_store_off = shmem_store_idx * Z_shmem_stage_off;

    W_shmem_idx = shm_weight_off + W_shmem_store_off + lane_id % w_lanes_per_col;
    W_lane_ptr = (int2 *)(qweight_ptr + (CHUNK_K * MMA_K / n_weights_per_int +  lane_id % w_lanes_per_col) * N
                         + warp_id * w_cols_per_warp 
                         + lane_id / w_lanes_per_col * w_cols_per_lane);

#pragma unroll
    for (size_t i = 0; i < weight_iters; ++i) {
        DEBUG_ADDR(W_lane_ptr, &weight[N * K / n_weights_per_int - 1], "Weight");

        uint32_t W_shmem_lane_addr = __cvta_generic_to_shared(&SHM_WEIGHT(W_shmem_idx, 
                warp_id * w_cols_per_warp + i * w_cols_per_iter)) + THREAD_COPY_WEIGHT_BYTES * (lane_id / w_lanes_per_col);

        CP_ASYNC_CA(W_shmem_lane_addr, W_lane_ptr, THREAD_COPY_WEIGHT_BYTES);
        W_lane_ptr = (int2 *)((TW *)W_lane_ptr + w_cols_per_iter);
    }

    if(warp_id < warps_load_s)
    {
        S_shmem_idx = shm_scales_off + S_shmem_store_off + lane_id % s_lanes_per_col;

        S_lane_ptr = (int *)(scales_ptr +  warp_id * s_cols_per_iter + (CHUNK_K * MMA_K / group_size + lane_id % s_lanes_per_col) * N  
                            + (lane_id / s_lanes_per_col) * s_cols_per_lane);
        DEBUG_ADDR(S_lane_ptr, &weight_scales[N * K / group_size - 1], "Scales");

        uint32_t S_shmem_lane_addr = __cvta_generic_to_shared(&SHM_SCALES(S_shmem_idx,
                warp_id * s_cols_per_iter)) + (lane_id / s_lanes_per_col) * THREAD_COPY_SCALE_BYTES;

        CP_ASYNC_CA(S_shmem_lane_addr, S_lane_ptr, THREAD_COPY_SCALE_BYTES);
    }



    CP_ASYNC_COMMIT_GROUP();

    shmem_store_idx = (shmem_store_idx + 1) % K_STAGE;
    A_shmem_store_off = shmem_store_idx * A_shmem_stage_off;

    A_shmem_idx = A_shmem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + 2 * CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    A_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_shmem_iters; ++i) {
        if((uint64_t)A_lane_ptr < (uint64_t)&A[M*K-1])
        {
            uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(&shmem[A_shmem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                        (A_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
                                        CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;

            CP_ASYNC_CG(A_shmem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }
    }

    W_shmem_store_off = shmem_store_idx * W_shmem_stage_off;
    S_shmem_store_off = shmem_store_idx * S_shmem_stage_off;
    Z_shmem_store_off = shmem_store_idx * Z_shmem_stage_off;

    W_shmem_idx = shm_weight_off + W_shmem_store_off + lane_id % w_lanes_per_col;
    W_lane_ptr = (int2 *)(qweight_ptr + (2 * CHUNK_K * MMA_K / n_weights_per_int + lane_id % w_lanes_per_col) * N + warp_id * w_cols_per_warp + 
                        lane_id / w_lanes_per_col * w_cols_per_lane);

#pragma unroll
    for (size_t i = 0; i < weight_iters; ++i) {
        uint32_t W_shmem_lane_addr = __cvta_generic_to_shared(&SHM_WEIGHT(W_shmem_idx, 
                warp_id * w_cols_per_warp + i * w_cols_per_iter)) + THREAD_COPY_WEIGHT_BYTES * (lane_id / w_lanes_per_col);
        DEBUG_ADDR(W_lane_ptr, &weight[N * K / n_weights_per_int - 1], "Weight");

        CP_ASYNC_CA(W_shmem_lane_addr, W_lane_ptr, THREAD_COPY_WEIGHT_BYTES);
        W_lane_ptr = (int2 *)((TW *)W_lane_ptr + w_cols_per_iter);
    }

    if(warp_id < warps_load_s)
    {
        S_shmem_idx = shm_scales_off + S_shmem_store_off + lane_id % s_lanes_per_col;

        S_lane_ptr = (int *)(scales_ptr + warp_id * s_cols_per_iter + (2 * CHUNK_K * MMA_K / group_size + lane_id % s_lanes_per_col) * N 
                            + (lane_id / s_lanes_per_col * s_cols_per_lane));
        DEBUG_ADDR(S_lane_ptr,  &weight_scales[N * K / group_size - 1], "Scales");

        uint32_t S_shmem_lane_addr = __cvta_generic_to_shared(&SHM_SCALES(S_shmem_idx,
                warp_id * s_cols_per_iter)) + (lane_id / s_lanes_per_col) * THREAD_COPY_SCALE_BYTES;

        CP_ASYNC_CA(S_shmem_lane_addr, S_lane_ptr, THREAD_COPY_SCALE_BYTES);
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(2);

    __syncthreads();

    uint32_t RA[2][WARP_COL_TILES][4];
    uint32_t RB[2][WARP_ROW_TILES][2];

    size_t reg_store_idx = 0;
    size_t reg_load_idx = 1;


#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        size_t A_shmem_idx = A_shmem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
        uint32_t A_shmem_lane_addr =
            __cvta_generic_to_shared(&shmem[A_shmem_idx + lane_id % 16]
                                           [((lane_id / 16) * 8 + (lane_id % 16 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) /
                                                                      SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                                            AB_SHMEM_STRIDE]);

        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                    A_shmem_lane_addr);

    }

#pragma unroll
    for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
        size_t W_shmem_idx = (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N + lane_id / 4;

        float fs = __half2float(SHM_SCALES(0, W_shmem_idx));
        TW qw = SHM_WEIGHT(0,  W_shmem_idx);


        TW qz0 = SHM_ZEROS(0,  (W_shmem_idx)/n_weights_per_int);

        half2 *RB_ptr = (half2*)&RB[reg_store_idx][j][0];
        int32_t w0 = (qw >> ((lane_id % 4)*4*2)) & B_MASK;
        int32_t w1 = (qw >> ((lane_id % 4)*4*2 + 4)) & B_MASK;
        int32_t w2 = (qw >> (32 + (lane_id % 4)*4*2)) & B_MASK;
        int32_t w3 = (qw >> (32 + (lane_id % 4)*4*2 + 4)) & B_MASK;

        int32_t z0 = (qz0 >> ((j * MMA_N + lane_id / 4) % n_weights_per_int * 4)) & B_MASK;

        w0 = w0 - (z0 + 1);
        w1 = w1 - (z0 + 1);
        w2 = w2 - (z0 + 1);
        w3 = w3 - (z0 + 1);
        float2 fw0 = {float(w0) * fs, float(w1) * fs};
        float2 fw1 = {float(w2) * fs, float(w3) * fs};
        RB_ptr[0] = __float22half2_rn(fw0);
        RB_ptr[1] = __float22half2_rn(fw1);

        // printf("id %d %lld %lld %lld %f\n", threadIdx.x, lane_id, weight[0], shm_zeros_ptr[0], f5);

            // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
            // {
            //     printf("id %d %lld %lld %lld %f %f\n", threadIdx.x, lane_id, weight[0], weight_zeros[0], float(w2) * fs, float(w3) * fs);

            //     printf("id %d %lld %f %f %f %f\n", threadIdx.x, lane_id, float(w0) * fs, float(w1) * fs, float(w2) * fs, float(w3) * fs);
            // }
    }


    // half * ra_h = (half*)&RA[reg_load_idx][0][0];
        // if (warp_id == 0 && RA[reg_load_idx][i][0] != 0 && lane_id % 4 == 0)
        // {
        //     printf("ldd %lld %lld %lld \n", lane_id, i, 0);
        // }
    // CP_ASYNC_WAIT_GROUP(0);

    // __syncthreads();

#pragma unroll
    for (size_t tile_k = CHUNK_K * (K_STAGE - 1); tile_k < K_tiles; tile_k += CHUNK_K) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_shmem_idx = A_shmem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(
                &shmem[A_shmem_idx + lane_id % 16]
                      [(MMA_K + (lane_id / 16) * 8 +
                        (lane_id % 16 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                       AB_SHMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2],
                        RA[reg_store_idx][i][3], A_shmem_lane_addr);
        }


#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t W_shmem_idx = (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N + lane_id / 4;

            float fs = __half2float(SHM_SCALES(S_shmem_load_off, W_shmem_idx));

            TW qw = SHM_WEIGHT(W_shmem_load_off + MMA_K / n_weights_per_int,  W_shmem_idx);

            TW qz0 = SHM_ZEROS(Z_shmem_load_off + (tile_k - (K_STAGE -1)*CHUNK_K + 1) * MMA_K / group_size,  (W_shmem_idx)/n_weights_per_int);

            half2 *RB_ptr = (half2*)&RB[reg_store_idx][j][0];
            int32_t w0 = (qw >> ((lane_id % 4)*2 * 4)) & B_MASK;
            int32_t w1 = (qw >> ((lane_id % 4)*2 * 4 + 4)) & B_MASK;
            int32_t w2 = (qw >> (32 + (lane_id % 4)*2 * 4)) & B_MASK;
            int32_t w3 = (qw >> (32 + (lane_id % 4)*2*4 + 4)) & B_MASK;

            int32_t z0 = (qz0 >> ((j * MMA_N + lane_id / 4) % n_weights_per_int * 4)) & B_MASK;


            w0 = w0 - (z0 + 1);
            w1 = w1 - (z0 + 1);
            w2 = w2 - (z0 + 1);
            w3 = w3 - (z0 + 1);
            float2 fw0 = {float(w0) * fs, float(w1) * fs};
            float2 fw1 = {float(w2) * fs, float(w3) * fs};
            RB_ptr[0] = __float22half2_rn(fw0);
            RB_ptr[1] = __float22half2_rn(fw1);

        }


#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            // if (warp_id == 0 && RA[reg_load_idx][i][0] != 0)
            // {
            //     printf("ldd %lld %lld %d %d\n", lane_id, i, 0, blockIdx.x);
            // }
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                          RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }

        shmem_store_idx = (shmem_store_idx + 1) % K_STAGE;
        A_shmem_store_off = shmem_store_idx * A_shmem_stage_off;

        A_shmem_idx = A_shmem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        A_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_shmem_iters / CHUNK_K; ++i) {
            if((uint64_t)A_lane_ptr < (uint64_t)&A[M*K-1])
            {
                uint32_t A_shmem_lane_addr =
                    __cvta_generic_to_shared(&shmem[A_shmem_idx][0]) +
                    ((lane_id % CHUNK_COPY_LINE_LANES +
                    (A_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
                    CHUNK_COPY_LINE_LANES) *
                        THREAD_COPY_BYTES;

                CP_ASYNC_CG(A_shmem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

                A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
                A_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }
        }

        W_shmem_store_off = shmem_store_idx * W_shmem_stage_off;
        S_shmem_store_off = shmem_store_idx * S_shmem_stage_off;
        Z_shmem_store_off = shmem_store_idx * Z_shmem_stage_off;


        W_shmem_idx = shm_weight_off + W_shmem_store_off + lane_id % w_lanes_per_col;
        W_lane_ptr = (int2 *)(qweight_ptr + (tile_k * MMA_K / n_weights_per_int + lane_id % w_lanes_per_col) * N 
                            + warp_id * w_cols_per_warp + 
                            lane_id / w_lanes_per_col * w_cols_per_lane);

    #pragma unroll
        for (size_t i = 0; i < weight_iters; ++i) {
            uint32_t W_shmem_lane_addr = __cvta_generic_to_shared(&SHM_WEIGHT(W_shmem_idx, 
                    warp_id * w_cols_per_warp + i * w_cols_per_iter)) + THREAD_COPY_WEIGHT_BYTES * (lane_id / w_lanes_per_col);
            DEBUG_ADDR(W_lane_ptr, &weight[N * K / n_weights_per_int - 1], "Weight");

            CP_ASYNC_CA(W_shmem_lane_addr, W_lane_ptr, THREAD_COPY_WEIGHT_BYTES);
            W_lane_ptr = (int2 *)((TW *)W_lane_ptr + w_cols_per_iter);
        }

        if(warp_id < warps_load_s)
        {
            S_shmem_idx = shm_scales_off + S_shmem_store_off + lane_id % s_lanes_per_col;

            S_lane_ptr = (int *)(scales_ptr + warp_id * s_cols_per_iter + (tile_k * MMA_K / group_size  + lane_id % s_lanes_per_col) * N 
                                + (lane_id / s_lanes_per_col * s_cols_per_lane));
            DEBUG_ADDR(S_lane_ptr,  &weight_scales[N * K / group_size - 1], "Scales");

            uint32_t S_shmem_lane_addr = __cvta_generic_to_shared(&SHM_SCALES(S_shmem_idx,
                    warp_id * s_cols_per_iter)) + (lane_id / s_lanes_per_col) * THREAD_COPY_SCALE_BYTES;

            CP_ASYNC_CA(S_shmem_lane_addr, S_lane_ptr, THREAD_COPY_SCALE_BYTES);
        }


#pragma unroll
        for (size_t i = (CHUNK_K - 1) * A_shmem_iters / CHUNK_K; i < A_shmem_iters; ++i) {
            uint32_t A_shmem_lane_addr =
                __cvta_generic_to_shared(&shmem[A_shmem_idx][0]) +
                ((lane_id % CHUNK_COPY_LINE_LANES +
                  (A_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
                 CHUNK_COPY_LINE_LANES) *
                    THREAD_COPY_BYTES;

            CP_ASYNC_CG(A_shmem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }


        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(2);

        __syncthreads();

        shmem_load_idx = (shmem_load_idx + 1) % K_STAGE;
        A_shmem_load_off = shmem_load_idx * A_shmem_stage_off;
        W_shmem_load_off = shmem_load_idx * W_shmem_stage_off;
        S_shmem_load_off = shmem_load_idx * S_shmem_stage_off;
        Z_shmem_load_off = shmem_load_idx * Z_shmem_stage_off;

        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_shmem_idx = A_shmem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(
                &shmem[A_shmem_idx + lane_id % 16]
                      [((lane_id / 16) * 8 +
                        (lane_id % 16 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                       AB_SHMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2],
                        RA[reg_store_idx][i][3], A_shmem_lane_addr);

        }

#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t W_shmem_idx = (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N + lane_id / 4;

            float fs = __half2float(SHM_SCALES(S_shmem_load_off, W_shmem_idx));
            TW qw = SHM_WEIGHT(W_shmem_load_off,  W_shmem_idx);
            TW qz0 = SHM_ZEROS(Z_shmem_load_off + (tile_k - (K_STAGE -1)*CHUNK_K + 2) * MMA_K / group_size,  (W_shmem_idx)/n_weights_per_int);

            half2 *RB_ptr = (half2*)&RB[reg_store_idx][j][0];
            int32_t w0 = (qw >> ((lane_id % 4)*4*2)) & B_MASK;
            int32_t w1 = (qw >> ((lane_id % 4)*4*2 + 4)) & B_MASK;
            int32_t w2 = (qw >> (32 + (lane_id % 4)*4*2)) & B_MASK;
            int32_t w3 = (qw >> (32 + (lane_id % 4)*4*2 + 4)) & B_MASK;

            int32_t z0 = (qz0 >> ((j * MMA_N + lane_id / 4) % n_weights_per_int * 4)) & B_MASK;

            w0 = w0 - (z0 + 1);
            w1 = w1 - (z0 + 1);
            w2 = w2 - (z0 + 1);
            w3 = w3 - (z0 + 1);
            float2 fw0 = {float(w0) * fs, float(w1) * fs};
            float2 fw1 = {float(w2) * fs, float(w3) * fs};
            RB_ptr[0] = __float22half2_rn(fw0);
            RB_ptr[1] = __float22half2_rn(fw1);

        }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            // if (warp_id == 0 && RA[reg_load_idx][i][0] != 0)
            // {
            //     printf("ldd %lld %lld %d %d\n", lane_id, i, 1,  blockIdx.x);
            // }
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                          RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }
    }


#pragma unroll
    for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_shmem_idx = A_shmem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(
                &shmem[A_shmem_idx + lane_id % 16]
                      [(((k_step + 1) % CHUNK_K) * MMA_K + (lane_id / 16) * 8 +
                        (lane_id % 16 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                       AB_SHMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2],
                        RA[reg_store_idx][i][3], A_shmem_lane_addr);
        }


    #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t W_shmem_idx = (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N + lane_id / 4;

            float fs = __half2float(SHM_SCALES(S_shmem_load_off, W_shmem_idx));

            TW qw = SHM_WEIGHT(W_shmem_load_off + ((k_step + 1) % CHUNK_K) * MMA_K / n_weights_per_int,  W_shmem_idx);
            TW qz0 = SHM_ZEROS(Z_shmem_load_off + (K_tiles - (5 - k_step)) * MMA_K / group_size,  (W_shmem_idx)/n_weights_per_int);

            half2 *RB_ptr = (half2*)&RB[reg_store_idx][j][0];
            int32_t w0 = (qw >> ((lane_id % 4)*2 * 4)) & B_MASK;
            int32_t w1 = (qw >> ((lane_id % 4)*2 * 4 + 4)) & B_MASK;
            int32_t w2 = (qw >> (32 + (lane_id % 4)*2 * 4)) & B_MASK;
            int32_t w3 = (qw >> (32 + (lane_id % 4)*2*4 + 4)) & B_MASK;

            int32_t z0 = (qz0 >> ((j * MMA_N + lane_id / 4) % n_weights_per_int * 4)) & B_MASK;

            w0 = w0 - (z0 + 1);
            w1 = w1 - (z0 + 1);
            w2 = w2 - (z0 + 1);
            w3 = w3 - (z0 + 1);
            float2 fw0 = {float(w0) * fs, float(w1) * fs};
            float2 fw1 = {float(w2) * fs, float(w3) * fs};
            RB_ptr[0] = __float22half2_rn(fw0);
            RB_ptr[1] = __float22half2_rn(fw1);

        }


#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                          RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }

        if (k_step + 2 == CHUNK_K) {
            shmem_load_idx = (shmem_load_idx + 1) % K_STAGE;
            A_shmem_load_off = shmem_load_idx * A_shmem_stage_off;
            W_shmem_load_off = shmem_load_idx * W_shmem_stage_off;
            S_shmem_load_off = shmem_load_idx * S_shmem_stage_off;
            Z_shmem_load_off = shmem_load_idx * Z_shmem_stage_off;

            CP_ASYNC_WAIT_GROUP(1);

            __syncthreads();
        }
    }


#pragma unroll
    for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_shmem_idx = A_shmem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(
                &shmem[A_shmem_idx + lane_id % 16]
                      [(((k_step + 1) % CHUNK_K) * MMA_K + (lane_id / 16) * 8 +
                        (lane_id % 16 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                       AB_SHMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2],
                        RA[reg_store_idx][i][3], A_shmem_lane_addr);
        }



#pragma unroll
    for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
        size_t W_shmem_idx = (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N + lane_id / 4;

        float fs = __half2float(SHM_SCALES(S_shmem_load_off, W_shmem_idx));
        TW qw = SHM_WEIGHT(W_shmem_load_off + ((k_step + 1) % CHUNK_K) * MMA_K / n_weights_per_int ,  W_shmem_idx);
        TW qz0 = SHM_ZEROS(Z_shmem_load_off + (K_tiles - (3 - k_step)) * MMA_K / group_size,  (W_shmem_idx)/n_weights_per_int);


        half2 *RB_ptr = (half2*)&RB[reg_store_idx][j][0];
        int32_t w0 = (qw >> ((lane_id % 4)*4*2)) & B_MASK;
        int32_t w1 = (qw >> ((lane_id % 4)*4*2 + 4)) & B_MASK;
        int32_t w2 = (qw >> (32 + (lane_id % 4)*4*2)) & B_MASK;
        int32_t w3 = (qw >> (32 + (lane_id % 4)*4*2 + 4)) & B_MASK;

        int32_t z0 = (qz0 >> ((j * MMA_N + lane_id / 4) % n_weights_per_int * 4)) & B_MASK;

        w0 = w0 - (z0 + 1);
        w1 = w1 - (z0 + 1);
        w2 = w2 - (z0 + 1);
        w3 = w3 - (z0 + 1);
        float2 fw0 = {float(w0) * fs, float(w1) * fs};
        float2 fw1 = {float(w2) * fs, float(w3) * fs};
        RB_ptr[0] = __float22half2_rn(fw0);
        RB_ptr[1] = __float22half2_rn(fw1);

    }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                          RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }

        if (k_step + 2 == CHUNK_K) {
            shmem_load_idx = (shmem_load_idx + 1) % K_STAGE;
            A_shmem_load_off = shmem_load_idx * A_shmem_stage_off;
            W_shmem_load_off = shmem_load_idx * W_shmem_stage_off;
            S_shmem_load_off = shmem_load_idx * S_shmem_stage_off;
            Z_shmem_load_off = shmem_load_idx * Z_shmem_stage_off;
            CP_ASYNC_WAIT_GROUP(0);

            __syncthreads();
        }
    }

#pragma unroll
    for (size_t k_step = 1; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_shmem_idx = A_shmem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(
                &shmem[A_shmem_idx + lane_id % 16]
                      [(k_step * MMA_K + (lane_id / 16) * 8 +
                        (lane_id % 16 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                       AB_SHMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2],
                        RA[reg_store_idx][i][3], A_shmem_lane_addr);
        }


#pragma unroll
    for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
        size_t W_shmem_idx = (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N + lane_id / 4;

            float fs = __half2float(SHM_SCALES(S_shmem_load_off, W_shmem_idx));

            TW qw = SHM_WEIGHT(W_shmem_load_off + k_step*MMA_K / n_weights_per_int,  W_shmem_idx);

            TW qz0 = SHM_ZEROS(Z_shmem_load_off + (K_tiles - k_step) * MMA_K / group_size,  (W_shmem_idx)/n_weights_per_int);

            half2 *RB_ptr = (half2*)&RB[reg_store_idx][j][0];
            int32_t w0 = (qw >> ((lane_id % 4)*2 * 4)) & B_MASK;
            int32_t w1 = (qw >> ((lane_id % 4)*2 * 4 + 4)) & B_MASK;
            int32_t w2 = (qw >> (32 + (lane_id % 4)*2 * 4)) & B_MASK;
            int32_t w3 = (qw >> (32 + (lane_id % 4)*2*4 + 4)) & B_MASK;

            int32_t z0 = (qz0 >> ((j * MMA_N + lane_id / 4) % n_weights_per_int * 4)) & B_MASK;

            w0 = w0 - (z0 + 1);
            w1 = w1 - (z0 + 1);
            w2 = w2 - (z0 + 1);
            w3 = w3 - (z0 + 1);
            float2 fw0 = {float(w0) * fs, float(w1) * fs};
            float2 fw1 = {float(w2) * fs, float(w3) * fs};
            RB_ptr[0] = __float22half2_rn(fw0);
            RB_ptr[1] = __float22half2_rn(fw1);

    }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                          RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }
    }

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

            HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                      RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], RB[reg_store_idx][j_s][0],
                      RB[reg_store_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
        }
    }
    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            half *lane_ptr0 =
                shmem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4) * C_SHMEM_STRIDE +
                ((warp_id % BLOCK_ROW_WARPS) * C_SHMEM_OFFSET + j * MMA_N +
                 (lane_id % 4) * sizeof(uint32_t) / sizeof(half) + ((lane_id / 4) % 8) * PERMUTED_OFFSET) %
                    C_SHMEM_STRIDE;
            half *lane_ptr1 =
                shmem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4 + 8) * C_SHMEM_STRIDE +
                ((warp_id % BLOCK_ROW_WARPS) * C_SHMEM_OFFSET + j * MMA_N +
                 (lane_id % 4) * sizeof(uint32_t) / sizeof(half) + ((lane_id / 4 + 8) % 8) * PERMUTED_OFFSET) %
                    C_SHMEM_STRIDE;

            *((uint32_t *)(lane_ptr0)) = RC[i][j][0];
            *((uint32_t *)(lane_ptr1)) = RC[i][j][1];
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < MMA_M; ++i) {
        if((uint64_t)src_gmem_warp_stream_ptr < (uint64_t)&C[M*N - 1])
        {
            half* shem_c_local =  (half*)((int4 *)(shmem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SHMEM_STRIDE) + 
                (lane_id % 16 + (i * 2 + lane_id / 16) % 8) % (C_SHMEM_STRIDE * sizeof(half) / THREAD_COPY_BYTES));
            half* c_local = (half*)((src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) 
                                    + (lane_id % 16) * THREAD_COPY_BYTES / sizeof(half));
            const half* bias_local = bias_ptr + lane_id % 16 * THREAD_COPY_BYTES / sizeof(half);
            half* residual_local = (half*)((int4 *)(residual_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16);
            for(size_t j = 0; j < THREAD_COPY_BYTES / sizeof(half); j ++)
            {
                if(add_bias)
                    shem_c_local[j] += bias_local[j];

                if(act_type == 1)
                    shem_c_local[j] = __float2half_rn(relu(__half2float(shem_c_local[j])));
                else if(act_type == 2)
                    shem_c_local[j] = __float2half_rn(gelu(__half2float(shem_c_local[j])));
                else if(act_type == 3)
                    shem_c_local[j] = __float2half_rn(silu(__half2float(shem_c_local[j])));

                if(add_residual)
                    shem_c_local[j] += residual_local[j];
        
            }

            *((int4 *)c_local) =
                *((int4 *)shem_c_local);
        }
    }

    __syncthreads();
}


static size_t initMmaAsyncStage4() {
    int dev_id = 0;
    cudaGetDevice(&dev_id);
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, dev_id);

    // size_t shmem_max_size = std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SHMEM_STRIDE * sizeof(half) * K_STAGE,
    //                                  BLOCK_ROWS * C_SHMEM_STRIDE * sizeof(half));

    size_t shm_size = 
    K_STAGE * (BLOCK_ROWS * AB_SHMEM_STRIDE * sizeof(half) + AB_SHMEM_STRIDE * sizeof(half) * BLOCK_COLS / 4 
             + BLOCK_COLS * sizeof(half) + AB_SHMEM_STRIDE * sizeof(int))
    + 128 * BLOCK_COLS / 2;

    size_t shmem_max_size = std::max(shm_size,
                                     BLOCK_ROWS * C_SHMEM_STRIDE * sizeof(half));

    printf("shmem_max_size: %.0f KBytes (%zu Bytes)\n", static_cast<float>(shmem_max_size / 1024.0f), shmem_max_size);



    if(dev_prop.sharedMemPerMultiprocessor < shmem_max_size)
    {
        printf("Error: CUDA shared memory size < %zu\n", shmem_max_size);
    }

    cudaFuncSetAttribute(gptq_bgemm_final<__half, uint64_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_max_size);

    return shmem_max_size;
}