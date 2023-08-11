
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
#else
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
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

#define BLOCK_ROW_WARPS (BLOCK_COLS/WARP_COLS)  // 2  BLOCK_COLS / WARP_COLS
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

#define C_SHMEM_STRIDE (BLOCK_COLS)  // 64 BLOCK_COLS
#define C_SHMEM_OFFSET (WARP_COLS)   // 64 WARP_COLS

#define BLOCK_STRIDE 16

#define SHMEM_BANK_ROWS 2  // 32 * 4 / (AB_SHMEM_STRIDE * sizeof(half))

#define PERMUTED_OFFSET 8
#define PERMUTED_COLS 4

#define K_STAGE 3

#define NW_PER_UINT64 16 // for 4bits 
// #define B_COL_STEP_PER_LANE 2
#define B_LANES_PER_CHUNK_K ((CHUNK_K * MMA_K) / NW_PER_UINT64)
#define B_COL_STEP_PER_WARP (WARP_SIZE / B_LANES_PER_CHUNK_K) // 16
inline __device__ __host__ static size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}




template <typename T, typename TW>
__global__ static void gptq_bgemm_v3(T* input,
                            TW* weight,
                            T* weight_scales,
                            TW* weight_zeros,
                            T* bias,
                            T* residual_ptr,
                            T* output,
                            size_t M, 
                            size_t N, 
                            size_t K,
                            uint64_t group_size,
                            int32_t act_type,
                            bool add_bias,
                            bool add_residual,
                            bool qkv_fused
                            )
{
    const size_t w_per_int  = sizeof(TW) * 2;
    const size_t w_bits = 4;
    const int32_t w_mask = (1 << w_bits) - 1; 
    const size_t grid_dim_z = div_ceil(N, BLOCK_COLS * BLOCK_STRIDE);
    const size_t qkv_offset = blockIdx.z / grid_dim_z; 
    const size_t block_id_z = blockIdx.z % grid_dim_z;
    const size_t M_tiles = div_ceil(M, MMA_M);
    const size_t N_tiles = div_ceil(N, MMA_N);
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t M_size = M > MMA_M? MMA_M: M;
    const size_t INPUT_SIZE = M_size * K;

    const size_t block_tile_i =
        (block_id_z % 2) ? ((gridDim.y - blockIdx.y - 1) * BLOCK_COL_TILES) : (blockIdx.y * BLOCK_COL_TILES);
    const size_t block_tile_j = (block_id_z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;

    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    extern __shared__ half shmem[][AB_SHMEM_STRIDE];

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    const size_t B_shmem_idx_off = BLOCK_ROWS;

    half *shmem_warp_tile_row_ptr = &shmem[0][0] + (warp_id / BLOCK_ROW_WARPS) * C_SHMEM_STRIDE * WARP_ROWS;
    const half *shmem_warp_stream_ptr = &shmem[0][0] + warp_id * MMA_M * 2 * C_SHMEM_STRIDE;

    const size_t gmem_idx = (block_tile_i * MMA_M + warp_id * MMA_M * 2) * N + block_tile_j * MMA_N;
    const half *src_gmem_warp_stream_ptr = &output[qkv_offset * M * N + gmem_idx];

    uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2];

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

    const half *A_warp_ptr = &input[block_tile_i * MMA_M * K] + BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id;
    // const half *B_warp_ptr = &B[block_tile_j * MMA_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;


    const TW *qB_warp_ptr = weight + qkv_offset * K * N / w_per_int + block_tile_j * MMA_N + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;

    const TW *qZ_warp_ptr = weight_zeros +  qkv_offset * K * N / w_per_int /group_size; //+ block_tile_j * MMA_N / w_per_int  +  BLOCK_COLS / WARPS_PER_BLOCK * warp_id /w_per_int;
    const T *qS_warp_ptr = weight_scales + qkv_offset * K * N /group_size + block_tile_j * MMA_N +  BLOCK_COLS / WARPS_PER_BLOCK * warp_id;


    const size_t A_shmem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    const size_t B_shmem_iters = BLOCK_COLS / (B_COL_STEP_PER_WARP*WARPS_PER_BLOCK);
    // const size_t B_load_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

#pragma unroll
    for (size_t tile_k = 0; tile_k < K_tiles; tile_k += CHUNK_K) {
        size_t A_shmem_idx = BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        int4 *A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                           (lane_id % CHUNK_COPY_LINE_LANES);
        A_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_shmem_iters; ++i) {

            if((void*)A_lane_ptr < (void*)&(input[M*K-1]))
            {
            *((int4 *)&shmem[A_shmem_idx][0] +
              ((lane_id % CHUNK_COPY_LINE_LANES) +
               (A_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
                  CHUNK_COPY_LINE_LANES) = *A_lane_ptr;
            }
            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }



// const half *B_warp_ptr = &B[block_tile_j * MMA_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

//         size_t B_shmem_idx = B_shmem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
//         int4 *B_lane_ptr = (int4 *)(B_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
//                            (lane_id % CHUNK_COPY_LINE_LANES);
//         B_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

// #pragma unroll
//         for (size_t i = 0; i < B_shmem_iters; ++i) {
//             *((int4 *)&shmem[B_shmem_idx][0] + (lane_id % CHUNK_COPY_LINE_LANES)) = *B_lane_ptr;

//             B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
//             B_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
//         }

//         __syncthreads();


        const TW* B_lane_ptr = qB_warp_ptr + lane_id / B_LANES_PER_CHUNK_K + (tile_k * MMA_K  + (lane_id % B_LANES_PER_CHUNK_K)) * N / w_per_int;
        const TW* Z_lane_ptr = qZ_warp_ptr +  (block_tile_j * MMA_N   +  BLOCK_COLS / WARPS_PER_BLOCK * warp_id +
          lane_id / B_LANES_PER_CHUNK_K) / w_per_int +  (tile_k * MMA_K  + (lane_id % B_LANES_PER_CHUNK_K)) * N / group_size / w_per_int;
        const T*  S_lane_ptr = qS_warp_ptr + lane_id / B_LANES_PER_CHUNK_K + (tile_k * MMA_K  + (lane_id % B_LANES_PER_CHUNK_K)) * N / group_size;

        {

            size_t B_shmem_idx = B_shmem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id + lane_id % B_LANES_PER_CHUNK_K;
            size_t off = (lane_id / B_LANES_PER_CHUNK_K);
            size_t  shift_z = (block_tile_j * MMA_N   +  BLOCK_COLS / WARPS_PER_BLOCK * warp_id +
                                lane_id / B_LANES_PER_CHUNK_K) % w_per_int;
            size_t shift_w = 0;
            TW qw = *B_lane_ptr;
            TW qz = *Z_lane_ptr;
            T qs = *S_lane_ptr;
            int32_t z = (qz >> (shift_z * w_bits)) & w_mask + 1;
            int32_t w0 = ((qw >> ((shift_w  + 0) * w_bits)) & w_mask);
            int32_t w1 = (qw >> ((shift_w  + 1) * w_bits)) & w_mask;
            int32_t w2 = (qw >> ((shift_w  + 2) * w_bits)) & w_mask;
            int32_t w3 = (qw >> ((shift_w  + 3) * w_bits)) & w_mask;
            int32_t w4 = (qw >> ((shift_w  + 4) * w_bits)) & w_mask;
            int32_t w5 = (qw >> ((shift_w  + 5) * w_bits)) & w_mask;
            int32_t w6 = (qw >> ((shift_w  + 6) * w_bits)) & w_mask;
            int32_t w7 = (qw >> ((shift_w  + 7) * w_bits)) & w_mask;
            w0 -= z;
            w1 -= z;
            w2 -= z;
            w3 -= z; 
            w4 -= z;
            w5 -= z;
            w6 -= z;
            w7 -= z;
            
            shmem[B_shmem_idx][off + 0] = conversion::to<T>(w0) * qs;
            shmem[B_shmem_idx][off + 1] = conversion::to<T>(w1) * qs;
            shmem[B_shmem_idx][off + 2] = conversion::to<T>(w2) * qs;
            shmem[B_shmem_idx][off + 3] = conversion::to<T>(w3) * qs;
            shmem[B_shmem_idx][off + 4] = conversion::to<T>(w4) * qs;
            shmem[B_shmem_idx][off + 5] = conversion::to<T>(w5) * qs;
            shmem[B_shmem_idx][off + 6] = conversion::to<T>(w6) * qs;
            shmem[B_shmem_idx][off + 7] = conversion::to<T>(w7) * qs;

        }

        __syncthreads();

#pragma unroll
        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
            uint32_t RA[WARP_COL_TILES][4];
            uint32_t RB[WARP_ROW_TILES][2];

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
                size_t A_shmem_idx = (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
                uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(
                    &shmem[A_shmem_idx + lane_id % 16]
                          [(k_step * MMA_K + (lane_id / 16) * 8 +
                            (lane_id % 16 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                           AB_SHMEM_STRIDE]);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], A_shmem_lane_addr);
            }

#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t B_shmem_idx = B_shmem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
                uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(
                    &shmem[B_shmem_idx + lane_id % 8]
                          [(k_step * MMA_K + ((lane_id / 8) % 2) * 8 +
                            (lane_id % 8 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                           AB_SHMEM_STRIDE]);

                LDMATRIX_X2(RB[j][0], RB[j][1], B_shmem_lane_addr);
            }

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
                for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                    size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;


                    HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j_s][0],
                              RB[j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
                }
            }
        }

        __syncthreads();
    }

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
    for (size_t i = 0; i < M_size; ++i) {

        *((int4 *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =
            *((int4 *)(shmem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SHMEM_STRIDE) +
              (lane_id % 16 + (i * 2 + lane_id / 16) % 8) % (C_SHMEM_STRIDE * sizeof(half) / THREAD_COPY_BYTES));
    }

    __syncthreads();

}




__global__ void mmaAsyncKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                               size_t N, size_t K) {
    const size_t M_tiles = div_ceil(M, MMA_M);
    const size_t N_tiles = div_ceil(N, MMA_N);
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t block_tile_i =
        (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * BLOCK_COL_TILES) : (blockIdx.y * BLOCK_COL_TILES);
    const size_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;

    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    extern __shared__ half shmem[][AB_SHMEM_STRIDE];

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    const size_t B_shmem_idx_off = BLOCK_ROWS;

    half *shmem_warp_tile_row_ptr = &shmem[0][0] + (warp_id / BLOCK_ROW_WARPS) * C_SHMEM_STRIDE * WARP_ROWS;
    const half *shmem_warp_stream_ptr = &shmem[0][0] + warp_id * MMA_M * 2 * C_SHMEM_STRIDE;

    const size_t gmem_idx = (block_tile_i * MMA_M + warp_id * MMA_M * 2) * N + block_tile_j * MMA_N;
    const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

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
    const half *B_warp_ptr = &B[block_tile_j * MMA_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

    const int A_shmem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    const int B_shmem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

#pragma unroll
    for (size_t tile_k = 0; tile_k < K_tiles; tile_k += CHUNK_K) {
        size_t A_shmem_idx = BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        int4 *A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                           (lane_id % CHUNK_COPY_LINE_LANES);
        A_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_shmem_iters; ++i) {
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

        size_t B_shmem_idx = B_shmem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        int4 *B_lane_ptr = (int4 *)(B_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                           (lane_id % CHUNK_COPY_LINE_LANES);
        B_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

// #pragma unroll
//         for (size_t i = 0; i < B_shmem_iters; ++i) {
//             uint32_t B_shmem_lane_addr =
//                 __cvta_generic_to_shared(&shmem[B_shmem_idx][0]) +
//                 ((lane_id % CHUNK_COPY_LINE_LANES +
//                   (B_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
//                  CHUNK_COPY_LINE_LANES) *
//                     THREAD_COPY_BYTES;

//             CP_ASYNC_CG(B_shmem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

//             B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
//             B_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
//         }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();

#pragma unroll
        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
            uint32_t RA[WARP_COL_TILES][4];
            uint32_t RB[WARP_ROW_TILES][2];

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
                size_t A_shmem_idx = (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
                uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(
                    &shmem[A_shmem_idx + lane_id % 16]
                          [(k_step * MMA_K + (lane_id / 16) * 8 +
                            (lane_id % 16 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                           AB_SHMEM_STRIDE]);

                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], A_shmem_lane_addr);
            }

#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t B_shmem_idx = B_shmem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
                uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(
                    &shmem[B_shmem_idx + lane_id % 8]
                          [(k_step * MMA_K + ((lane_id / 8) % 2) * 8 +
                            (lane_id % 8 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                           AB_SHMEM_STRIDE]);

                LDMATRIX_X2(RB[j][0], RB[j][1], B_shmem_lane_addr);
            }

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
                for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                    size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                    HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j_s][0],
                              RB[j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
                }
            }
        }

        __syncthreads();
    }

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
        *((int4 *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =
            *((int4 *)(shmem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SHMEM_STRIDE) +
              (lane_id % 16 + (i * 2 + lane_id / 16) % 8) % (C_SHMEM_STRIDE * sizeof(half) / THREAD_COPY_BYTES));
    }

    __syncthreads();
}



// __global__ static void gptq_bgemm_final(T* input,
//                             TW* weight,
//                             T* weight_scales,
//                             TW* weight_zeros,
//                             T* bias,
//                             T* residual_ptr,
//                             T* output,
//                             size_t M, 
//                             size_t N, 
//                             size_t K,
//                             uint64_t group_size,
//                             int32_t act_type,
//                             bool add_bias,
//                             bool add_residual,
//                             bool qkv_fused
//                             )

__global__ void mmaAsyncStage4Kernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
                                     size_t M, size_t N, size_t K) {
    const size_t M_tiles = div_ceil(M, MMA_M);
    const size_t N_tiles = div_ceil(N, MMA_N);
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t block_tile_i =
        (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * BLOCK_COL_TILES) : (blockIdx.y * BLOCK_COL_TILES);
    const size_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;

    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    extern __shared__ half shmem[][AB_SHMEM_STRIDE];

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    // const size_t B_shmem_idx_off = BLOCK_ROWS;
    // const size_t shmem_stage_off = BLOCK_ROWS + BLOCK_COLS;

    const size_t B_shmem_idx_off = K_STAGE * BLOCK_ROWS;
    const size_t A_shmem_stage_off = BLOCK_ROWS;
    const size_t B_shmem_stage_off = BLOCK_COLS;


    half *shmem_warp_tile_row_ptr = &shmem[0][0] + (warp_id / BLOCK_ROW_WARPS) * C_SHMEM_STRIDE * WARP_ROWS;
    const half *shmem_warp_stream_ptr = &shmem[0][0] + warp_id * MMA_M * 2 * C_SHMEM_STRIDE;

    const size_t gmem_idx = (block_tile_i * MMA_M + warp_id * MMA_M * 2) * N + block_tile_j * MMA_N;
    const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

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
    const half *B_warp_ptr = &B[block_tile_j * MMA_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

    const size_t A_shmem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    const size_t B_shmem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

    size_t shmem_store_idx = 0;
    size_t shmem_load_idx = 0;

    // size_t shmem_store_off = 0;
    // size_t shmem_load_off = 0;

    size_t A_shmem_store_off = 0;
    size_t A_shmem_load_off = 0;

    size_t B_shmem_store_off = 0;
    size_t B_shmem_load_off = 0;

    size_t A_shmem_idx = 0;
    int4 *A_lane_ptr = nullptr;

    size_t B_shmem_idx = 0;
    int4 *B_lane_ptr = nullptr;

    A_shmem_idx = A_shmem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    A_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_shmem_iters; ++i) {
        uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(&shmem[A_shmem_idx][0]) +
                                     ((lane_id % CHUNK_COPY_LINE_LANES +
                                       (A_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
                                      CHUNK_COPY_LINE_LANES) *
                                         THREAD_COPY_BYTES;

        CP_ASYNC_CG(A_shmem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

        A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    B_shmem_idx = B_shmem_store_off + B_shmem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4 *)(B_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    B_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < B_shmem_iters; ++i) {
        uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(&shmem[B_shmem_idx][0]) +
                                     ((lane_id % CHUNK_COPY_LINE_LANES +
                                       (B_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
                                      CHUNK_COPY_LINE_LANES) *
                                         THREAD_COPY_BYTES;

        CP_ASYNC_CG(B_shmem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

        B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    CP_ASYNC_COMMIT_GROUP();

    shmem_store_idx = (shmem_store_idx + 1) % K_STAGE;
    A_shmem_store_off = shmem_store_idx * A_shmem_stage_off;
    B_shmem_store_off = shmem_store_idx * B_shmem_stage_off;

    A_shmem_idx = A_shmem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    A_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_shmem_iters; ++i) {
        uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(&shmem[A_shmem_idx][0]) +
                                     ((lane_id % CHUNK_COPY_LINE_LANES +
                                       (A_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
                                      CHUNK_COPY_LINE_LANES) *
                                         THREAD_COPY_BYTES;

        CP_ASYNC_CG(A_shmem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

        A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    B_shmem_idx = B_shmem_store_off + B_shmem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4 *)(B_warp_ptr + CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    B_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < B_shmem_iters; ++i) {
        uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(&shmem[B_shmem_idx][0]) +
                                     ((lane_id % CHUNK_COPY_LINE_LANES +
                                       (B_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
                                      CHUNK_COPY_LINE_LANES) *
                                         THREAD_COPY_BYTES;

        CP_ASYNC_CG(B_shmem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

        B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    CP_ASYNC_COMMIT_GROUP();

    shmem_store_idx = (shmem_store_idx + 1) % K_STAGE;
    A_shmem_store_off = shmem_store_idx * A_shmem_stage_off;
    B_shmem_store_off = shmem_store_idx * B_shmem_stage_off;

    A_shmem_idx = A_shmem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + 2 * CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    A_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_shmem_iters; ++i) {
        uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(&shmem[A_shmem_idx][0]) +
                                     ((lane_id % CHUNK_COPY_LINE_LANES +
                                       (A_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
                                      CHUNK_COPY_LINE_LANES) *
                                         THREAD_COPY_BYTES;

        CP_ASYNC_CG(A_shmem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

        A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    B_shmem_idx = B_shmem_store_off + B_shmem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4 *)(B_warp_ptr + 2 * CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    B_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < B_shmem_iters; ++i) {
        uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(&shmem[B_shmem_idx][0]) +
                                     ((lane_id % CHUNK_COPY_LINE_LANES +
                                       (B_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
                                      CHUNK_COPY_LINE_LANES) *
                                         THREAD_COPY_BYTES;

        CP_ASYNC_CG(B_shmem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

        B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
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
        size_t B_shmem_idx = B_shmem_load_off + B_shmem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
        uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(
            &shmem[B_shmem_idx + lane_id % 8]
                  [(((lane_id / 8) % 2) * 8 +
                    (lane_id % 8 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                   AB_SHMEM_STRIDE]);

        LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_shmem_lane_addr);
    }

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
            size_t B_shmem_idx = B_shmem_load_off + B_shmem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(
                &shmem[B_shmem_idx + lane_id % 8]
                      [(MMA_K + ((lane_id / 8) % 2) * 8 +
                        (lane_id % 8 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                       AB_SHMEM_STRIDE]);

            LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_shmem_lane_addr);
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

        shmem_store_idx = (shmem_store_idx + 1) % K_STAGE;
        A_shmem_store_off = shmem_store_idx * A_shmem_stage_off;
        B_shmem_store_off = shmem_store_idx * B_shmem_stage_off;

        A_shmem_idx = A_shmem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        A_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_shmem_iters / CHUNK_K; ++i) {
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

        B_shmem_idx = shmem_store_off + B_shmem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        B_lane_ptr = (int4 *)(B_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        B_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < B_shmem_iters / CHUNK_K; ++i) {
            uint32_t B_shmem_lane_addr =
                __cvta_generic_to_shared(&shmem[B_shmem_idx][0]) +
                ((lane_id % CHUNK_COPY_LINE_LANES +
                  (B_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
                 CHUNK_COPY_LINE_LANES) *
                    THREAD_COPY_BYTES;

            CP_ASYNC_CG(B_shmem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        shmem_load_idx = (shmem_load_idx + 1) % K_STAGE;
        A_shmem_load_off = shmem_load_idx * A_shmem_stage_off;
        B_shmem_load_off = shmem_load_idx * B_shmem_stage_off;

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

#pragma unroll
        for (size_t i = (CHUNK_K - 1) * B_shmem_iters / CHUNK_K; i < B_shmem_iters; ++i) {
            uint32_t B_shmem_lane_addr =
                __cvta_generic_to_shared(&shmem[B_shmem_idx][0]) +
                ((lane_id % CHUNK_COPY_LINE_LANES +
                  (B_shmem_idx % (CHUNK_COPY_LINE_LANES * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS) %
                 CHUNK_COPY_LINE_LANES) *
                    THREAD_COPY_BYTES;

            CP_ASYNC_CG(B_shmem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(2);

        __syncthreads();

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
            size_t B_shmem_idx = B_shmem_load_off + B_shmem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(
                &shmem[B_shmem_idx + lane_id % 8]
                      [(((lane_id / 8) % 2) * 8 +
                        (lane_id % 8 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                       AB_SHMEM_STRIDE]);

            LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_shmem_lane_addr);
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
            size_t B_shmem_idx = B_shmem_load_off + B_shmem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(
                &shmem[B_shmem_idx + lane_id % 8]
                      [(((k_step + 1) % CHUNK_K) * MMA_K + ((lane_id / 8) % 2) * 8 +
                        (lane_id % 8 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                       AB_SHMEM_STRIDE]);

            LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_shmem_lane_addr);
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
            B_shmem_load_off = shmem_load_idx * B_shmem_stage_off;

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
            size_t B_shmem_idx = B_shmem_load_off + B_shmem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(
                &shmem[B_shmem_idx + lane_id % 8]
                      [(((k_step + 1) % CHUNK_K) * MMA_K + ((lane_id / 8) % 2) * 8 +
                        (lane_id % 8 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                       AB_SHMEM_STRIDE]);

            LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_shmem_lane_addr);
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
            B_shmem_load_off = shmem_load_idx * B_shmem_stage_off;

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
            size_t B_shmem_idx = B_shmem_load_off + B_shmem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(
                &shmem[B_shmem_idx + lane_id % 8]
                      [(k_step * MMA_K + ((lane_id / 8) % 2) * 8 +
                        (lane_id % 8 % (PERMUTED_COLS * SHMEM_BANK_ROWS)) / SHMEM_BANK_ROWS * PERMUTED_OFFSET) %
                       AB_SHMEM_STRIDE]);

            LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_shmem_lane_addr);
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
        *((int4 *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =
            *((int4 *)(shmem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SHMEM_STRIDE) +
              (lane_id % 16 + (i * 2 + lane_id / 16) % 8) % (C_SHMEM_STRIDE * sizeof(half) / THREAD_COPY_BYTES));
    }

    __syncthreads();
}	
static size_t initMmaAsyncStage4() {
    int dev_id = 0;
    cudaGetDevice(&dev_id);
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, dev_id);

    size_t shmem_max_size = std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SHMEM_STRIDE * sizeof(half) * K_STAGE,
                                     BLOCK_ROWS * C_SHMEM_STRIDE * sizeof(half));
    printf("shmem_max_size: %.0f KBytes (%zu Bytes)\n", static_cast<float>(shmem_max_size / 1024.0f), shmem_max_size);

    if(dev_prop.sharedMemPerMultiprocessor < shmem_max_size)
    {
        printf("Error: CUDA shared memory size < %zu\n", shmem_max_size);
    }

    cudaFuncSetAttribute(gptq_bgemm_v3<__half, uint64_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_max_size);

    return shmem_max_size;
}
