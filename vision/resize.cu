__global__ void
resizeNearestKernel(float *in, float *out, int src_W, int src_H, int dst_W, int dst_H) {

    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;

    int src_x = __float2ll_rn((dst_x + 0.5f) * src_W / dst_W - 0.5f);
    int src_y = __float2ll_rn((dst_y + 0.5f) * src_H / dst_H - 0.5f);
    src_x = min(max(src_x, 0), src_W - 1);
    src_y = min(max(src_y, 0), src_H - 1);

    if (dst_x < dst_W && dst_y < dst_H) {

        int dst_offset = batch_idx * dst_H * dst_W + dst_y * dst_W + dst_x;
        int src_offset = batch_idx * src_H * src_W + src_y * src_W + src_x;

        out[dst_offset] = in[src_offset];
    }
}

void resizeNearest(float *d_src, float *d_dst, int B, int src_H, int src_W, int dst_H, int dst_W) {

    dim3 dim_grid((dst_W + 15) / 16, (dst_H + 15) / 16, B);
    dim3 dim_block(16, 16, 1);
    resizeNearestKernel<<<dim_grid, dim_block>>>(d_src, d_dst, src_W, src_H, dst_W, dst_H);
}

__global__ void
resizeBilinearKernel(float *in, float *out, int src_W, int src_H, int dst_W, int dst_H) {

    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;

    if (dst_x < dst_W && dst_y < dst_H) {

        float src_x = (dst_x + 0.5f) * src_W / dst_W - 0.5f;
        float src_y = (dst_y + 0.5f) * src_H / dst_H - 0.5f;

        int x0 = max(int(src_x), 0);
        int x1 = min(int(src_x + 1.0f), src_W - 1);
        int y0 = max(int(src_y), 0);
        int y1 = min(int(src_y + 1.0f), src_H - 1);

        int batch_offset_src = batch_idx * src_H * src_W;
        float f00 = in[batch_offset_src + y0 * src_W + x0];
        float f10 = in[batch_offset_src + y0 * src_W + x1];
        float f01 = in[batch_offset_src + y1 * src_W + x0];
        float f11 = in[batch_offset_src + y1 * src_W + x1];

        float wx = src_x - floor(src_x);
        float wy = src_y - floor(src_y);

        int dst_offset = batch_idx * dst_H * dst_W + dst_y * dst_W + dst_x;
        out[dst_offset] =
            f00 * (1 - wx) * (1 - wy) + f10 * wx * (1 - wy) + f01 * (1 - wx) * wy + f11 * wx * wy;
    }
}

void resizeBilinear(
    float *d_src,
    float *d_dst,
    int B,
    int src_H,
    int src_W,
    int dst_H,
    int dst_W) {

    dim3 dim_grid((dst_W + 15) / 16, (dst_H + 15) / 16, B);
    dim3 dim_block(16, 16, 1);
    resizeBilinearKernel<<<dim_grid, dim_block>>>(d_src, d_dst, src_W, src_H, dst_W, dst_H);
}
