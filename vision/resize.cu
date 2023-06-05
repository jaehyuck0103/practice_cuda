__global__ void
subsampleKernel(float *in, float *out, int src_W, int src_H, int dst_W, int dst_H) {

    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;

    float scale_x = float(src_W) / dst_W;
    int src_x = (dst_x + 0.499f) * scale_x;

    float scale_y = float(src_H) / dst_H;
    int src_y = (dst_y + 0.499f) * scale_y;

    if (dst_x < dst_W && dst_y < dst_H) {

        int dst_offset = batch_idx * dst_H * dst_W + dst_y * dst_W + dst_x;
        int src_offset = batch_idx * src_H * src_W + src_y * src_W + src_x;

        out[dst_offset] = in[src_offset];
    }
}

void subsample(float *d_src, float *d_dest, int W, int H, int B) {

    dim3 dim_grid(ceil(W / 4.0 / 16.0), ceil(H / 4.0 / 16.0), B);
    dim3 dim_block(16, 16, 1);
    subsampleKernel<<<dim_grid, dim_block>>>(d_src, d_dest, W, H, W / 4, H / 4);
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
        int x1 = min(int(src_x + 0.5f), src_W - 1);
        int y0 = max(int(src_y), 0);
        int y1 = min(int(src_y + 0.5f), src_H - 1);

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

void resizeBilinear(float *d_src, float *d_dest, int W, int H, int B) {
    dim3 dim_grid(ceil(W / 16.0), ceil(H / 16.0), B);
    dim3 dim_block(16, 16, 1);
    resizeBilinearKernel<<<dim_grid, dim_block>>>(d_src, d_dest, W / 4, H / 4, W, H);
}

void resizeBilinear2(float *d_src, float *d_dest, int W, int H, int batch_size) {
    dim3 dim_grid(ceil(W / 4.0 / 16.0), ceil(H / 4.0 / 16.0), batch_size);
    dim3 dim_block(16, 16, 1);
    resizeBilinearKernel<<<dim_grid, dim_block>>>(d_src, d_dest, W, H, W / 4, H / 4);
}
