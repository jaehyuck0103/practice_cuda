/*
  Perform a fast box filter using the sliding window method.

  As the kernel moves from left to right, we add in the contribution of the
  new sample on the right, and subtract the value of the exiting sample on the
  left. This only requires 2 adds and a mul per output value, independent of the
  filter radius. The box filter is separable, so to perform a 2D box filter we
  perform the filter in the x direction, followed by the same filter in the y
  direction. Applying multiple iterations of the box filter converges towards a
  Gaussian blur. Using CUDA, rows or columns of the image are processed in
  parallel. This version duplicates edge pixels.

  Note that the x (row) pass suffers from uncoalesced global memory reads,
  since each thread is reading from a different row. For this reason it is
  better to use texture lookups for the x pass.
  The y (column) pass is perfectly coalesced.

  Parameters
  id - pointer to input data in global memory
  od - pointer to output data in global memory
  w  - image width
  h  - image height
  r  - filter radius

  e.g. for r = 2, w = 8:

  0 1 2 3 4 5 6 7
  x - -
  - x - -
  - - x - -
    - - x - -
      - - x - -
        - - x - -
          - - x -
            - - x
*/

// process row
__device__ void d_boxfilter_x(float *id, float *od, int w, int h, int r) {
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
    t = id[0] * r;

    for (int x = 0; x < (r + 1); x++) {
        t += id[x];
    }

    od[0] = t * scale;

    for (int x = 1; x < (r + 1); x++) {
        t += id[x + r];
        t -= id[0];
        od[x] = t * scale;
    }

    // main loop
    for (int x = (r + 1); x < w - r; x++) {
        t += id[x + r];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }

    // do right edge
    for (int x = w - r; x < w; x++) {
        t += id[w - 1];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }
}

// process column
__device__ void d_boxfilter_y(float *id, float *od, int w, int h, int r) {
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
    t = id[0] * r;

    for (int y = 0; y < (r + 1); y++) {
        t += id[y * w];
    }

    od[0] = t * scale;

    for (int y = 1; y < (r + 1); y++) {
        t += id[(y + r) * w];
        t -= id[0];
        od[y * w] = t * scale;
    }

    // main loop
    for (int y = (r + 1); y < (h - r); y++) {
        t += id[(y + r) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }

    // do right edge
    for (int y = h - r; y < h; y++) {
        t += id[(h - 1) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }
}

__global__ void d_boxfilter_x_global(float *id, float *od, int w, int h, int r) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (y < h) {
        d_boxfilter_x(&id[batch_idx * h * w + y * w], &od[batch_idx * h * w + y * w], w, h, r);
    }
}

__global__ void d_boxfilter_y_global(float *id, float *od, int w, int h, int r) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (x < w) {
        d_boxfilter_y(&id[batch_idx * h * w + x], &od[batch_idx * h * w + x], w, h, r);
    }
}

/*
    Perform 2D box filter on image using CUDA

    Parameters:
    d_src  - pointer to input image in device memory
    d_temp - pointer to temporary storage in device memory
    d_dest - pointer to destination image in device memory
    width  - image width
    height - image height
    radius - filter radius
    iterations - number of iterations

*/
void boxFilterLarge(
    float *d_src,
    float *d_temp,
    float *d_dest,
    int B,
    int H,
    int W,
    int radius,
    int nthreads) {

    const dim3 dim_grid_x((H + nthreads - 1) / nthreads, B);
    d_boxfilter_x_global<<<dim_grid_x, nthreads, 0>>>(d_src, d_temp, W, H, radius);

    const dim3 dim_grid_y((W + nthreads - 1) / nthreads, B);
    d_boxfilter_y_global<<<dim_grid_y, nthreads, 0>>>(d_temp, d_dest, W, H, radius);
}

__global__ void boxFilterSmallKernel(float *in, float *out, int radius, int W, int H) {

    int xx = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;
    int batch_offset = batch_idx * H * W;

    if (xx < W && yy < H) {
        float sum = 0;
        float cnt = 0;

        for (int box_y = yy - radius; box_y < yy + radius + 1; ++box_y) {
            for (int box_x = xx - radius; box_x < xx + radius + 1; ++box_x) {
                if (box_y >= 0 && box_y < H && box_x >= 0 && box_x < W) {
                    sum += in[batch_offset + box_y * W + box_x];
                    cnt += 1;
                }
            }
        }
        out[batch_offset + yy * W + xx] = sum / cnt;
    }
}

void boxFilterSmall(float *d_src, float *d_dest, int B, int H, int W, int radius) {

    dim3 dim_grid((W + 15) / 16, (H + 15) / 16, B);
    dim3 dim_block(16, 16, 1);

    boxFilterSmallKernel<<<dim_grid, dim_block>>>(d_src, d_dest, radius, W, H);
}

__global__ void tiledBoxFilterKernel(float *in, float *out, int radius, int W, int H) {

    int xx = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;
    int batch_offset = batch_idx * H * W;

    __shared__ float tile[100][100];
    int tile_W = blockDim.x + radius + radius;
    int tile_H = blockDim.y + radius + radius;
    int tile_size = tile_W * tile_H;
    int num_threads = blockDim.x * blockDim.y;
    int iters = (tile_size + num_threads - 1) / num_threads;
    int thread_id = blockDim.x * threadIdx.y + threadIdx.x;
    for (int i = 0; i < iters; ++i) {
        int tile_idx = i * num_threads + thread_id;
        if (tile_idx >= tile_size) {
            break;
        }
        int tile_x = tile_idx % tile_H;
        int tile_y = tile_idx / tile_H;
        int tile_xx = min(max(blockIdx.x * blockDim.x - radius + tile_x, 0), W - 1);
        int tile_yy = min(max(blockIdx.y * blockDim.y - radius + tile_y, 0), W - 1);
        tile[tile_y][tile_x] = in[batch_offset + tile_yy * W + tile_xx];
    }
    __syncthreads();

    if (xx < W && yy < H) {
        float sum = 0;

        for (int box_y = threadIdx.y; box_y < threadIdx.y + radius + radius + 1; ++box_y) {
            for (int box_x = threadIdx.x; box_x < threadIdx.x + radius + radius + 1; ++box_x) {
                sum += tile[box_y][box_x];
            }
        }
        out[batch_offset + yy * W + xx] = sum / ((2 * radius + 1) * (2 * radius + 1));
    }
}

void tiledBoxFilter(float *d_src, float *d_dest, int B, int H, int W, int radius) {

    dim3 dim_grid((W + 31) / 32, (H + 31) / 32, B);
    dim3 dim_block(32, 32, 1);

    tiledBoxFilterKernel<<<dim_grid, dim_block>>>(d_src, d_dest, radius, W, H);
}
