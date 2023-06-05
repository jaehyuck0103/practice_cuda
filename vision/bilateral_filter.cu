__constant__ float cGaussian[64]; // gaussian array in device side

/*
    Perform a simple bilateral filter.

    Bilateral filter is a nonlinear filter that is a mixture of range
    filter and domain filter, the previous one preserves crisp edges and
    the latter one filters noise. The intensity value at each pixel in
    an image is replaced by a weighted average of intensity values from
    nearby pixels.

    The weight factor is calculated by the product of domain filter
    component(using the gaussian distribution as a spatial distance) as
    well as range filter component(Euclidean distance between center pixel
    and the current neighbor pixel). Because this process is nonlinear,
    the sample just uses a simple pixel by pixel step.

    Texture fetches automatically clamp to edge of image. 1D gaussian array
    is mapped to a 1D texture instead of using shared memory, which may
    cause severe bank conflict.

    Threads are y-pass(column-pass), because the output is coalesced.

    Parameters
    od - pointer to output data in global memory
    d_f - pointer to the 1D gaussian array
    e_d - euclidean delta
    w  - image width
    h  - image height
    r  - filter radius
*/

// Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float a, float b, float d) {
    float mod = (b - a) * (b - a);

    return __expf(-mod / (2.f * d * d));
}

// column pass using coalesced global memory reads
__global__ void d_bilateral_filter(float *in, float *out, int w, int h, float e_d, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = y * w + x;

    if (x >= w || y >= h) {
        return;
    }

    float sum = 0.0f;
    float factor;
    float t = 0.f;
    float center = in[offset];

    for (int j = -r; j <= r; j++) {
        for (int i = -r; i <= r; i++) {
            int xx = x + i;
            int yy = y + j;

            if (xx >= 0 && xx < w && yy >= 0 && yy < h) {

                float curPix = in[yy * w + xx];
                factor = cGaussian[i + r] * cGaussian[j + r] * // domain factor
                         euclideanLen(curPix, center, e_d);    // range factor

                t += factor * curPix;
                sum += factor;
            }
        }
    }

    out[y * w + x] = t / sum;
}

/*
    Because a 2D gaussian mask is symmetry in row and column,
    here only generate a 1D mask, and use the product by row
    and column index later.

    1D gaussian distribution :
        g(x, d) -- C * exp(-x^2/d^2), C is a constant amplifier

    parameters:
    og - output gaussian array in global memory
    delta - the 2nd parameter 'd' in the above function
    radius - half of the filter size
             (total filter size = 2 * radius + 1)
*/
void updateGaussian(float delta, int radius) {
    float fGaussian[64];

    for (int i = 0; i < 2 * radius + 1; ++i) {
        float x = (float)(i - radius);
        fGaussian[i] = expf(-(x * x) / (2 * delta * delta));
    }

    cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float) * (2 * radius + 1));
}

/*
    Perform 2D bilateral filter on image using CUDA

    Parameters:
    d_dest - pointer to destination image in device memory
    width  - image width
    height - image height
    e_d    - euclidean delta
    radius - filter radius
    iterations - number of iterations
*/

void bilateralFilter(float *in, float *out, int width, int height, float e_d, int radius) {

    dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
    dim3 blockSize(16, 16);

    d_bilateral_filter<<<gridSize, blockSize>>>(in, out, width, height, e_d, radius);
}
