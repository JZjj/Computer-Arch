#include "helper.hpp"
const int M          = 32; // the height of the output matrix
const int N          = 32; // the width of the output matrix
const int K          = 4;  // the length of the intermediate dimension in A*B
const int UNROLL     = 32 / K;
const int BLOCKDIM_Y = M / UNROLL;
const int BLOCKDIM_X = N;
const int BLOCKDIM_Z = 1;

// Sequential code for the forward path of the convolution layer
// You should not modify this code
static void conv_forward_valid(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims) {
  std::fill(Y, Y + ydims.flattened_length(), 0);

  for (auto i : range(0, ydims.num)) {
    for (auto m : range(0, ydims.depth)) {    // for each output feature map
      for (auto h : range(0, ydims.height)) { // for each output element
        for (auto w : range(0, ydims.width)) {
          const auto yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
          for (auto c : range(0, xdims.depth)) {     // sum over all input feature maps
            for (auto p : range(0, wdims.height)) {  // filter height
              for (auto q : range(0, wdims.width)) { // filter width
                const auto xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
                const auto woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Baseline GPU kernel code for forward convolution.
// One thread per output index
// You should not modify this kernel as it is used for correctness comparison.
// Instead, define a new one below
__global__ void conv_forward_baseline_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y,
                                             const shape ydims) {

  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = gx; i < ydims.num * ydims.depth * ydims.height * ydims.width; i += blockDim.x * gridDim.x) {
    Y[i] = 0.f;
  }

  for (size_t i = gx; i < ydims.num; i += gridDim.x * blockDim.x) {
    for (auto m : range(0, ydims.depth)) {    // for each output feature map
      for (auto h : range(0, ydims.height)) { // for each output element
        for (auto w : range(0, ydims.width)) {
          const size_t yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
          for (auto c : range(0, xdims.depth)) {     // sum over all input feature maps
            for (auto p : range(0, wdims.height)) {  // filter height
              for (auto q : range(0, wdims.width)) { // filter width
                const size_t xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
                const size_t woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_baseline(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims) {

  dim3 dimGrid(1);
  dim3 dimBlock(32);

  conv_forward_baseline_kernel<<<dimGrid, dimBlock>>>(X, xdims, W, wdims, Y, ydims);
  THROW_IF_ERROR(cudaGetLastError());
}

// Implement your optimized kernel here.
// Make any modifications you wish.
// Don't forget to modify the host code below, if needed!
__global__ void conv_forward_opt_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y, const shape ydims) {

  // X[b, c, h+p, w+q] = X[((b * xdims.depth + c) * xdims.height + (h + p)) * xdims.width + (w + q)]
  // W[m, c, p, q] = W[((m * wdims.depth + c) * wdims.height + p) * wdims.width + q]
  // Y[b, m, h, w] = Y[((b * ydims.depth + m) * ydims.height + h) * ydims.width + w]
  const int b     = ydims.num;
  const int wh    = wdims.height;
  const int ww    = wdims.width;
  const int h_in  = ydims.height + wh - 1;
  const int w_in  = ydims.width + ww - 1;
  const int c     = xdims.depth;
  const int m     = ydims.depth;
  const int h_out = h_in - wh + 1;
  const int w_out = w_in - ww + 1;

  int idxXBlockCol = blockIdx.x * m + threadIdx.x;
  int idxM         = blockIdx.y * N + threadIdx.y;

  int numXBlockRows = c * wh * ww;
  int numXBlockCols = h_out * w_out;

  __shared__ float blockMemA[M][K];
  __shared__ float blockMemB[K][N];

  int idxB    = blockIdx.z * BLOCKDIM_Z + threadIdx.z;
  int idxHOut = idxXBlockCol / w_out;
  int idxWOut = idxXBlockCol % w_out;

  if (idxB < b) {
    float PValue[UNROLL];
    for (int unroll = 0; unroll < UNROLL; unroll++) {
      PValue[unroll] = 0.0;
    }

    for (int idx = 0; idx < (numXBlockRows + K - 1) / K; idx++) {
// simplify the matrix indexing
#define Y4d(i3, i2, i1, i0) Y[(i3) * (m * h_out * w_out) + (i2) * (h_out * w_out) + (i1) * (w_out) + i0]
#define W4d(i3, i2, i1, i0) X[(i3) * (c * h_in * w_in) + (i2) * (h_in * w_in) + (i1) * (w_in) + i0]
#define X4d(i3, i2, i1, i0) W[(i3) * (c * wh * ww) + (i2) * (wh * ww) + (i1) * (ww) + i0]

      const int threadIdx1D      = threadIdx.y * BLOCKDIM_X + threadIdx.x;
      const int threadNumInBlock = BLOCKDIM_X * BLOCKDIM_Y;
      int max                    = (M * K + threadNumInBlock - 1) / threadNumInBlock;
      if (((K * N + threadNumInBlock - 1) / threadNumInBlock) > max)
        max = (K * N + threadNumInBlock - 1) / threadNumInBlock;
      for (int idxLoad = 0; idxLoad < max; idxLoad++) {
        int idxYBlockMemALoad = (threadIdx1D + idxLoad * threadNumInBlock) / K;
        int idxXBlockMemALoad = (threadIdx1D + idxLoad * threadNumInBlock) % K;
        int idxYBlockMemBLoad = (threadIdx1D + idxLoad * threadNumInBlock) / N;
        int idxXBlockMemBLoad = (threadIdx1D + idxLoad * threadNumInBlock) % N;

        int idxWColumnToLoad   = idx * K + idxXBlockMemALoad;
        int idxXBlockColToLoad = blockIdx.x * N + idxXBlockMemBLoad;
        int idxXBlockRowToLoad = (idxYBlockMemBLoad + idx * K);

        int idxHOutToLoad = idxXBlockColToLoad / w_out;
        int idxWOutToLoad = idxXBlockColToLoad % w_out;

        int idxCWColumn   = idxWColumnToLoad / (wh * ww);
        int idxMToLoad    = blockIdx.y * M + idxYBlockMemALoad;
        int idxCXBlockRow = idxXBlockRowToLoad / (wh * ww);
        if ((idxCWColumn < c) && (idxMToLoad < m)) {
          int idxKHeightWColumn                           = (idxWColumnToLoad % (wh * ww)) / ww;
          int idxKWidthWColumn                            = (idxWColumnToLoad % (wh * ww)) % ww;
          blockMemA[idxYBlockMemALoad][idxXBlockMemALoad] = X4d(idxMToLoad, idxCWColumn, idxKHeightWColumn, idxKWidthWColumn);
        } else {
          blockMemA[idxYBlockMemALoad][idxXBlockMemALoad] = 0.0;
        }
        if ((idxCXBlockRow < c) && (idxHOutToLoad < h_out)) {
          int idxKHeightXBlockRow = (idxXBlockRowToLoad % (wh * ww)) / ww;
          int idxKWidthXBlockRow  = (idxXBlockRowToLoad % (wh * ww)) % ww;
          blockMemB[idxYBlockMemBLoad][idxXBlockMemBLoad] =
              W4d(idxB, idxCXBlockRow, idxHOutToLoad + idxKHeightXBlockRow, idxWOutToLoad + idxKWidthXBlockRow);
        } else {
          blockMemB[idxYBlockMemBLoad][idxXBlockMemBLoad] = 0.0;
        }
      }
      __syncthreads();
      for (int idx2 = 0; idx2 < K; idx2++) {
        for (int unroll = 0; unroll < UNROLL; unroll++) {
          PValue[unroll] += blockMemA[threadIdx.y + unroll * BLOCKDIM_Y][idx2] * blockMemB[idx2][threadIdx.x];
        }
      }
      __syncthreads();
    }
    if (idxM < M && idxXBlockCol < numXBlockCols) {
      for (int unroll = 0; unroll < UNROLL; unroll++) {
        Y4d(idxB, idxM + unroll * BLOCKDIM_Y, idxHOut, idxWOut) = PValue[unroll];
      }
    }
  }
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_opt(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims) {
  // X[b, c, h+p, w+q] = X[((b * xdims.depth + c) * xdims.height + (h + p)) * xdims.width + (w + q)]
  // W[m, c, p, q] = W[((m * wdims.depth + c) * wdims.height + p) * wdims.width + q]
  // Y[b, m, h, w] = Y[((b * ydims.depth + m) * ydims.height + h) * ydims.width + w]
  const int wh      = wdims.height;
  const int ww      = wdims.width;
  const int m       = ydims.depth;
  const int h_in    = ydims.height + wh - 1;
  const int w_in    = ydims.width + ww - 1;
  const int h_out   = h_in - wh + 1;
  const int w_out   = w_in - ww + 1;
  const int b       = ydims.num;
  int numWtiledRows = m;
  int numXBlockCols = h_out * w_out;
  dim3 blockUnrollDimGolden(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
  dim3 gridUnrollDimGolden((numXBlockCols + BLOCKDIM_X - 1) / BLOCKDIM_X,
                           (numWtiledRows + BLOCKDIM_Y * UNROLL - 1) / (BLOCKDIM_Y * UNROLL),
                           (b + BLOCKDIM_Z - 1) / BLOCKDIM_Z);
  THROW_IF_ERROR(cudaMemset(Y, 0, sizeof(float) * ydims.num * ydims.depth * ydims.height * ydims.width));
  conv_forward_opt_kernel<<<gridUnrollDimGolden, blockUnrollDimGolden>>>(X, xdims, W, wdims, Y, ydims);
  THROW_IF_ERROR(cudaGetLastError());
}

static int eval(const shape wDims, const shape xDims, bool doVerify) {

  // Generate model
  const auto conf_info = std::string("conv[wDims:") + std::to_string(wDims.num) + "," + std::to_string(wDims.depth) + "," +
                         std::to_string(wDims.height) + "," + std::to_string(wDims.width) + " xDims:" + std::to_string(xDims.num) + "," +
                         std::to_string(xDims.depth) + "," + std::to_string(xDims.height) + "," + std::to_string(xDims.width) + "]";
  INFO("Running " << conf_info);

  // Generate convolution weights
  float *hostW = allocate<float>(wDims);
  generate_convfilters(hostW, wDims);

  // generate input feature map
  float *hostX = allocate<float>(xDims);
  generate_data(hostX, xDims);

  // generate output feature map for verification
  const shape ydims = {xDims.num, wDims.num, (xDims.height - wDims.height + 1), (xDims.width - wDims.width + 1)};
  INFO("Allocating output tensor [" << ydims.num << "," << ydims.depth << "," << ydims.height << "," << ydims.width << "]");
  float *hostY    = allocate<float>(ydims);
  float *expected = allocate<float>(ydims);
  generate_data(hostY, ydims);

  const size_t wByteCount = wDims.flattened_length() * sizeof(float);
  const size_t xByteCount = xDims.flattened_length() * sizeof(float);
  const size_t yByteCount = ydims.flattened_length() * sizeof(float);

  float *deviceW = nullptr, *deviceX = nullptr, *deviceY = nullptr;
  timer_start("Allocating GPU memory.");
  THROW_IF_ERROR(cudaMalloc((void **) &deviceW, wByteCount));
  THROW_IF_ERROR(cudaMalloc((void **) &deviceX, xByteCount));
  THROW_IF_ERROR(cudaMalloc((void **) &deviceY, yByteCount));
  timer_stop();

  timer_start("Copying inputs to the GPU.");
  THROW_IF_ERROR(cudaMemcpy(deviceW, hostW, wByteCount, cudaMemcpyDefault));
  THROW_IF_ERROR(cudaMemcpy(deviceX, hostX, xByteCount, cudaMemcpyDefault));
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  convlayer_gpu_opt(deviceX, xDims, deviceW, wDims, deviceY, ydims);
  THROW_IF_ERROR(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  if (doVerify) {
    timer_start("Copying output to the CPU");
    THROW_IF_ERROR(cudaMemcpy(hostY, deviceY, yByteCount, cudaMemcpyDefault));
    timer_stop();

    convlayer_gpu_baseline(deviceX, xDims, deviceW, wDims, deviceY, ydims);
    THROW_IF_ERROR(cudaDeviceSynchronize());
    THROW_IF_ERROR(cudaMemcpy(expected, deviceY, yByteCount, cudaMemcpyDefault));
    // conv_forward_valid(hostX, xDims, hostW, wDims, expected, ydims);
    verify(expected, hostY, ydims);
  }

  THROW_IF_ERROR(cudaFree(deviceW));
  THROW_IF_ERROR(cudaFree(deviceX));
  THROW_IF_ERROR(cudaFree(deviceY));
  free(hostW);
  free(hostX);
  free(hostY);
  free(expected);

  return 0;
}

TEST_CASE("Convlayer", "[convlayer]") {
#if 1
  // test five times in case code errors depend on data
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32, 1, 5, 5}, {20, 1, 28, 28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32, 1, 5, 5}, {20, 1, 28, 28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32, 1, 5, 5}, {20, 1, 28, 28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32, 1, 5, 5}, {20, 1, 28, 28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32, 1, 5, 5}, {20, 1, 28, 28}, true);
  }
#else
  SECTION("[wDims:32,1,5,5 xDims:50000,1,28,28]") {
    eval({32, 1, 5, 5}, {50000, 1, 28, 28}, false);
  }
#endif
}

