/*
 * greentea.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: Fabian Tschopp
 */

#include "caffe/common.hpp"
#include "caffe/greentea/greentea.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {

#ifdef USE_GREENTEA

viennacl::ocl::handle<cl_mem> WrapHandle(cl_mem in,
                                         viennacl::ocl::context *ctx) {
  if (in != NULL) {
    viennacl::ocl::handle<cl_mem> memhandle(in, *ctx);
    memhandle.inc();
    return memhandle;
  } else {
    cl_int err;
    cl_mem dummy = clCreateBuffer(ctx->handle().get(), CL_MEM_READ_WRITE, 0,
    NULL,
                                  &err);
    viennacl::ocl::handle<cl_mem> memhandle(dummy, *ctx);
    return memhandle;
  }
}
  
void greentea_malloc(void ** devPtr, int_tp size, int device_id) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(device_id);
  ctx.get_queue().finish();
  cl_int err;
  cl_mem cl_data_ = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE, size, nullptr, &err);
  CHECK_EQ(0, err) << "OpenCL buffer allocation of size " << size << " failed.";
  *devPtr = reinterpret_cast<void *> (cl_data_);
  ctx.get_queue().finish();
}

void greentea_free(void * devPtr, int device_id) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(device_id);
  ctx.get_queue().finish();
  CHECK_EQ(CL_SUCCESS, clReleaseMemObject((cl_mem) devPtr))
    << "OpenCL memory corruption";
  ctx.get_queue().finish();
}
  
#endif // USE_GREENTEA


}  // namespace caffe
