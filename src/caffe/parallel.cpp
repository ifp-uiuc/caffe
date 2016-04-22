#ifdef CMAKE_BUILD
  #include "caffe_config.h"
#endif

#ifndef CPU_ONLY
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif  // USE_CUDA
#endif  // !CPU_ONLY
#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"
#include "caffe/greentea/greentea.hpp"

namespace caffe {

enum Op {
  copy,
  replace_cpu,
  replace_gpu,
  replace_cpu_diff,
  replace_gpu_diff
};

uint_tp _align_size(uint_tp size, int align) {
  if (size % align == 0) {
    return size;
  } else {
    return size + align - (size % align);
  }
}
  
template<typename Dtype>
static void apply_buffers(const vector<Blob<Dtype>*>& blobs, Dtype* buffer,
                          uint_tp total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int_tp size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
        caffe_copy(size,
                   reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                   ptr);
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);
        break;
    }
    ptr += size;
  }
  // total_size is at least one byte
  CHECK_EQ(total_size, (ptr == buffer ? 1 : ptr - buffer));
}

#ifdef USE_GREENTEA
template<typename Dtype>
static void apply_buffers_greentea(const vector<Blob<Dtype>*>& blobs, cl_mem buffer,
				   uint_tp total_size, Op op, int device_id) {
  uint_tp offset = 0;

  size_t mem_size, ret_size;
  DLOG(INFO) << "total_size: " << total_size;
  clGetMemObjectInfo(buffer,  CL_MEM_SIZE , sizeof(size_t),  &mem_size, &ret_size);
  DLOG(INFO) << "Incoming buffer size : " << mem_size; 
  for (int i = 0; i < blobs.size(); ++i) {
    int_tp size = blobs[i]->count()*sizeof(Dtype);
    cl_buffer_region region;
    region.origin = offset;
    DLOG(INFO) << "Region offset " << offset;
    DLOG(INFO) << "Region size " << size;
    region.size = size;
    cl_int r_code;
    cl_mem sub_buffer = clCreateSubBuffer(buffer,
					  CL_MEM_READ_WRITE,
					  CL_BUFFER_CREATE_TYPE_REGION,
					      &region,
					  &r_code);
    CHECK_EQ(r_code, CL_SUCCESS);
    
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
	greentea_gpu_memcpy(device_id, size, blobs[i]->data()->cpu_data(), buffer, offset);
        break;
      }
      case replace_cpu:
        CHECK(0);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(sub_buffer);
        break;
      case replace_cpu_diff:
        CHECK(0);
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(sub_buffer);
        break;
    }
    offset += size;
    offset = _align_size(offset, 256);
  }
  // total_size is at least one byte
  //CHECK_EQ(total_size, (offset==0 ? 1 : offset));
}
#endif // USE_GREENTEA
  
// Buffer size necessary to store given blobs
template<typename Dtype>
static uint_tp total_size(const vector<Blob<Dtype>*>& params) {
  uint_tp size = 0;
  for (int i = 0; i < params.size(); ++i) {
    size += params[i]->count();
    // TODO: verify the alignment.
    size = _align_size(size, 256/4);
  }
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters.
  return (size > 0) ? size : 1;
}

template<typename Dtype>
Params<Dtype>::Params(shared_ptr<Solver<Dtype> > root_solver)
    : size_(total_size<Dtype>(root_solver->net()->learnable_params())), data_(),
      diff_() {
}

template<typename Dtype>
GPUParams<Dtype>::GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device)
    : Params<Dtype>(root_solver) {
#ifndef CPU_ONLY
#ifdef USE_CUDA
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));

  // Allocate device buffers
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(Dtype)));

  // Copy blob values
  const vector<Blob<Dtype>*>& net = root_solver->net()->learnable_params();
  apply_buffers(net, data_, size_, copy);

  CUDA_CHECK(cudaMalloc(&diff_, size_ * sizeof(Dtype)));
  caffe_gpu_set(size_, Dtype(0), diff_);

  CUDA_CHECK(cudaSetDevice(initial_device));
#elif USE_GREENTEA
  DLOG(INFO) << "GREENTEA MALLOC " << size_*sizeof(Dtype) << " bytes for data and diff";
  greentea_malloc((void**) &data_, size_*sizeof(Dtype), device);
  greentea_malloc((void**) &diff_, size_*sizeof(Dtype), device);
  const vector<Blob<Dtype>*>& net = root_solver->net()->learnable_params();

  int dbg = 0;
  for (int i = 0; i < net.size(); ++i)
    dbg += net[i]->count();

  DLOG(INFO) << "total of " << dbg << " parameters";
  apply_buffers_greentea(net, (cl_mem) data_, size_, copy, device);
  
  DLOG(INFO) << "greentea_gpu_set diff_ to 0";
  greentea_gpu_set(device, size_, Dtype(0), (cl_mem) diff_, 0);
  DLOG(INFO) << "complete initialization of device " << device;
#endif // USE_GREENTEA or USE_CUDA
#else
  NO_GPU;
#endif
}

template<typename Dtype>
GPUParams<Dtype>::~GPUParams() {
#ifndef CPU_ONLY
#ifdef USE_CUDA
  CUDA_CHECK(cudaFree(data_));
  CUDA_CHECK(cudaFree(diff_));
#endif  // USE_CUDA  
#ifdef USE_GREENTEA
  // TODO: check whether this is called in the right thread.
  int cur_device = Caffe::GetDefaultDevice()->id();
  greentea_free(data_, cur_device);
  greentea_free(diff_, cur_device);
#endif // USE_GREENTEA
#endif  // !CPU_ONLY
}

template<typename Dtype>
void GPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  const vector<Blob<Dtype>*>& net = solver->net()->learnable_params();
#ifdef USE_GREENTEA
  int device_id = solver->param().device_id();
  DLOG(INFO) << "GPU Params: configure " << device_id;
  apply_buffers_greentea(net, (cl_mem) data_, size_, replace_gpu, device_id);
  apply_buffers_greentea(net, (cl_mem) diff_, size_, replace_gpu_diff, device_id);
#else
  apply_buffers(net, data_, size_, replace_gpu);
  apply_buffers(net, diff_, size_, replace_gpu_diff);
#endif // USE_GREENTEA
}

void DevicePair::compute(const vector<device*> devices,
                         vector<DevicePair>* pairs) {
#ifndef CPU_ONLY
  vector<device*> remaining(devices);

  // Depth for reduction tree
  int remaining_depth = static_cast<int>(ceil(log2(remaining.size())));

  // Group GPUs by board
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      for (int j = i + 1; j < remaining.size(); ++j) {
        // Currently, dual-chip device only on CUDA
        if (remaining[i]->backend() == BACKEND_CUDA
            && remaining[j]->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
          cudaDeviceProp a, b;
          CUDA_CHECK(cudaGetDeviceProperties(&a, remaining[i]->id()));
          CUDA_CHECK(cudaGetDeviceProperties(&b, remaining[j]->id()));
          if (a.isMultiGpuBoard && b.isMultiGpuBoard) {
            if (a.multiGpuBoardGroupID == b.multiGpuBoardGroupID) {
              pairs->push_back(DevicePair(remaining[i], remaining[j]));
              DLOG(INFO)<< "GPU board: " << remaining[i] << ":" << remaining[j];
              remaining.erase(remaining.begin() + j);
              break;
            }
          }
#endif  // USE_CUDA
        }
      }
    }
  }
  ostringstream s;
  for (int i = 0; i < remaining.size(); ++i) {
    s << (i ? ", " : "") << remaining[i];
  }
  DLOG(INFO)<< "GPUs paired by boards, remaining: " << s.str();

  // Group by P2P accessibility
  remaining_depth = ceil(log2(remaining.size()));
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      for (int j = i + 1; j < remaining.size(); ++j) {
        int access = 0;
        // Currently, P2P access only on CUDA
        if (remaining[i]->backend() == BACKEND_CUDA
            && remaining[j]->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
          CUDA_CHECK(
              cudaDeviceCanAccessPeer(&access, remaining[i]->id(),
                                      remaining[j]->id()));
#endif  // USE_CUDA
        }
        if (access) {
          pairs->push_back(DevicePair(remaining[i], remaining[j]));
          DLOG(INFO)<< "P2P pair: " << remaining[i] << ":" << remaining[j];
          remaining.erase(remaining.begin() + j);
          break;
        }
      }
    }
  }
  s.str("");
  for (int i = 0; i < remaining.size(); ++i) {
    s << (i ? ", " : "") << remaining[i];
  }
  DLOG(INFO)<< "GPUs paired by P2P access, remaining: " << s.str();

  // Group remaining
  remaining_depth = ceil(log2(remaining.size()));
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      pairs->push_back(DevicePair(remaining[i], remaining[i + 1]));
      DLOG(INFO)<< "Remaining pair: " << remaining[i] << ":"
      << remaining[i + 1];
      remaining.erase(remaining.begin() + i + 1);
    }
  }

  // Should only be the parent node remaining
  CHECK_EQ(remaining.size(), 1);

  pairs->insert(pairs->begin(),
                DevicePair(Caffe::Get().GetCPUDevice(), remaining[0]));

  CHECK(pairs->size() == devices.size());
  for (int i = 0; i < pairs->size(); ++i) {
    CHECK((*pairs)[i].get_parent() != (*pairs)[i].get_device());
    for (int j = i + 1; j < pairs->size(); ++j) {
      CHECK((*pairs)[i].get_device() != (*pairs)[j].get_device());
    }
  }
#else
  NO_GPU;
#endif
}

//

template<typename Dtype>
P2PSync<Dtype>::P2PSync(shared_ptr<Solver<Dtype> > root_solver,
                        P2PSync<Dtype>* parent, const SolverParameter& param)
    : GPUParams<Dtype>(root_solver, param.device_id()), parent_(parent),
      children_(), queue_(), initial_iter_(root_solver->iter()), solver_() {
#ifndef CPU_ONLY
#ifdef  USE_CUDA
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = param.device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent == NULL) {
    solver_ = root_solver;
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(new WorkerSolver<Dtype>(param, root_solver.get()));
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get());
  solver_->add_callback(this);

  if (parent) {
    // Enable p2p access between devices
    const int peer = parent->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
    } else {
      LOG(INFO)<< "GPU " << self << " does not have p2p access to GPU " << peer;
    }
    // Allocate receiving buffer on parent
    CUDA_CHECK(cudaSetDevice(peer));
    CUDA_CHECK(cudaMalloc(&parent_grads_, size_ * sizeof(Dtype)));
    CUDA_CHECK(cudaSetDevice(self));
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#endif // USE_CUDA
  
#ifdef USE_GREENTEA
  int initial_device = Caffe::GetDefaultDevice()->id();
  if (parent == NULL) {
    solver_ = root_solver;
  } else {
    Caffe::set_root_solver(false);
    Caffe::SetDevice(param.device_id());
    DLOG(INFO) << "Constructing WorkerSolver on device " << param.device_id();
    solver_.reset(new WorkerSolver<Dtype>(param, root_solver.get()));
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get());
  solver_->add_callback(this);

  if (parent) {
    parent_cpu_data_ = parent->parent_cpu_data_;    
  } else {
    // Root solver allocate params on host.
    parent_cpu_data_ = new Dtype[size_];
  }

  // Everyone has a grad buffer on host.
  cpu_grads_ = new Dtype[size_];
  // Everyone has a grad buffer on device, too.
  // In this greentea case, this is actually 'child_grad_'
  const int self = param.device_id();
  greentea_malloc((void**) &parent_grads_, size_*sizeof(Dtype), self);

  Caffe::SetDevice(initial_device);

#endif //USE_GREENTEA
#else
  NO_GPU;
#endif
}

template<typename Dtype>
P2PSync<Dtype>::~P2PSync() {
#ifndef CPU_ONLY
#ifdef USE_CUDA
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent_) {
    CUDA_CHECK(cudaFree(parent_grads_));
    const int peer = parent_->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
    }
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#endif  // USE_CUDA

#ifdef USE_GREENTEA
  if (parent_ == NULL) {
    // root solver own the host param copy. 
    delete [] parent_cpu_data_;
  } 
  delete [] cpu_grads_;
#endif // USE_GREENTEA
#endif  // !CPU_ONLY
}

template<typename Dtype>
void P2PSync<Dtype>::InternalThreadEntry() {
  Caffe::SetDevice(solver_->param().device_id());
  CHECK(Caffe::root_solver());
  Caffe::set_root_solver(false);
  // See if there is a defined seed and reset random state if so
  if (solver_->param().random_seed() >= 0) {
    // Fetch random seed and modulate by device ID to make sure
    // everyone doesn't have the same seed.  We seem to have some
    // solver instability if we have everyone with the same seed
    Caffe::set_random_seed(
        solver_->param().random_seed() + solver_->param().device_id(),
        solver_->get_device());
  }
  solver_->Step(solver_->param().max_iter() - initial_iter_);
}

template<typename Dtype>
void P2PSync<Dtype>::on_start() {
#ifndef CPU_ONLY
#ifdef USE_CUDA
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#else
//  CHECK(false);
#endif

  // Wait for update from parent
  if (parent_) {
    P2PSync<Dtype> *parent = queue_.pop();
    CHECK(parent == parent_);
  }

  // Update children
  for (int i = children_.size() - 1; i >= 0; i--) {
    Dtype* src = data_;
    Dtype* dst = children_[i]->data_;

#ifdef DEBUG
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == children_[i]->solver_->param().device_id());
#endif

    CUDA_CHECK(
        cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),
                        cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    children_[i]->queue_.push(this);
  }
#endif  // USE_CUDA

#ifdef USE_GREENTEA
  // Wait for update from parent
  int device_id = solver_->param().device_id();
  if (parent_) {
    P2PSync<Dtype> *parent = queue_.pop();
    CHECK(parent == parent_);
    // Copy from the host to GPU
    greentea_gpu_memcpy(device_id, size_ * sizeof(Dtype), parent_cpu_data_,  (cl_mem) data_, 0);
  } else {
    // This is the root, read params to host.
    greentea_gpu_memcpy(device_id, size_ * sizeof(Dtype), (cl_mem) data_, 0, parent_cpu_data_);
  }

  // Sync device
  Caffe::Synchronize(device_id);
  // notify children. 
  for (int i = children_.size() - 1; i >= 0; i--) {
    children_[i]->queue_.push(this);
  }
#endif  // USE_GREENTEA

#endif  // !CPU_ONLY
}

// TODO: Rewrite this function for OpenCL
template<typename Dtype>
void P2PSync<Dtype>::on_gradients_ready() {
#ifndef CPU_ONLY
#ifdef USE_CUDA
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#endif

  // Sum children gradients as they appear in the queue
  for (int i = 0; i < children_.size(); ++i) {
    P2PSync<Dtype> *child = queue_.pop();
    Dtype* src = child->parent_grads_;
    Dtype* dst = diff_;

#ifdef DEBUG
    bool ok = false;
    for (int j = 0; j < children_.size(); ++j) {
      if (child == children_[j]) {
        ok = true;
      }
    }
    CHECK(ok);
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == device);
#endif

    caffe_gpu_add(size_, src, dst, dst);
  }

  // Send gradients to parent
  if (parent_) {
    Dtype* src = diff_;
    Dtype* dst = parent_grads_;

#ifdef DEBUG
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == parent_->solver_->param().device_id());
#endif

    CUDA_CHECK(cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),  //
        cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    parent_->queue_.push(this);
  } else {
    // Loss functions divide gradients by the batch size, so to compensate
    // for split batch, the root solver divides by number of solvers.
    caffe_gpu_scal(size_, Dtype(1.0 / Caffe::solver_count()), diff_);
  }
#endif  // USE_CUDA

#ifdef USE_GREENTEA
  // Sum children gradients as they appear in the queue
  int device_id = solver_->param().device_id();
  for (int i = 0; i < children_.size(); ++i) {
    P2PSync<Dtype> *child = queue_.pop();
    Dtype* src = child->cpu_grads_;
    greentea_copy(device_id, size_ * sizeof(Dtype), src, (cl_mem) parent_grads_, 0);
    greentea_gpu_add<Dtype>(device_id, size_, (cl_mem) parent_grads_, 0, (cl_mem) diff_, 0, (cl_mem) diff_, 0);
  }

  // Send gradients to parent
  if (parent_) {
    greentea_copy(device_id, size_ * sizeof(Dtype), (cl_mem) diff_, 0, cpu_grads_);
    Caffe::Synchronize(device_id);
    parent_->queue_.push(this);
  } else {
    // Loss functions divide gradients by the batch size, so to compensate
    // for split batch, the root solver divides by number of solvers.
    greentea_gpu_scale(device_id, size_, Dtype(1.0 / Caffe::solver_count()), (cl_mem) diff_, 0, (cl_mem) diff_, 0);
  }
#endif  // USE_GREENTEA
  
#endif  // !CPU_ONLY
}

template<typename Dtype>
void P2PSync<Dtype>::Prepare(const vector<device*>& gpus,
            vector<shared_ptr<P2PSync<Dtype> > >* syncs) {
  // Pair devices for map-reduce synchronization
  vector<DevicePair> pairs;
  DevicePair::compute(gpus, &pairs);
  ostringstream s;
  for (int i = 1; i < pairs.size(); ++i) {
    s << (i == 1 ? "" : ", ") << pairs[i].get_parent()->id() << ":"
      << pairs[i].get_device()->id();
     
  }
  LOG(INFO)<< "GPUs pairs " << s.str();

  SolverParameter param(solver_->param());

  // Build the GPU tree by finding the parent for each solver
  for (int attempts = 0; attempts < pairs.size(); ++attempts) {
    for (int i = 1; i < pairs.size(); ++i) {
      if (!syncs->at(i).get()) {
        P2PSync<Dtype>* parent = NULL;
        for (int j = 0; j < syncs->size(); ++j) {
          P2PSync<Dtype>* sync = j == 0 ? this : syncs->at(j).get();
          if (sync) {
            const SolverParameter& p = sync->solver()->param();
            if (p.device_id() == pairs[i].get_parent()->id()) {
              parent = sync;
            }
          }
        }
        if (parent) {
          param.set_device_id(pairs[i].get_device()->id());
          syncs->at(i).reset(new P2PSync<Dtype>(solver_, parent, param));
          parent->children_.push_back((P2PSync<Dtype>*) syncs->at(i).get());
        }
      }
    }
  }
}

template<typename Dtype>
void P2PSync<Dtype>::Run(const vector<device*>& gpus) {
  vector<shared_ptr<P2PSync<Dtype> > > syncs(gpus.size());
  Prepare(gpus, &syncs);

  LOG(INFO)<< "Starting Optimization";

  DLOG(INFO) << "syncs array";
  
  for (int i = 1; i < syncs.size(); ++i) {
    DLOG(INFO) << "syncs " << i << ": " << syncs[i];
    //syncs[i]->StartInternalThread(solver_->get_device());
    syncs[i]->StartInternalThread(gpus[i]);
  }

  // Run root solver on current thread
  solver_->Solve();

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StopInternalThread();
  }
}

INSTANTIATE_CLASS(Params);
INSTANTIATE_CLASS(GPUParams);
INSTANTIATE_CLASS(P2PSync);

}  // namespace caffe
