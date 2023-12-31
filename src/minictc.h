#ifndef INC_MINICTC_MINICTC_H
#define INC_MINICTC_MINICTC_H

#include <vector>

namespace minictc {

std::vector<float> compute_ctc_grad(
  const float *const logprobs,
  const int *const targets,
  const int batch_size,
  const int *const num_frames,
  const int num_tokens,
  const int *const num_targets,
  const int blank_idx,
  float *const loss_value
);

void compute_ctc_grad_single(
  const float *const logprobs,
  const int *const targets,
  const int num_frames,
  const int num_tokens,
  const int num_targets,
  const int blank_idx,
  float *const loss_value,
  float *const grad
);

std::vector<float> compute_alpha(
  const float * const logprobs,
  const int * const targets,
  const int num_frames,
  const int num_tokens,
  const int num_targets,
  const int blank_idx
  );

std::vector<float> compute_beta(
  const float * const logprobs,
  const int * const targets,
  const int num_frames,
  const int num_tokens,
  const int num_targets,
  const int blank_idx
  );

}

#endif //INC_MINICTC_MINICTC_H

