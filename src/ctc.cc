#include <cmath>
#include <numeric>

#include "ctc.h"

namespace tcrowns {

namespace internal {

constexpr float logsumexp(const float x, const float y) {
  const auto maxv = std::max(x, y);
  if (std::isinf(maxv)) {
    return -INFINITY;
  }
  return maxv + std::log(std::exp(x - maxv) + std::exp(y - maxv));
}

template<typename Vec>
float get(const Vec &arr, const int S, const int T, const int s, const int t) {
  if (-1 < s && s < S && -1 < t && t < T) {
    return arr[s + S*t];
  }
  return -INFINITY;
}

std::vector<int> with_blanks(const int * const targets, const int num_targets, const int blank_idx) {
  std::vector x{blank_idx};
  for (int i = 0; i != num_targets; ++i) {
    x.push_back(targets[i]);
    x.push_back(blank_idx);
  }
  return x;
}
}

std::vector<float> compute_alpha(
  const float *const logprobs,
  const int *const targets,
  const int num_frames,
  const int num_tokens,
  const int num_targets,
  const int blank_idx
) {
  using internal::logsumexp;
  using std::vector;
  using std::span;

  // indexing alpha: [target_label_idx, time_idx]
  // indexing probs: [vocabulary_label_idx, time_idx]

  const auto target_blank = internal::with_blanks(targets, num_targets, blank_idx);
  const int path_len = static_cast<int>(target_blank.size());

  vector alpha(num_frames * path_len, -INFINITY); // S x T
  alpha[0] = logprobs[blank_idx];
  alpha[1] = logprobs[target_blank[1]];

  for (int t = 1; t != num_frames; ++t) {
    for (int s = 0; s != path_len; ++s) {
      const auto current_label = target_blank[s];
      const auto previous_label = s - 2 > -1 ? target_blank[s - 2] : -1;
      // likelihood of getting here via any possible path
      const auto previous_alpha = logsumexp(
        internal::get(alpha, path_len, num_frames, s, t - 1),  // likelihood of the same path from previous time index [non emitting transition]
        internal::get(
          alpha, path_len, num_frames, s - 1, t - 1
        )  // likelihood of the path prefix (minus last one label) from previous time index [emitting transition]
      );
      const auto current_observation_likelihood = logprobs[target_blank[s] + num_tokens * t];
      if (current_label == blank_idx or current_label == previous_label) {
        // [non emitting transition]
        alpha[s + path_len * t] = previous_alpha + current_observation_likelihood;
      }
      else {
        // [emitting transition]
        alpha[s + path_len * t] = logsumexp(previous_alpha, internal::get(alpha, path_len, num_frames, s - 2, t - 1)) + current_observation_likelihood;
      }
    }
  }

  // zero out the prob of paths that did not reach the end
  // t=-1 => s=-3::-1
  // t=-2 => s=-5::-1
  // t=-3 => s=-7::-1
  int t = -1;
  while(true) {
    int s = 2 * t - 1;
    if (path_len + s < 0) {
      break;
    }
    while (path_len + s > -1) {
      alpha[(path_len + s) + path_len  * (num_frames + t)] = -INFINITY;
      s -= 1;
    }
    t -= 1;
  }

  return alpha;
}


std::vector<float> compute_beta(
  const float *const logprobs,
  const int *const targets,
  const int num_frames,
  const int num_tokens,
  const int num_targets,
  const int blank_idx
) {
  using internal::logsumexp;
  using std::vector;
  using std::span;

  // indexing beta: [target_label_idx, time_idx]
  // indexing probs: [vocabulary_label_idx, time_idx]

  const auto target_blank = internal::with_blanks(targets, num_targets, blank_idx);
  const int path_len = static_cast<int>(target_blank.size());

  vector beta(num_frames * path_len, -INFINITY); // S x T
  beta[path_len - 1 + path_len * (num_frames - 1)] = logprobs[blank_idx + num_tokens * (num_frames - 1)];
  beta[path_len - 2 + path_len * (num_frames - 1)] = logprobs[target_blank.back() + num_tokens * (num_frames - 1)];

  for (int t = num_frames - 2; t != -1; --t) {
    for (int s = path_len - 1; s != -1; --s) {
      const auto current_label = target_blank[s];
      const auto next_label = s + 2 < path_len ? target_blank[s + 2] : -1;
      const auto next_beta = logsumexp(
        internal::get(beta, path_len, num_frames, s, t + 1),
        internal::get(beta, path_len, num_frames, s + 1, t + 1)
      );
      const auto current_observation_likelihood = logprobs[target_blank[s] + num_tokens * t];
      if (current_label == blank_idx or current_label == next_label) {
        // [non emitting transition]
        beta[s + path_len * t] = next_beta + current_observation_likelihood;
      }
      else {
        // [emitting transition]
        beta[s + path_len * t] = logsumexp(next_beta, internal::get(beta, path_len, num_frames, s + 2, t + 1)) + current_observation_likelihood;
      }
    }
  }

  // zero out the prob of paths that did not reach the beginning
  // s = 2t + 2
  // t=0 => s=2:
  // t=1 => s=4:
  // t=2 => s=6:
  int t = 0;
  while(true) {
    int s = 2 * t + 2;
    if (path_len - s < 0) {
      break;
    }
    while (path_len - s > 0) {
      beta[s + path_len * t] = -INFINITY;
      s += 1;
    }
    t += 1;
  }

  return beta;
}

std::vector<float> compute_ctc_grad_single(
  const float *const logprobs,
  const int *const targets,
  const int num_frames,
  const int num_tokens,
  const int num_targets,
  const int blank_idx,
  float *const loss_value
) {
  const auto target_blank = internal::with_blanks(targets, num_targets, blank_idx);
  const int path_len = static_cast<int>(target_blank.size());

  auto gamma = compute_alpha(logprobs, targets, num_frames, num_tokens, num_targets, blank_idx);
  if (loss_value != nullptr) {
    // it's still alpha at this point despite the name
    *loss_value = internal::logsumexp(
      gamma[path_len - 1 + path_len * (num_frames - 1)],
      gamma[path_len - 2 + path_len * (num_frames - 1)]
    );
  }
  const auto beta = compute_beta(logprobs, targets, num_frames, num_tokens, num_targets, blank_idx);
  for (int t = 0; t < num_frames; ++t) {
    for (int s = 0; s < path_len; ++ s) {
      gamma[s + path_len * t] += beta[s + path_len * t];
    }
  }

  const auto numel = num_frames * num_tokens;
  std::vector<float> d_logprobs;
  for (int i = 0; i != numel; ++i) { d_logprobs.push_back(std::exp(logprobs[i])); }

  for (int t = 0; t != num_frames; ++t) {
    float Z_t = 0;
    for (int s = 0; s != path_len; ++s) {
      Z_t += std::exp(gamma[s + path_len * t] - logprobs[target_blank[s] + num_tokens * t]);
    }

    for (int k = 0; k != num_tokens; ++k) {
      float sum_alpha_beta_s_t = 0;
      for (int s = 0; s != path_len; ++s) {
        if (target_blank[s] == k) {
          sum_alpha_beta_s_t += std::exp(gamma[s + path_len * t]);
        }
      }
      if (sum_alpha_beta_s_t > 0) {
        d_logprobs[k + num_tokens * t] -= 1 / (d_logprobs[k + num_tokens * t] * Z_t) * sum_alpha_beta_s_t;
      }
    }
  }

  return d_logprobs;
}

std::vector<float> compute_ctc_grad(
  const float *const logprobs,
  const int *const targets,
  const int batch_size,
  const int *const num_frames,
  const int num_tokens,
  const int *const num_targets,
  const int blank_idx,
  float *const loss_value
) {

  // I ran into some issue with -Xclang -fopenmp and structured bindings here
  // const auto [logprob_offsets, target_offsets] = [=] {
  const auto tpl = [=] {
    std::vector<int> lo{0};
    std::vector<int> to{0};
    for (int b = 0; b != batch_size; ++b) {
      lo.push_back(lo.back() + num_tokens * num_frames[b]);
      to.push_back(to.back() + num_targets[b]);
    }
    return std::make_tuple(lo, to);
  }();
  const auto &logprob_offsets = get<0>(tpl);
  const auto &target_offsets = get<1>(tpl);
  const auto numel = logprob_offsets.back();

  std::vector<float> grad(numel);

  // I "borrowed" this simple parallelization trick here from warp_ctc
#pragma omp parallel for
  for (int b = 0; b != batch_size; ++b) {
    const auto grad_b = compute_ctc_grad_single(
      logprobs + logprob_offsets[b],
      targets + target_offsets[b],
      num_frames[b],
      num_tokens,
      num_targets[b],
      blank_idx,
      loss_value + b
    );
    // To be maximally efficient the API needs to be reworked to allow writing directly into the output.
    std::copy(grad_b.cbegin(), grad_b.cend(), grad.begin() + logprob_offsets[b]);
  }

  return grad;
}
}
