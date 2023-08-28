#include <iostream>

#include "ctc.h"

void printmtx(const auto &mtx, const int n_rows, const int n_cols, std::string_view name = "") {
  using std::cout;
  if (!name.empty()) cout << name << ":\n";
  cout << "[";
  for (int i = 0; i != n_rows; ++i) {
    cout << (i == 0 ? "[" : " [");
    for (int j = 0; j != n_cols; ++j) {
      cout << mtx[i + n_rows * j];
      if (j != n_cols - 1) { cout << ", "; }
    }
    cout << "]";
    if (i != n_rows - 1) { cout << "\n"; }
  }
  cout << "]\n";
}

int main(int argc, const char *const *const argv) {
  using namespace std;

  const vector vocabulary{0, 1, 2, 3, 4};
  const int blank_idx = 0;
  const int T = 10;
  const int K = vocabulary.size();
  const vector target{3, 1, 2};
  const auto logprobs = [=]() {
    vector<float> x(T * K, 1.0 / K);
    for (auto &elem: x) {
      elem = std::log(elem);
    }
    return x;
  }();

  printmtx(logprobs, K, T, "logprobs");

  const auto alpha = tcrowns::compute_alpha(
    logprobs.data(), target.data(), T, vocabulary.size(), target.size(), blank_idx
  );

  const auto S = alpha.size() / T;

  printmtx(alpha, S, T, "alpha");

  const auto beta = tcrowns::compute_beta(
    logprobs.data(), target.data(), T, vocabulary.size(), target.size(), blank_idx
  );

  printmtx(beta, S, T, "beta");

  // single item CTC benchmark
  {
    auto begin = std::chrono::high_resolution_clock::now();
    uint32_t iterations = 10000;
    for (uint32_t i = 0; i < iterations; ++i) {
      float loss;
      vector<float> grad(logprobs.size());
      tcrowns::compute_ctc_grad_single(
        logprobs.data(), target.data(), T, vocabulary.size(), target.size(), blank_idx, &loss, grad.data()
      );
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << duration << "us total, average : " << static_cast<double>(duration) / iterations << "us." << std::endl;
  }

  // single item CTC grad
  {
    float loss;
    vector<float> grad(logprobs.size());
    tcrowns::compute_ctc_grad_single(
      logprobs.data(), target.data(), T, vocabulary.size(), target.size(), blank_idx, &loss, grad.data()
    );

    std::cout << "loss=" << loss << "\n";
    printmtx(grad, K, T, "grad");
  }

  // benchmark batch of items CTC grad
  {
    const auto batch_size = 32;
    const auto [logprobs_batch, target_batch, Ts, Ss] = [&] {
      vector lps = logprobs;
      vector ts = target;
      vector<int> Ts;
      vector<int> Ss;
      for (int b = 0; b != batch_size; ++b) {
        lps.insert(lps.end(), logprobs.cbegin(), logprobs.cend());
        lps.back() -= static_cast<float>(b);  // make it a little different
        ts.insert(ts.end(), target.cbegin(), target.cend());
        Ts.push_back(T);
        Ss.push_back(static_cast<int>(target.size()));
      }
      return std::make_tuple(lps, ts, Ts, Ss);
    }();
    vector<float> loss(batch_size, 0);

    const auto begin = std::chrono::high_resolution_clock::now();
    const uint32_t iterations = 10000;
    for (uint32_t i = 0; i < iterations; ++i) {
      vector<float> loss(batch_size, 0);
      const auto grad = tcrowns::compute_ctc_grad(
        logprobs_batch.data(),
        target_batch.data(),
        batch_size,
        Ts.data(),
        vocabulary.size(),
        Ss.data(),
        blank_idx,
        loss.data());
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    const auto mean = static_cast<double>(duration) / iterations;
    const auto per_item = mean / batch_size;

    std::cout << duration << "us total, average: " << mean << "us, per item: " << per_item << "us." << std::endl;
  }


  // batch of items CTC grad
  {
    const auto batch_size = 2;
    const auto [logprobs_batch, target_batch] = [&] {
      vector lps = logprobs;
      vector ts = target;
      for (int b = 0; b != batch_size; ++b) {
        lps.insert(lps.end(), logprobs.cbegin(), logprobs.cend());
        lps.back() -= static_cast<float>(b);  // make it a little different
        ts.insert(ts.end(), target.cbegin(), target.cend());
      }
      return std::make_tuple(lps, ts);
    }();
    vector Ts = {T, T};
    vector Ss = {static_cast<int>(target.size()), static_cast<int>(target.size())};
    vector<float> loss(batch_size, 0);

    const auto grad = tcrowns::compute_ctc_grad(
      logprobs_batch.data(),
      target_batch.data(),
      batch_size,
      Ts.data(),
      vocabulary.size(),
      Ss.data(),
      blank_idx,
      loss.data());

    for (int b = 0; b != batch_size; ++b) {
      std::cout << "batch=" << b << ", loss=" << loss[b] << "\n";
      printmtx(std::span(grad.data() + b * (K * T), K * T), K, T, "grad");
    }
  }

  return 0;
}