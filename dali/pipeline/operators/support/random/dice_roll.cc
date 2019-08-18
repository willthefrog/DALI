// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "dali/pipeline/operators/support/random/dice_roll.h"

namespace dali {

void DiceRoll::RunImpl(SupportWorkspace * ws) {
  auto &output = ws->Output<CPUBackend>(0);
  output.Resize({batch_size_});

  float * out_data = output.template mutable_data<float>();

  int face = 0;
  for (int i = 0; i < batch_size_; ++i) {
    face = (!batch_wise_ || i == 0) ? dis_(rng_) : face;
    out_data[i] = values_[face];
  }
}

DALI_REGISTER_OPERATOR(DiceRoll, DiceRoll, Support);

DALI_SCHEMA(DiceRoll)
  .DocStr("Produce tensor filled with random choices of values by rolling a dice.")
  .NumInput(0)
  .NumOutput(1)
  .AddArg("face_values",
          R"code(Values of each dice face. List of floats.)code", DALI_FLOAT_VEC)
  .AddOptionalArg("face_probs",
                  R"code(Probability of each dice face. List of floats.)code", std::vector<float>)
  .AddOptionalArg("batch_wise",
                  R"code(Same dice roll result is used for all samples in a batch.)code", false);

}  // namespace dali
