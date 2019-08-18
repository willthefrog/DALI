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

#ifndef DALI_PIPELINE_OPERATORS_SUPPORT_RANDOM_DICE_ROLL_H_
#define DALI_PIPELINE_OPERATORS_SUPPORT_RANDOM_DICE_ROLL_H_

#include <random>
#include <vector>

#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/common.h"

namespace dali {

class DiceRoll : public Operator<SupportBackend> {
 public:
  inline explicit DiceRoll(const OpSpec &spec) :
    Operator<SupportBackend>(spec),
    rng_(spec.GetArgument<int64_t>("seed")) {
    values_ = spec.GetArgument<vector<float>>("face_values");
    batch_wise_ = spec.GetArgument<bool>("batch_wise");
    auto num_faces = values_.size();
    auto probs = spec.GetArgument<vector<float>>("face_probs");
    if (probs.size() == 0) {
      probs.resize(num_faces, 1./num_faces);
    }
    dis_ = std::discrete_distribution<int>(probs.begin(), probs.end());
  }

  inline ~DiceRoll() override = default;

  DISABLE_COPY_MOVE_ASSIGN(DiceRoll);

  USE_OPERATOR_MEMBERS();
  using Operator<SupportBackend>::RunImpl;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const SupportWorkspace &ws) override {
    return false;
  }

  void RunImpl(Workspace<SupportBackend> * ws) override;

 private:
  std::discrete_distribution<int> dis_;
  std::mt19937 rng_;
  vector<float> values_;
  bool batch_wise_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_SUPPORT_RANDOM_DICE_ROLL_H_
