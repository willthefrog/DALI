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

#ifndef DALI_PIPELINE_OPERATORS_DETECTION_BOX_ENCODER_H_
#define DALI_PIPELINE_OPERATORS_DETECTION_BOX_ENCODER_H_

#include <algorithm>
#include <cstring>
#include <vector>
#include <utility>

#include "dali/kernels/tensor_shape.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/util/bounding_box.h"

namespace dali {

template<typename Backend>
class BoxEncoder;

template <>
class BoxEncoder<CPUBackend>: public Operator<CPUBackend> {
 public:
  explicit BoxEncoder(const OpSpec &spec)
      : Operator<CPUBackend>(spec), criteria_(spec.GetArgument<float>("criteria")),
        offset_(spec.GetArgument<bool>("offset")),
        scale_(spec.GetArgument<float>("scale")) {
    DALI_ENFORCE(
      criteria_ >= 0.f,
      "Expected criteria >= 0, actual value = " + std::to_string(criteria_));
    DALI_ENFORCE(
      criteria_ <= 1.f,
      "Expected criteria <= 1, actual value = " + std::to_string(criteria_));

    auto anchors = spec.GetArgument<vector<float>>("anchors");

    DALI_ENFORCE(
      (anchors.size() % BoundingBox::kSize) == 0,
      "Anchors size must be divisible by 4, actual value = " + std::to_string(anchors.size()));

    anchors_ = ReadBoxesFromInput(anchors.data(), anchors.size() / BoundingBox::kSize);

    means_ = spec.GetArgument<vector<float>>("means");
    DALI_ENFORCE(means_.size() == 4,
      "means size must be a list of 4 values.");

    stds_ = spec.GetArgument<vector<float>>("stds");
    DALI_ENFORCE(stds_.size() == 4,
      "stds size must be a list of 4 values.");
    DALI_ENFORCE(std::find(stds_.begin(), stds_.end(), 0) == stds_.end(),
       "stds values must be != 0.");
  }

  ~BoxEncoder() override = default;

  DISABLE_COPY_MOVE_ASSIGN(BoxEncoder);

 protected:
  void RunImpl(Workspace<CPUBackend> *ws, const int idx) override;
  using Operator<CPUBackend>::RunImpl;

 private:
  const float criteria_;
  vector<BoundingBox> anchors_;

  bool offset_;
  vector<float> means_;
  vector<float> stds_;
  float scale_;

  vector<float> CalculateIous(const vector<BoundingBox> &boxes) const;

  void CalculateIousForBox(float *ious, const BoundingBox &box) const;

  vector<BoundingBox> ReadBoxesFromInput(const float *in_boxes, unsigned num_boxes) const;

  void WriteAnchorsToOutput(float *out_boxes, int *out_labels) const;

  void WriteBoxToOutput(const std::array<float, BoundingBox::kSize>& box,
                        float *out_box_data) const;

  void WriteMatchesToOutput(const vector<std::pair<unsigned, unsigned>> matches,
    const vector<BoundingBox> &boxes, const int *labels, float *out_boxes, int *out_labels) const;

  vector<std::pair<unsigned, unsigned>> MatchBoxesWithAnchors(
    const vector<BoundingBox> &boxes) const;

  unsigned FindBestBoxForAnchor(
    unsigned anchor_idx, const vector<float> &ious, unsigned num_boxes) const;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DETECTION_BOX_ENCODER_H_
