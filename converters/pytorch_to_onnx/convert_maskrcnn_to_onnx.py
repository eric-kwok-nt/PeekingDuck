# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module to convert PyTorch Mask R-CNN models to Onnx"""

import logging
import numpy as np
import onnx
import onnxruntime
import torch
import torch.onnx
from pathlib import Path
from time import perf_counter
from torch import inference_mode, nn
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.detector import (
    Detector,
)
import pdb


####################
# Globals
####################
MODEL_WEIGHTS_DIR = Path("peekingduck_weights/mask_rcnn/pytorch")
MIN_SIZE = 800
MAX_SIZE = 1333
MODEL_TYPE = "r50-fpn"
DATA_SHAPE = (1, 3, MIN_SIZE, MAX_SIZE)
DEVICE = torch.device("cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model:
    def __init__(self):
        self.model_config = {
            "model_dir": MODEL_WEIGHTS_DIR,
            "class_names": {"null": "null"},  # Non-critical
            "detect_ids": ["*"],  # Non critical
            "model_type": MODEL_TYPE,
            "num_classes": 91,
            "model_file": {"r50-fpn": "mask-rcnn-r50-fpn.pth"},
            "min_size": MIN_SIZE,
            "max_size": MAX_SIZE,
            "nms_iou_threshold": 0.5,
            "max_num_detections": 100,
            "score_threshold": 0.5,
            "mask_threshold": 0.5,  # Non critical
        }
        MaskRCNN = Detector(**self.model_config)
        self.model = MaskRCNN.mask_rcnn
        self.model.to(DEVICE)

    @property
    def get_model(self):
        return self.model

MaskRCNN = Model().get_model


def convert_model(model_code: str) -> None:
    """Convert model given by 'model_code' from PyTorch to Onnx format.

    Args:
        model_code (str): supported codes
                          { "r50-fpn" }
    """
    logger.info(f"Convert Mask R-CNN {model_code} to Onnx")
    onnx_model_save_path = f"{MODEL_WEIGHTS_DIR.parent}/{model_code}.onnx"
    
    logger.info(f"Converting model to {onnx_model_save_path}")
    inp_random = torch.randn(*DATA_SHAPE, dtype=torch.float32, device=DEVICE)
    torch.onnx.export(
        MaskRCNN,
        inp_random,
        onnx_model_save_path,
        # export_params=True,
        # verbose=True,
        opset_version=11,
        # do_constant_folding=False,
        input_names=["images"],
        output_names=["boxes", "labels", "scores", "masks"],
    )

    # check converted model
    logger.info("Checking converted model")
    onnx_model = onnx.load(onnx_model_save_path)
    onnx.checker.check_model(onnx_model)

    logger.info("All good")


def test_converted_model(onnx_model_path: str):
    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    inp_random = torch.randn(*DATA_SHAPE, dtype=torch.float32, device=DEVICE) * 255

    with torch.no_grad():
        ref_output = MaskRCNN(inp_random)
    ref_output = ref_output[0]
    for key, value in ref_output.items():
        ref_output[key] = to_numpy(value)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inp_random)}
    output_keys = ["boxes", "labels", "scores", "masks"]
    ort_outs = ort_session.run(output_keys, ort_inputs)

    np.testing.assert_allclose(
        ref_output["boxes"], ort_outs[0], rtol=1e-03, atol=1e-05
    )
    np.testing.assert_array_equal(ref_output["scores"], ort_outs[2])
    np.testing.assert_array_equal(ref_output["labels"], ort_outs[1])
    np.testing.assert_allclose(
        ref_output["masks"], ort_outs[3], rtol=1e-03, atol=1e-05
    )

    logger.info(
        "Exported model has been tested with ONNXRuntime, and the result looks good!"
    )


if __name__ == "__main__":
    """Main entry point"""
    # convert_model(MODEL_TYPE)
    onnx_model_save_path = f"{MODEL_WEIGHTS_DIR.parent}/r50-fpn.onnx"
    test_converted_model(onnx_model_save_path)
