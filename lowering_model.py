import torch
from torch.export import export, ExportedProgram
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge
from executorch.exir.backend.backend_api import to_backend

import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import mobilenet_v3_small

from typing import Literal
import os
from GazeTrackingKeyboard.model import TinyTracker
from tqdm import tqdm


def lowering_model(model: nn.Module, model_path: str, backend: Literal['xnnpack'] = 'xnnpack'):
    tqdm.write('Lowering the Model...')
    model = model.eval()
    dummy_data = (torch.randn(1, 3, 112, 112),)
    exported_program: ExportedProgram = export(model, dummy_data)
    edge: EdgeProgramManager = to_edge(exported_program)

    edge = edge.to_backend(XnnpackPartitioner())

    exec_prog: ExecutorchProgramManager = edge.to_executorch()

    with open(os.path.join(model_path, f'{backend}_model.pte'), 'wb') as file:
        exec_prog.write_to_file(file)

    tqdm.write(
        f'Lowering Model Done! \nModel in {os.path.join(model_path, f"{backend}_model.pte")}')

if __name__ == '__main__':

    model = TinyTracker(3).eval()
    dummy_data = (torch.randn(1, 3, 112, 112),)
    exported_program: ExportedProgram = export(model, dummy_data)
    edge: EdgeProgramManager = to_edge(exported_program)

    edge = edge.to_backend(XnnpackPartitioner())

    # print(edge.exported_program().graph_module)

    exec_prog = edge.to_executorch()

    # TODO: Quantizing -> https://url.kr/h15din
    with open('model/xnnpack_model.pte', 'wb') as file:
        exec_prog.write_to_file(file)
