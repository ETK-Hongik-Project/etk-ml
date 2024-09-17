import torch
from torch.export import export, ExportedProgram
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge
from executorch.exir.backend.backend_api import to_backend

import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import mobilenet_v3_small

from model import TinyTracker

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
