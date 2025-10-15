# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""File containing NASNet-series search space.

The implementation is based on NDS.
It's called ``nasnet.py`` simply because NASNet is the first to propose such structure.
"""

from typing import Tuple, List, Union, Iterable, Optional, cast



try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import nni.nas.nn.pytorch as nn
from nni.nas import model_wrapper

from nni.nas.hub.pytorch.utils.fixed import FixedFactory

from nni.nas.hub.pytorch.nasnet import CellBuilder, NDSStage, DropPath_

class AuxiliaryHead(nn.Module):
    def __init__(self, C: int, num_labels: int, dataset: str):
        super().__init__()

        stride = 3 if dataset == 'cifar10' or dataset == 'cifar100' else 2

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

class NDS(nn.Module):

    def __init__(self,
                 op_candidates: List[str],
                 num_labels: int = 10,
                 merge_op: Literal['all', 'loose_end'] = 'all',
                 num_nodes_per_cell: int = 4,
                 width: Union[Tuple[int, ...], int] = 16,
                 num_cells: Union[Tuple[int, ...], int] = 20,
                 dataset: str = 'cifar10',
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.):
        super().__init__()

        self.dataset = dataset
        # self.num_labels = 10 if dataset == 'cifar' else 1000
        self.num_labels = num_labels
        self.auxiliary_loss = auxiliary_loss
        self.drop_path_prob = drop_path_prob
        self.op_candidates = op_candidates

        # preprocess the specified width and depth
        if isinstance(width, Iterable):
            C = nn.ValueChoice(list(width), label='width')
        else:
            C = width

        self.num_cells: nn.MaybeChoice[int] = cast(int, num_cells)
        if isinstance(num_cells, Iterable):
            self.num_cells = nn.ValueChoice(list(num_cells), label='depth')
        num_cells_per_stage = [(i + 1) * self.num_cells // 3 - i * self.num_cells // 3 for i in range(3)]

        # auxiliary head is different for network targetted at different datasets
        if dataset in ['mnist', 'fashion-mnist']:
            self.stem = nn.Sequential(
                nn.Conv2d(1, cast(int, 3 * C), 3, padding=1, bias=False),
                nn.BatchNorm2d(cast(int, 3 * C))
            )
            C_pprev = C_prev = 3 * C
            C_curr = C
            last_cell_reduce = False
        elif dataset in ['cifar10', 'cifar100']:
            self.stem = nn.Sequential(
                nn.Conv2d(3, cast(int, 3 * C), 3, padding=1, bias=False),
                nn.BatchNorm2d(cast(int, 3 * C))
            )
            C_pprev = C_prev = 3 * C
            C_curr = C
            last_cell_reduce = False
        else :
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, cast(int, C // 2), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cast(int, C // 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(cast(int, C // 2), cast(int, C), 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(cast(int, C), cast(int, C), 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            C_pprev = C_prev = C_curr = C
            last_cell_reduce = True

        self.stages = nn.ModuleList()
        for stage_idx in range(3):
            if stage_idx > 0:
                C_curr *= 2
            # For a stage, we get C_in, C_curr, and C_out.
            # C_in is only used in the first cell.
            # C_curr is number of channels for each operator in current stage.
            # C_out is usually `C * num_nodes_per_cell` because of concat operator.
            cell_builder = CellBuilder(op_candidates, C_pprev, C_prev, C_curr, num_nodes_per_cell,
                                       merge_op, stage_idx > 0, last_cell_reduce, drop_path_prob)
            stage: Union[NDSStage, nn.Sequential] = NDSStage(cell_builder, num_cells_per_stage[stage_idx])

            if isinstance(stage, NDSStage):
                stage.estimated_out_channels_prev = cast(int, C_prev)
                stage.estimated_out_channels = cast(int, C_curr * num_nodes_per_cell)
                stage.downsampling = stage_idx > 0

            self.stages.append(stage)

            # NOTE: output_node_indices will be computed on-the-fly in trial code.
            # When constructing model space, it's just all the nodes in the cell,
            # which happens to be the case of one-shot supernet.

            # C_pprev is output channel number of last second cell among all the cells already built.
            if len(stage) > 1:
                # Contains more than one cell
                C_pprev = len(cast(nn.Cell, stage[-2]).output_node_indices) * C_curr
            else:
                # Look up in the out channels of last stage.
                C_pprev = C_prev

            # This was originally,
            # C_prev = num_nodes_per_cell * C_curr.
            # but due to loose end, it becomes,
            C_prev = len(cast(nn.Cell, stage[-1]).output_node_indices) * C_curr

            # Useful in aligning the pprev and prev cell.
            last_cell_reduce = cell_builder.last_cell_reduce

            if stage_idx == 2:
                C_to_auxiliary = C_prev

        if auxiliary_loss:
            assert isinstance(self.stages[2], nn.Sequential), 'Auxiliary loss can only be enabled in retrain mode.'
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, self.num_labels, dataset=self.dataset)  # type: ignore

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(cast(int, C_prev), self.num_labels)

    def forward(self, inputs):
        if self.dataset not in ['cifar10', 'cifar100', 'mnist', 'fashion-mnist']:
            s0 = self.stem0(inputs)
            s1 = self.stem1(s0)
        else:
            s0 = s1 = self.stem(inputs)

        for stage_idx, stage in enumerate(self.stages):
            if stage_idx == 2 and self.auxiliary_loss and self.training:
                assert isinstance(stage, nn.Sequential), 'Auxiliary loss is only supported for fixed architecture.'
                for block_idx, block in enumerate(stage):
                    # auxiliary loss is attached to the first cell of the last stage.
                    s0, s1 = block([s0, s1])
                    if block_idx == 0:
                        logits_aux = self.auxiliary_head(s1)
            else:
                s0, s1 = stage([s0, s1])

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self.training and self.auxiliary_loss:
            return logits, logits_aux  # type: ignore
        else:
            return logits

    def set_drop_path_prob(self, drop_prob):
        """
        Set the drop probability of Drop-path in the network.
        Reference: `FractalNet: Ultra-Deep Neural Networks without Residuals <https://arxiv.org/pdf/1605.07648v4.pdf>`__.
        """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.drop_prob = drop_prob

    @classmethod
    def fixed_arch(cls, arch: dict) -> FixedFactory:
        return FixedFactory(cls, arch)


@model_wrapper
class DARTS(NDS):
    __doc__ = """Search space proposed in `Darts: Differentiable architecture search <https://arxiv.org/abs/1806.09055>`__.

    It is built upon :class:`~nni.retiarii.nn.pytorch.Cell`, and implemented based on :class:`~nni.retiarii.hub.pytorch.nasnet.NDS`.
    Its operator candidates are :attr:`~DARTS.DARTS_OPS`.
    It has 4 nodes per cell, and the output is concatenation of all nodes in the cell.

    .. note::

        ``none`` is not included in the operator candidates.
        It has already been handled in the differentiable implementation of cell.

    """

    DARTS_OPS = [
        # 'none',
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
    ]
    """The candidate operations."""

    def __init__(self,
                 num_labels: int = 10,
                 width: Union[Tuple[int, ...], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
                 dataset: str = 'cifar10',
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.,
                 darts_ops: Optional[List[str]] = None
                 ):
        # if darts_ops is None or empty, use the default DARTS_OPS
        if not darts_ops:
            darts_ops = self.DARTS_OPS
        super().__init__(darts_ops,
                         num_labels=num_labels,
                         merge_op='all',
                         num_nodes_per_cell=4,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss,
                         drop_path_prob=drop_path_prob)
