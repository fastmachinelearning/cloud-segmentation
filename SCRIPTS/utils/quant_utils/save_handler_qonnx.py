import os
import numbers
import stat
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Mapping, Union, BinaryIO, IO
import ignite.distributed as idist
from ignite.engine import Engine, Events
import torch
from torch import nn
from ignite.handlers import Checkpoint, DiskSaver
from ignite.handlers.checkpoint import BaseSaveHandler
from utils.quant_utils.export.export_qonnx import export_model_qonnx

from copy import deepcopy


def checkpoint_qonnx(
    checkpoint: Mapping,
    export_path: Union[str, os.PathLike, BinaryIO, IO[bytes]],
):
    model_shape = checkpoint["model_shape"]
    if "student_model" in checkpoint.keys():
        model = checkpoint["student_model"]
    else:
        model = checkpoint["model"]

    model.eval()

    device = None
    for p in model.parameters():
        device = p.device

    x = torch.randn(
        model_shape[0],
        model_shape[1],
        model_shape[2],
        model_shape[3],
        requires_grad=False,
    ).to(device)

    export_path = Path(export_path)
    # export_path_student = export_path.with_name(export_path)

    export_model_qonnx(
        model=model,
        device=device,
        inp=x,
        export_path=export_path,
    )

    model.train()

    return


class CheckpointQONNX(Checkpoint):
    """CheckpointQONNX handler can be used to periodically save and load models to QONNX (a.k.a. Brevitas ONNX).
    This class can use specific save handlers to store on the disk or a cloud storage, etc.
    Args:
        see ignite.handlers.Checkpoint
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ext = "onnx"

    def __call__(self, engine: Engine) -> None:

        global_step = None
        if self.global_step_transform is not None:
            global_step = self.global_step_transform(
                engine, engine.last_event_name
            )

        if self.score_function is not None:
            priority = self.score_function(engine)
            if not isinstance(priority, numbers.Number):
                raise ValueError("Output of score_function should be a number")
        else:
            if global_step is None:
                global_step = engine.state.get_event_attrib_value(
                    Events.ITERATION_COMPLETED
                )
            priority = global_step

        if self._check_lt_n_saved() or self._compare_fn(priority):

            priority_str = (
                f"{priority}"
                if isinstance(priority, numbers.Integral)
                else f"{priority:.4f}"
            )

            checkpoint = self.to_save

            name = "checkpoint"

            filename_pattern = self._get_filename_pattern(global_step)

            filename_dict = {
                "filename_prefix": self.filename_prefix,
                "ext": self.ext,
                "name": name,
                "score_name": self.score_name,
                "score": (
                    priority_str if (self.score_function is not None) else None
                ),
                "global_step": global_step,
            }
            filename = filename_pattern.format(**filename_dict)

            metadata = {
                "basename": f"{self.filename_prefix}{'_' * int(len(self.filename_prefix) > 0)}{name}",
                "score_name": self.score_name,
                "priority": priority,
            }

            try:
                index = list(
                    map(lambda it: it.filename == filename, self._saved)
                ).index(True)
                to_remove = True
            except ValueError:
                index = 0
                to_remove = not self._check_lt_n_saved()

            if to_remove:
                item = self._saved.pop(index)
                if isinstance(self.save_handler, BaseSaveHandler):
                    self.save_handler.remove(item.filename)
                else:
                    os.remove(item.filename)

            self._saved.append(Checkpoint.Item(priority, filename))
            self._saved.sort(key=lambda it: it[0])

            if self.include_self:
                # Now that we've updated _saved, we can add our own state_dict.
                # checkpoint["checkpointer"] = self.state_dict()
                print(
                    f"CheckpointQONNX: doesn't support self.include_self = {self.include_self}"
                )

            try:
                self.save_handler(checkpoint, filename, metadata)
            except TypeError:
                self.save_handler(checkpoint, filename)

    @staticmethod
    def _check_objects(objs: Mapping, attr: str) -> None:
        def func(obj: Any, **kwargs: Any) -> None:
            if not hasattr(obj, attr):
                raise TypeError(
                    f"Object {type(obj)} should have `{attr}` method"
                )

        # only model has state_dict
        func(objs["model"])

    @staticmethod
    def load_objects(
        to_load: Mapping, checkpoint: Mapping, **kwargs: Any
    ) -> None:
        ###NOT IMPLEMENTED
        raise RuntimeError("load_object for CheckpointQONNX not implemented")


class DiskSaverQONNX(DiskSaver):
    """Handler that saves input model into QONNX format on a disk.
    Args:
        see ignite.handlers.DiskSaver
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        checkpoint: Mapping,
        filename: str,
        metadata: Optional[Mapping] = None,
    ) -> None:
        path = os.path.join(self.dirname, filename)

        if idist.has_xla_support:
            # all tpu procs should enter here as internally performs sync across device
            print("DiskSaverQONNX: xla not supported for ONNX save")
        else:
            self._save_func(checkpoint, path, checkpoint_qonnx)

    def _save_func(
        self, checkpoint: Mapping, path: Path, func: Callable
    ) -> None:
        if not self._atomic:
            func(checkpoint, path, **self.kwargs)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=self.dirname)
            tmp_file = tmp.file
            tmp_name = tmp.name
            try:
                func(checkpoint, tmp_name, **self.kwargs)
            except BaseException:
                tmp.close()
                os.remove(tmp_name)
                raise
            else:
                tmp.close()
                path = Path(path)
                export_path = path.with_name(
                    path.stem + checkpoint["model_suffix"] + path.suffix
                )

                os.replace(tmp_name, export_path)
                # # append group/others read mode
                os.chmod(
                    export_path,
                    os.stat(export_path).st_mode | stat.S_IRGRP | stat.S_IROTH,
                )

    @idist.one_rank_only()
    def remove(self, filename: str) -> None:
        path = os.path.join(self.dirname, filename)
        os.remove(path)
