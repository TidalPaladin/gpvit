import pytest
import pytorch_lightning as pl

from gpvit.train.tasks import MAE, MultiTask, QueryPatch


class TestMultiTask:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        tasks = [
            ("mae", MAE(backbone, optimizer_init=optimizer_init)),
            ("query-patch", QueryPatch(backbone, optimizer_init=optimizer_init)),
        ]
        task = MultiTask(tasks, optimizer_init=optimizer_init)
        return task

    def test_fit(self, task, datamodule, logger):
        trainer = pl.Trainer(
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)
