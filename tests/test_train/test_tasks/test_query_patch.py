import pytest
import pytorch_lightning as pl

from gpvit.train.tasks.query_patch import QueryPatch


class TestQueryPatch:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        return QueryPatch(backbone, optimizer_init=optimizer_init)

    def test_fit(self, task, datamodule, logger):
        trainer = pl.Trainer(
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)
