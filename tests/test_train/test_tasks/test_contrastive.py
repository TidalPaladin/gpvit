import pytest
import pytorch_lightning as pl

from gpvit.train.tasks.contrastive import ContrastiveEmbedding


class TestContrastiveEmbedding:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        return ContrastiveEmbedding(backbone, optimizer_init=optimizer_init)

    def test_fit(self, task, datamodule, logger):
        trainer = pl.Trainer(
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)

    def test_predict(self, task, datamodule, logger):
        trainer = pl.Trainer(
            fast_dev_run=True,
            logger=logger,
        )
        trainer.predict(task, datamodule=datamodule)
