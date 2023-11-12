import pytest
import pytorch_lightning as pl

from gpvit.train.tasks.jepa import JEPA


@pytest.fixture(params=[False, True])
def loss_includes_unmasked(request):
    return request.param


class TestJEPA:
    @pytest.fixture
    def task(self, optimizer_init, backbone, loss_includes_unmasked):
        return JEPA(backbone, optimizer_init=optimizer_init, loss_includes_unmasked=loss_includes_unmasked)

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
