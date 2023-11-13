from deep_helpers.tasks import MultiTask as MultiTaskBase


class MultiTask(MultiTaskBase):
    r"""MultiTask wrapper for training on multiple tasks at once."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.share_attribute("backbone")
