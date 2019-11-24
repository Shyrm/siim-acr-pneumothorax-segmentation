from fastai.vision import *


class AccumulateOptimWrapper(OptimWrapper):
    def step(self):           pass

    def zero_grad(self):      pass

    def real_step(self):      super().step()

    def real_zero_grad(self): super().zero_grad()


class AccGradLearner(Learner):

    def create_opt(self, lr: Floats, wd: Floats = 0.) -> None:
        "Create optimizer with `lr` learning rate and `wd` weight decay."
        self.opt = AccumulateOptimWrapper.create(self.opt_func, lr, self.layer_groups,
                                                 wd=wd, true_wd=self.true_wd, bn_wd=self.bn_wd)
