from mxnet.gluon.loss import Loss


class smooth_l1(Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(smooth_l1, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label):
        loss = F.smooth_l1(pred-label, scalar=1.)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class FocalLoss(Loss):
    def __init__(self,axis=-1,alpha=0.25,gamma=2,batch_axis=0,**kwargs):
        super(FocalLoss,self).__init__(None,batch_axis,**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.axis = axis
        self.batch_axis = batch_axis

    def hybrid_forward(self, F, y, label):
        y = F.softmax(y)
        pt = F.pick(y, label, axis=self.axis, keepdims=True)
        loss = -self.alpha * ((1 - pt) ** self.gamma) * F.log(pt)
        return F.mean(loss, axis=self._batch_axis, exclude=True)