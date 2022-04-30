import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Layer):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, axis=-1)
        nll_loss = -logprobs[paddle.arange(logprobs.shape[0]).unsqueeze(1), target.unsqueeze(1)]
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(axis=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class _SoftTargetCrossEntropy(nn.Layer):

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N == N_rep:
            target = target.repeat(N_rep // N,1)
        loss = paddle.sum(-target * F.log_softmax(x, axis=-1), axis=-1)
        return loss.mean()


class TokenLabelCrossEntropy(nn.Layer):
    """
    Token labeling loss.
    """

    def __init__(self, dense_weight=1.0, cls_weight=1.0,
                 mixup_active=True, classes=1000, ground_truth=False):
        """
        Constructor Token labeling loss.
        """
        super(TokenLabelCrossEntropy, self).__init__()

        self.CE = _SoftTargetCrossEntropy()

        self.dense_weight = dense_weight
        self.mixup_active = mixup_active
        self.classes = classes
        self.cls_weight = cls_weight
        self.ground_truth = ground_truth
        assert dense_weight + cls_weight > 0

    def forward(self, x, target):
        output, aux_output, bb = x
        bbx1, bby1, bbx2, bby2 = bb

        B, N, C = aux_output.shape
        if len(target.shape) == 2:
            target_cls = target
            target_aux = target.repeat([1, N]).reshape([B * N, C])
        else:
            target_cls = target[:, :, 1]
            if self.ground_truth:
                # use ground truth to help correct label.
                # rely more on ground truth if target_cls is incorrect.
                ground_truth = target[:, :, 0]
                ratio = (0.9 - 0.4 * (ground_truth.max(-1)
                         [1] == target_cls.max(-1)[1])).unsqueeze(-1)
                target_cls = target_cls * ratio + ground_truth * (1 - ratio)
            target_aux = target[:, :, 2:]
            target_aux = target_aux.transpose([0, 2, 1]).reshape([-1, C])
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / N)
        if lam < 1:
            target_cls = lam * target_cls + (1 - lam) * target_cls.flip(0)

        aux_output = aux_output.reshape([-1, C])
        loss_cls = self.CE(output, target_cls)
        loss_aux = self.CE(aux_output, target_aux)
        return self.cls_weight * loss_cls + self.dense_weight * loss_aux
