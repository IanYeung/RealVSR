import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'VideoSR_AllPair_YCbCr_Combine':
        from .VideoSR_AllPair_model_YCbCr_Combine import VideoSRModel as M
    elif model == 'VideoSR_AllPair_YCbCr_Split':
        from .VideoSR_AllPair_model_YCbCr_Split import VideoSRModel as M
    elif model == 'VideoSRGAN_AllPair_YCbCr_Split':
        from .VideoSRGAN_AllPair_model_YCbCr_Split import VideoSRGANModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
