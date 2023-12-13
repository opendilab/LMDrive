import imgaug as ia
from imgaug import augmenters as iaa

def augment(prob=0.2):
    augmenter = iaa.Sequential([
        iaa.Sometimes(prob, iaa.GaussianBlur((0, 0.5))),
        iaa.Sometimes(prob, iaa.AdditiveGaussianNoise(loc=0, scale=(0., 0.05*255), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.Dropout((0.01, 0.1), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.Multiply((1/1.2, 1.2), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.LinearContrast((1/1.2, 1.2), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.Grayscale((0.0, 1))),
        iaa.Sometimes(prob, iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
    ], random_order=True)
    return augmenter
