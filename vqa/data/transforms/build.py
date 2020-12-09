from . import transforms as T
import torchvision.transforms as transforms


def build_transforms(cfg, mode='train'):
    assert mode in ['train', 'test', 'val']
    min_size = cfg.SCALES[0]
    max_size = cfg.SCALES[1]
    assert min_size <= max_size

    if mode == 'train':
        flip_prob = cfg.TRAIN.FLIP_PROB
    elif mode == 'test':
        flip_prob = cfg.TEST.FLIP_PROB
    else:
        flip_prob = cfg.VAL.FLIP_PROB

    to_bgr255 = True

    normalize_transform = T.Normalize(
        mean=cfg.NETWORK.PIXEL_MEANS, std=cfg.NETWORK.PIXEL_STDS, to_bgr255=to_bgr255
    )

    # transform = T.Compose(
    #     [
    #         T.Resize(min_size, max_size),
    #         T.RandomHorizontalFlip(flip_prob),
    #         T.ToTensor(),
    #         normalize_transform,
    #         T.FixPadding(min_size, max_size, pad=0)
    #     ]
    # )

    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )

    return transform

def build_transforms_second_image(cfg, mode='train'):
    # TODO:
    #   - Modify transformations to use ones from SimCLR
    #   - Add transformations to transforms.py

    assert mode in ['train', 'test', 'val']
    min_size = cfg.SCALES[0]
    max_size = cfg.SCALES[1]
    assert min_size <= max_size

    if mode == 'train':
        flip_prob = cfg.TRAIN.FLIP_PROB
    elif mode == 'test':
        flip_prob = cfg.TEST.FLIP_PROB
    else:
        flip_prob = cfg.VAL.FLIP_PROB

    to_bgr255 = True
 
    # SimCLR Transformations
    # https://github.com/sthalles/SimCLR/blob/e8a690ae4f4359528cfba6f270a9226e3733b7fa/data_aug/dataset_wrapper.py#L61
    data_transforms = T.Compose([
        # first image transformations
        T.Resize(min_size, max_size),
        T.RandomHorizontalFlip(flip_prob),
        # second image transformations
        T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=(int(0.1 * min_size), int(0.1 * min_size))),
        T.RandomErasing(),
        # tensor and normalize
        T.ToTensor(),
        T.Normalize(mean=cfg.NETWORK.PIXEL_MEANS, std=cfg.NETWORK.PIXEL_STDS, to_bgr255=to_bgr255),
    ])

    return data_transforms