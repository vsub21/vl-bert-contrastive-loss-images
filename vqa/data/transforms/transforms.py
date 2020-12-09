import math
import numbers
import random
import warnings

import numpy as np

import torch
import torchvision
from torchvision.transforms import functional as F
# from torchvision.transforms import functional_pil as F_pil # available in future version of torchvision
# from torchvision.transforms import functional_tensor as F_t

from PIL import Image, ImageFilter

# try:
#     import accimage
# except ImportError:
#     accimage = None

# NOTE: VQA doesn't use masks, so any new custom transforms will not modify masks in any way.

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):
        for t in self.transforms:
            # print(f'transform t: {t}')
            image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout = t(image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout)
        return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomApply(object):
    def __init__(self, transforms, p=0.5):
        # super().__init__()
        self.transforms = transforms
        self.p = p

    def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):
        if self.p < torch.rand(1):
            return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout
        for t in self.transforms:
            image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout = t(image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout)
        return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomGaussianBlur(object):
    def __init__(self, radius=5, p=0.5):
        self.gaussian_filter = ImageFilter.GaussianBlur(radius=radius) # using PIL.ImageFilter.GaussianBlur
        self.prob = p

    def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):
        if random.random() < self.prob:
            image = image.filter(self.gaussian_filter)
            blur = True
        return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(max_size * min_original_size / max_original_size)

        if (w <= h and w == size) or (h <= w and h == size):
            return (w, h)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (ow, oh)

    def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):
        origin_size = im_info[:2]
        size = self.get_size(origin_size)
        if image is not None:
            image = F.resize(image, (size[1], size[0]))

        ratios = [size[0] * 1.0 / origin_size[0], size[1] * 1.0 / origin_size[1]]
        if boxes is not None:
            boxes[:, [0, 2]] *= ratios[0]
            boxes[:, [1, 3]] *= ratios[1]
        im_info[0], im_info[1] = size
        im_info[2], im_info[3] = ratios
        return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.prob = p

    def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):
        if random.random() < self.prob:
            w, h = im_info[:2]
            if image is not None:
                image = F.hflip(image)
            if boxes is not None:
                boxes[:, [0, 2]] = w - 1 - boxes[:, [2, 0]]
            if masks is not None:
                masks = torch.as_tensor(masks.numpy()[:, :, ::-1].tolist())
            flipped = not flipped
        return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout


class ToTensor(object):
    def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):
        return F.to_tensor(image) if image is not None else image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):
        if image is not None:
            if self.to_bgr255:
                image = image[[2, 1, 0]] * 255
            image = F.normalize(image, mean=self.mean, std=self.std)
        return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):
        image = self.color_jitter(image)
        color_jitter = True
        return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout


# class FixPadding(object):
#     def __init__(self, min_size, max_size, pad=0):
#         self.min_size = min_size
#         self.max_size = max_size
#         self.pad = pad

#     def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):

#         if image is not None:
#             # padding to fixed size for determinacy
#             c, h, w = image.shape
#             if h <= w:
#                 h1 = self.min_size
#                 w1 = self.max_size
#             else:
#                 h1 = self.max_size
#                 w1 = self.min_size
#             padded_image = image.new_zeros((c, h1, w1)).fill_(self.pad)
#             padded_image[:, :h, :w] = image
#             image = padded_image

#         return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout


class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    Args:
        p (float): probability that image should be converted to grayscale.
    Returns:
        PIL Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b
    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.
        Returns:
            PIL Image: Randomly grayscaled image.
        """
        num_output_channels = 1 if image.mode == 'L' else 3
        if random.random() < self.p:
            image = F.to_grayscale(image, num_output_channels=num_output_channels)
            grayscale = True
        return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.
    Returns:
        Erased Image.
    # Examples:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.ToTensor(),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        """Get parameters for ``erase`` for a random erasing.
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.
        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for _ in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                if isinstance(value, numbers.Number):
                    v = value
                elif isinstance(value, torch._six.string_classes):
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                elif isinstance(value, (list, tuple)):
                    v = torch.tensor(value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
        Returns:
            img (Tensor): Erased Tensor image.
        """
        if random.uniform(0, 1) < self.p:
            x, y, h, w, v = self.get_params(image, scale=self.scale, ratio=self.ratio, value=self.value)
            image = F.erase(image, x, y, h, w, v, self.inplace)
            cutout = True
        return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout

# class RandomErasing(object):
#     """ Randomly selects a rectangle region in an image and erases its pixels.
#     'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

#     Args:
#          p: probability that the random erasing operation will be performed.
#          scale: range of proportion of erased area against input image.
#          ratio: range of aspect ratio of erased area.
#          value: erasing value. Default is 0. If a single int, it is used to
#             erase all pixels. If a tuple of length 3, it is used to erase
#             R, G, B channels respectively.
#             If a str of 'random', erasing each pixel with random values.
#          inplace: boolean to make this transform inplace. Default set to False.

#     Returns:
#         Erased Image.

#     Example:
#         >>> transform = transforms.Compose([
#         >>>   transforms.RandomHorizontalFlip(),
#         >>>   transforms.ToTensor(),
#         >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         >>>   transforms.RandomErasing(),
#         >>> ])
#     """

#     def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
#         super().__init__()
#         if not isinstance(value, (numbers.Number, str, tuple, list)):
#             raise TypeError("Argument value should be either a number or str or a sequence")
#         if isinstance(value, str) and value != "random":
#             raise ValueError("If value is str, it should be 'random'")
#         if not isinstance(scale, (tuple, list)):
#             raise TypeError("Scale should be a sequence")
#         if not isinstance(ratio, (tuple, list)):
#             raise TypeError("Ratio should be a sequence")
#         if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
#             warnings.warn("Scale and ratio should be of kind (min, max)")
#         if scale[0] < 0 or scale[1] > 1:
#             raise ValueError("Scale should be between 0 and 1")
#         if p < 0 or p > 1:
#             raise ValueError("Random erasing probability should be between 0 and 1")

#         self.p = p
#         self.scale = scale
#         self.ratio = ratio
#         self.value = value
#         self.inplace = inplace

#     @staticmethod
#     def get_params(img, scale, ratio, value=None):
#     #         img: Tensor, scale: Tuple[float, float], ratio: Tuple[float, float], value: Optional[List[float]] = None
#     # ) -> Tuple[int, int, int, int, Tensor]:
#         """Get parameters for ``erase`` for a random erasing.

#         Args:
#             img (Tensor): Tensor image to be erased.
#             scale (tuple or list): range of proportion of erased area against input image.
#             ratio (tuple or list): range of aspect ratio of erased area.
#             value (list, optional): erasing value. If None, it is interpreted as "random"
#                 (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
#                 i.e. ``value[0]``.

#         Returns:
#             tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
#         """
#         img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
#         area = img_h * img_w

#         for _ in range(10):
#             erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
#             aspect_ratio = torch.empty(1).uniform_(ratio[0], ratio[1]).item()

#             h = int(round(math.sqrt(erase_area * aspect_ratio)))
#             w = int(round(math.sqrt(erase_area / aspect_ratio)))
#             if not (h < img_h and w < img_w):
#                 continue

#             if value is None:
#                 v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
#             else:
#                 v = torch.tensor(value)[:, None, None]

#             i = torch.randint(0, img_h - h + 1, size=(1, )).item()
#             j = torch.randint(0, img_w - w + 1, size=(1, )).item()
#             return i, j, h, w, v

#         # Return original image
#         return 0, 0, img_h, img_w, img


#     def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):
#         """
#         Args:
#             img (Tensor): Tensor image to be erased.

#         Returns:
#             img (Tensor): Erased Tensor image.
#         """
#         if torch.rand(1) < self.p:

#             # cast self.value to script acceptable type
#             if isinstance(self.value, (int, float)):
#                 value = [self.value, ]
#             elif isinstance(self.value, str):
#                 value = None
#             elif isinstance(self.value, tuple):
#                 value = list(self.value)
#             else:
#                 value = self.value

#             if value is not None and not (len(value) in (1, image.shape[-3])):
#                 raise ValueError(
#                     "If value is a sequence, it should have either a single value or "
#                     "{} (number of input channels)".format(image.shape[-3])
#                 )

#             x, y, h, w, v = self.get_params(image, scale=self.scale, ratio=self.ratio, value=value)
#             image = F.erase(image, x, y, h, w, v, self.inplace)
#             cutout = True
#         return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout


# class RandomResizedCrop(torch.nn.Module):
#     """Crop the given image to random size and aspect ratio.
#     The image can be a PIL Image or a Tensor, in which case it is expected
#     to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

#     A crop of random size (default: of 0.08 to 1.0) of the original size and a random
#     aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
#     is finally resized to given size.
#     This is popularly used to train the Inception networks.

#     Args:
#         size (int or sequence): expected output size of each edge. If size is an
#             int instead of sequence like (h, w), a square output size ``(size, size)`` is
#             made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
#         scale (tuple of float): range of size of the origin size cropped
#         ratio (tuple of float): range of aspect ratio of the origin aspect ratio cropped.
#         interpolation (int): Desired interpolation enum defined by `filters`_.
#             Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
#             and ``PIL.Image.BICUBIC`` are supported.
#     """

#     def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
#         super().__init__()
#         # self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
#         if len(size) == 2:
#             self.size = size

#         # if not isinstance(scale, Sequence):
#         #     raise TypeError("Scale should be a sequence")
#         # if not isinstance(ratio, Sequence):
#         #     raise TypeError("Ratio should be a sequence")
#         if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
#             warnings.warn("Scale and ratio should be of kind (min, max)")

#         self.interpolation = interpolation
#         self.scale = scale
#         self.ratio = ratio

#     @staticmethod
#     def get_params(img, scale, ratio):
#         """Get parameters for ``crop`` for a random sized crop.

#         Args:
#             img (PIL Image or Tensor): Input image.
#             scale (list): range of scale of the origin size cropped
#             ratio (list): range of aspect ratio of the origin aspect ratio cropped

#         Returns:
#             tuple: params (i, j, h, w) to be passed to ``crop`` for a random
#                 sized crop.
#         """
#         width, height = F._get_image_size(img)
#         area = height * width

#         for _ in range(10):
#             target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
#             log_ratio = torch.log(torch.tensor(ratio))
#             aspect_ratio = torch.exp(
#                 torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
#             ).item()

#             w = int(round(math.sqrt(target_area * aspect_ratio)))
#             h = int(round(math.sqrt(target_area / aspect_ratio)))

#             if 0 < w <= width and 0 < h <= height:
#                 i = torch.randint(0, height - h + 1, size=(1,)).item()
#                 j = torch.randint(0, width - w + 1, size=(1,)).item()
#                 return i, j, h, w

#         # Fallback to central crop
#         in_ratio = float(width) / float(height)
#         if in_ratio < min(ratio):
#             w = width
#             h = int(round(w / min(ratio)))
#         elif in_ratio > max(ratio):
#             h = height
#             w = int(round(h * max(ratio)))
#         else:  # whole image
#             w = width
#             h = height
#         i = (height - h) // 2
#         j = (width - w) // 2
#         return i, j, h, w

#     def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):
#         """
#         Args:
#             img (PIL Image or Tensor): Image to be cropped and resized.

#         Returns:
#             PIL Image or Tensor: Randomly cropped and resized image.
#         """
#         i, j, h, w = self.get_params(image, self.scale, self.ratio)
#         image = F.resized_crop(image, i, j, h, w, self.size, self.interpolation)

#         # Implement resized crop on bounding box

#         return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout


#     def __repr__(self):
#         _pil_interpolation_to_str = {
#             Image.NEAREST: 'PIL.Image.NEAREST',
#             Image.BILINEAR: 'PIL.Image.BILINEAR',
#             Image.BICUBIC: 'PIL.Image.BICUBIC',
#             Image.LANCZOS: 'PIL.Image.LANCZOS',
#             Image.HAMMING: 'PIL.Image.HAMMING',
#             Image.BOX: 'PIL.Image.BOX',
#         }
#         interpolate_str = _pil_interpolation_to_str[self.interpolation]
#         format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
#         format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
#         format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
#         format_string += ', interpolation={0})'.format(interpolate_str)
#         return format_string
#
## class RandomGrayscale(object):
#     """Randomly convert image to grayscale with a probability of p (default 0.1).
#     The image can be a PIL Image or a Tensor, in which case it is expected
#     to have [..., 3, H, W] shape, where ... means an arbitrary number of leading
#     dimensions

#     Args:
#         p (float): probability that image should be converted to grayscale.

#     Returns:
#         PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
#         with probability (1-p).
#         - If input image is 1 channel: grayscale version is 1 channel
#         - If input image is 3 channel: grayscale version is 3 channel with r == g == b

#     """

#     def __init__(self, p=0.1):
#         # super().__init__()
#         self.p = p

#     def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):
#         """
#         Args:
#             img (PIL Image or Tensor): Image to be converted to grayscale.

#         Returns:
#             PIL Image or Tensor: Randomly grayscaled image.
#         """
#         num_output_channels = self._get_image_num_channels(image) # from future version of torchvision
#         if torch.rand(1) < self.p:
#             image = F.rgb_to_grayscale(image, num_output_channels=num_output_channels)
#             grayscale = True
#         return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout

#     def _get_image_num_channels(self, img):
#         if isinstance(img, torch.Tensor):
#             return F_t._get_image_num_channels(img)

#         return self._get_image_num_channels(img) # F_pil (functional_pil) does not exist in this version of torchvision

#     def _get_image_num_channels(self, img):
#         if self._is_pil_image(img):
#             return 1 if img.mode == 'L' else 3
#         raise TypeError("Unexpected type {}".format(type(img)))

#     def _is_pil_image(self, img):
#         if accimage is not None:
#             return isinstance(img, (Image.Image, accimage.Image))
#         else:
#             return isinstance(img, Image.Image)

#     def __repr__(self):
#         return self.__class__.__name__ + '(p={0})'.format(self.p)
#
#
# class GaussianBlur(object):
#     def __init__(self, kernel_size, sigma=(0.1, 2.0)):
#         # super().__init__()
#         self.kernel_size = list(kernel_size)

#         for idx, ks in enumerate(self.kernel_size):
#             if ks <= 0 or ks % 2 == 0:
#                 if ks > 1:
#                     self.kernel_size[idx] = ks - 1
#                 else:
#                     self.kernel_size[idx] = ks + 1
#                 # raise ValueError("Kernel size value should be an odd and positive number.")

#         self.kernel_size = tuple(kernel_size)

#         # if isinstance(sigma, numbers.Number):
#         #     if sigma <= 0:
#         #         raise ValueError("If sigma is a single number, it must be positive.")
#         #     sigma = (sigma, sigma)
#         # elif isinstance(sigma, Sequence) and len(sigma) == 2:
#         #     if not 0. < sigma[0] <= sigma[1]:
#         #         raise ValueError("sigma values should be positive and of the form (min, max).")
#         # else:
#         #     raise ValueError("sigma should be a single number or a list/tuple with length 2.")

#         self.sigma = sigma

#     @staticmethod
#     def get_params(sigma_min: float, sigma_max: float) -> float:
#         """Choose sigma for random gaussian blurring.

#         Args:
#             sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
#             sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

#         Returns:
#             float: Standard deviation to be passed to calculate kernel for gaussian blurring.
#         """
#         return torch.empty(1).uniform_(sigma_min, sigma_max).item()

#     def __call__(self, image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout):
#         """
#         Args:
#             img (PIL Image or Tensor): image to be blurred.

#         Returns:
#             PIL Image or Tensor: Gaussian blurred image
#         """
#         sigma = self.get_params(self.sigma[0], self.sigma[1])
#         image = F.gaussian_blur(image, self.kernel_size, [sigma, sigma])

#         return image, boxes, masks, im_info, flipped, resize, color_jitter, grayscale, blur, cutout