import numpy as np 
import torch
from common.utils.clip_pad import *


class BatchCollator(object):
    def __init__(self, dataset, append_ind=False):
        self.dataset = dataset
        self.test_mode = self.dataset.test_mode
        self.data_names = self.dataset.data_names
        self.append_ind = append_ind

    def __call__(self, batch):
        if not isinstance(batch, list):
            batch = list(batch)

        if batch[0][self.data_names.index('image')] is not None:
            max_shape = tuple(max(s) for s in zip(*[data[self.data_names.index('image')].shape for data in batch]))
            image_none = False
        else:
            image_none = True
        max_boxes = max([data[self.data_names.index('boxes')].shape[0] for data in batch])
        max_question_length = max([len(data[self.data_names.index('question')]) for data in batch])

        for i, ibatch in enumerate(batch):
            out = {}

            if image_none:
                out['image'] = None
                out['image1'] = None
            else:
                image = ibatch[self.data_names.index('image')]
                image1 = ibatch[self.data_names.index('image1')] 
                out['image'] = clip_pad_images(image, max_shape, pad=0)
                out['image1'] = clip_pad_images(image1, max_shape, pad=0) 

            boxes = ibatch[self.data_names.index('boxes')]
            out['boxes'] = clip_pad_boxes(boxes, max_boxes, pad=-2)

            question = ibatch[self.data_names.index('question')]
            out['question'] = clip_pad_1d(question, max_question_length, pad=0)

            array = [0 for i in range(15)]
            if ibatch[self.data_names.index('index')] % 7 == 0:
                array[7] = 1
            out['index'] = torch.tensor(array)

            other_names = [data_name for data_name in self.data_names if data_name not in out]
            for name in other_names:
                if name == 'boxes1':
                    boxes1 = ibatch[self.data_names.index('boxes1')]
                    out['boxes1'] = clip_pad_boxes(boxes1, max_boxes, pad=-2)
                elif name == 'question1':
                    question1 = ibatch[self.data_names.index('question1')]
                    out['question1'] = clip_pad_1d(question1, max_question_length, pad=0)
                else: 
                    out[name] = torch.as_tensor(ibatch[self.data_names.index(name)])

            batch[i] = tuple(out[data_name] for data_name in self.data_names)
            if self.append_ind:
                batch[i] += (torch.tensor(i, dtype=torch.int64),)

        out_tuple = ()
        for items in zip(*batch):
            if items[0] is None or items[3] is None:
                out_tuple += (None,)
            else:
                out_tuple += (torch.stack(tuple(items), dim=0), )

        return out_tuple

