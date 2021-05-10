# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RefCoco datasets."""

import collections
import json
from operator import itemgetter

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """
A collection of 3 referring expression datasets based off images in the
COCO dataset. A referring expression is a piece of text that describes a
unique object in an image. These datasets are collected by asking human raters
to disambiguate objects delineated by bounding boxes in the COCO dataset.

RefCoco and RefCoco+ are from Kazemzadeh et al. 2014. RefCoco+ expressions
are strictly appearance based descriptions, which they enforced by preventing
raters from using location based descriptions (e.g., "person to the right" is
not a valid description for RefCoco+). RefCocoG is from Mao et al. 2016, and
has more rich description of objects compared to RefCoco due to differences
in the annotation process. In particular, RefCoco was collected in an
interactive game-based setting, while RefCocoG was collected in a
non-interactive setting. On average, RefCocoG has 8.4 words per expression
while RefCoco has 3.5 words.

Each dataset has different split allocations that are typically all reported
in papers. The "testA" and "testB" sets in RefCoco and RefCoco+ contain only
people and only non-people respectively. Images are partitioned into the various
splits. In the "google" split, objects, not images, are partitioned between the
train and non-train splits. This means that the same image can appear in both
the train and validation split, but the objects being referred to in the image
will be different between the two sets. In contrast, the "unc" and "umd" splits
partition images between the train, validation, and test split.
In RefCocoG, the "google" split does not have a canonical test set,
and the validation set is typically reported in papers as "val*".

Stats for each dataset and split ("refs" is the number of referring expressions,
and "images" is the number of images):

  dataset  partition  split   refs   images
 ===========================================
   refcoco   google   train   40000   19213
   refcoco   google     val    5000    4559
   refcoco   google    test    5000    4527
   refcoco      unc   train   42404   16994
   refcoco      unc     val    3811    1500
   refcoco      unc   testA    1975     750
   refcoco      unc   testB    1810     750
  refcoco+      unc   train   42278   16992
  refcoco+      unc     val    3805    1500
  refcoco+      unc   testA    1975     750
  refcoco+      unc   testB    1798     750
  refcocog   google   train   44822   24698
  refcocog   google     val    5000    4650
  refcocog      umd   train   42226   21899
  refcocog      umd     val    2573    1300
  refcocog      umd    test    5023    2600
"""

_CITATION = """
@inproceedings{kazemzadeh2014referitgame,
  title={Referitgame: Referring to objects in photographs of natural scenes},
  author={Kazemzadeh, Sahar and Ordonez, Vicente and Matten, Mark and Berg, Tamara},
  booktitle={Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)},
  pages={787--798},
  year={2014}
}
@inproceedings{yu2016modeling,
  title={Modeling context in referring expressions},
  author={Yu, Licheng and Poirson, Patrick and Yang, Shan and Berg, Alexander C and Berg, Tamara L},
  booktitle={European Conference on Computer Vision},
  pages={69--85},
  year={2016},
  organization={Springer}
}
@inproceedings{mao2016generation,
  title={Generation and Comprehension of Unambiguous Object Descriptions},
  author={Mao, Junhua and Huang, Jonathan and Toshev, Alexander and Camburu, Oana and Yuille, Alan and Murphy, Kevin},
  booktitle={CVPR},
  year={2016}
}
@inproceedings{nagaraja2016modeling,
  title={Modeling context between objects for referring expression understanding},
  author={Nagaraja, Varun K and Morariu, Vlad I and Davis, Larry S},
  booktitle={European Conference on Computer Vision},
  pages={792--807},
  year={2016},
  organization={Springer}
}
"""


def _build_bbox(image_info, x, y, width, height):
  """Calculates the coordinates of a bbox."""
  return tfds.features.BBox(
      ymin=y / image_info['height'],
      xmin=x / image_info['width'],
      ymax=(y + height) / image_info['height'],
      xmax=(x + width) / image_info['width'],
  )


def _extract_annotation(ann, image_info):
  """Extracts the bounding box annotation information."""
  return {
      'id': ann['id'],
      'area': ann['area'],
      'bbox': _build_bbox(image_info, *ann['bbox']),
      'label': ann['category_id'],
  }


def generate_examples(refcoco_json, dataset, dataset_partition, split):
  """Generates examples of images and its refexps & ground truth bboxes.

  Args:
    refcoco_json: contents of the annotation file.
    dataset: str specifying the dataset (refcoco, refcoco+, refcocog)
    dataset_partition: str specifying the partition for the dataset
    split: str specifying the split of the dataset_partition

  Yields:
    image_id and example tuple
  """
  refs = refcoco_json['ref']
  coco = refcoco_json['coco_anns']

  # Collect all referring expressions for a given image.
  imageid2annref = collections.defaultdict(list)
  for r in refs:
    if r['dataset'] == dataset and r[
        'dataset_partition'] == dataset_partition and r['split'] == split:
      imageid2annref[r['image_id']].append(r)

  # Process all the referring expressions and ground truth annotations for
  # a given COCO image.
  for image_id in sorted(imageid2annref.keys()):
    coco_image = coco[str(image_id)]
    image_info = coco_image['info']
    example = {
        'image_filename': image_info['file_name'],
        'image/id': image_id,
        'coco_annotations': [],
        'objects': [],
    }

    # Collect ground truth bboxes.
    for ann in sorted(coco_image['anns'], key=itemgetter('id')):
      example['coco_annotations'].append(_extract_annotation(ann, image_info))

    # Collect referring expressions.
    for r in sorted(imageid2annref[image_id], key=itemgetter('ref_id')):
      obj = _extract_annotation(r['ann'], image_info)

      refexp = []
      for s in sorted(r['sentences'], key=itemgetter('sent_id')):
        refexp.append({
            'raw': s['raw'],
            'refexp_id': s['sent_id'],
        })

      # Match the referring expression to its corresponding bbox in the ground
      # truth list.
      gt_box_index = [
          i for i, v in enumerate(example['coco_annotations'])
          if v['id'] == r['ann']['id']
      ]
      if len(gt_box_index) != 1:
        raise ValueError(f'gt_box_index does not have length 1: {gt_box_index}')
      gt_box_index = gt_box_index[0]

      obj.update({
          'refexp': refexp,
          'gt_box_index': gt_box_index,
      })
      example['objects'].append(obj)

    yield image_id, example


class RefCocoConfig(tfds.core.BuilderConfig):
  """Config to specify each RefCoco variant."""

  def __init__(self, dataset, dataset_partition, **kwargs):
    name = f'{dataset}/{dataset_partition}'
    super(RefCocoConfig, self).__init__(name=name, **kwargs)
    self.dataset = dataset
    self.dataset_partition = dataset_partition


class RefCoco(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for RefCoco datasets."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  1. Follow the instructions in https://github.com/lichengunc/refer and
  download the annotations and the images, matching the data/ directory
  specified in the repo.

  2. Follow the instructions of PythonAPI in
  https://github.com/cocodataset/cocoapi to get pycocotools and the
  instances_train2014 annotations file from https://cocodataset.org/#download

  3. Add both refer.py from (1) and pycocotools from (2) to your PYTHONPATH.

  4. Run the following code to generate refcoco.json, replacing `ref_data_root`,
  `coco_annotations_file`, and `out_file` with the values corresponding to
  where you have downloaded / want to save these files:

    import json
    from refer import REFER
    from pycocotools.coco import COCO

    ref_data_root = '<path/to/refer/data/folder>'
    all_refs = []
    for dataset, split_bys in [
        ('refcoco', ['google', 'unc']),
        ('refcoco+', ['unc']),
        ('refcocog', ['google', 'umd'])
    ]:
        for split_by in split_bys:
            refer = REFER(ref_data_root, dataset, split_by)
            for ref_id in refer.getRefIds():
                ref = refer.Refs[ref_id]
                ann = refer.refToAnn[ref_id]
                ref['ann'] = ann
                ref['dataset'] = dataset
                ref['dataset_partition'] = split_by
                all_refs.append(ref)

    coco_annotations_file = '<path/to/instances_train2014.json>'
    coco = COCO(coco_annotations_file)
    ref_image_ids = set(x['image_id'] for x in all_refs)
    coco_anns = {image_id: {'info': coco.imgs[image_id],
                            'anns': coco.imgToAnns[image_id]}
                 for image_id in ref_image_ids}

    out_file = '<path/to/refcoco.json>'
    with open(out_file, 'w') as f:
      json.dump({'ref': all_refs, 'coco_anns': coco_anns}, f)

  5. Download the COCO training set from https://cocodataset.org/#download
  and stick it into a folder called `coco_train2014/`. Move `refcoco.json`
  to the same level as `coco_train2014`.

  6. Run this pipeline.
  """

  BUILDER_CONFIGS = [
      RefCocoConfig(dataset='refcoco', dataset_partition='unc'),
      RefCocoConfig(dataset='refcoco', dataset_partition='google'),
      RefCocoConfig(dataset='refcoco+', dataset_partition='unc'),
      RefCocoConfig(dataset='refcocog', dataset_partition='google'),
      RefCocoConfig(dataset='refcocog', dataset_partition='umd'),
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        homepage='https://github.com/lichengunc/refer',
        features=tfds.features.FeaturesDict({
            'image':
                tfds.features.Image(encoding_format='jpeg'),
            'image/id':
                tf.int64,
            'objects':
                tfds.features.Sequence({
                    'id': tf.int64,
                    'area': tf.int64,
                    'bbox': tfds.features.BBoxFeature(),
                    'label': tf.int64,
                    'gt_box_index': tf.int64,
                    'refexp':
                        tfds.features.Sequence({
                            'refexp_id': tf.int64,
                            'raw': tfds.features.Text(),
                        }),
                }),
            'coco_annotations':
                tfds.features.Sequence({
                    'id': tf.int64,
                    'area': tf.int64,
                    'bbox': tfds.features.BBoxFeature(),
                    'label': tf.int64,
                }),
        }),
        supervised_keys=None,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    allowed_splits = {
        ('refcoco', 'google'): [
            tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST],
        ('refcoco', 'unc'): [
            tfds.Split.TRAIN, tfds.Split.VALIDATION, 'testA', 'testB'],
        ('refcoco+', 'unc'): [
            tfds.Split.TRAIN, tfds.Split.VALIDATION, 'testA', 'testB'],
        ('refcocog', 'google'): [
            tfds.Split.TRAIN, tfds.Split.VALIDATION],
        ('refcocog', 'umd'): [
            tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST],
    }
    bc = self.builder_config
    splits = allowed_splits[(bc.dataset, bc.dataset_partition)]

    return [
        tfds.core.SplitGenerator(
            name=split,
            gen_kwargs={
                'dataset': bc.dataset,
                'dataset_partition': bc.dataset_partition,
                'split': split,
                'dl_manager': dl_manager,
            },
        ) for split in splits
    ]

  def _generate_examples(self, dataset, dataset_partition, split, dl_manager):
    refcoco_json = json.loads(
        (dl_manager.manual_dir / 'refcoco.json').read_text())
    coco_dir = dl_manager.manual_dir / 'coco_train2014'

    if split == tfds.Split.VALIDATION:
      split = 'val'

    for image_id, example in generate_examples(refcoco_json, dataset,
                                               dataset_partition, split):
      example['image'] = coco_dir / example['image_filename']
      del example['image_filename']
      yield image_id, example
