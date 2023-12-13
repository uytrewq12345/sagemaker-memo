import typer
from groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm
import torchvision
import torch
import fiftyone as fo
from demo.inference_on_a_image import get_grounding_output
from collections import deque
import yaml

def find_char_all_index(text, find_char):
    indexes = []
    for i, char in enumerate(text):
        if char == find_char:
            indexes.append(i)
    return indexes

def calc_token_spans(caption):
    tmp_max = 10000000000

    space_indexes = deque(find_char_all_index(caption, " "))
    space_indexes.append(tmp_max)
    sep_indexes = deque(find_char_all_index(caption, "."))

    token_spans = [ ]
    space_index = space_indexes.popleft()
    sep_index = sep_indexes.popleft()
    start = 0

    tokens = []
    while True:
        if space_index < sep_index:
            tokens.append( [start, space_index])
            start = space_index + 1
            space_index = space_indexes.popleft()
        else :
            tokens.append( [start, sep_index ] )
            token_spans.append(tokens)
            start = sep_index + 1
            tokens = []

            if len(sep_indexes) > 0:
                sep_index = sep_indexes.popleft()
            else:
                break

    return  token_spans

def separate_phrase_logits(input_string):
    start = input_string.find('(')
    end = input_string.find(')')

    if start == -1 or end == -1:
        raise ValueError("format error:", input_string)

    phrase = input_string[:start].strip()
    logits = input_string[start+1:end]

    return phrase, float(logits) 


def main(
        image_directory: str = 'test_grounding_dino',
        text_prompt: str = 'Red Car.Blue car.white Car.yellow car.black car.',
        box_threshold: float = 0.3, 
        text_threshold: float = 0.25,
        export_dataset: bool = True,
        view_dataset: bool = False,
        export_annotated_images: bool = True,
        weights_path : str = "./weights/groundingdino_swint_ogc.pth",
        config_path: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        subsample: int = None,
        cpu_only: bool = True,
        token_spans = None,
    ):

    model = load_model(config_path, weights_path)

    dataset = fo.Dataset.from_images_dir(image_directory)

    samples = []

    if subsample is not None: 
        
        if subsample < len(dataset):
            dataset = dataset.take(subsample).clone()
    
    for sample in tqdm(dataset):

        image_source, image = load_image(sample.filepath)

        phrases = []
        logits  = []
        boxes, phrases_with_logits = get_grounding_output(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            cpu_only=cpu_only,
            token_spans=token_spans
        )
        for pl in phrases_with_logits:
            phrase, logit = separate_phrase_logits(pl)
            phrases.append(phrase)
            logits.append(logit)

        detections = [] 

        for box, logit, phrase in zip(boxes, logits, phrases):

            rel_box = torchvision.ops.box_convert(box, 'cxcywh', 'xywh')

            detections.append(
                fo.Detection(
                    label=phrase, 
                    bounding_box=rel_box,
                    confidence=logit,
            ))

        # Store detections in a field name of your choice
        sample["detections"] = fo.Detections(detections=detections)
        sample.save()

    # loads the voxel fiftyone UI ready for viewing the dataset.
    if view_dataset:
        session = fo.launch_app(dataset)
        session.wait()
        
    # exports COCO dataset ready for training
    if export_dataset:
        dataset.export(
            'coco_dataset',
            dataset_type=fo.types.COCODetectionDataset,
        )
        
    # saves bounding boxes plotted on the input images to disk
    if export_annotated_images:
        dataset.draw_labels(
            'images_with_bounding_boxes',
            label_fields=['detections']
        )


if __name__ == '__main__':
    with open('config.yaml', 'r') as yml:
        cfg = yaml.safe_load(yml)
    text_prompt = ".".join(cfg["class_list"])
    text_prompt += "."
    text_prompt = text_prompt.lower()
    token_spans = calc_token_spans(text_prompt)
    print(token_spans)

    main(
        image_directory = cfg["image_directory"],
        text_prompt = text_prompt,
        box_threshold = cfg["box_threshold"],
        text_threshold = cfg["text_threshold"],
        export_dataset = cfg["export_dataset"],
        view_dataset = cfg["view_dataset"],
        export_annotated_images = cfg["export_annotated_images"],
        weights_path = cfg["weights_path"],
        config_path = cfg["config_path"],
        subsample = cfg["subsample"],
        cpu_only = cfg["cpu_only"],
        token_spans = token_spans,
    )
