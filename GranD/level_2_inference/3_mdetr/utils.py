import re
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours
from matplotlib.patches import Polygon
import nltk
from nltk.corpus import words
import spacy

spacy.prefer_gpu()
# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def plot_results(pil_img, scores, boxes, labels, masks=None, output_file_path="mdetr_prediction.jpg"):
    plt.figure(figsize=(16, 10))
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
        masks = [None for _ in range(len(scores))]
    assert len(scores) == len(boxes) == len(labels) == len(masks)
    for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{l}: {s:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

        if mask is None:
            continue
        np_image = apply_mask(np_image, mask, c)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=c)
            ax.add_patch(p)

    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig(output_file_path)


nltk.download("words")
word_set = set(words.words())


def process_phrase(phrase):
    words = phrase.split()
    result = []
    current_word = ""

    for word in words:
        current_word += word
        if current_word in word_set:
            result.append(current_word)
            current_word = ""

    if current_word:
        result.append(current_word)

    return " ".join(result)


def get_blip2_phrases(blip2_caption):
    doc = nlp(blip2_caption.capitalize())
    nouns, phrases = [], []
    for token in doc:
        if token.pos_ == "NOUN":
            noun = token.text
            phrase = " ".join([child.text for child in token.subtree])
            if noun != phrase:
                phrase = phrase.lower().strip()
                if phrase.endswith("."):
                    phrase = phrase[:-1]
                phrases.append(phrase)
                nouns.append(noun)

    return nouns, phrases


def get_llava_phrases(llava_caption):
    doc = nlp(llava_caption.capitalize())

    all_phrase_string = ""
    nouns, phrases = [], []
    for token in doc:
        if token.pos_ == "NOUN":
            noun = token.text
            phrase_list = [child for child in token.subtree]
            if len(phrase_list) > 1:
                if len(phrase_list) == 2 and phrase_list[0].pos_ in ['DET', 'PRON'] and phrase_list[1].pos_ == 'NOUN':
                    continue
                phrase = " ".join([child.text for child in phrase_list])
                if "," in phrase:
                    sub_phrase_list = phrase.split(',')
                    sub_phrase = []
                    for sub in sub_phrase_list:
                        if noun in sub:
                            sub_phrase.append(sub.strip())
                            break
                        sub_phrase.append(sub.strip())
                    if len(sub_phrase) > 0:
                        phrase = ",".join([child for child in sub_phrase])
                if noun != phrase:
                    if phrase not in all_phrase_string:
                        all_phrase_string = all_phrase_string + " " + phrase
                        nouns.append(noun)
                        phrases.append(phrase)

    return nouns, phrases
