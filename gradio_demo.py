import gradio as gr
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T
import cv2
from torchvision.ops import box_convert
import torch
from functools import cmp_to_key
import numpy as np
from PIL import Image
from llava.ocr import acs_ocr, py_ocr

def image_transform_grounding(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return init_image, image

def image_transform_grounding_for_vis(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return image


def is_contained(text_box, frame_box):
    '''
    Returns true if the text_box is contained in the frame_box
    text_box: [x1, y1, x2, y2]
    frame_box: [x1, y1, x2, y2]
    '''
    xt_1, yt_1, xt_2, yt_2 = text_box
    xf_1, yf_1, xf_2, yf_2 = frame_box
    return xt_1 >= xf_1 and yt_1 >= yf_1 and xt_2 <= xf_2 and yt_2 <= yf_2

# compare function for sorting contours from left to right
def compare(panel1, panel2):
    x1, y1, _, _ = cv2.boundingRect(panel1)
    x2, y2, _, _ = cv2.boundingRect(panel2)
    return x1 - x2

def get_frames(im):
    # im = cv2.imread(img_path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    avg = sum(areas) / len(areas)
    panel_contours = [contour for contour in contours if cv2.contourArea(contour) > avg]
    panel_contours.sort(key=cmp_to_key(compare))
    # get bounding boxes of panels
    frames = []
    for contour in panel_contours:
        x, y, w, h = cv2.boundingRect(contour)
        frames.append([x, y, x + w, y + h])
    return frames

def get_text_extracts(im):
    model = load_model("llava/model/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "llava/model/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    TEXT_PROMPT = "text . frame . character"
    BOX_THRESHOLD = 0.2 # needs to be low to detect the text boxes
    TEXT_THRESHOLD = 0.25
    init_image = im.convert("RGB")
    original_size = init_image.size

    _, image_tensor = image_transform_grounding(init_image)
    image_pil: Image = image_transform_grounding_for_vis(init_image)

    # run grounidng
    boxes, logits, phrases = predict(
        device="cpu",
        model=model,
        image=image_tensor,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    image_source = np.array(image_pil)
    # convert from cxcywh to xyxy
    h, w, _ = image_source.shape
    bboxes = boxes * torch.Tensor([w, h, w, h])
    bboxes = box_convert(boxes=bboxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    text_boxes = [bboxes[i] for i, phrase in enumerate(phrases) if phrase == "text"]
    char_boxes = [bboxes[i] for i, phrase in enumerate(phrases) if phrase == "character"]

    frame_boxes = get_frames(image_source) # use opencv to get the frames
    text_extracts_acs = {}
    text_extracts_py = {}
    for i, frame_box in enumerate(sorted(frame_boxes, key=lambda x: x[0])):
        text_extracts_acs["panel " + str(i + 1)] = []
        text_extracts_py["panel " + str(i + 1)] = []
        for text_box in sorted(text_boxes, key=lambda x: x[0]):
            if is_contained(text_box, frame_box):
                # Extract text from the image using OpenCV
                x1, y1, x2, y2 = map(int, text_box)
                
                text_via_acs_ocr = acs_ocr(image_source[y1:y2, x1:x2])
                text_via_py_ocr = py_ocr(image_source[y1:y2, x1:x2])
                # Extract text from the image using Azure Cognitive services OCR
                text_extracts_acs["panel " + str(i + 1)].append(text_via_acs_ocr)
                text_extracts_py["panel " + str(i + 1)].append(text_via_py_ocr)

    for panel in text_extracts_py:
        for i, text in enumerate(text_extracts_py[panel]):
            text_extracts_py[panel][i] = text.replace("\n\x0c", "").replace("\n", " ").replace("\x0c", "")

    annotated_image = image_source.copy()
    for box in text_boxes:
        cv2.rectangle(annotated_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    for box in frame_boxes:
        cv2.rectangle(annotated_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    for box in char_boxes:
        cv2.rectangle(annotated_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    return text_extracts_acs, text_extracts_py, annotated_image

def ocr_app(image):
    # image_path = "uploaded_image.jpg"
    # cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image = convert_cv_to_pil(image)
    text_extracts_acs, text_extracts_py, annotated_image = get_text_extracts(image)
    return text_extracts_acs, text_extracts_py, annotated_image

def convert_cv_to_pil(image):
    return Image.fromarray(image)
if __name__ == "__main__":
    iface = gr.Interface(fn=ocr_app, inputs="image", outputs=[gr.JSON(label="ACS OCR"), gr.JSON(label="Pytesseract OCR"), "image"])
    iface.launch()
