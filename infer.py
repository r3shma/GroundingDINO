from groundingdino.util.inference import load_model, load_image, predict, annotate
from PIL import Image
import pytesseract
import cv2
from pprint import pprint
from torchvision.ops import box_convert
import torch
import os
from functools import cmp_to_key


model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "weights/2022-07-27.PNG"
TEXT_PROMPT = "text . frame . character"
BOX_TRESHOLD = 0.25 #needs to be low to detect the text boxes
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)
pil_img = Image.open(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)


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
def compare(panel1,panel2):
    x1,y1,_,_ = cv2.boundingRect(panel1)
    x2,y2,_,_ = cv2.boundingRect(panel2)
    return x1-x2

def get_frames(img_path):
    im = cv2.imread(img_path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
    (contours, hierarchy) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    avg = sum(areas)/len(areas)
    panel_contours = [contour for contour in contours if cv2.contourArea(contour) > avg]
    panel_contours.sort(key=cmp_to_key(compare))
    #get bounding boxes of panels
    frames = []
    for contour in panel_contours:
        x,y,w,h = cv2.boundingRect(contour)
        frames.append([x,y,x+w,y+h])
    return frames

def get_text_extracts():
    '''
    text_extracts = {"panel 1" : ["sorry I'm late. I had a car problem", "panel 2": ["What kind of car problem?","I didn't get in it soon enough"], "panel 3": ["That sounds like a 'you' problem","Then my stupid car took me to starbucks."]]
    '''
    # convert from cxcywh to xyxy
    h, w, _ = image_source.shape
    bboxes = boxes * torch.Tensor([w, h, w, h])
    bboxes = box_convert(boxes=bboxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    text_boxes = [bboxes[i] for i, phrase in enumerate(phrases) if phrase == "text"]
    #frame_boxes = [bboxes[i] for i, phrase in enumerate(phrases) if phrase == "frame"]
    frame_boxes = get_frames(IMAGE_PATH) # use opencv to get the frames
    text_extracts = {}
    for i, frame_box in enumerate(sorted(frame_boxes, key=lambda x: x[0])):
        text_extracts["panel " + str(i+1)] = []
        for text_box in sorted(text_boxes, key=lambda x: x[0]):
            if is_contained(text_box, frame_box):
                # Extract text from the text_box
                ocr = pytesseract.image_to_string(pil_img.crop(text_box))
                text_extracts["panel " + str(i+1)].append(ocr)

    for panel in text_extracts:
        for i, text in enumerate(text_extracts[panel]):
            text_extracts[panel][i] = text.replace("\n\x0c", "").replace("\n", " ")
    return text_extracts

def test_get_text_extracts():
    text_extracts = get_text_extracts()
    print(text_extracts)

test_get_text_extracts()