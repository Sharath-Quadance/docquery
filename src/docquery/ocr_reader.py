import abc
import logging
from typing import Any, List, Optional, Tuple
import time

import numpy as np
from pydantic.fields import ModelField

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials


class NoOCRReaderFound(Exception):
    def __init__(self, e):
        self.e = e

    def __str__(self):
        return f"Could not load OCR Reader: {self.e}"


OCR_AVAILABLE = {
    "tesseract": False,
    "easyocr": False,
    "dummy": True,
    "azureocr":True,
}

try:
    import pytesseract  # noqa

    pytesseract.get_tesseract_version()
    OCR_AVAILABLE["tesseract"] = True
except ImportError:
    pass
except pytesseract.TesseractNotFoundError as e:
    logging.warning("Unable to find tesseract: %s." % (e))
    pass

try:
    import easyocr  # noqa

    OCR_AVAILABLE["easyocr"] = True
except ImportError:
    pass


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class OCRReader(metaclass=SingletonMeta):
    def __init__(self):
        # TODO: add device here
        self._check_if_available()

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field: ModelField):
        if not isinstance(v, cls):
            raise TypeError("Invalid value")
        return v

    @abc.abstractmethod
    def apply_ocr(self, image: "Image.Image") -> Tuple[List[Any], List[List[int]]]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _check_if_available():
        raise NotImplementedError


class TesseractReader(OCRReader):
    def __init__(self):
        super().__init__()

    def apply_ocr(self, image: "Image.Image") -> Tuple[List[str], List[List[int]]]:
        """
        Applies Tesseract on a document image, and returns recognized words + normalized bounding boxes.
        This was derived from LayoutLM preprocessing code in Huggingface's Transformers library.
        """
        data = pytesseract.image_to_data(image, output_type="dict")
        words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

        # filter empty words and corresponding coordinates
        irrelevant_indices = set(idx for idx, word in enumerate(words) if not word.strip())
        words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
        left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
        top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
        width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
        height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

        # turn coordinates into (left, top, left+width, top+height) format
        actual_boxes = [[x, y, x + w, y + h] for x, y, w, h in zip(left, top, width, height)]

        return words, actual_boxes

    @staticmethod
    def _check_if_available():
        if not OCR_AVAILABLE["tesseract"]:
            raise NoOCRReaderFound(
                "Unable to use pytesseract (OCR will be unavailable). Install tesseract to process images with OCR."
            )
            
            
class AzureOCRReader(OCRReader):
    def __init__(self,subscription_key,endpoint):
        self.subscription_key = subscription_key
        self.endpoint = endpoint
        #super().__init__()
    
    def apply_ocr(self, image: "Image.Image") -> Tuple[List[str], List[List[int]]]:
        """Applies Azure OCR on a document image, and returns recognized words + normalized bounding boxes."""
        subscription_key = self.subscription_key
        endpoint = self.endpoint
        
        image_path = "image.jpg"
        image.save(image_path, 'JPEG')
        byte_image = open(image_path, "rb")
        
        for attemt in range(10):
            try:
                computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
                read_response = computervision_client.read_in_stream(byte_image, raw=True)
            except Exception as e:
                print("error in azure OCR.. Retrying: "+str(attemt) +str(e))
                time.sleep(20)
            else:
                break
        else:
            raise Exception("exception in azure ocr")
        
        read_operation_location = read_response.headers["Operation-Location"]
        # Grab the ID from the URL
        operation_id = read_operation_location.split("/")[-1]
        # Retrieve the results
        while True:
          read_result = computervision_client.get_read_result(operation_id)
          if read_result.status not in ['notStarted', 'running']:
              break
          time.sleep(1)
        width, height = image.size
        w_scale = 1000 / width
        h_scale = 1000 / height
        
        az_data=[]
        texts,words, boxes,lineBoxes, actual_boxes=[],[],[],[],[]
        
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                if "MALE" in line.text.upper() or "FEMALE" in line.text.upper():
                    line.text = "Gender: "+ line.text
                texts.append(line.text)
                lineBox = [round(min(line.bounding_box[0],line.bounding_box[6])),round(min(line.bounding_box[1],line.bounding_box[3])),round(max(line.bounding_box[2],line.bounding_box[4])),round(max(line.bounding_box[5],line.bounding_box[7]))]
                lineBoxes.append(lineBox)
                for word in (line.words):
                    az_data.append([word.text,
                                    [round(min(word.bounding_box[0],word.bounding_box[6])*w_scale),round(min(word.bounding_box[1],word.bounding_box[3])*h_scale),round(max(word.bounding_box[2],word.bounding_box[4])*w_scale),round(max(word.bounding_box[5],word.bounding_box[7])*h_scale)],
                                    [round(min(word.bounding_box[0],word.bounding_box[6])),round(min(word.bounding_box[1],word.bounding_box[3])),round(max(word.bounding_box[2],word.bounding_box[4])),round(max(word.bounding_box[5],word.bounding_box[7]))]])
                    words.append(word.text)
                    actual_box=[round(min(word.bounding_box[0],word.bounding_box[6])),round(min(word.bounding_box[1],word.bounding_box[3])),round(max(word.bounding_box[2],word.bounding_box[4])),round(max(word.bounding_box[5],word.bounding_box[7]))]
                    actual_boxes.append(actual_box)
                    #words, boxes, actual_boxes=[],[],[]
        time.sleep(20)
        return texts, lineBoxes
                    

class EasyOCRReader(OCRReader):
    def __init__(self):
        super().__init__()
        self.reader = None

    def apply_ocr(self, image: "Image.Image") -> Tuple[List[str], List[List[int]]]:
        """Applies Easy OCR on a document image, and returns recognized words + normalized bounding boxes."""
        if not self.reader:
            # TODO: expose language currently setting to english
            self.reader = easyocr.Reader(["en"])  # TODO: device here example: gpu=self.device > -1)

        # apply OCR
        data = self.reader.readtext(np.array(image))
        boxes, words, acc = list(map(list, zip(*data)))

        # filter empty words and corresponding coordinates
        irrelevant_indices = set(idx for idx, word in enumerate(words) if not word.strip())
        words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
        boxes = [coords for idx, coords in enumerate(boxes) if idx not in irrelevant_indices]

        # turn coordinates into (left, top, left+width, top+height) format
        actual_boxes = [tl + br for tl, tr, br, bl in boxes]

        return words, actual_boxes

    @staticmethod
    def _check_if_available():
        if not OCR_AVAILABLE["easyocr"]:
            raise NoOCRReaderFound(
                "Unable to use easyocr (OCR will be unavailable). Install easyocr to process images with OCR."
            )


class DummyOCRReader(OCRReader):
    def __init__(self):
        super().__init__()
        self.reader = None

    def apply_ocr(self, image: "Image.Image") -> Tuple[(List[str], List[List[int]])]:
        raise NoOCRReaderFound("Unable to find any OCR reader and OCR extraction was requested")

    @staticmethod
    def _check_if_available():
        logging.warning("Falling back to a dummy OCR reader since none were found.")


OCR_MAPPING = {
    "tesseract": TesseractReader,
    "easyocr": EasyOCRReader,
    "dummy": DummyOCRReader,
    "azureocr": AzureOCRReader,
}


def get_ocr_reader(ocr_reader_name: Optional[str] = None, subscription_key = None,endpoint = None):
    if not ocr_reader_name:
        for name, reader in OCR_MAPPING.items():
            if OCR_AVAILABLE[name]:
                return reader()
            
    if ocr_reader_name == "azureocr":
        return OCR_MAPPING[ocr_reader_name](subscription_key,endpoint)

    if ocr_reader_name in OCR_MAPPING.keys():
        if OCR_AVAILABLE[ocr_reader_name]:
            return OCR_MAPPING[ocr_reader_name]()
        else:
            raise NoOCRReaderFound(f"Failed to load: {ocr_reader_name} Please make sure its installed correctly.")
    else:
        raise NoOCRReaderFound(
            f"Failed to find: {ocr_reader_name} in the available ocr libraries. The choices are: {list(OCR_MAPPING.keys())}"
        )
