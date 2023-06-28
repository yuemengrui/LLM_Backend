# *_*coding:utf-8 *_*
# @Author : YueMengRui
import cv2
import fitz
import base64
import requests
import numpy as np
from config import OCR_URL
from flask import current_app
from typing import Any, List, Optional
from langchain.docstore.document import Document
from .chinese_text_splitter import ChineseTextSplitter
from langchain.document_loaders.pdf import PyMuPDFLoader, BasePDFLoader
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, UnstructuredMarkdownLoader, \
    UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader, CSVLoader, \
    UnstructuredODTLoader

LOADER_MAPPING = {
    "csv": (CSVLoader, {}),
    "doc": (UnstructuredWordDocumentLoader, {}),
    "docx": (UnstructuredWordDocumentLoader, {}),
    "html": (UnstructuredHTMLLoader, {}),
    "md": (UnstructuredMarkdownLoader, {}),
    "odt": (UnstructuredODTLoader, {}),
    "ppt": (UnstructuredPowerPointLoader, {}),
    "pptx": (UnstructuredPowerPointLoader, {}),
    "txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def get_ocr_res(image):
    if isinstance(image, np.ndarray):
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
    elif isinstance(image, bytes):
        image_bytes = image
    else:
        raise ('not supported type: {}, type should be np.ndarray or bytes'.format(type(image)))

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    try:
        res = requests.post(
            url=OCR_URL,
            json={'image': image_base64})

        text_list = [x['text'][0] for x in res.json()['data']['results']]
        text = ''.join(text_list)
        return text
    except Exception as e:
        current_app.logger.error(str({'ocr_error': e}) + '\n')
        return ''


class PDFOCRLoader(BasePDFLoader):
    """Loader that uses OCR to load PDF files."""

    def __init__(self, file_path: str):
        super().__init__(file_path)

    def load(self, **kwargs: Optional[Any]) -> List[Document]:

        doc = fitz.open(self.file_path)
        file_path = self.file_path if self.web_path is None else self.web_path

        doc_pages = doc.page_count

        docs = []
        for i in range(doc_pages):
            try:
                zoom_x = 2.0
                zoom_y = 2.0
                trans = fitz.Matrix(zoom_x, zoom_y)
                pm = doc[i].get_pixmap(matrix=trans)
                page_text = get_ocr_res(pm.tobytes())
            except Exception as e:
                current_app.logger.error(str({'pdf_image_ocr_error': e}) + '\n')
                page_text = ''

            docs.append(Document(page_content=page_text,
                                 metadata={"source": file_path, "page_number": i + 1, "total_pages": doc_pages}))

        return docs


def check_pdf_can_extract(pdf_path):
    doc = fitz.open(pdf_path)
    doc_text = ''
    for page in doc:
        doc_text += page.get_text().strip()

    if len(doc_text) > 10:
        return True

    return False


def load_file(filepath, pdf=False):
    if filepath.lower().endswith(".pdf"):
        if check_pdf_can_extract(filepath):
            loader = PyMuPDFLoader(filepath)
            pdf = True
        else:
            loader = PDFOCRLoader(filepath)
            pdf = True
    else:
        ext = filepath.lower().split('.')[-1]
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(filepath, **loader_args)
        else:
            current_app.logger.warning(str({'Unsupported file extension': '{}'.format(ext)}) + '\n')
            loader = UnstructuredFileLoader(filepath, mode="elements")

    textsplitter = ChineseTextSplitter(pdf=pdf)
    docs = loader.load_and_split(text_splitter=textsplitter)

    return docs
