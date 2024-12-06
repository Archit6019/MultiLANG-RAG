import io
import os 
import base64
import uuid
import traceback
import pytesseract
from PIL import Image, ImageOps
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid 

class Processor:
    def __init__(self, embedder : SentenceTransformer) -> None:
        self.embedder = embedder
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=500, 
            separators=[" ", ",", "\n"]
        )

    def get_embeddings(self, text: str):
        return self.embedder.encode(text).tolist()

    def ocr_page(self, page):
        try:
            temp_grayscale = ImageOps.grayscale(page)
            return pytesseract.image_to_string(temp_grayscale)
        except Exception as e:
            print(f"[ERROR] Error in OCR for page: {e}")
            traceback.print_exc()
            return ""
        
    def extract(self, content: bytes) -> str:
        extracted_content = ""
        try:
            pdf_reader = PdfReader(io.BytesIO(content), strict=False)
            for page in pdf_reader.pages:
                extracted_content += page.extract_text()
        except Exception as e:
            print(f"[ERROR] PyPDF2 text extraction failed: {e}")
            extracted_content = ""

        if not extracted_content.strip():
            print("[WARNING] No text extracted using PyPDF2. Falling back to OCR.")
            
            try:
                pdf_pages = convert_from_bytes(content)
                
                ocr_texts = []
                for page in pdf_pages:
                    page_text = self.ocr_page(page)
                    if page_text.strip():
                        ocr_texts.append(page_text)
                
                extracted_content = "\n".join(ocr_texts)
            
            except Exception as e:
                print(f"[ERROR] OCR processing failed: {e}")
                traceback.print_exc()
                return ""
            
        return extracted_content
    
    def Preprocess(self, file_content: bytes):
        unique_id = uuid.uuid4()

        text = self.extract(content=file_content)
        if text is None:
            print(f"[WARNING] No text extracted from")
            return None , 0
        
        embeddings = []
        chunks = self.text_splitter.split_text(text)
        for i, j in enumerate(chunks):
            chunk = f"passage: {j}"
            embedding = self.get_embeddings(text=chunk)
            output = {
                "document_id": str(unique_id),
                "text": j,
                "embedding": embedding
            }
            embeddings.append(output)

        print(f"[INGO] Chunks generated, {len(chunks)}")
        return embeddings, len(chunks)
    
      
    