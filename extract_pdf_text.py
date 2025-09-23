#!/usr/bin/env python3
"""
Extract text from PDF file for JOSS conversion
"""

import PyPDF2
import sys

def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n--- PAGE {page_num + 1} ---\n"
                text += page.extract_text()
                text += "\n"
            
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

if __name__ == "__main__":
    pdf_path = "/Users/karthik/Offline Projects/BIE/BIE__Research_Paper_.pdf"
    text = extract_pdf_text(pdf_path)
    
    if text:
        # Save to text file for easier viewing
        with open("/Users/karthik/Offline Projects/BIE/extracted_paper_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Text extracted successfully and saved to extracted_paper_text.txt")
        
        # Print first 2000 characters to see the structure
        print("\n--- FIRST 2000 CHARACTERS ---")
        print(text[:2000])
    else:
        print("Failed to extract text from PDF")