import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import PyPDF2

# Define the paths
data_input_folder = 'data_input'

# Ensure data_input folder exists
if not os.path.exists(data_input_folder):
    os.makedirs(data_input_folder)

def convert_epub_to_text(file_path):
    book = epub.read_epub(file_path)
    text = ''
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text += soup.get_text()
    return text

def convert_pdf_to_text(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def convert_files_to_text(input_folder):
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        try:

            if filename.endswith('.txt'):
                # TXT file, just move it to data_input
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()

            elif filename.endswith('.epub'):
                # EPUB file, convert to text
                text_content = convert_epub_to_text(file_path)

            elif filename.endswith('.pdf'):
                # PDF file, convert to text
                text_content = convert_pdf_to_text(file_path)

            else:
                print(f"Unsupported file format: {filename}")
                continue
        except: print(f"{filename} is invalid")

        # Save the text content to data_input folder
        output_file = os.path.join(data_input_folder, filename.replace('.pdf', '.txt').replace('.epub', '.txt'))
        with open(output_file, 'w', encoding='utf-8') as out_file:
            out_file.write(text_content)
            print(f"Converted {filename} to {output_file}")

if __name__ == "__main__":
    input_folder = "input_files"  # Path where original PDFs, EPUBs, and TXTs are stored
    convert_files_to_text(input_folder)
