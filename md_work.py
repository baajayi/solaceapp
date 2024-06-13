import os
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def write_to_md(text, md_file):
    try:
        with open(md_file, 'w') as file:
            file.write(text)
    except Exception as e:
        print(f"Error writing to {md_file}: {e}")

if __name__ == "__main__":
    pdf_directory = "covchurch"  # Change this to the directory containing your PDF files
    output_directory = "MDFiles"  # Change this to the directory where you want to save MD files

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through PDF files in the directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            md_filename = os.path.splitext(filename)[0] + ".md"
            md_path = os.path.join(output_directory, md_filename)

            # Extract text from PDF and write to MD file
            text = extract_text_from_pdf(pdf_path)
            if text:  # Only write if text extraction was successful
                write_to_md(text, md_path)
