import fitz


"""
THIS IS A TEST SNIPEET FOR ASSESING THE METHOD USED EXTRACTING PDFS
BASED ON THEIR FONT SIZE

NO NEED TO RUN THIS CODE IN THE MAIN PROGRAM
"""


def analyze_fonts(filePath):
    font_data = {}
    pdf = fitz.open(filePath)
    for page in pdf:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block.keys():
                lines = block['lines']
                for line in lines:
                    spans = line['spans']
                    for span in spans:
                        font_size = span['size']
                        if font_size in font_data:
                            font_data[font_size] += 1
                        else:
                            font_data[font_size] = 1
    pdf.close()

    if font_data:
        sorted_fonts = sorted(font_data.items(), key=lambda item: item[1], reverse=True)
        main_font_size = sorted_fonts[0][0]
        next_size = sorted_fonts[1][0] if len(sorted_fonts) > 1 else main_font_size
    else:
        main_font_size = next_size = None

    return main_font_size, next_size


if __name__ == "__main__":
    filePath_siratimustakim = '../texture/data_pdfs/siratimustakim/cilt_3.pdf'
    filePath_sebil = '../texture/data_pdfs/sebiluressad/cilt_11.pdf'
    main_font_size, next_size = analyze_fonts(filePath_sebil)
    print(f"Main text font size: {main_font_size}")
    print(f"Next largest font size: {next_size}")