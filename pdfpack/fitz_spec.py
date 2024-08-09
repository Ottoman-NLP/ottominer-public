import os
import re
import logging
logging.basicConfig(filename='removal_script.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def remove_footnotes(text):
    footnotes = [
        "Abone bedeli peşindir.",
        "Mesleğe muvafık asar",
        "maal-memnüniyye kabul olunur.",
        "Derc edilmeyen yazılar iade olunmaz.",
        "Adres tebdilinde ayrıca beş kuruş gönderilmelidir.",
        "Mektubların imzaları vazıh ve okunaklı olması ve abone sıra numarasını muhtevi bulunması lazımdır.",
        "Memalik-i ecnebiyye için abone olanların adreslerinin Fransızca yazılması rica olunur.",
        "Para gönderildiği zaman neye dair olduğu bildirilmesi rica olunur.",
        "Her yer için seneliği (400), altı aylığı (225) memalik-i ecnebiyye için seneliği (450) altı aylığı (250) kuruşdur.",
        "Nüshası 7,5 kuruşdur.",
        "Seneliği 50 adeddir."
    ]
    for footnote in footnotes:
        text = text.replace(footnote, '')
    return text

def capture_and_format_titles(txt_input_path, txt_output_path):
    try:
        with open(txt_input_path, 'r', encoding='utf-8') as input_file:
            content = input_file.readlines()
    except Exception as e:
        print(f"Failed to read {txt_input_path}. Error: {e}")
        return

    formatted_lines = []
    i = 0

    while i < len(content):
        line = content[i].strip()
        line = re.sub(r'\s+', ' ', line)

        if re.match(r'^(?:[A-ZİÖÜÇŞĞ ]+)$', line):
            title_lines = [line]
            while i + 1 < len(content) and re.match(r'^(?:[A-ZİÖÜÇŞĞ ]+)$', content[i + 1].strip()):
                i += 1
                title_lines.append(re.sub(r'\s+', ' ', content[i].strip()))

            full_title = ' '.join(title_lines)
            formatted_lines.append(f"\n\n####\n{full_title}\n####\n\n")
        elif re.match(r'^([A-ZİÖÜÇŞĞ][a-zâêîôûäëïöüçğşİ]*\s*)+$', line):
            formatted_lines.append(f"\n\n****\n{line}\n****\n\n")

        else:
            if line:
                formatted_lines.append(line)
        i += 1

    updated_content = "\n".join(formatted_lines)

    try:
        with open(txt_input_path, 'w', encoding='utf-8') as output_file:
            output_file.write(updated_content)
        print(f"Updated titles in {txt_input_path}")
    except Exception as e:
        print(f"Failed to write to {txt_input_path}. Error: {e}")

def process_all_txt_files(source_directory, destination_directory):

    specific_files = ["cilt_25.txt", "cilt_24.txt", "cilt_23.txt", "cilt_22.txt",
                      "cilt_20.txt", "cilt_19.txt", "cilt_17.txt", "cilt_16.txt", "cilt_14.txt"]

    for root, _, files in os.walk(source_directory):
        for file in files:
            if file in specific_files:
                txt_input_path = os.path.join(root, file)
                txt_output_path = os.path.join(destination_directory, file)
                print(f"Processing: {txt_input_path} to {txt_output_path}")
                capture_and_format_titles(txt_input_path, txt_output_path)



source_directory = '../texture/data_txt/sebiluressad'
destination_directory = '../texture/data_txt/sebiluressad'
process_all_txt_files(source_directory, destination_directory)
