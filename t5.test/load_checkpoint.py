from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]
checkpoint_dir = base_dir / 'corpus-texts' / 'tokenizer_bpe' / 'checkpoint-19485'

model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
hf_tokenizer = T5Tokenizer.from_pretrained("t5-small")

print("Model and tokenizer loaded successfully from checkpoint.")


"""
this is a test checkpoint. earlier versions of the model were trained on a corpus of 3 million token texts.
"""

noisy_text = """Mamafih onun verdiği cevabdan mağmûm oldum. Çünkü burada adama44 METİN Merv Seyahatnamesi kıllı otel veyahut misafirhane yok imiş. Böylece ben ister istemez bir takım Türkmen veya Özbek hancıları ma‘rifetleriyle idare kılınan hanlardan birine inmeğe mecbûr olacağımı anladım. Ama sonradan salifu’l-beyan hanların tertibatlarından mahzûz olduğumu da inkar edemem. Zira işbu çöl adamları artık bizim Petersburg ’da pansiyon ismi verilen usûlde hareket ediyorlar. [31] Yani Avrupa’nın her şehrinde mikdar-ı mukanne1 ücret-i yevmiyye veyahut ücret-i mahiyye ile aileler içinde istenildiği müddet ikamet edilmek mümkün olduğu misillü buranın hanlarında da aynı usûl caridir. Ancak ma‘lûm olduğu üzere gerek Londra ’da ve gerek Paris ’te bu gibi aileler esasen bellenmiş ve hem de türlü türlü tavsiye-nameleri de ha’iz olduklarından artık oraları ziyaret eden misafirlerce tanınmış bulundukları halde Merv ’de bilakis insan hiç bilmediği ve mu‘arefesi olmadığı aileler nezdine nazil olmak mecbûriyyetinden kurtulamamaktadır. Nitekim ben de demin ta‘rif eylediğim yerlinin cevabı akibinde2 hah na-hah3 bunu ihtiyara mecbûr oldum. [32] İşte ben kendi kendime: “Hele bakalım tali‘ ve kaderim beni kime tesadüf ettirecektir!” diye bir hana getirilip odadan odaya gezmekte iken an-asıl Petersburg’dan uzak olmayan İşçadriye kazası sekenesinden bir doktora rast geldim. Bu ise elli beş yaşlarında olup tahminen elli iki yaşında bulunan zevcesiyle orada mukim idi. Ve hem de başka kimsesi de olmadığından beni ayrıca han odaları kiralamaktan ise kendilerinde misafir olmağa da‘vet etti. Ve fazla olarak orada iki sene kadar zamandan beri ikamet eylediğini beyanla kendi refakatinde daha iyi gezebileceğimi ve ziyaretler icrası mümkün olacağını da beyan eyledi. [33] Böylece merkûm bana kendisinin iş odasını tahsis etti."""
inputs = hf_tokenizer(noisy_text, return_tensors="pt", max_length=256, padding="max_length", truncation=True)

outputs = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=560,
    num_beams=5,
    early_stopping=True
)

cleaned_text = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Cleaned text:", cleaned_text)