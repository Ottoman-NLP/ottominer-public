## Experimental Setup

#### Large Language Models:
- Electra, BERTurk, distillBERTurk, ConvBERTurk, Turna, mBERT, XLM-R, mGPT


#### Datasets:
- Ottoman Turkish source: OTA_BOUN-UD-Treebank: 107 Training, 200 Test sent. (for now)
- Modern Turkish source: TR_BOUN-UD-Treebank: 7803 Training, 979 Dev, 979 Test sent.

#### Experiments:

##### Finetuning:
- LLM+Finetune with TR_BOUN, test on OTA-BOUN Test set
- LLM+Finetune with TR_BOUN+Further finetune with OTA-BOUN Training set, test on OTA-BOUN Test set
- LLM+Finetune with OTA-BOUN Training set, test on OTA-BOUN Test set

##### Further Pretraining:
- LLM+Further pretrain using Ottoman Corpus = OtaLLM

- OtaLLM+Finetune with TR_BOUN, test on OTA-BOUN Test set
- OtaLLM+Finetune with TR_BOUN+Further finetune with OTA-BOUN Training set, test on OTA-BOUN Test set
- OtaLLM+Finetune with OTA-BOUN Training set, test on OTA-BOUN Test set

##### Research Questions:
- Q1- Çok az da olsa Osmanlıca annotated data ile modelleri fine-tune etmek modelin Osmanlıca dependency annotation'daki başarısını etkiliyor mu?
-  Q1.1- Modern Türkçe ile finetuning ne kadar gerekli?
- Q2- Osmanlıca Corpus ile modelleri further pretrain etmek modellerin Osmanlıca dependency annotation'daki başarısını etkiliyor mu?

  
  
