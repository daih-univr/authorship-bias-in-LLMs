import os
import json
import pandas as pd
import re
import torch

from llama_cpp import Llama
from huggingface_hub import hf_hub_download


for name in ["JSONoutput", "CSVoutput"]:
  try:
      os.mkdir(name)
  except Exception:
    pass
  
percorso_testi = r'storie_generate.txt'
with open(percorso_testi,'r',encoding='utf-8') as file:
    testo = [t for t in file.read().split('|separator|')]
testo_Streghe, testo_James, testo_Matilde = testo[0], testo[1], testo[2]

percorso_questionari = r'questionari.txt'
with open(percorso_questionari,'r',encoding='utf-8') as file:
    questionario = file.read()

sezione_fiabe_HU = questionario.replace("{author_template}",'un autore umano per bambini')
sezione_fiabe_AI = questionario.replace("{author_template}","un'Intelligenza Artificiale")
sezione_fiabe_HU_AI = questionario.replace("{author_template}","un autore umano per bambini che utilizza l'intelligenza artificiale")

combinazioni = [
    'storia_1+autore_umano',
    'storia_1+autore_AI',
    'storia_1+autore_umano_E_AI',
    'storia_2+autore_umano',
    'storia_2+autore_AI',
    'storia_2+autore_umano_E_AI',
    'storia_3+autore_umano',
    'storia_3+autore_AI',
    'storia_3+autore_umano_E_AI'
]


model_id = "unsloth/Llama-3.1-8B-Instruct-GGUF"
model_basename = "Llama-3.1-8B-Instruct-BF16.gguf" # This is the specific GGUF file name


TEMPERATURE=1.0
NUMBER_OF_ITERATIONS = 2


# 2. Download the GGUF model
model_path = hf_hub_download(
    repo_id=model_id,
    filename=model_basename,
    resume_download=True
)

print(f"Model downloaded to: {model_path}")


# 2. Download the GGUF model
model_path = hf_hub_download(
    repo_id=model_id,
    filename=model_basename,
    resume_download=True
)

print(f"Model downloaded to: {model_path}")

# 3. Load the model using llama_cpp.Llama
# Key parameters for memory management:
#   n_gpu_layers: Number of layers to offload to the GPU.
#                 Set to -1 to offload all layers possible.
#                 Set to 0 to run entirely on CPU.
#   n_ctx: Context window size (how many tokens the model can "remember").
#          Larger values require more memory.
#   n_batch: How many tokens are processed in parallel during generation.
#            Larger values can speed up generation but use more VRAM.

# Adjust n_gpu_layers based on your GPU VRAM:
# For a 7B model Q4_K_M, you might need ~4.5GB VRAM.
# You'll need to experiment with this value for your specific GPU.
# If you have 8GB VRAM, you might try 20, 30, or even -1 (all).
# If you only have system RAM, set n_gpu_layers=0
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1, # <--- This is where you pass the parameter
    n_ctx=8192,
    n_batch=512,
    #verbose=True     # <--- IMPORTANT: This will print messages about GPU offloading
)
print("\nModel loading complete. Please review the output above this line for details on GPU offloading.")
print("Look for lines like 'llm_load_tensors: offloading X/Y layers to GPU'.")
print("If no such lines appear, or if X is 0, then no layers were offloaded to the GPU (or you set n_gpu_layers=0).")






def questionario_LLM(testo, questionario_autore, llm, TEMPERATURE=1.0) -> str:
  """
  Ritorna una stringa di risposte stile "q1___risposta | q2___risposta | q3___risposta"
  """
  

  prompt = f"""Rispondi al seguente questionario. Dovrai rispondere in modo specifico. La risposta a questo messaggio dovrà contenere SOLO e SOLTANTO la risposta al questionario che troverai in fondo a questo messaggio, nient'altro.
Verrai sottoposto a domande in cui dovrai indicare una scala numerica e domande in cui dovrai rispondere con Sì/No: nel primo caso rispondi SOLAMENTE con la scala numerica richiesta; nel secondo caso dovrai rispondere solo con Sì o No, senza argomentare.
Le tue risposte devono essere strutturate come se fossero una stringa di Python, dove unirai due elementi: qN= (dove "N" è il numero della risposta, comunque troverai nel questionario la domanda numerata), e la tua risposta.
Ogni risposta dovrà essere separata da una pipe |, e non dovranno esserci tabulazioni e ritorni a capo. Dopo l'ultima risposta, non mettere la pipe. Ti faccio un esempio, a livello di struttura, di un paio di risposte che mi aspetto: q1=3 | q2=5 | q3=Sì
Ricorda di rispondere *una sola volta* ad ogni domanda del questionario.
  
Ecco il questionario che devi compilare:
  {questionario_autore}
  
  """

  def clean(text):
    clean = text.strip('\n')
    cleaner = clean.replace('\\','')
    no_double_quotes = cleaner.replace('"',"''")
    return no_double_quotes


  prompt = f"""
   <|begin_of_text|><|start_header_id|>system<|end_header_id|>

  Sei un assistente servizievole e preciso. Segui diligentemente le istruzioni date.<|eot_id|><|start_header_id|>user<|end_header_id|>

  {prompt}

  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
  """


  #print(f"Prompt:\n{prompt}\n")

  output = llm(
        prompt,
        max_tokens=5000,
        temperature=TEMPERATURE,
        top_p=0.95,
        stop=["###"],
        echo=False,
        stream=False
  )

  risposta = clean(output["choices"][0]["text"].strip())
  return risposta


def correct_structure_checker(item:str, expected_len:int) -> bool:
  if len(item.split('|')) == expected_len:
    return True
  else:
    return False
  
index = 0
for storia in [testo_Streghe, testo_James, testo_Matilde]:
#for storia in [testo_Streghe]:

  #for autore in [sezione_fiabe_HU]:
  for autore in [sezione_fiabe_HU, sezione_fiabe_AI, sezione_fiabe_HU_AI]: 

    questionario_autore_storia = autore.replace("[FIABA]",f"[INIZIO_FIABA]\n{storia}\n[FINE_FIABA]")
    risposte_strutturate = []

    n = 0
    while n < NUMBER_OF_ITERATIONS:
    #for n in range(100): #qui inserire il numero di questionari per combinazione

      print(f"Inizio risposta numero {n+1}")

      risposta_str = questionario_LLM(storia, questionario_autore_storia, llm, TEMPERATURE)
      print(f"Finito risposta numero {n+1}")

      print(f"Risposta numero {n+1}:\n{risposta_str}\n")
      print("Controllo che la risposta sia completa")
      if correct_structure_checker(risposta_str,19) == True:
        print("La risposta è corretta")

        print("Converto in dizionario")
        risposta_dict = {}
        for item in risposta_str.split("|"):
            item = item.strip()
            qn, risposta = item.split("=", maxsplit=1)
            risposta_dict[qn] = risposta.strip()
        risposte_strutturate.append(risposta_dict)

        print('Scrivo su file\n')
        with open(f"JSONoutput/{combinazioni[index]}.json", "w", encoding="utf-8") as f:
            json.dump(risposte_strutturate, f, ensure_ascii=False, indent=2)
    

        n += 1
      else:
        print('Risposta non valida, passo al prossimo tentativo')
    df = pd.read_json(f'JSONoutput/{combinazioni[index]}.json')
    df.to_csv(f'CSVoutput/{combinazioni[index]}.csv', index=False)   
    index = index + 1