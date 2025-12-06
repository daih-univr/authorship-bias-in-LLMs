import os
import json
from time import sleep
import pandas as pd
import re
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

client = genai.Client(api_key=GEMINI_API_KEY)

# Choose an OpenAI model
GEMINI_MODEL = "gemini-2.0-flash" 


TEMPERATURE=1.0
NUMBER_OF_ITERATIONS = 5



# --- Original Code (adapted for OpenAI) ---

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

# Removed model downloading and loading specific to llama-cpp

def clean(text):
    """Cleans the text by stripping newlines, backslashes, and replacing double quotes."""
    clean_text = text.strip('\n')
    clean_text = clean_text.replace('\\','')
    no_double_quotes = clean_text.replace('"',"''")
    return no_double_quotes

def questionario_LLM(testo: str, questionario_autore: str, client, TEMPERATURE: float = 1.0) -> str:
 
  prompt = f"""Rispondi al seguente questionario. Dovrai rispondere in modo specifico. La risposta a questo messaggio dovrà contenere SOLO e SOLTANTO la risposta al questionario che troverai in fondo a questo messaggio, nient'altro.
Verrai sottoposto a domande in cui dovrai indicare una scala numerica e domande in cui dovrai rispondere con Sì/No: nel primo caso rispondi SOLAMENTE con la scala numerica richiesta; nel secondo caso dovrai rispondere solo con Sì o No, senza argomentare.
Le tue risposte devono essere strutturate come se fossero una stringa di Python, dove unirai due elementi: qN= (dove "N" è il numero della risposta, comunque troverai nel questionario la domanda numerata), e la tua risposta.
Ogni risposta dovrà essere separata da una pipe |, e non dovranno esserci tabulazioni e ritorni a capo. Dopo l'ultima risposta, non mettere la pipe. Ti faccio un esempio, a livello di struttura, di un paio di risposte che mi aspetto: q1=3 | q2=5 | q3=Sì
Ricorda di rispondere *una sola volta* ad ogni domanda del questionario.
  
Ecco il questionario che devi compilare:
  {questionario_autore}
  
  """

  while True:
      try:
          response = client.models.generate_content(
              model=GEMINI_MODEL,
              contents=prompt,
              #max_output_tokens=1000, # Increased max_tokens for potentially longer responses
              config=types.GenerateContentConfig(
                temperature=TEMPERATURE,
                top_p=0.95,
                max_output_tokens=1000 # Increased max_tokens for potentially longer responses
              )
              
              # stop parameters are handled differently in OpenAI, usually by including them in the prompt
              # or letting the model naturally conclude. For a specific format, ensure prompt guides it.
          )
          risposta = clean(response.text.strip())
          break
      except Exception as e:
          print(f'Eccezione {e}. Retrying...')
          sleep(1)
          continue
      
  return risposta


def correct_structure_checker(item:str, expected_len:int) -> bool:
  if len(item.split('|')) == expected_len:
    return True
  else:
    return False
  
index = 0
for storia in [testo_Streghe, testo_James, testo_Matilde]:
#for storia in [testo_Streghe]: # Uncomment this line for testing with a single story

  for autore in [sezione_fiabe_HU, sezione_fiabe_AI, sezione_fiabe_HU_AI]: # Uncomment this line for full loop
  #for autore in [sezione_fiabe_HU]: # Uncomment this line for testing with a single author type

    # Ensure the story is embedded correctly within the author section for the prompt
    questionario_autore_storia = autore.replace("[FIABA]",f"[INIZIO_FIABA]\n{storia}\n[FINE_FIABA]")
    risposte_strutturate = []

    n = 0
    while n < NUMBER_OF_ITERATIONS: # This loop generates 100 responses per combination
    #for n in range(1): # Use this for quick testing, generates 1 response

      print(f"Inizio risposta numero {n+1} per combinazione {combinazioni[index]}")      

      risposta_str = questionario_LLM(storia, questionario_autore_storia, client, TEMPERATURE) # Pass the full questionnaire
      print(f"Finito risposta numero {n+1}")

      print(f"Risposta numero {n+1}:\n{risposta_str}\n")
      print("Controllo che la risposta sia completa")
      # You need to verify `expected_len` based on the total number of questions across all sections.
      # Count them precisely from your `questionari.txt` and `sezione_fiabe`.
      # Assuming 103 is still correct based on your original logic.
      if correct_structure_checker(risposta_str, 19) == True: # Make sure this '103' matches your actual question count
        print("La risposta è corretta")

        print("Converto in dizionario")
        risposta_dict = {}
        try:
            for item in risposta_str.split("|"):
                item = item.strip()
                if "=" in item: # Ensure the separator exists before splitting
                    qn, risposta = item.split("=", maxsplit=1)
                    risposta_dict[qn] = risposta.strip()
                else:
                    print(f"Warning: Item '{item}' does not contain '___' separator. Skipping.")
        except Exception as e:
            print(f"Error parsing response string to dictionary: {e}")
            print(f"Problematic response part: {item}")
            continue # Skip saving this response if parsing fails

        risposte_strutturate.append(risposta_dict)

        print('Scrivo su file\n')
        with open(f"JSONoutput/{combinazioni[index]}.json", "w", encoding="utf-8") as f:
            json.dump(risposte_strutturate, f, ensure_ascii=False, indent=2)

        n += 1
      else:
        print('Risposta non valida, passo al prossimo tentativo')
        # Consider adding a small delay here to avoid hammering the API if many retries are expected
        # import time
        # time.sleep(1)

    # After attempting all N generations for a specific combination, process the results
    if risposte_strutturate: # Only proceed if there's data to process
        print(f"Processing JSON to CSV for {combinazioni[index]}.json")
        try:
            df = pd.read_json(f'JSONoutput/{combinazioni[index]}.json')
            df.to_csv(f'CSVoutput/{combinazioni[index]}.csv', index=False)
            print(f"Successfully converted {combinazioni[index]}.json to CSV.")
        except Exception as e:
            print(f"Error converting JSON to CSV for {combinazioni[index]}.json: {e}")
    else:
        print(f"No valid responses generated for {combinazioni[index]}. Skipping CSV conversion.")

    index = index + 1