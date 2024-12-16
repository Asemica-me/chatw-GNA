import os
import requests
from dotenv import load_dotenv
import mwparserfromhell

# Carica le variabili di ambiente dal file .env
load_dotenv()

# URL dell'API del GNA
url = "https://gna.cultura.gov.it/wiki/api.php"

# Parametri della richiesta API (recupero contenuto della pagina "Il progetto")
params = {
    "action": "query",
    "format": "json",
    "titles": "Il_progetto",
    "prop": "revisions",
    "rvprop": "content"
}

# Richiesta per ottenere i dati
response = requests.get(url, params=params)
data = response.json()

# Estrazione contenuto wikitext dalla risposta
page_id = next(iter(data['query']['pages']))
page = data['query']['pages'][page_id]
wikitext = page['revisions'][0]['*']

# Parsing del wikitext per rimuovere i tag mwparser
parsed = mwparserfromhell.parse(wikitext)
html_content = parsed.strip_code()  # Ottieni il contenuto in formato leggibile

# Salva il contenuto in un nuovo file
with open(('project_content.txt'), 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Dati recuperati e salvati con successo!")
