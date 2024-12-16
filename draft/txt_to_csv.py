import pandas as pd
import re

# Dataset
with open("project_content.txt", "r", encoding="utf-8") as file:
    # Leggi il contenuto del file e assegnalo alla variabile text
    text = file.read()

# Prima regex: Titoli tra due ritorni a capo
sections = re.split(r'(?<=\n)([A-Za-z\s]+)\n', text)

# Seconda regex: Titoli preceduti da una riga vuota e seguiti da un ritorno a capo
# Usa findall per trovare tutti i titoli seguiti dal contenuto
sections2 = re.findall(r'(\n\s*\n)([A-Za-z\s]+)\n', text)

# Unisci entrambe le sezioni in un'unica lista di titoli e contenuti
# Combinando i titoli ottenuti da entrambe le regex
titles = []
contents = []

# Usa la prima regex per raccogliere titoli e contenuti
for i in range(1, len(sections), 2):
    title = sections[i].strip()
    content = sections[i + 1].strip() if (i + 1) < len(sections) else ""
    titles.append(title)
    contents.append(content)

# Aggiungi i titoli e contenuti della seconda regex, se presenti
for match in sections2:
    title = match[1].strip()
    content = match[1].strip() if len(match) > 1 else ""
    if title not in titles:  # Evita duplicati
        titles.append(title)
        contents.append(content)


data = list(zip(titles, contents))
df = pd.DataFrame(data, columns=['Title', 'Content'])
print(df)
df.to_csv("data/dataframe.csv", index=False, encoding="utf-8")
