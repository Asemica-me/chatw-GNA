import time
import pandas as pd
from tqdm import tqdm
import mwparserfromhell
from trafilatura import fetch_url, extract
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# Carica gli URL dalla sitemap XML    
def get_urls_from_sitemap_file(file_path: str) -> list:
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Handle namespaces in XML if present
        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}  # Namespace della sitemap
        
        # Debug: stampa la struttura dell'XML
        print(ET.tostring(root, encoding='unicode', method='xml'))
        
        # Extract URLs, accounting for the namespace
        urls = []
        for url in root.findall('.//ns:url', namespaces):  # Trova tutti i tag <url> con il namespace
            loc = url.find('ns:loc', namespaces)  # Trova il tag <loc> all'interno di <url>
            if loc is not None:
                urls.append(loc.text)  # Aggiungi l'URL trovato alla lista
        
        print(f"Found {len(urls)} URLs in sitemap.")
        return urls
    except Exception as e:
        print(f"Error reading the sitemap: {e}")
        return []

def extract_metadata_with_bs4(html: str) -> dict:
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.title.string if soup.title else "No title"
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    description = meta_desc['content'] if meta_desc else "No description"
    return {"title": title, "description": description}

def parse_mediawiki_content(content: str) -> dict:
    try:
        wiki_code = mwparserfromhell.parse(content)
        sections = [str(section.title) for section in wiki_code.get_sections()]
        plain_text = wiki_code.strip_code()
        return {
            "sections": sections,
            "plain_text": plain_text
        }
    except Exception as e:
        print(f"Failed to parse MediaWiki content: {e}")
        return {"sections": [], "plain_text": "Parsing failed"}

def create_dataset(file_path: str) -> pd.DataFrame:
    data = []
    urls = get_urls_from_sitemap_file(file_path)
    if not urls:
        print("No URLs found in sitemap.")
        return pd.DataFrame()  # Return empty DataFrame if no URLs found
    for url in tqdm(urls, desc="URLs"):
        try:
            html = fetch_url(url)
            body = extract(html)
            if not body:
                raise ValueError("No content extracted.")
            metadata = extract_metadata_with_bs4(html)
            
            # Fallback: MediaWiki parsing
            parsed_data = parse_mediawiki_content(html)
            sections = parsed_data["sections"]
            plain_text = parsed_data["plain_text"]

            # Aggiungi i dati alla lista
            data.append({
                'url': url,
                'body': body,
                'title': metadata['title'],
                'description': metadata['description'],
                'sections': sections,
                'plain_text': plain_text
            })
        except Exception as e:
            print(f"Failed to process URL {url}: {e}")
        time.sleep(0.5)
    
    # Converti le liste in stringhe per evitare l'errore
    for row in data:
        row['sections'] = str(row['sections'])  # Converte la lista 'sections' in stringa
    
    # Crea il DataFrame e rimuovi i duplicati
    df = pd.DataFrame(data).drop_duplicates().dropna()
    return df

if __name__ == "__main__":
    file_path = 'GNA_sitemap.xml'  # Path to your sitemap XML file
    df = create_dataset(file_path)
    if not df.empty:
        dataset_name = "./data/gna_kg_dataset.csv"
        df.to_csv(dataset_name, index=False)
        print(f"Dataset saved successfully as {dataset_name}")
    else:
        print("No data to save.")
