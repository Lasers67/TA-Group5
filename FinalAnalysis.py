import os
from pypdf import PdfReader
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import csv
from tabulate import tabulate
import pandas as pd
import nltk
import sys
from collections import Counter
from tqdm import tqdm


# Function to ensure NLTK resources are available
def download_nltk_resources():
    nltk_data_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "nltk_data")
    print(f"NLTK data path: {nltk_data_path}")
    try:
        nltk.download('punkt', quiet=True, force=True)
        nltk.download('punkt_tab', quiet=True, force=True)
        for lang in ['english', 'spanish', 'french']:
            nltk.data.find(f"tokenizers/punkt_tab/{lang}/")
        print("NLTK resources verified successfully.")
    except LookupError as e:
        print(f"Failed to download or verify NLTK resources: {e}")
        print("Please ensure internet access and write permissions in nltk_data directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error verifying NLTK resources: {e}")
        sys.exit(1)


# Test tokenization
def test_tokenization():
    try:
        sample_text = "This is a test sentence. Another sentence."
        for lang in ['english', 'spanish', 'french']:
            sent_tokenize(sample_text, language=lang)
            word_tokenize(sample_text, language=lang)
        print("Tokenization test passed for all languages.")
    except Exception as e:
        print(f"Tokenization test failed: {e}")
        sys.exit(1)


# Run NLTK setup
download_nltk_resources()
test_tokenization()


# Load SpaCy models
def load_spacy_model(model_name):
    try:
        return spacy.load(model_name)
    except OSError as e:
        print(f"Error loading SpaCy model {model_name}: {e}")
        print(f"Please install the model by running: python -m spacy download {model_name}")
        sys.exit(1)


nlp_en = load_spacy_model("en_core_web_sm")
nlp_es = load_spacy_model("es_core_news_sm")
nlp_fr = load_spacy_model("fr_core_news_sm")

# Base directory (relative path)
BASE_DIR = "Documents"

# Language-specific configurations
CONFIG = {
    'en': {
        'nlp': nlp_en,
        'function_words': {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'over', 'after', 'is', 'are', 'was', 'were', 'be', 'been',
            'that', 'this', 'these', 'those', 'I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
        },
        'hedging_expressions': {
            'possibly', 'might', 'seems', 'perhaps', 'likely', 'probably', 'apparently', 'may', 'could',
            'would', 'should', 'tend', 'appear', 'suggest', 'indicate', 'assume', 'believe', 'think',
            'suppose', 'guess', 'estimate'
        },
        'folders': [os.path.join(BASE_DIR, 'EN-His'), os.path.join(BASE_DIR, 'EN-Psy')]
    },
    'es': {
        'nlp': nlp_es,
        'function_words': {
            'el', 'la', 'lo', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'a', 'en', 'con', 'por', 'para',
            'y', 'o', 'ni', 'pero', 'sino', 'que', 'si', 'cuando', 'como', 'porque', 'pues', 'aunque', 'mientras',
            'hasta', 'desde', 'sobre', 'bajo', 'entre', 'hacia', 'sin', 'durante', 'según', 'tras', 'ante',
            'yo', 'tú', 'él', 'ella', 'nosotros', 'vosotros', 'ellos', 'ellas', 'me', 'te', 'se', 'le', 'les',
            'mi', 'tu', 'su', 'mis', 'tus', 'sus', 'nuestro', 'vuestro', 'este', 'esta', 'esto', 'estos', 'estas',
            'ese', 'esa', 'eso', 'esos', 'esas', 'aquel', 'aquella', 'aquellos', 'aquellas', 'quien', 'que', 'cual',
            'es', 'era', 'eras', 'éramos', 'erais', 'eran', 'fui', 'fuiste', 'fue', 'fuimos', 'fuisteis', 'fueron',
            'he', 'has', 'ha', 'hemos', 'habéis', 'han', 'no', 'sí', 'todo', 'toda', 'todos', 'todas', 'cada',
            'ningún', 'ninguna', 'algún', 'alguna', 'ciertos', 'cierta', 'bien', 'más', 'menos', 'tan', 'tanto'
        },
        'hedging_expressions': {
            'quizás', 'tal vez', 'parece', 'parecer', 'probablemente', 'posiblemente', 'aparentemente', 'podría',
            'debería', 'tender', 'sugerir', 'indicar', 'suponer', 'creer', 'pensar', 'estimar', 'parecería',
            'probable', 'eventualmente', 'hipotéticamente'
        },
        'folders': [os.path.join(BASE_DIR, 'ES-His'), os.path.join(BASE_DIR, 'ES-Psy')]
    },
    'fr': {
        'nlp': nlp_fr,
        'function_words': {
            'le', 'la', "l'", 'les', 'un', 'une', 'des', 'du', 'de', 'la', 'et', 'ou', 'mais', 'donc', 'ni',
            'car', 'que', 'si', 'quand', 'comme', 'puisque', 'lorsque', 'à', 'en', 'sur', 'dans', 'par',
            'pour', 'avec', 'chez', 'contre', 'vers', "jusqu'à", 'depuis', 'je', 'tu', 'il', 'elle', 'on',
            'nous', 'vous', 'ils', 'elles', 'me', 'te', 'se', 'lui', 'leur', 'ce', 'cette', 'cet', 'ces',
            'celui', 'celle', 'ceux', 'celles', 'qui', 'dont', 'où', 'mon', 'ton', 'son', 'ma', 'ta', 'sa',
            'mes', 'tes', 'ses', 'notre', 'votre', 'leur', 'est', 'es', 'sont', 'étais', 'était', 'étaient',
            'ai', 'as', 'a', 'ont', 'avais', 'avait', 'avaient', 'ne', 'pas', 'ceci', 'cela', 'y', 'en',
            'tout', 'toute', 'tous', 'toutes', 'chaque', 'aucun', 'quelque', 'certains', 'certaine',
            'bien', 'même', 'trop', 'soit', 'fût', "t'"
        },
        'hedging_expressions': {
            'peut-être', 'semble', 'probablement', 'possiblement', 'apparemment', 'pourrait',
            'devrait', 'tendre', 'suggérer', 'indiquer', 'supposer', 'croire', 'penser', 'estimer',
            'paraît', 'vraisemblablement', 'éventuellement', 'hypothétiquement'
        },
        'folders': [os.path.join(BASE_DIR, 'FR-His'), os.path.join(BASE_DIR, 'FR-Psy')]
    }
}


# Function to count syllables (language-specific)
def count_syllables(word, lang):
    word = word.lower()
    vowels = {
        'en': "aeiouy",
        'es': "aeiouáéíóúü",
        'fr': "aeiouyàâäéèêëîïôöùûüÿ"
    }[lang]
    syllable_count = 0
    prev_char_was_vowel = False
    for char in word:
        if char in vowels:
            if not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False
    if lang == 'en' and word.endswith('e'):
        syllable_count = max(1, syllable_count - 1)
    elif lang == 'es' and word.endswith(('e', 'es')) and not word.endswith(('ée', 'ées')):
        syllable_count = max(1, syllable_count - 1)
    elif lang == 'fr' and word.endswith(('e', 'es', 'ent')) and not word.endswith(('ée', 'ées')):
        syllable_count = max(1, syllable_count - 1)
    return max(1, syllable_count)


# Function to load gender metadata from Excel
def load_gender_metadata(excel_path):
    try:
        xls = pd.ExcelFile(excel_path)
    except Exception as e:
        print(f"Error reading Excel file {excel_path}: {e}")
        return {}

    gender_map = {}
    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if 'FileName' not in df.columns:
                print(f"Sheet {sheet_name} does not have 'FileName' column. Skipping...")
                continue

            gender_col = 'Gender' if 'Gender' in df.columns else 'GENDER' if 'GENDER' in df.columns else None
            if not gender_col:
                print(f"Sheet {sheet_name} does not have 'Gender' or 'GENDER' column. Skipping...")
                continue

            df['FileName'] = df['FileName'].astype(str).str.replace('.pdf', '', regex=False)
            df['FileName'] = df['FileName'].apply(
                lambda x: str(int(float(x))) if x.replace('.', '', 1).isdigit() else x)
            for _, row in df.iterrows():
                if pd.notna(row['FileName']) and pd.notna(row[gender_col]):
                    key = (sheet_name, row['FileName'])
                    gender_map[key] = row[gender_col]
        except Exception as e:
            print(f"Error processing sheet {sheet_name}: {e}")
    return gender_map


# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path, lang):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        start_page = 1 if lang in ['es', 'fr'] else 0
        for page in reader.pages[start_page:]:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + " "
        if not text.strip():
            print(f"No text extracted from {pdf_path}")
            return ""
        token_count = len(word_tokenize(text, language={'en': 'english', 'es': 'spanish', 'fr': 'french'}[lang]))
        if token_count < 100:
            print(f"Short text extracted from {pdf_path}: {token_count} tokens")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""
    return text


# Function to calculate Type-Token Ratio (TTR)
def calculate_ttr(text, lang):
    if not text.strip():
        return 0.0
    try:
        language = {'en': 'english', 'es': 'spanish', 'fr': 'french'}[lang]
        tokens = word_tokenize(text.lower(), language=language)
        tokens = [token for token in tokens if token not in string.punctuation]
        total_words = len(tokens)
        if total_words == 0:
            return 0.0
        unique_words = len(set(tokens))
        ttr = unique_words / total_words
        return ttr
    except Exception as e:
        print(f"Error in TTR calculation for {lang}: {e}")
        return 0.0


# Function to calculate Hapax Legomena Ratio (HL)
def calculate_hl(text, lang):
    if not text.strip():
        return 0.0
    try:
        language = {'en': 'english', 'es': 'spanish', 'fr': 'french'}[lang]
        tokens = word_tokenize(text.lower(), language=language)
        tokens = [token for token in tokens if token not in string.punctuation]
        total_words = len(tokens)
        if total_words == 0:
            return 0.0
        word_counts = Counter(tokens)
        hapax_count = sum(1 for word, count in word_counts.items() if count == 1)
        hl = hapax_count / total_words
        return hl
    except Exception as e:
        print(f"Error in HL calculation for {lang}: {e}")
        return 0.0


# Function to calculate Mean Sentence Length (MSL)
def calculate_msl(text, lang):
    if not text.strip():
        return 0.0
    try:
        language = {'en': 'english', 'es': 'spanish', 'fr': 'french'}[lang]
        sentences = sent_tokenize(text, language=language)
        total_sentences = len(sentences)
        if total_sentences == 0:
            return 0.0
        tokens = word_tokenize(text.lower(), language=language)
        tokens = [token for token in tokens if token not in string.punctuation]
        total_words = len(tokens)
        msl = total_words / total_sentences if total_sentences > 0 else 0.0
        return msl
    except Exception as e:
        print(f"Error in MSL calculation for {lang}: {e}")
        return 0.0


# Function to calculate Function Word Frequency (FWF)
def calculate_fwf(text, lang):
    if not text.strip():
        return 0.0
    nlp = CONFIG[lang]['nlp']
    try:
        doc = nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_punct]
        total_words = len(tokens)
        if total_words == 0:
            return 0.0
        function_word_count = sum(1 for token in tokens if token in CONFIG[lang]['function_words'])
        fwf = function_word_count / total_words
        if fwf < 0.05 or fwf > 0.4:
            print(f"FWF ({fwf:.4f}) for current document, total words: {total_words}")
        return fwf
    except Exception as e:
        print(f"Error in FWF calculation for {lang}: {e}")
        return 0.0


# Function to calculate Hedging Expression Rate (HER)
def calculate_her(text, lang):
    if not text.strip():
        return 0.0
    nlp = CONFIG[lang]['nlp']
    try:
        doc = nlp(text)
        total_words = len([token for token in doc if not token.is_punct])
        if total_words == 0:
            return 0.0
        hedge_count = sum(1 for token in doc if token.text.lower() in CONFIG[lang]['hedging_expressions'])
        if lang in ['es', 'fr']:
            multi_hedges = 0
            triggers = ['parece', 'parecería'] if lang == 'es' else ['semble', 'paraît']
            for sent in doc.sents:
                sent_tokens = [token.text.lower() for token in sent]
                if any(hedge in sent_tokens for hedge in triggers) and 'que' in sent_tokens:
                    multi_hedges += 1
                if 'puede' in sent_tokens and 'suponer' in sent_tokens and 'que' in sent_tokens and lang == 'es':
                    multi_hedges += 1
                elif 'peut' in sent_tokens and 'supposer' in sent_tokens and 'que' in sent_tokens and lang == 'fr':
                    multi_hedges += 1
            hedge_count += multi_hedges
        her = hedge_count / total_words
        if her < 0.001:
            print(f"Low HER ({her:.4f}) detected in {lang} text.")
        return her
    except Exception as e:
        print(f"Error in HER calculation for {lang}: {e}")
        return 0.0


# Function to calculate Passive Voice Usage (PVU)
def calculate_pvu(text, lang):
    if not text.strip():
        return 0.0
    try:
        language = {'en': 'english', 'es': 'spanish', 'fr': 'french'}[lang]
        sentences = sent_tokenize(text, language=language)
        total_sentences = len(sentences)
        if total_sentences == 0:
            return 0.0
        passive_sentences = 0
        nlp = CONFIG[lang]['nlp']
        for sentence in sentences:
            doc = nlp(sentence)
            if lang == 'en':
                for token in doc:
                    if token.dep_ == "nsubjpass" and token.head.pos_ == "VERB":
                        passive_sentences += 1
                        break
            else:
                for token in doc:
                    lemma = "ser" if lang == 'es' else "être"
                    if token.lemma_ == lemma and token.pos_ == "AUX":
                        for child in token.children:
                            if child.pos_ == "VERB" and child.tag_.endswith("PP"):
                                passive_sentences += 1
                                break
                    if token.text.lower() == "se" and token.head.pos_ == "VERB":
                        passive_sentences += 1
                        break
        pvu = passive_sentences / total_sentences
        if pvu < 0.001:
            print(f"Low PVU ({pvu:.4f}) detected in {lang} text.")
        return pvu
    except Exception as e:
        print(f"Error in PVU calculation for {lang}: {e}")
        return 0.0


# Function to calculate Readability Score (FK for all languages per request)
def calculate_fk(text, lang):
    if not text.strip():
        return 0.0
    try:
        language = {'en': 'english', 'es': 'spanish', 'fr': 'french'}[lang]
        sentences = sent_tokenize(text, language=language)
        total_sentences = len(sentences)
        if total_sentences == 0:
            return 0.0
        tokens = word_tokenize(text.lower(), language=language)
        tokens = [token for token in tokens if token not in string.punctuation]
        total_words = len(tokens)
        if total_words == 0:
            return 0.0
        asl = total_words / total_sentences
        total_syllables = sum(count_syllables(token, lang) for token in tokens)
        asw = total_syllables / total_words if total_words > 0 else 0
        fk = 206.835 - (1.015 * asl) - (84.6 * asw)
        return fk
    except Exception as e:
        print(f"Error in FK calculation for {lang}: {e}")
        return 0.0


# Function to calculate Pronoun Usage (PU)
def calculate_pu(text, lang):
    if not text.strip():
        return 0.0
    try:
        language = {'en': 'english', 'es': 'spanish', 'fr': 'french'}[lang]
        tokens = word_tokenize(text.lower(), language=language)
        tokens = [token for token in tokens if token not in string.punctuation]
        total_words = len(tokens)
        if total_words == 0:
            return 0.0
        nlp = CONFIG[lang]['nlp']
        doc = nlp(text)
        pronoun_count = sum(1 for token in doc if token.pos_ == "PRON")
        pu = (pronoun_count / total_words) * 100 if total_words > 0 else 0.0
        if pu < 1.0:
            print(f"Low PU ({pu:.2f}%) detected in {lang} text.")
        return pu
    except Exception as e:
        print(f"Error in PU calculation for {lang}: {e}")
        return 0.0


# Function to calculate Involvement/Informational Rate (IIR) for English only
def calculate_iir(text, lang):
    if lang != 'en' or not text.strip():
        return None
    try:
        nlp = CONFIG['en']['nlp']
        doc = nlp(text)
        total_words = len([token for token in doc if not token.is_punct])
        if total_words == 0:
            return 0.0
        personal_pronouns = sum(1 for token in doc if token.pos_ == "PRON" and token.tag_ in {"PRP", "PRP$"})
        questions = sum(1 for token in doc if token.text == "?")
        coordination = sum(1 for token in doc if token.dep_ == "cc")
        involvement = (personal_pronouns + questions + coordination) / total_words if total_words else 0
        noun_phrases = len(list(doc.noun_chunks))
        technical_terms = sum(1 for token in doc if token.pos_ == "NOUN" and len(token.text) > 6)
        references = sum(
            1 for token in doc if token.like_num or token.text.startswith("http") or token.text.startswith("@"))
        informational = (noun_phrases + technical_terms + references) / total_words if total_words else 0
        iir = involvement / informational if informational else float('inf')
        return iir
    except Exception as e:
        print(f"Error in IIR calculation: {e}")
        return 0.0


# Main function to process PDFs
def process_pdfs_in_folders(excel_path):
    gender_map = load_gender_metadata(excel_path)
    results = []
    counts = {os.path.basename(folder): {'M': 0, 'F': 0} for lang in CONFIG for folder in CONFIG[lang]['folders']}

    for lang in CONFIG:
        for folder_path in CONFIG[lang]['folders']:
            if not os.path.exists(folder_path):
                print(f"Folder {folder_path} does not exist. Skipping...")
                continue
            folder_name = os.path.basename(folder_path)
            # Get and sort PDF files numerically
            pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
            pdf_files.sort(
                key=lambda x: int(x.replace('.pdf', '')) if x.replace('.pdf', '').isdigit() else float('inf'))

            # Process PDFs with tqdm progress bar
            for filename in tqdm(pdf_files, desc=f"Processing {folder_name} ({lang})"):
                pdf_path = os.path.join(folder_path, filename)
                text = extract_text_from_pdf(pdf_path, lang)
                if not text.strip():
                    print(f"Skipping metrics calculation for {filename} due to empty text")
                    continue

                ttr = calculate_ttr(text, lang)
                hl = calculate_hl(text, lang)
                msl = calculate_msl(text, lang)
                fwf = calculate_fwf(text, lang)
                her = calculate_her(text, lang)
                pvu = calculate_pvu(text, lang)
                fk = calculate_fk(text, lang)
                pu = calculate_pu(text, lang)
                iir = calculate_iir(text, lang) if lang == 'en' else None

                file_key = filename.replace('.pdf', '')
                gender_key = (folder_name, file_key)
                gender = gender_map.get(gender_key, 'Unknown')

                if gender in ['M', 'F']:
                    counts[folder_name][gender] += 1

                result = {
                    'language': lang.upper(),
                    'folder': folder_name,
                    'filename': filename,
                    'TTR': ttr,
                    'HL': hl,
                    'MSL': msl,
                    'FWF': fwf,
                    'HER': her,
                    'PVU': pvu,
                    'FK': fk,
                    'PU': pu,
                    'IIR': iir,
                    'Gender': gender
                }
                results.append(result)

    print("\nDocument Counts by Gender and Folder:")
    for folder, gender_counts in counts.items():
        print(f"{folder}: M={gender_counts['M']}, F={gender_counts['F']}")
    return results


# Main execution
excel_path = "Metadata-GENDER.xlsx"
try:
    results = process_pdfs_in_folders(excel_path)
except Exception as e:
    print(f"Error during processing: {e}")
    sys.exit(1)

# Create Results folder if it doesn't exist
results_dir = "Results"
os.makedirs(results_dir, exist_ok=True)

# Prepare data for tabulate
table_data = []
for result in results:
    iir_str = f"{result['IIR']:.2f}" if result['IIR'] is not None else "-"
    table_data.append([
        result['language'],
        result['folder'],
        result['filename'],
        f"{result['TTR']:.4f}",
        f"{result['HL']:.4f}",
        f"{result['MSL']:.2f}",
        f"{result['FWF']:.4f}",
        f"{result['HER']:.4f}",
        f"{result['PVU']:.4f}",
        f"{result['FK']:.2f}",
        f"{result['PU']:.2f}",
        iir_str,
        result['Gender']
    ])

# Print results
print("\nResults:")
print(tabulate(
    table_data,
    headers=["Lang", "Folder", "Filename", "TTR", "HL", "MSL", "FWF", "HER", "PVU", "FK", "PU", "IIR", "Gender"],
    tablefmt="fancy_grid",
    stralign="center",
    numalign="center"
))

# Save results to CSV in Results folder
try:
    csv_path = os.path.join(results_dir, "metrics_results_all_languages.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Language', 'Folder', 'Filename', 'TTR', 'HL', 'MSL', 'FWF', 'HER', 'PVU', 'FK', 'PU', 'IIR',
                      'Gender']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                'Language': result['language'],
                'Folder': result['folder'],
                'Filename': result['filename'],
                'TTR': result['TTR'],
                'HL': result['HL'],
                'MSL': result['MSL'],
                'FWF': result['FWF'],
                'HER': result['HER'],
                'PVU': result['PVU'],
                'FK': result['FK'],
                'PU': result['PU'],
                'IIR': result['IIR'] if result['IIR'] is not None else '',
                'Gender': result['Gender']
            })
    print(f"\nResults saved to '{csv_path}'")
except Exception as e:
    print(f"Error saving CSV: {e}")