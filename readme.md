# Linguistic Analysis Project

This project analyzes PDF documents in English, Spanish, and French to compute linguistic metrics such as Type-Token Ratio (TTR), Hapax Legomena Ratio (HL), Mean Sentence Length (MSL), Function Word Frequency (FWF), Hedging Expression Rate (HER), Passive Voice Usage (PVU), Flesch-Kincaid Readability Score (FK), Pronoun Usage (PU), and Involvement/Informational Rate (IIR, English only). It also generates gender-based visualizations for MSL, FWF, and FK using violin and bar plots.

The project consists of two main scripts:
- `FinalAnalysis.py`: Processes PDFs, computes metrics, and saves results to a CSV file.
- `visualize_gender_metrics.py`: Creates visualizations based on the CSV output.

A metadata file, `Metadata-GENDER.xlsx`, provides gender information for the documents.

## Prerequisites

To run the project, ensure the following requirements are met:

### Software
- **Python 3.8 or higher**
- **pip** (Python package manager)

### Python Packages
Install the required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```
## Commands to Run the Files

Ensure you are in the project directory (e.g., `linguistic-analysis/`) containing `FinalAnalysis.py`, `visualize_gender_metrics.py`, `Documents/`, and `Metadata-GENDER.xlsx`.

1. **Run the Analysis Script**  
   Process the PDFs and compute linguistic metrics:

   ```bash
   python FinalAnalysis.py
   ```