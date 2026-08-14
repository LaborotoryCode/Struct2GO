# Uncertainty-Aware Cross Attention Fusion for Improved Protein Function Prediction
# Abstract
We propose UAX, a novel Uncertainty-Aware Cross Attention Fusion architecture that integrates per-token reliability into the fusion process via the attention mechanism using the epistemic confidence score of predicted Local Distance Difference Test (pLDDT).

![UAX architecture](/model.png)

*Figure 1. A graphical representation of the UAX framework*

# Data
- Formatted Dataset: [UAX_Training_Data](https://drive.google.com/file/d/1NpQLOKgOLk77B0tF5LW8djwZDrutLuj3/view?usp=sharing)
- Protein structure: To download from the [AlphaFold Protein Struct Database](https://alphafold.ebi.ac.uk/download)
- Protein sequence: To download from the [UniProt website](https://www.uniprot.org/) 
- Protein annotion: To download from the [GOA website](https://www.ebi.ac.uk/GOA/)
- Gene Ontology: To download from the [GO website](http://geneontology.org/)

# Instructions for usage
## Training
 ```python
 python3 run-code.ps1
 ```

## Validation
``` python
python3 run_valid.ps1
```

### Models
1. Struct2Go
2. GAT-GO

## Scores

| Model | Ontology | Method | Fmax | AUC | AUPR |
|:---|:---|:---|---:|---:|---:|
| Struct2GO | BPO | Baseline | 0.4542 | 0.8704 | 0.4953 |
| Struct2GO | BPO | **UAX** | **0.4677** | **0.8734** | **0.5039** |
| Struct2GO | CCO | Baseline | 0.6208 | **0.9354** | 0.6969 |
| Struct2GO | CCO | **UAX** | **0.6295** | 0.9345 | **0.7043** |
| Struct2GO | MFO | Baseline | 0.6880 | 0.9701 | 0.7732 |
| Struct2GO | MFO | **UAX** | **0.7379** | **0.9802** | **0.8293** |
| GAT-GO | BPO | Baseline | 0.4802 | 0.8762 | 0.5617 |
| GAT-GO | BPO | **UAX** | **0.4956** | **0.8805** | **0.5827** |
| GAT-GO | CCO | Baseline | 0.6587 | 0.9410 | 0.7839 |
| GAT-GO | CCO | **UAX** | **0.6666** | **0.9424** | **0.7930** |
| GAT-GO | MFO | Baseline | 0.7164 | 0.9617 | 0.8439 |
| GAT-GO | MFO | **UAX** | **0.7433** | **0.9666** | **0.8692** |

*Table 3. Scores from Ablation 3. Bold values indicate the better result between the baseline and UAX methods.*





