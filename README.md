# Deep Visual Proteomics links vascular smooth muscle cell phenotypes to atherosclerotic plaque stability

Authors: Elena Kratz, Lars Maegdefessel, and Matthias Mann.

Preprint: https://www.biorxiv.org/content/10.64898/2026.07.17.738909v1

This repository contains the code for the analysis performed in the preprint article.

## Abstract
Vascular smooth muscle cells (VSMCs) drive atherosclerosis through phenotypic switching, yet their spatial organization and protein signatures within plaques remain poorly characterized. Here, we applied Deep Visual Proteomics (DVP) to dissect more than 500 VSMC neighborhoods across 24 human carotid plaques and profile VSMC plasticity in disease. To functionally interpret these tissue proteomes, we built a reference atlas of primary VSMCs driven toward five phenotypes (by TGF-β, PDGF-BB, osteogenic stimuli, IL-1β, or cholesterol), quantifying over 10,000 proteins. We integrated tissue and reference proteomes by building a deep-learning framework thereby assigning functional phenotypes to each neighborhood. This revealed spatially distinct phenotype distributions and a shift toward dedifferentiated states in unstable plaques. Knockdown of four candidates (TNC, TNFAIP2, AEBP1, PLK1) validated operational roles during phenotypic switching. Our approach functionally annotates spatial proteomes and links VSMC plasticity to plaque instability in carotid artery disease.

## Data availability
The mass spectrometry proteomics data have been deposited to the ProteomeXchange Consortium via the PRIDE69 partner repository (identifier: PXD079843). Image annotation files (XML) and immunofluorescence images (CZI) are available at the BioImage Archive (accession [S-BIAD3625]) (https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD3625)).

## Naming conventions
File and variable names in this repository use the internal identifiers from data acquisition:
- `EK{n}` refers to plaque ID n in the manuscript (n = 1–24).
- `Niche{k}` refers to neighborhood k of that plaque.

## Annotation files (XML)
XML files are provided following the following scheme:

    xml/
    ├── EK1/
    │   ├── EK1_All.xml        # all segmented ACTA2+ shapes of the slide
    │   ├── EK1_Niche1.xml     # shapes of neighborhood 1
    │   └── ...
    ├── ...
    └── EK24/

In total, 24 slide-level files and 545 neighborhood files (all acquired neighborhoods) are provided.

## Neighborhood clustering (`cellNeighbourhoods/`)
Interactive Streamlit application to define neighborhoods around manually placed centroids.
Inputs:
- `EK{n}_All.xml` of the respective plaque
- centroid coordinates from `cellNeighbourhoods/centroids/centroids_all_plaques.csv` (columns: plaque_ID, slide, neighborhood, x, y, acquired)

Run:

    pip install -r cellNeighbourhoods/requirements.txt
    streamlit run cellNeighbourhoods/st_spatialvsmc.py

Upload `EK{n}_All.xml`, then enter the centroids of that plaque in the "Centroid coordinates" field as one line per centroid in the format `neighborhood, x, y` (e.g. `1, -100000, 690000`). Neighborhoods were defined with 400 shapes each.

## Luminal-to-periphery score (`Luminal_Periphery_Score_Atherosclerosis_Proteomics.ipynb`)
Requires the slide-level and neighborhood XML files of each plaque as shown above (as downloaded from the BioImage Archive). Set `DATA_DIR` at the top of the notebook to the location of the downloaded `xml/` folder and run all cells. Per-slide offsets used to correct the estimated plaque center are defined in the notebook.
