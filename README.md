# KEGNI

**KEGNI (Knowledge graph-Enhanced Gene regulatory Network Inference)** is a knowledge-guided framework for inferring cell type-specific gene regulatory networks (GRNs) from scRNA-seq data by integrating prior biological knowledge. 

<p align="center">
  <img src="docs/workflow/worklow.bmp" alt="KEGNI Workflow" width="600" height="400"/>
</p>


---

##  Quick Start

### Run Example Scripts

To get started quickly, run the following scripts:

```bash
bash run_mESC.sh              
bash run_pbmc_naiveCD4T.sh    
```

---

##  Input Files

All required data files—including **scRNA-seq datasets**, **ground truth networks**, and **cell type-specific knowledge graphs**—can be downloaded from: [Zenodo Repository](https://zenodo.org/records/14028211)

---

## Documentation & Tutorials

We provide several tutorials and user guide for construct [cell type specific knowledge graph](./docs/native_CD4T_cell_KG.html), [benchmark with multi-omics methods with ground truth from cistrome](./docs/benchmark_cistrome.ipynb), [GO enrichment results and ARI (Adjusted Rand Index) calculations](docs/ARI_calculation.html).

---

