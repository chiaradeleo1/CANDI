# 🍬 CANDI
**C**ode for **AN**alysis of modified **D**istance-duality with cosmological Inference

Welcome to the CANDI code!

This repository provides a cosmological analysis code to place constraints from **supernovae (SN)**, **baryon acoustic oscillations (BAO)**, and **gravitational waves (GW)** through distance measurements.  

It allows you to:  
- Run likelihood analyses using current and simulated data.  
- Implement custom cosmologies beyond ΛCDM, including models that break the distance–duality relation (DDR).  
- Use different samplers (MCMC or nested sampling) to obtain posterior constraints.  

---
## ✨ What’s inside?

Here you can find:  

- 📈 Likelihoods for **SN**, **BAO**, and **simulated Einstein Telescope (ET) standard sirens**.  
- 🧩 A custom cosmology **API** (Application Programming Interface) that lets you **plug in your own models** easily.  
- 🔓 Tools to test **DDR-breaking parametrizations**.  
- 🎲 Built-in support for **Cobaya MCMC** and **Nautilus nested sampling**.  
- 📂 Ready-to-use **example settings** and **demo notebooks** to get you started quickly.  

---

## ⚙️ Installation

### Requirements
- Python ≥ 3.9  
- `numpy`, `scipy`, `pandas`, `pyyaml`  
- `cobaya`, `camb`, `getdist`, `nautilus`  
- Optional: `matplotlib`, `jupyter`  


### ⚡ Quickstart

```bash
# Clone the repository
git clone https://github.com/your-username/CANDi.git
cd CANDi

# Install dependencies
pip install -r requirements.txt
```
## 📂 Data

The data are **not included** in this repository.  

For reproducibility of the results in our paper ([arXiv:2505.13613](https://arxiv.org/abs/2505.13613)), the **simulated datasets** and **MCMC chains** can be downloaded from [LINK HERE].  

The repository contains:  
- 🟠 **LSST Supernovae (SN)**  
- 🔵 **SKAO Baryon Acoustic Oscillations (BAO)**  
- 🟣 **Einstein Telescope (ET) Standard Sirens (GW)**  

The chains correspond to the different scenarios discussed in the paper.  
The corresponding configuration files are provided in the `settings/` folder.  

## How to run it?

Each run is controlled via a YAML settings file.  

Key entries include:  
- `output`: name of the run / output folder  
- `SN_data`, `BAO_data`, `GW_data`: specify which datasets to include  
- `cosmology`: choose the expansion model  
- `DDR_options`: enable DDR-breaking models  
- `sampler`: choose between `mcmc` or `nautilus`  

Examples can be found in the `settings/` folder for different cosmological models, while interactive notebook examples are provided under the name **DEMO_**


## 🧩 Custom Cosmology

You can implement your own cosmological model by following  
`theory_code/expansion_models/example_custom_cosmology.py`.  

A valid cosmology must provide at least:  
- `H(z)` — Hubble expansion rate  
- `comoving(z)` — comoving distance  

With this interface you can:  
- ✨ Implement your own **DDR-breaking model** (see e.g. [arXiv:/2505.13613](https://arxiv.org/abs//2505.13613)).  
- 🌌 Modify the **expansion history of the universe** by introducing alternative cosmologies (see e.g. [arXiv:2507.13890](https://arxiv.org/abs/2507.13890)).  

This makes it easy to go **beyond ΛCDM** and test custom scenarios with SN, BAO, and GW probes.  

## 📜 License & Citation

- License: to be specified (e.g. MIT).  
- If you use this code, please cite our papers:
  
  > C. De Leo et. al , *Distinguishing Distance Duality breaking models using electromagnetic and gravitational waves measurements*, [arXiv:2505.13613](https://arxiv.org/abs/2505.13613)

  > E. Fazzari, C. De Leo et. al , *Investigating f(R)-Inflation: background evolution and constraints*, [arXiv:2507.13890](https://arxiv.org/abs/2507.13890)  

---

