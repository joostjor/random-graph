# Covid-19 Modelling on GIRGs

This repository contains the main code for two projects related to spreading models on Geometric Inhomogeneous Random Graphs.
It consists of two Python packages: `graph_lib` and `spreading_lib`.
This code is used for the simulations in the paper "[Increasing efficacy of contact-tracing applications by user referrals and stricter quarantining"](https://doi.org/10.1371/journal.pone.0250435).) by Leslie Ann Goldberg, Joost Jorritsma, Júlia Komjáthy, and John Lapinskas, [this link] If you want to re-use our code, then please do not to forget to add a citation to this codebase:

```
@software{JorritsmaLapinskasSpreadingGirgs,
  author       = {Joost Jorritsma and John Lapinskas},
  title        = {Software for "Increasing efficacy of contact-tracing applications by user referrals and stricter quarantining"},
  month        = apr,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.4675115},
  url          = {https://doi.org/10.5281/zenodo.4675115}
}
```

## How to install
0. If needed, create a clean Python 3 environment by running `python3 -m venv <name-of-env>`. Activate the environment by runnning `source <name-of-env>/bin/activate`.
1. Clone the repository by running `git clone git@github.com:joostjor/covid-girg.git` .
2. Install the packages using `pip install .`


## How to uninstall
1. Ensure you have activated the correct environment. Run `pip uninstall girg-spreading`.
