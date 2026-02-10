<div align="center">
  <br />
  <p>
    <a href="https://github.com/Chokyotager/Boltz-Metadiffusion"><img src="/docs/metadiffusion_banner.png" alt="banner" /></a>
  </p>
  <br />
  <p>
  </p>
</div>



## Introduction

Metadiffusion is implemented on top of [Boltz-2](https://github.com/jwohlwend/boltz) (Passaro et al., 2025). In essence, metadiffusion is a gradient-based diffusion guidance method applied to biomolecular structure generation. It allows optimisation, steering, and exploration of collective variables that control a structure's property such as its overall compactness and surface accessibility.

By maximising the pairwise RMSD between samples, highly diverse conformers can be generated. These conformers can also be fitted to experimental data including NMR chemical shifts and SAXS.

## Installation

Install Boltz-Metadiffusion directly from GitHub:

```
git clone https://github.com/Chokyoterger/Boltz-Metadiffusion.git
cd boltz; pip install -e .[cuda]
```

A fresh environment is recommended for installation.

If you are installing on CPU-only or non-CUDA GPUs hardware, remove `[cuda]` from the above commands. Note that the CPU version is significantly slower than the GPU version.

## Inference

> [!IMPORTANT]  
> Metadiffusion is **NOT COMPATIBLE** with Boltz-1

You can run inference using Boltz with:

```
boltz predict input_path --use_msa_server
```

Do note that for certain methods, such as generating highly diverse conformers, you should run Boltz with the `--diffusion_samples` argument specified:

```
boltz predict input_path --use_msa_server --diffusion_samples 8
```

`input_path` should point to a YAML file, or a directory of YAML files for batched processing, describing the biomolecules you want to model and the properties you want to predict (e.g. affinity, metadiffusion collective variables). To see all available options: `boltz predict --help` and for more information on these input formats, see our [prediction instructions](docs/prediction.md).

For metadiffusion parameters, the YAML can be edited to include the biases, see [metadiffusion](docs/metadiffusion.md) for more details and [collective variables](docs/collective_variables.md).


## License

The code is released under the MIT license as per the original Boltz family of models.


## Cite

If you use this code or the models in your research, please cite the following papers:

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}

@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}
```

In addition if you use the automatic MSA generation, please cite:

```bibtex
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  year={2022},
}
```
