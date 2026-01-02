# DPLR in PyTorch backend

[![Test Python](https://github.com/ChiahsinChu/torch-dplr/actions/workflows/test_python.yml/badge.svg)](https://github.com/ChiahsinChu/torch-dplr/actions/workflows/test_python.yml) [![codecov](https://codecov.io/gh/chiahsinchu/torch-dplr/graph/badge.svg?token=PlTQbpLNLj)](https://codecov.io/gh/chiahsinchu/torch-dplr)

## Introduction

`torch-dplr` is a PyTorch implementation (training only yet) of the [DPLR (Deep potential long-range) model](https://docs.deepmodeling.com/projects/deepmd/en/master/model/dplr.html). This package is used as a plugin of DeepMD-kit package.

TODO:

- [ ] backend convert between tf and pt
- [ ] lammps interface

## Installation

### Prerequisites

Before installing torch-dplr, ensure you have the following custom deepmd-kit package:

```bash
pip install "deepmd-kit[gpu,cu12,torch] @ git+https://github.com/ChiahsinChu/deepmd-kit.git@devel-pt-dplr"
```

This custom branch, developed on the basis of offical devel branch, includes features of:

- supporting data modifier plugin for PyTorch backend;
- allowing atomic_weight in dipole model, which is required to calculate force/virial correction in DPLR.

### Standard Installation

You can install torch-dplr using pip:

```bash
# you can choose the optional dependencies
pip install "torch-dplr[torch_admp,test] @ git+https://github.com/ChiahsinChu/torch-dplr"
```

## Usage

### Basic Usage

The usage of DPLR model in PyTorch backend is the same as that in TensorFlow backend, i.e., adding `modifier` section in training setup:

```json
{
  "model": {
    "type_map": ["O", "H"],
    "descriptor": {
      "type": "se_e2_a",
      "sel": [46, 92],
      "rcut_smth": 0.5,
      "rcut": 4.0,
      "neuron": [25, 50, 100],
      "resnet_dt": false,
      "axis_neuron": 16,
      "type_one_side": true
    },
    "fitting_net": {
      "type": "dipole",
      "neuron": [100, 100, 100],
      "resnet_dt": true,
      "precision": "float64"
    },
    "modifier": {
      "type": "dipole_charge",
      "model_name": "dw_model.pth",
      "model_charge_map": [-8.0],
      "sys_charge_map": [6.0, 1.0],
      "ewald_h": 1.0,
      "ewald_beta": 1.0
    }
  }
}
```

## License

This project is licensed under the LGPL-3.0-or-later license. See the [LICENSE](LICENSE) file for details.

## Citation

If you use torch-dplr in your research, please cite it as:

```bibtex
@software{zhu_jiaxin_2025_10582233,
  author       = {Zhu, Jia-Xin},
  title        = {DPLR in PyTorch backend},
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v0.1.0},
  doi          = {10.5281/zenodo.10582233},
  url          = {https://github.com/ChiahsinChu/torch-dplr}
}
```

Or refer to the full citation details in the [CITATION.cff](CITATION.cff) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

For questions and support, please open an issue on the [GitHub repository](https://github.com/ChiahsinChu/torch-dplr).
