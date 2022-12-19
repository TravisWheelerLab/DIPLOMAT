# DIPLOMAT

Deep learning-based Identity Preserving Labeled-Object Multi-Animal Tracking.

**NOTE:** DIPLOMAT is currently alpha software, there may be minor bugs and issues.

## About

DIPLOMAT provides algorithms and tools for performing multi-animal identity preserving tracking on top of single animal and multi animal CNN based tracking packages. Currently, it supports running on both DeepLabCut and SLEAP projects.
Unlike other multi-animal tracking packages, DIPLOMAT's algorithms work directly off confidence maps instead of running peak detection, allowing for more nuanced tracking results compared to other methods. 

|                                                            |                                                  |
|------------------------------------------------------------|--------------------------------------------------|
| ![Example of tracking 2 Degus in a Box](docs/source/_static/imgs/example1.png) | ![Example of tracking 3 Rats](docs/source/_static/imgs/example2.png) |

DIPLOMAT also includes a UI for performing tracking and several other tools for storing and visualizing confidence maps. 

![UI Demo Showing user correcting tracking in a video](docs/source/_static/imgs/UIDemo.png)

## Installation

To install DIPLOMAT with PIP right now, you can and install it with pip using one of the following commands below:
```bash
# For working with SLEAP projects:
pip install "diplomat-track[sleap] @ git+https://github.com/TravisWheelerLab/DIPLOMAT.git"
# For working with DeepLabCut projects:
pip install "diplomat-track[dlc] @ git+https://github.com/TravisWheelerLab/DIPLOMAT.git"
```
To install DIPLOMAT with GUI elements and supervised tracking support, use one of the commands below:
```bash
# For using DIPLOMAT with SLEAP
pip install "diplomat-track[sleap, gui] @ git+https://github.com/TravisWheelerLab/DIPLOMAT.git"
# Again, replace sleap with dlc to install with DeepLabCut support.
pip install "diplomat-track[dlc, gui] @ git+https://github.com/TravisWheelerLab/DIPLOMAT.git"
```

**NOTE:** DIPLOMAT also includes two environment configuration files for setting up DIPLOMAT with conda.
To create an environment using conda, run one of these two commands:
```bash
# Create the environment for using DIPLOMAT with DeepLabCut
conda env create -f https://raw.githubusercontent.com/TravisWheelerLab/DIPLOMAT/main/conda-environments/DIPLOMAT-DEEPLABCUT.yaml
# OR Create an environment for using DIPLOMAT with SLEAP instead...
conda env create -f https://raw.githubusercontent.com/TravisWheelerLab/DIPLOMAT/main/conda-environments/DIPLOMAT-SLEAP.yaml
```
And then activate the environment with one of these two commands:
```bash
# Activate the DeepLabCut/DIPLOMAT environment...
conda activate DIPLOMAT-DEEPLABCUT
# Activate the SLEAP/DIPLOMAT environment...
conda activate DIPLOMAT-SLEAP
```

For a more thorough explanation of the installation process, see the [documentation](https://diplomat.readthedocs.io/en/latest/installation.html).

## Usage

#### Running DIPLOMAT

To run DIPLOMAT on a video once it is installed, simply use DIPLOMAT's `unsupervised` and `supervised` commands to track a video:
```bash
# Run DIPLOMAT with no UI...
diplomat unsupervised -c path/to/config -v path/to/video
# Run DIPLOMAT with UI...
diplomat supervised -c path/to/config -v path/to/video
```

Multiple videos can be tracked by passing them as a list:
```bash
diplomat unsupervised -c path/to/config -v [path/to/video1, path/to/video2, "path/to/video3"]
```

Once tracking is done, DIPLOMAT can create labeled videos via it's `annotate` subcommand:
```bash
diplomat annotate -c path/to/config -v path/to/video
```

If you need to make minor modifications after tracking a video, you can do so using the tweak subcommand:
```bash
diplomat tweak -c path/to/config -v path/to/video
```
This will display a stripped down version of the supervised editing UI, allowing for minor tweaks to be made to the tracks, and then
saved back to the same file.

For a list of additional ways DIPLOMAT can be used, see the [documentation](https://diplomat.readthedocs.io/en/latest/basic_usage.html).

#### Additional Help

All DIPLOMAT commands are documented via help strings. To get more information about a diplomat subcommand or command, simply run it with the `-h` or `--help` flag.

```bash
# Help for all of diplomat (lists sub commands of diplomat):
diplomat --help 
# Help for the track subcommand:
diplomat track --help
# Help for the predictors subcommand space:
diplomat predictors --help
```

## Documentation

DIPLOMAT has documentation on ReadTheDocs at [https://diplomat.readthedocs.io/en/latest](https://diplomat.readthedocs.io/en/latest).

## Development

DIPLOMAT is written entirely in python. To set up an environment for developing DIPLOMAT, you can simply pull down this repository and install its
requirements using poetry. For a further description of how to set up DIPLOMAT for development, see the 
[Development Usage](https://diplomat.readthedocs.io/en/latest/advanced_usage.html#development-usage) section in the documentation.

## Contributing

We welcome external contributions, although it is a good idea to contact the
maintainers before embarking on any significant development work to make sure
the proposed changes are a good fit.

Contributors agree to license their code under the license in use by this
project (see `LICENSE`).

To contribute:

  1. Fork the repo
  2. Make changes on a branch
  3. Create a pull request

## License

See `LICENSE` for details.

## Authors

If you have any questions, feel free to reach out to Isaac Robinson, at [isaac.k.robinson2000@gmail.com](mailto:isaac.k.robinson2000@gmail.com)

See `AUTHORS` the full list of authors.

