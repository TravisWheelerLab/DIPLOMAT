# DIPLOMAT

DeepLabCut-based Identity Preserving Labeled-Object Multi-Animal Tracking.

## About

DIPLOMAT provides algorithms and tools for performing multi-animal identity preserving tracking on top of single animal and multi animal CNN based tracking packages. Currently, it supports running on both single and multi animal DeepLabCut projects, but can be extended to support other tracking
packages. Unlike other multi-animal tracking packages, DIPLOMAT's algorithms work directly off confidence maps instead of running peak detection, allowing for more nuanced tracking results compared to other methods. 

|                                                            |                                                  |
|------------------------------------------------------------|--------------------------------------------------|
| ![Example of tracking 2 Degus in a Box](imgs/example1.png) | ![Example of tracking 3 Rats](imgs/example2.png) |

DIPLOMAT also includes a UI for performing tracking and several other tools for storing and visualizing confidence maps. 

![UI Demo Showing user correcting tracking in a video](imgs/UIDemo.png)

## Installation

To install DIPLOMAT right now, you can and install it with pip using the following command:
```bash
pip install git+https://github.com/TravisWheelerLab/DIPLOMAT.git
```
To install DIPLOMAT with GUI elements and supervised tracking support, use the command below:
```bash
pip install "diplomat-track[gui] @ git+https://github.com/TravisWheelerLab/DIPLOMAT.git"
```

**NOTE:** You'll likely want to install diplomat in a virtual environment.

To create an environment using conda, and activate it, run these commands:
```bash
conda create -n DIPLOMAT python
conda activate DIPLOMAT
```

To create and activate an environment using venv, follow the commands below, depending on your platform:
```bash
python -m venv DIPLOMAT
# On mac os and linux...
source DIPLOMAT/bin/activate
# On windows...
.\DIPLOMAT\bin\activate.bat
```

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

#### Frame Store Workflow

DIPLOMAT is capable of grabbing model outputs and dumping them to a file, which can improve performance
when analyzing the same video multiple times or allow analysis to be completed somewhere else. To create
a frame store for later analysis, run tracking with the frame store predictor:

```bash
diplomat track -c path/to/config -v path/to/video -p FrameExporter
```

The above command will generate a `.dlfs` file. To run tracking on it, run one of DIPLOMAT's tracking methods, but with the `-fs` flag passing in
the frame store instead of the video.
```bash
# Run DIPLOMAT with no UI...
diplomat unsupervised -c path/to/config -fs path/to/fstore.dlfs
# Run DIPLOMAT with UI...
diplomat supervised -c path/to/config -fs path/to/fstore.dlfs
# Run DIPLOMAT with some other prediction algorithm
diplomat track -c path/to/config -fs path/to/fstore.dlfs -p NameOfPredictorPlugin
```

#### Inspecting DIPLOMAT Plugins

DIPLOMAT was developed with extensibility in mind, so core functionality can be extended via plugins. DIPLOMAT has two kinds of plugins:
 - Predictors: Plugins that take in model outputs and predict poses, or animal locations from them. Some of these also have additional side effects such as plotting or frame export.
 - Frontends: These are plugins that grab frames from another tracking software and pipe them into the predictor the user has selected. Currently, there is only one for deeplabcut.

To get information about predictors, one can use the subcommands of `diplomat predictors`:
```bash
# List predictor names and their descriptions (Names are passed to -p flag of track).
diplomat predictors list
# List the settings of a predictor plugin (Can be passed to -ps flag of track to configure them).
diplomat predictors list_settings PredictorName
```

To get information about frontends, use subcommands of `diplomat frontends`:
```bash
# List all frontends available
diplomat frontends list all
# List loaded frontends...
diplomat frontends list loaded
```

#### Additional Help

All diplomat commands are documented via help strings. To get more information about a diplomat subcommand or command, simply run it with the `-h` or `--help` flag.

```bash
# Help for all of diplomat (lists sub commands of diplomat):
diplomat --help 
# Help for the track subcommand:
diplomat track --help
# Help for the predictors subcommand space:
diplomat predictors --help
```

## Development

DIPLOMAT is written entirely in python. To set up an environment for developing DIPLOMAT, you can simply pull down this repository and install its
requirements.txt dependencies to your virtual environment.

```bash
git clone https://github.com/TravisWheelerLab/DIPLOMAT.git
cd DIPLOMAT
pip install -r requirements.txt
```

For most development, you'll most likely want to add additional predictor plugins. Predictors can be found in `diplomat/predictors`. Classes that
extend Predictor are automatically loaded from this directory. To test predictors, you can use DIPLOMAT's testing command:
```bash
diplomat predictors test PredictorName
```

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

