# DIPLOMAT

DeepLabCut-based Identity Preserving Labeled-Object Multi-Animal Tracking.

## About

**REPLACE:** Describe your project in more detail. What does it do? Who are the intended
users? Why is it important or meaningful? How does it improve upon similar
software? Is it a component of or extension to a larger piece of software?

DIPLOMAT provides algorithms and tools for performing multi-animal identity preserving tracking on top of single animal and multi animal CNN based tracking packages. Currently, it supports running on both single and multi animal DeepLabCut projects, but can be extended to support other tracking
packages. Unlike other multi-animal tracking packages, DIPLOMAT's algorithms work directly off confidence maps instead of running peak detection, allowing for more naunced tracking results compared to other methods. 

**TODO:** Add pictures of tracking results....

DIPLOMAT also includes a UI for performing tracking and several other tools for storing and visualizing confidence maps. 

**TODO:** Add quick UI GIF

## Installation

DIPLOMAT will eventually be installable using pip. To install it right now, you can and install it with pip using the following command:
```bash
pip install git+https://github.com/TravisWheelerLab/DIPLOMAT.git
```
To install DIPLOMAT with GUI elements and supervised tracking support, use the command below:
```bash
pip install "diplomat-track[gui] @ git+https://github.com/TravisWheelerLab/DIPLOMAT.git"
```

**NOTE:** 

## Usage

**REPLACE:** How is the software run (or consumed, for libraries)? Are there any command line
flags the user should know about? What do they do, exactly? What do the input
data look like? Are there special file formats in use, what are they? What does
the output look like?

## Development

**REPLACE:** What language(s) are in use? What does a user need to install for development
purposes? This might include build systems, Docker, a particular compiler or
runtime version, test libraries or tools, linters, code formatters, or other
tools. Are there any special requirements, like processor architecture? What
commands should developers use to accomplish common tasks like building, running
the test suite, and so on?

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

**REPLACE:** Who should people contact with questions?

See `AUTHORS` the full list of authors.

