# Installation

To download the code and all dependencies, paste the following
into a terminal. Make sure you have Anaconda installed, since it
is required to manage dependencies. This set of commands will
create a new Anaconda environment called "teaching" for this
project.

```sh
git clone --recurse-submodules \
    https://github.com/nauralcodinglab/linear-nonlinear-dendrites.git \
    && conda env create -f environment.yml \
    && conda activate teaching \
    && pip install -e ./ez-ephys
```

# Minimal workflow

When you want to start work, run `git pull && conda activate
teaching && jupyter notebook` to fetch updates from github and
start jupyter in your web browser.

When you're finished, run `git commit -am "Short description of
changes" && git push origin master` to publish your changes to
github.
