# Installation

To download the code and all dependencies, paste the following
into a terminal. Make sure you have Anaconda installed, since it
is required to manage dependencies. This set of commands will
create a new Anaconda environment called "teaching" for this
project.

```sh
git clone --recurse-submodules \
    https://github.com/nauralcodinglab/linear-nonlinear-dendrites.git \
    && cd linear-nonlinear-dendrites \
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

# Troubleshooting

## Git pull doesn't work

**Problem:** git pull produces a message like "local branch master and upstream
origin/master have diverged" or "unable to fast-forward".

**Cause:** You've made commits since your last git pull, and someone else has
pushed changes to the same branch since then.

**Solution:** Merge your commits into the upstream (GitHub) version. Running
the following two commands in a terminal will fix this automatically most of
the time.

```sh
git commit -am "Commit everything before merge" \
    && git checkout -b local-version \
    && git reset --hard origin/master \
    && git merge local-version
```

If the above command produces warnings about merge conflicts, you have to fix
them by editing the affected files manually and committing the result before
running the next commant.

```sh
git push origin master \
    && git branch -d local-version
```
