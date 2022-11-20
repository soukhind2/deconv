#fMRI Deconvolution Toolbox
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.com/project/pip/)
[![Github all releases](https://img.shields.io/github/downloads/Naereen/StrapDown.js/total.svg)](https://GitHub.com/Naereen/StrapDown.js/releases/)
[![GitLab last commit](https://badgen.net/gitlab/last-commit/NickBusey/HomelabOS/)](https://gitlab.com/NickBusey/HomelabOS/-/commits)




The fMRI deconvolution toolbox is a Python package that has been developed on top of the fMRIsim module of BrainIAK. provide some additional information that will permit cognitive neuroscience researchers to develop optimal designs for many common experimental designs used in fMRI.  

Currently, the toolbox only supports two conditions that are alternating in a sequence such as A-B-A-B-A-B... where A and B represent a cue and a target respectively, similiar to attention related fMRI designs (Hopfinger 2000, Kastner 1999).

A future release will include support for multiple stimuli in a trial similar to a event related trial by trial fMRI design.  
&nbsp;  
*Authors: Soukhin Das (UC Davis) & Weigang Yi (UC Davis), 2022*. 

# Installation

If you have ```git```, you can fetch a local copy by:
```
git clone https://github.com/soukhind2/deconv.git
```

Otherwise, you can download from here: [Download Toolbox](https://github.com/soukhind2/deconv/archive/refs/heads/master.zip)  
&nbsp;

To run jupyter notebooks, open a Terminal and ```cd``` to the toolbox directory and run:
```


# Examples

To get started with the toobox, refer to the Jupyter notebooks provided.  
&nbsp;  
[Testing Different Parameter Combinations](https://github.com/soukhind2/deconv/blob/master/deconvolve_optimization.ipynb)  
[Testing Different Null Ratio Combinations](https://github.com/soukhind2/deconv/blob/master/null_events_optimization.ipynb)  
