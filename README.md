# mfmri

Module for **extracting multifractal features** from **Magnetic Resonance Imaging** scans by utilizng a 2D-to-1D map based on the [Hilbert Space-Filling Curve](https://en.wikipedia.org/wiki/Hilbert_curve).
Fractal calculations are performed using the [Multifractal Detrended Fluctuation Analysis](https://www.sciencedirect.com/science/article/pii/S0378437102013833).

## Usage
We provide full and lite versions of the pipeline. The full version includes a machine-learning preprocessing module while the lite version is lightweight but assumes you input an mri scan preprocessed elsewhere.

#### Full pipeline

To extract Hurst exponents from a **raw (i.e. not preprocessed) MRI scan**:
```
from mfmri.mfmri_full import MFractalMRI

mri = MFractalMRI()
mri.pipeline(scan_file = 'test-data/scanfile.nii.gz', slice_axis = 'z')
hurst_exps = mri.get_hurst()
```
The output is a set of exponents along the z axis.

#### Lite pipeline

To extract Hurst exponents from a **preprocessed MRI scan**, use instead:
```
from mfmri.mfmri_lite import LiteMFractalMRI

mri = LiteMFractalMRI()
...
```

## Requirements
* [MFDFA](https://github.com/mlaib/MFDFA)
* [nibabel](https://github.com/nipy/nibabel) [used by LiteMFractalMRI] 
* [antspyx](https://github.com/ANTsX/ANTsPy) [used by MFractalMRI]
* [antspynet](https://github.com/ANTsX/ANTsPyNet) [used by MFractalMRI]
