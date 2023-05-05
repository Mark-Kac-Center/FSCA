# mfmri

Module for **extracting multifractal features** from **Magnetic Resonance Imaging** scans by utilizng a 2D-to-1D map based on the [Hilbert Space-Filling Curve](https://en.wikipedia.org/wiki/Hilbert_curve).
Fractal calculations are performed using the [Multifractal Detrended Fluctuation Analysis](https://www.sciencedirect.com/science/article/pii/S0378437102013833).

## Usage
To extract Hurst exponents for brain slices along the z-axis:
```
from mfmri import MFractalMRI

mri = MFractalMRI()
mri.pipeline(scan_file = 'test-data/scanfile.nii.gz', slice_axis = 'z')
hurst_exps = mri.get_hurst()
```

## Requirements
* [MFDFA](https://github.com/mlaib/MFDFA)
* [antspyx](https://github.com/ANTsX/ANTsPy)
* [antspynet](https://github.com/ANTsX/ANTsPyNet)
