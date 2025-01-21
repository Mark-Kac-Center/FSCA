# Fractal Space-Curve Analysis (FSCA) package


Fractal Space-Curve Analysis is a method designed for the analysis of multidimensional data. FSCA combines the [Space-Filling Curve](https://en.wikipedia.org/wiki/Space-filling_curve) mapping for dimensionality reduction with the [Detrended Fluctuation Analysis](https://www.sciencedirect.com/science/article/pii/S0378437102013833) fractal analysis for interpretable results.
<p align="center">
  <img alt="FSCA flowchart" src="imgs/flowchart.png" width="40%" >
</p>

<!-- ![FSCA flowchart]( | width=100) -->

Method is suitable for diverse data formats with special submodule for neuroimaging data. An example of FSCA application to identification of Alzheimer's Disease:
<p align="center">
  <img src="https://github.com/grelade/fractal-alzheimer/blob/master/dataplot.png" alt="Example: Alzheimers' disease" width="70%" >
</p>

## Citation
The method's description and testing is fully described in

Grela, J., Drogosz, Z., Janarek, J., Ochab, J.K., Cifre, I., Gudowska-Nowak, E., Nowak, M.A., Oswiecimka, P., Chialvo, D., 2025. Using space-filling curves and fractals to reveal spatial and temporal patterns in neuroimaging data. J. Neural Eng. [https://doi.org/10.1088/1741-2552/ada705](https://doi.org/10.1088/1741-2552/ada705)

Please cite as:

``` bibtex
@article{grela2025fsca,
	author={Grela, Jacek and Drogosz, Zbigniew and Janarek, Jakub and Ochab, Jeremi K and Cifre, Ignacio and Gudowska-Nowak, Ewa and Nowak, Maciej A and Oswiecimka, Pawel and Chialvo, Dante},
	title={Using space-filling curves and fractals to reveal spatial and temporal patterns in neuroimaging data},
	journal={Journal of Neural Engineering},
	url={http://iopscience.iop.org/article/10.1088/1741-2552/ada705},
	year={2025}
}
```

----

## Usage

### Apply FSCA to MRI scan

For an out-of-the-box analysis of a structural MRI scan:
```
import fsca
pipe = fsca.neuro.pipeline_mri()
pipe.run(scan_file = 'test-data/scanfile.nii.gz', slice_axis = 'z')
hurst_profile = pipe.get_result()
```
The outcome of FSCA is a Hurst profile measuring local fractal properties of the MRI scan along the `slice_axis` axis.

----

### FSCA for generic data structures
Method is applicable to most popular 2D, 3D and 4D structured data. Basic pipelines are:

- `pipeline_2d` : 2D data
- `pipeline_2x1d` : 2+1D  data (f.e. animation or video)
- `pipeline_3d`: 3D data (temperature grids)
- `pipeline_3x1d` 3+1D data (dynamical fields)

----

## Dedicated neuroimaging tools

### MRI preprocessing module
Package comes with a additional, ML-powered preprocessing module for MRI scans using the `ants` package. This module is especially useful for systematic analysis of unprocessed MRI scans. This module is integrated into a dedicated pipeline `fsca.neuro.pipeline_ext.pipeline_mri_preprocessing`:
```
from fsca.neuro.pipeline_ext import pipeline_mri_preprocessing
pipe = pipeline_mri_preprocessing()
pipe.run(scan_file = 'test-data/raw_scan.nii.gz', slice_axis = 'z')
hurst_profile = pipe.get_result()
```

----

## Requirements

TO use the FSCA method: 
```
pip install -r requirements.txt
```

To use the additional MRI preprocessing module:
```
pip install -r requirements_ext.txt
```
