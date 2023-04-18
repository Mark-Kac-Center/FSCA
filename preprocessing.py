import pathlib
import ants
import antspynet
from typing import Union
from pathlib import Path

import numpy


class Preprocessing:
    """
    Set of wrappers for ANTs and ANTsPyNet routines for brain extraction from NiFTi structural scans of human brains.

    ANTs images can be easily converted to numpy arrays by calling .numpy() function.
    """
    def __init__(self,
                 save_brain_img: bool = False,
                 save_bias_corrected: bool = False,
                 brain_threshold: float = 0.99,
                 n4_bias_corr_kwds: dict = None,
                 ) -> None:

        self.save_brain_img = save_brain_img
        self.save_bias_corrected = save_bias_corrected
        self.brain_threshold = brain_threshold
        if n4_bias_corr_kwds is None:
            self.n4_bias_corr_kwds = dict(shrink_factor=4,
                                          convergence={'iters': [50, 50, 50, 50], 'tol': 1e-07},
                                          spline_param=200, )
        else:
            self.n4_bias_corr_kwds = n4_bias_corr_kwds

    @staticmethod
    def read_img(img_path: Union[str, Path]) -> ants.core.ants_image.ANTsImage:
        """
        Read NiFTi scan from a given path, return an ANTsImage object
        """
        assert isinstance(img_path, (str, pathlib.Path)), \
            "img_path should be string of pathlib.Path"

        if type(img_path) == str:
            img_path = Path(img_path)
        if img_path.exists():
            img = ants.image_read(str(img_path))
            return img
        else:
            print("File does not exist.")

    @staticmethod
    def save_img(img_path: Union[str, Path], img: ants.core.ants_image.ANTsImage) -> None:
        """
        Save ANTs image to a .nii.gz file using ANTs
        """

        assert isinstance(img, ants.core.ants_image.ANTsImage), "img has to be ANTs Image"
        assert isinstance(img_path, (str, Path)), ""

        if isinstance(img_path, str):
            img_path = Path(img_path)

        dir_name = img_path.parent
        filename = img_path.stem.split('.')[0] # get true stem
        suffix = "".join(img_path.suffixes)
        if suffix == '':
            suffix = '.nii.gz'
        elif suffix in [".nii", ".nii.gz"]:
            suffix = suffix
        else:
            suffix = '.nii.gz'
        output_path = dir_name / (filename + suffix)

        ants.image_write(img, str(output_path))

    @staticmethod
    def show_img_path(img_path: Union[str, Path]) -> None:
        """
        Simple wrapper for ants.plot_ortho method.
        Plots image from a given path containing NiFTi image.
        """
        img = Preprocessing.read_img(img_path)
        ants.plot_ortho(img)

    @staticmethod
    def show_img(img: ants.core.ants_image.ANTsImage,
                 overlay: ants.core.ants_image.ANTsImage = None) -> None:
        """
        Simple wrapper for ants.plot_ortho method.
        Plot image from a given an ANTs image. An overlay image can be specified for comparisons.
        """

        ants.plot_ortho(img, overlay, overlay_cmap="CMRmap")

    def bias_correction(self,
                        img: ants.core.ants_image.ANTsImage,
                        n4_bias_corr_kwds: dict = None,
                        save_bias_corrected: bool = None,
                        img_path: Union[str, Path] = None,
                        ) -> ants.core.ants_image.ANTsImage:
        """
        Perform magnetic-field bias correction using ANTs N4 bias correction routine

        :return: ANTs image
        """

        if n4_bias_corr_kwds is None:
            n4_bias_corr_kwds = self.n4_bias_corr_kwds

        img_corrected = ants.n4_bias_field_correction(img, **n4_bias_corr_kwds)

        if save_bias_corrected is None:
            save_bias_corrected = self.save_bias_corrected

        if save_bias_corrected:
            if img_path is None:
                img_path = Path("bias_corrected_image.nii.gz")
            self.save_img(img_path, img_corrected)

        return img_corrected

    def brain_extract(self,
                      img: ants.core.ants_image.ANTsImage,
                      modality: str = 't1',
                      save_brain_img: bool = None,
                      img_path: Union[str, Path] = None,
                      brain_threshold: float = None,
                      ) -> ants.core.ants_image.ANTsImage:
        """
        Extract brain from the scan using ANN based antspynet routine

        :return: ANTs image
        """
        brain_mask = antspynet.utilities.brain_extraction(img, modality)

        if brain_threshold is None:
            brain_threshold = self.brain_threshold
        binary_mask = ants.utils.threshold_image(brain_mask, low_thresh=brain_threshold)
        brain_img = ants.mask_image(img, binary_mask)

        if save_brain_img is None:
            save_brain_img = self.save_brain_img

        if save_brain_img:
            if img_path is None:
                img_path = Path("brain.nii.gz")
            self.save_img(img_path, brain_img)

        return brain_img

    def preprocess_structural_img(self,
                                  img: ants.core.ants_image.ANTsImage,
                                  n4_bias_corr_kwds: dict = None,
                                  brain_threshold: float = None,
                                  show_overlay: bool = False,
                                  to_numpy: bool = False,
                                  ) -> Union[ants.core.ants_image.ANTsImage, numpy.ndarray]:
        """
        Compute magnetic field bias correction for a given ANTs image and
        extract brain from a scan using ANTs and ANTsPyNet routines.

        If to_numpy=True, the method returns numpy array.

        :return: ANTs brain image / numpy array
        """
        if n4_bias_corr_kwds is None:
            n4_bias_corr_kwds = self.n4_bias_corr_kwds
        img_corrected = self.bias_correction(img,
                                             n4_bias_corr_kwds=n4_bias_corr_kwds,
                                             save_bias_corrected=False)

        img_brain = self.brain_extract(img_corrected,
                                       brain_threshold=brain_threshold,
                                       save_brain_img=False)
        if show_overlay:
            self.show_img(img, img_brain)

        if to_numpy:
            img_brain = img_brain.numpy()

        return img_brain

    def preprocess_structural_img_path(self,
                                       img_path: Union[str, Path],
                                       n4_bias_corr_kwds: dict = None,
                                       brain_threshold: float = None,
                                       save_bias_corrected: bool = None,
                                       save_brain_img: bool = None,
                                       show_overlay: bool = False,
                                       to_numpy: bool = False,
                                       ) -> Union[ants.core.ants_image.ANTsImage, numpy.ndarray]:
        """
        Compute magnetic field bias correction for a given path of an ANTs image and
        extract brain from a scan using ANTs and ANTsPyNet routines.

        If to_numpy=True, the method returns numpy array.

        It is possible to save bias corrected and brain images in original file directory.

        :return: ANTs brain image / numpy array
        """
        if isinstance(img_path, str):
            img_path = Path(img_path)
        dir_name = img_path.parent
        file_name = img_path.stem.split('.')[0] # get 'true' stem
        suffix = "".join(img_path.suffixes)

        img = self.read_img(img_path)
        if n4_bias_corr_kwds is None:
            n4_bias_corr_kwds = self.n4_bias_corr_kwds
        img_corrected = self.bias_correction(img,
                                             n4_bias_corr_kwds=n4_bias_corr_kwds,
                                             save_bias_corrected=False)

        if save_bias_corrected is None:
            save_bias_corrected = self.save_bias_corrected

        if save_bias_corrected:
            output_path = dir_name / (file_name + "_bias_corrected" + suffix)
            self.save_img(output_path, img_corrected)

        if brain_threshold is None:
            brain_threshold = self.brain_threshold

        img_brain = self.brain_extract(img_corrected,
                                       brain_threshold=brain_threshold,
                                       save_brain_img=False)
        if save_brain_img is None:
            save_brain_img = self.save_brain_img

        if save_brain_img:
            output_path = dir_name / (file_name + "_bias_corrected_brain" + suffix)
            self.save_img(output_path, img_brain)

        if show_overlay:
            self.show_img(img, img_brain)

        if to_numpy:
            img_brain = img_brain.numpy()

        return img_brain
