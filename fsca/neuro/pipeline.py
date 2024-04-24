from fsca import pipeline_3d

def pipeline_mri_lite():
    import nibabel as nib
    print(pipeline_3d())
    return 'pipeline_mri_lite()'

def pipeline_mri_full():
    import antspyx
    print(pipeline_3d())
    return 'pipeline_mri_full()'

def pipeline_fmri():
    print(pipeline_3x1d())
    return 'pipeline_fmri()'