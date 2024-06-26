from yacs.config import CfgNode as CN
import os 

#Declare some nodes
_C = CN()
_C.SYSTEM = CN()
_C.PATHS = CN()
_C.FMRI=CN()
_C.BACKBONE=CN()

#Paths
_C.PATHS.NSD_ROOT=os.environ.get('NSD_ROOT_PATH') #set NSD_ROOT_PATH as environment variable 
_C.PATHS.BETA_ROOT=os.path.join(_C.PATHS.NSD_ROOT,'nsddata_betas','ppdata')
_C.PATHS.MASK_ROOT=os.path.join(_C.PATHS.NSD_ROOT,'nsddata','ppdata')
_C.PATHS.STIM_ROOT=os.path.join(_C.PATHS.NSD_ROOT,'nsddata_stimuli','stimuli','nsd')
_C.PATHS.FREESURFER_ROOT=os.path.join(_C.PATHS.NSD_ROOT,'freesurfer') #freesurfer path for nsd subjects
#NSD_PREPROC=os.path.join(NSD_ROOT,'nsd_preproc')
_C.PATHS.SCRATCH_PATH=os.environ.get('SCRATCH_PATH') #using scratch for now
_C.PATHS.NSD_PREPROC=os.path.join(_C.PATHS.SCRATCH_PATH,'nsd_preproc') #using scratch for now
_C.PATHS.BACKBONE_FILES=os.environ.get('BACKBONE_ROOT_PATH')

#Files
_C.PATHS.EXP_DESIGN_FILE=os.path.join(_C.PATHS.NSD_ROOT,'nsddata','experiments','nsd','nsd_expdesign.mat')
_C.PATHS.STIM_FILE=os.path.join(_C.PATHS.STIM_ROOT,'nsd_stimuli.hdf5')

#Fmri data parameters
_C.FMRI.FUNC_RES='func1pt8mm'
_C.FMRI.FUNC_PREPROC='betas_fithrf_GLMdenoise_RR'

#File extension for data
_C.FMRI.LOAD_EXT='.nii.gz'

#Default backbone
_C.BACKBONE.NAME='alexnet'
_C.BACKBONE.FILE= 'alexnet.pt'   
_C.BACKBONE.FINETUNE= False
_C.BACKBONE.LAYERS_TO_EXTRACT= ['features.2','features.5','features.7','features.9','features.12']
_C.BACKBONE.PERCENT_OF_CHANNELS= 100
_C.BACKBONE.INPUT_SIZE= (1,3,256,256)

  

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

