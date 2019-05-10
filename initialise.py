"""
When you run the algorithm, all model and memory files are saved in the run folder, in the root directory.

To restart the algorithm from this checkpoint later,
transfer the run folder to the run_archive folder, attaching a run number to the folder name. Then, enter the run number, model version number and memory version number into the initialise.py file, 
corresponding to the location of the relevant files in the run_archive folder. Running the algorithm as usual will then start from this checkpoint.
"""
INITIAL_RUN_NUMBER = None
INITIAL_MODEL_VERSION = None
INITIAL_MEMORY_VERSION = None