### Files And Directories Hierarchy
Let's understand the hierarchy of files:

after cloning the repo and $cd fairseq/ , you are in the root directory.
every upcoming path will be inside and relevant to it.

* fairseq/ : it contains main python scripts that you may need.

* fairseq_cli : this folder contains main python scripts of preprocess.py, train.py, generate.py, ...
In other words, the files you use when you call $ fairseq-${file anem without '.py'} shell commands.

* scripts/ : it contains scripts which are not directly related to deep learning. For example, scripts/spm_train.py is used in learning BPE on multilingual data. You may check examples/translation/prepare-iwslt17-multilingual.sh for details about usage of this specific file.

## examples/ :
This directory contains examples for each use case for the fairseq library, e.g. find preprocessing, training and evaluating translation model in examples/translation/README.
**Note**: README file does not contain scripts to prepare the data before getting it into $ fairseq-preprocess. They should reference them, e.g. examples/translation/README.md references examples/translation/prepare-iwslt14.sh for learning and creating BPE files. You need to follow them and adapt them on your pathes and ata.




