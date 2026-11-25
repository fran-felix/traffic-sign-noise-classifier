# traffic-sign-noise-classifier
University Project
A CNN trained to classify traffic signs from the GTSRB, but with noisy and degraded images.

Ignored git files at the end of .gitignore
No image files are uploaded to github, so in order to properly use the code, one must use the material_alunos.zip provided with
the assignment to get the data. Exception for data/ folder, which is downloaded through code.

Brief explanation:
The presented solution uses the base model provided and implements the data augmentation peculiarities to improve its performance.
Implementation can be found at noise.py and it mainly uses PIL native methods to apply the correct image transformations.
The code is correctly integrated to the rest of the base code provided with the assignment.

To generate predictions:
python src/predizer.py --pasta teste --saida NOME_SOBRENOME_RESULTS.txt