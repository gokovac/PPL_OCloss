# PPL_OCloss
This repository contains necessary files for the implementation of Pull-Push Loss Function. The implementation is based on the previous studies; [OC-Softmax](https://github.com/yzyouzhang/AIR-ASVspoof), [TOC-Softmax](https://github.com/gylin2/ocnet), [One Class Contrastive Loss](https://gitlab.idiap.ch/bob/bob.paper.oneclass_mccnn_2019/-/tree/master?ref_type=heads).

For wav2vec 2.0 - AASIST model, the implementation given in [this](https://github.com/TakHemlata/SSL_Anti-spoofing) repository is followed. 

# Pre-trained model for LA 
You can download the pre-trained model [here](https://drive.google.com/file/d/1ulhJhSk0-paSentTgNVhaWwJhRmu7oJk/view?usp=drive_link).
PPL layer can be downloaded [here](https://drive.google.com/file/d/1hgfED0Ul5zJJxV-WSCcxlVvOFK6Thg-T/view?usp=drive_link).

Results with the pre-trained model: 
EER: 0.13%, min t-DCF: 0.0039 for ASVspoof 2019 LA evaluation.
EER: 0.87%, min t-DCF: 0.2089 for ASVspoof 2021 LA evaluation.

Score files for different random seeds are provided in the Scores folder.
