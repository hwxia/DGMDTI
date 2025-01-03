# DGMDTI
DGMDTI: Efficient Drug-Target Interaction Prediction via a Sequence-Based Framework with Attention Mechanism Fusion

## Tips
First, use the ESM2 model to extract protein features(cd ./ESM2_Feature_Extration); 

Second, use the MolFormer model to extract small molecule features(cd ./MolFormer_Feature_Extration); 

Finally, the drug-target interaction prediction can be performed(cd ./DGMDTI_Predict);
## Requirements
The code has been tested running under Python 3.11.3, with the following packages and their dependencies installed:
###
    torch==2.1.0
    torch_geometric==2.5.3
    torch-cluster==1.6.3+pt21cu121
    torch-scatter==2.1.2+pt21cu121
    torch-sparse==0.6.18+pt21cu121
    torch-spline-conv==1.2.2+pt21cu121
    lightning==2.2.3
    lightning-utilities==0.11.2
    numpy==1.26.4
    pandas==2.2.2
    scikit-learn==1.4.2
    pyg-lib==0.4.0+pt21cu121
    pytorch-fast-transformers==0.4.0
    pytorch-lightning==1.1.5
## Usage
    cd ./DGMDTI_Predict
    python main.py 
