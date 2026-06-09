.\.venv\Scripts\activate

# MF
python train_Struct2GO_new.py --network_file network --train_data new_mf_train_plddt.pkl --valid_data new_mf_valid_plddt.pkl --branch mf -labels_num 273 -label_network mf_label_network.dgl --seeds 0 1 2

# BP
python train_Struct2GO_new.py --network_file network --train_data new_bp_train_plddt.pkl --valid_data new_bp_valid_plddt.pkl --branch bp -labels_num 809 -label_network bp_label_network.dgl --seeds 0 1 2

# CC
python train_Struct2GO_new.py --network_file network --train_data new_cc_train_plddt.pkl --valid_data new_cc_valid_plddt.pkl --branch cc -labels_num 298 -label_network cc_label_network.dgl --seeds 0 1 2