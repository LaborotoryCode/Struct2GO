.\.venv\Scripts\activate

# MF
python eval_Struct2GO.py --test_data new_mf_test_plddt.pkl --branch mf --network_file network --labels_num 273 --label_network mf_label_network.dgl --seeds 3

# BP
python eval_Struct2GO.py --test_data new_bp_test_plddt.pkl --branch bp --network_file network --labels_num 809 --label_network bp_label_network.dgl --seeds 3
# CC
python eval_Struct2GO.py --test_data new_cc_test_plddt.pkl --branch cc --network_file network --labels_num 298 --label_network cc_label_network.dgl --seeds 3