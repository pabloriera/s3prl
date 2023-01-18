python run_downstream.py -m train -n epa_hubert -u hubert -c downstream/pronscor_classification/config_epa.yaml  -d pronscor_classification 
python run_downstream.py -m train -n epa_wav2vec2 -u wav2vec2_base_960 -c downstream/pronscor_classification/config_epa.yaml  -d pronscor_classification 
python run_downstream.py -m train -n epa_xlsr_53 -u xlsr_53 -c downstream/pronscor_classification/config_epa.yaml  -d pronscor_classification 
python run_downstream.py -m train -n l2arctic_hubert -u hubert -c downstream/pronscor_classification/config_l2arctic.yaml  -d pronscor_classification 
python run_downstream.py -m train -n l2arctic_wav2vec2 -u wav2vec2_base_960 -c downstream/pronscor_classification/config_l2arctic.yaml  -d pronscor_classification 
python run_downstream.py -m train -n l2arctic_xlsr_53 -u xlsr_53 -c downstream/pronscor_classification/config_l2arctic.yaml  -d pronscor_classification 
python run_downstream.py -f -s last_hidden_state -m train -n epa_hubert_ft -u hubert -c downstream/pronscor_classification/config_epa.yaml  -d pronscor_classification 
python run_downstream.py -f -s last_hidden_state -m train -n epa_wav2vec2_ft -u wav2vec2_base_960 -c downstream/pronscor_classification/config_epa.yaml  -d pronscor_classification 
python run_downstream.py -f -s last_hidden_state -m train -n epa_xlsr_53_ft -u xlsr_53 -c downstream/pronscor_classification/config_epa.yaml  -d pronscor_classification 
python run_downstream.py -f -s last_hidden_state -m train -n l2arctic_hubert_ft -u hubert -c downstream/pronscor_classification/config_l2arctic.yaml  -d pronscor_classification 
python run_downstream.py -f -s last_hidden_state -m train -n l2arctic_wav2vec2_ft -u wav2vec2_base_960 -c downstream/pronscor_classification/config_l2arctic.yaml  -d pronscor_classification 
python run_downstream.py -f -s last_hidden_state -m train -n l2arctic_xlsr_53_ft -u xlsr_53 -c downstream/pronscor_classification/config_l2arctic.yaml  -d pronscor_classification 
