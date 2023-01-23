python run_evaluation.py -c result/downstream/epa_distilhubert_base/best-states-dev.ckpt  -s dev test
python run_evaluation.py -c result/downstream/epa_hubert/best-states-dev.ckpt -s dev test
python run_evaluation.py -c result/downstream/epa_hubert_testigo/best-states-dev.ckpt -s dev test
python run_evaluation.py -c result/downstream/epa_hubert_testigo_clean/best-states-dev.ckpt -s dev test
python run_evaluation.py -c result/downstream/epa_unispeech_sat_base_plus/best-states-dev.ckpt -s dev test
python run_evaluation.py -c result/downstream/epa_wav2vec2/best-states-dev.ckpt -s dev test
python run_evaluation.py -c result/downstream/epa_wavlm/best-states-dev.ckpt -s dev test
python run_evaluation.py -c result/downstream/epa_xls_r_300m/best-states-dev.ckpt -s dev test
python run_evaluation.py -c result/downstream/l2arctic_distilhubert_base/best-states-dev.ckpt -s dev test
python run_evaluation.py -c result/downstream/l2arctic_hubert/best-states-dev.ckpt -s dev test
python run_evaluation.py -c result/downstream/l2arctic_unispeech_sat_base_plus/best-states-dev.ckpt -s dev test
python run_evaluation.py -c result/downstream/l2arctic_wav2vec2/best-states-dev.ckpt -s dev test
python run_evaluation.py -c result/downstream/l2arctic_wavlm/best-states-dev.ckpt -s dev test
python run_evaluation.py -c result/downstream/l2arctic_xls_r_300m/best-states-dev.ckpt -s dev test


python run_evaluation.py -c result/downstream/epa_hubert/best-states-dev.ckpt -s dev test 
python run_evaluation.py -c result/downstream/epa_hubert_ft/best-states-dev.ckpt -s dev test 
python run_evaluation.py -c result/downstream/epa_wav2vec2/best-states-dev.ckpt -s dev test 
python run_evaluation.py -c result/downstream/epa_wav2vec2_ft/best-states-dev.ckpt -s dev test 
python run_evaluation.py -c result/downstream/epa_xlsr_53/best-states-dev.ckpt -s dev test 
python run_evaluation.py -c result/downstream/epa_xlsr_53_ft/best-states-dev.ckpt -s dev test 
python run_evaluation.py -c result/downstream/l2arctic_hubert/best-states-dev.ckpt -s dev test 
python run_evaluation.py -c result/downstream/l2arctic_wav2vec2/best-states-dev.ckpt -s dev test 
python run_evaluation.py -c result/downstream/l2arctic_xlsr_53/best-states-dev.ckpt -s dev test 
python run_evaluation.py -c result/downstream/l2arctic_hubert_ft/best-states-dev.ckpt -s dev test 
python run_evaluation.py -c result/downstream/l2arctic_wav2vec2_ft/best-states-dev.ckpt -s dev test 

# train pending
# python run_evaluation.py -c result/downstream/l2arctic_xlsr_53_ft /best-states-dev.ckpt -s dev test 

python run_evaluation.py -c result/downstream/timit_hubert/best-states-dev.ckpt -s dev test  --config result/downstream/epa_hubert/config_2023-01-19-17-49-20.yaml --phone-db-map TIMIT EPA --output-dir result/downstream/timit_hubert_epa
python run_evaluation.py -c result/downstream/timit_hubert_ft/best-states-dev.ckpt -s dev test  --config result/downstream/epa_hubert/config_2023-01-19-17-49-20.yaml --phone-db-map TIMIT EPA --output-dir result/downstream/timit_hubert_ft_epa
python run_evaluation.py -c result/downstream/timit_wav2vec2/best-states-dev.ckpt -s dev test  --config result/downstream/epa_hubert/config_2023-01-19-17-49-20.yaml --phone-db-map TIMIT EPA --output-dir result/downstream/timit_wav2vec2_epa
python run_evaluation.py -c result/downstream/timit_wav2vec2_ft/best-states-dev.ckpt -s dev test  --config result/downstream/epa_hubert/config_2023-01-19-17-49-20.yaml --phone-db-map TIMIT EPA --output-dir result/downstream/timit_wav2vec2_ft_epa

python run_evaluation.py -c result/downstream/timit_hubert/best-states-dev.ckpt -s dev test  --config result/downstream/l2arctic_hubert/config_2023-01-19-19-33-38.yaml --phone-db-map TIMIT ARCTIC --output-dir result/downstream/timit_hubert_l2arctic
python run_evaluation.py -c result/downstream/timit_hubert_ft/best-states-dev.ckpt -s dev test  --config result/downstream/l2arctic_hubert/config_2023-01-19-19-33-38.yaml --phone-db-map TIMIT ARCTIC --output-dir result/downstream/timit_hubert_ft_l2arctic
python run_evaluation.py -c result/downstream/timit_wav2vec2/best-states-dev.ckpt -s dev test  --config result/downstream/l2arctic_hubert/config_2023-01-19-19-33-38.yaml --phone-db-map TIMIT ARCTIC --output-dir result/downstream/timit_wav2vec2_l2arctic
python run_evaluation.py -c result/downstream/timit_wav2vec2_ft/best-states-dev.ckpt -s dev test  --config result/downstream/l2arctic_hubert/config_2023-01-19-19-33-38.yaml --phone-db-map TIMIT ARCTIC --output-dir result/downstream/timit_wav2vec2_ft_l2arctic

# old configs
# python run_evaluation.py -c result/downstream/timit_hubert/best-states-dev.ckpt -s dev test  --config result/downstream/epa_hubert/config_2023-01-18-15-07-44.yaml --phone-db-map TIMIT EPA --output-dir result/downstream/timit_hubert_epa
# python run_evaluation.py -c result/downstream/timit_hubert_ft/best-states-dev.ckpt -s dev test  --config result/downstream/epa_hubert/config_2023-01-18-15-07-44.yaml --phone-db-map TIMIT EPA --output-dir result/downstream/timit_hubert_ft_epa
# python run_evaluation.py -c result/downstream/timit_wav2vec2/best-states-dev.ckpt -s dev test  --config result/downstream/epa_hubert/config_2023-01-18-15-07-44.yaml --phone-db-map TIMIT EPA --output-dir result/downstream/timit_wav2vec2_epa
# python run_evaluation.py -c result/downstream/timit_wav2vec2_ft/best-states-dev.ckpt -s dev test  --config result/downstream/epa_hubert/config_2023-01-18-15-07-44.yaml --phone-db-map TIMIT EPA --output-dir result/downstream/timit_wav2vec2_ft_epa

# python run_evaluation.py -c result/downstream/timit_hubert/best-states-dev.ckpt -s dev test  --config result/downstream/l2arctic_hubert/config_2023-01-18-17-11-52.yaml --phone-db-map TIMIT ARCTIC --output-dir result/downstream/timit_hubert_l2arctic
# python run_evaluation.py -c result/downstream/timit_hubert_ft/best-states-dev.ckpt -s dev test  --config result/downstream/l2arctic_hubert/config_2023-01-18-17-11-52.yaml --phone-db-map TIMIT ARCTIC --output-dir result/downstream/timit_hubert_ft_l2arctic
# python run_evaluation.py -c result/downstream/timit_wav2vec2/best-states-dev.ckpt -s dev test  --config result/downstream/l2arctic_hubert/config_2023-01-18-17-11-52.yaml --phone-db-map TIMIT ARCTIC --output-dir result/downstream/timit_wav2vec2_l2arctic
# python run_evaluation.py -c result/downstream/timit_wav2vec2_ft/best-states-dev.ckpt -s dev test  --config result/downstream/l2arctic_hubert/config_2023-01-18-17-11-52.yaml --phone-db-map TIMIT ARCTIC --output-dir result/downstream/timit_wav2vec2_ft_l2arctic
