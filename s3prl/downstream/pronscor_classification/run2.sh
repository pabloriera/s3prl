# python run_downstream.py -m train -n timit2A_hubert -u hubert  -d timit_phone_to_arctic
# python run_downstream.py -m train -n timit2A_wav2vec2_base_960 -u wav2vec2_base_960  -d timit_phone_to_arctic
# python run_downstream.py -m train -n timit2A_xls_r_300m -u xls_r_300m  -d timit_phone_to_arctic
# python run_downstream.py -m train -n timit2A_wavlm -u wavlm  -d timit_phone_to_arctic
# python run_downstream.py -m train -n timit2A_distilhubert_base -u distilhubert_base  -d timit_phone_to_arctic
# python run_downstream.py -m train -n timit2A_unispeech_sat_base_plus -u unispeech_sat_base_plus  -d timit_phone_to_arctic

# python run_downstream.py -m train -n timit2A_hubert_epa -u hubert -c downstream/pronscor_classification/config_epa.yaml  -d pronscor_classification  -w result/downstream/timit2A_hubert/best-states-dev.ckpt
# python run_downstream.py -m train -n timit2A_wav2vec2_base_960_epa -u wav2vec2_base_960 -c downstream/pronscor_classification/config_epa.yaml  -d pronscor_classification -w result/downstream/timit2A_wav2vec2_base_960/best-states-dev.ckpt
# python run_downstream.py -m train -n timit2A_xls_r_300m_epa -u xls_r_300m -c downstream/pronscor_classification/config_epa.yaml  -d pronscor_classification -w result/downstream/timit2A_xls_r_300m/best-states-dev.ckpt
# python run_downstream.py -m train -n timit2A_wavlm_epa -u wavlm -c downstream/pronscor_classification/config_epa.yaml -d pronscor_classification -w result/downstream/timit2A_wavlm/best-states-dev.ckpt
# python run_downstream.py -m train -n timit2A_distilhubert_base_epa -u distilhubert_base -c downstream/pronscor_classification/config_epa.yaml -d pronscor_classification -w result/downstream/timit2A_distilhubert_base/best-states-dev.ckpt
# python run_downstream.py -m train -n timit2A_unispeech_sat_base_plus_epa -u unispeech_sat_base_plus -c downstream/pronscor_classification/config_epa.yaml -d pronscor_classification -w result/downstream/timit2A_unispeech_sat_base_plus/best-states-dev.ckpt

# python run_downstream.py -m train -n timit2A_hubert_l2arctic -u hubert -c downstream/pronscor_classification/config_l2arctic.yaml  -d pronscor_classification  -w result/downstream/timit2A_hubert/best-states-dev.ckpt
# python run_downstream.py -m train -n timit2A_wav2vec2_base_960_l2arctic -u wav2vec2_base_960 -c downstream/pronscor_classification/config_l2arctic.yaml  -d pronscor_classification -w result/downstream/timit2A_wav2vec2_base_960/best-states-dev.ckpt
# python run_downstream.py -m train -n timit2A_xls_r_300m_l2arctic -u xls_r_300m -c downstream/pronscor_classification/config_l2arctic.yaml  -d pronscor_classification -w result/downstream/timit2A_xls_r_300m/best-states-dev.ckpt
# python run_downstream.py -m train -n timit2A_wavlm_l2arctic -u wavlm -c downstream/pronscor_classification/config_l2arctic.yaml -d pronscor_classification -w result/downstream/timit2A_wavlm/best-states-dev.ckpt
# python run_downstream.py -m train -n timit2A_distilhubert_base_l2arctic -u distilhubert_base -c downstream/pronscor_classification/config_l2arctic.yaml -d pronscor_classification -w result/downstream/timit2A_distilhubert_base/best-states-dev.ckpt
# python run_downstream.py -m train -n timit2A_unispeech_sat_base_plus_l2arctic -u unispeech_sat_base_plus -c downstream/pronscor_classification/config_l2arctic.yaml -d pronscor_classification -w result/downstream/timit2A_unispeech_sat_base_plus/best-states-dev.ckpt

FT="config.runner.total_steps=2000,,config.optimizer.lr=5e-6"

# python run_downstream.py -f -m train -n timit2A_hubert_epa_ft -u hubert -c downstream/pronscor_classification/config_epa.yaml  -d pronscor_classification  -w result_ene_24_timit2A/downstream/timit2A_hubert/best-states-dev.ckpt -o $FT
# python run_downstream.py -f -m train -n timit2A_wav2vec2_base_960_epa_ft -u wav2vec2_base_960 -c downstream/pronscor_classification/config_epa.yaml  -d pronscor_classification -w result_ene_24_timit2A/downstream/timit2A_wav2vec2_base_960/best-states-dev.ckpt -o $FT
# python run_downstream.py -f -m train -n timit2A_xls_r_300m_epa_ft -u xls_r_300m -c downstream/pronscor_classification/config_epa.yaml  -d pronscor_classification -w result_ene_24_timit2A/downstream/timit2A_xls_r_300m/best-states-dev.ckpt -o $FT
# python run_downstream.py -f -m train -n timit2A_wavlm_epa_ft -u wavlm -c downstream/pronscor_classification/config_epa.yaml -d pronscor_classification -w result_ene_24_timit2A/downstream/timit2A_wavlm/best-states-dev.ckpt -o $FT
# python run_downstream.py -f -m train -n timit2A_distilhubert_base_epa_ft -u distilhubert_base -c downstream/pronscor_classification/config_epa.yaml -d pronscor_classification -w result_ene_24_timit2A/downstream/timit2A_distilhubert_base/best-states-dev.ckpt -o $FT
# python run_downstream.py -f -m train -n timit2A_unispeech_sat_base_plus_epa_ft -u unispeech_sat_base_plus -c downstream/pronscor_classification/config_epa.yaml -d pronscor_classification -w result_ene_24_timit2A/downstream/timit2A_unispeech_sat_base_plus/best-states-dev.ckpt -o $FT

python run_downstream.py -f -m train -n timit2A_hubert_l2arctic_ft -u hubert -c downstream/pronscor_classification/config_l2arctic.yaml  -d pronscor_classification  -w result_ene_24_timit2A/downstream/timit2A_hubert/best-states-dev.ckpt -o $FT
python run_downstream.py -f -m train -n timit2A_wav2vec2_base_960_l2arctic_ft -u wav2vec2_base_960 -c downstream/pronscor_classification/config_l2arctic.yaml  -d pronscor_classification -w result_ene_24_timit2A/downstream/timit2A_wav2vec2_base_960/best-states-dev.ckpt -o $FT
python run_downstream.py -f -m train -n timit2A_xls_r_300m_l2arctic_ft -u xls_r_300m -c downstream/pronscor_classification/config_l2arctic.yaml  -d pronscor_classification -w result_ene_24_timit2A/downstream/timit2A_xls_r_300m/best-states-dev.ckpt -o $FT
python run_downstream.py -f -m train -n timit2A_wavlm_l2arctic_ft -u wavlm -c downstream/pronscor_classification/config_l2arctic.yaml -d pronscor_classification -w result_ene_24_timit2A/downstream/timit2A_wavlm/best-states-dev.ckpt -o $FT
# python run_downstream.py -f -m train -n timit2A_distilhubert_base_l2arctic_ft -u distilhubert_base -c downstream/pronscor_classification/config_l2arctic.yaml -d pronscor_classification -w result_ene_24_timit2A/downstream/timit2A_distilhubert_base/best-states-dev.ckpt -o $FT
python run_downstream.py -f -m train -n timit2A_unispeech_sat_base_plus_l2arctic_ft -u unispeech_sat_base_plus -c downstream/pronscor_classification/config_l2arctic.yaml -d pronscor_classification -w result_ene_24_timit2A/downstream/timit2A_unispeech_sat_base_plus/best-states-dev.ckpt -o $FT
