set -e

# python run_downstream.py -m train -n l2arctic_wav2vec2_linear_ns -u wav2vec2 \
#                         -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev'],,config.runner.train_dataloader=train" 
                    
# python run_downstream.py -m train -n l2arctic_data2vec_large_ll60k_linear_ns -u data2vec_large_ll60k \
#                         -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev'],,config.runner.train_dataloader=train,,config.optimizer.lr=1e-4" 
                    
# python run_downstream.py -m train -n l2arctic_wav2vec2_linear_lpp -u wav2vec2 \
#                         -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
#                         -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev'],,config.runner.train_dataloader=train" 
                    
# python run_downstream.py -m train -n l2arctic_data2vec_large_ll60k_linear_lpp -u data2vec_large_ll60k \
#                         -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
#                         -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev'],,config.runner.train_dataloader=train,,config.optimizer.lr=1e-4" 
                                        
# python run_downstream.py -m train -n l2arctic_wav2vec2_linear_lpp_cw -u wav2vec2 \
#                         -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
#                         -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev'],,config.runner.train_dataloader=train" 
                    
# python run_downstream.py -m train -n l2arctic_data2vec_large_ll60k_linear_lpp_cw -u data2vec_large_ll60k \
#                         -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
#                         -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev'],,config.runner.train_dataloader=train,,config.optimizer.lr=1e-4" 
                                       
# python run_downstream.py -m train -n l2arctic_wav2vec2_linear_ns_cw -u wav2vec2 \
#                         -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
#                         -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev'],,config.runner.train_dataloader=train" 
                    
# python run_downstream.py -m train -n l2arctic_data2vec_large_ll60k_linear_ns_cw -u data2vec_large_ll60k \
#                         -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
#                         -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev'],,config.runner.train_dataloader=train,,config.optimizer.lr=1e-4" 
                    
# python run_downstream.py -m train -n epa_hubert_linear_ns_0 -u hubert \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0" 
                    
# python run_downstream.py -m train -n epa_hubert_linear_ns_1 -u hubert \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1" 
                    
# python run_downstream.py -m train -n epa_hubert_linear_ns_2 -u hubert \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2" 
                    
# python run_downstream.py -m train -n epa_hubert_linear_ns_3 -u hubert \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3" 
                    
# python run_downstream.py -m train -n epa_hubert_linear_ns_4 -u hubert \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4" 
                    
# python run_downstream.py -m train -n epa_hubert_linear_ns_5 -u hubert \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5" 
                    
# python run_downstream.py -m train -n epa_wavlm_linear_ns_0 -u wavlm \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0" 
                    
# python run_downstream.py -m train -n epa_wavlm_linear_ns_1 -u wavlm \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1" 
                    
# python run_downstream.py -m train -n epa_wavlm_linear_ns_2 -u wavlm \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2" 
                    
# python run_downstream.py -m train -n epa_wavlm_linear_ns_3 -u wavlm \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3" 
                    
# python run_downstream.py -m train -n epa_wavlm_linear_ns_4 -u wavlm \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4" 
                    
# python run_downstream.py -m train -n epa_wavlm_linear_ns_5 -u wavlm \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5" 
                    
# python run_downstream.py -m train -n epa_wav2vec2_linear_ns_0 -u wav2vec2 \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0" 
                    
# python run_downstream.py -m train -n epa_wav2vec2_linear_ns_1 -u wav2vec2 \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1" 
                    
# python run_downstream.py -m train -n epa_wav2vec2_linear_ns_2 -u wav2vec2 \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2" 
                    
# python run_downstream.py -m train -n epa_wav2vec2_linear_ns_3 -u wav2vec2 \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3" 
                    
# python run_downstream.py -m train -n epa_wav2vec2_linear_ns_4 -u wav2vec2 \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4" 
                    
# python run_downstream.py -m train -n epa_wav2vec2_linear_ns_5 -u wav2vec2 \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_ns_0 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_ns_1 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_ns_2 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_ns_3 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_ns_4 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_ns_5 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_hubert_linear_lpp_0 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0" 
                    
python run_downstream.py -m train -n epa_hubert_linear_lpp_1 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1" 
                    
python run_downstream.py -m train -n epa_hubert_linear_lpp_2 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2" 
                    
python run_downstream.py -m train -n epa_hubert_linear_lpp_3 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3" 
                    
python run_downstream.py -m train -n epa_hubert_linear_lpp_4 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4" 
                    
python run_downstream.py -m train -n epa_hubert_linear_lpp_5 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_lpp_0 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_lpp_1 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_lpp_2 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_lpp_3 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_lpp_4 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_lpp_5 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_lpp_0 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_lpp_1 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_lpp_2 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_lpp_3 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_lpp_4 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_lpp_5 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_lpp_0 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_lpp_1 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_lpp_2 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_lpp_3 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_lpp_4 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_lpp_5 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_hubert_linear_lpp_cw_0 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0" 
                    
python run_downstream.py -m train -n epa_hubert_linear_lpp_cw_1 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1" 
                    
python run_downstream.py -m train -n epa_hubert_linear_lpp_cw_2 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2" 
                    
python run_downstream.py -m train -n epa_hubert_linear_lpp_cw_3 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3" 
                    
python run_downstream.py -m train -n epa_hubert_linear_lpp_cw_4 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4" 
                    
python run_downstream.py -m train -n epa_hubert_linear_lpp_cw_5 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_lpp_cw_0 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_lpp_cw_1 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_lpp_cw_2 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_lpp_cw_3 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_lpp_cw_4 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_lpp_cw_5 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_lpp_cw_0 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_lpp_cw_1 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_lpp_cw_2 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_lpp_cw_3 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_lpp_cw_4 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_lpp_cw_5 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_lpp_cw_0 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_lpp_cw_1 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_lpp_cw_2 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_lpp_cw_3 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_lpp_cw_4 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_lpp_cw_5 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_hubert_linear_ns_cw_0 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0" 
                    
python run_downstream.py -m train -n epa_hubert_linear_ns_cw_1 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1" 
                    
python run_downstream.py -m train -n epa_hubert_linear_ns_cw_2 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2" 
                    
python run_downstream.py -m train -n epa_hubert_linear_ns_cw_3 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3" 
                    
python run_downstream.py -m train -n epa_hubert_linear_ns_cw_4 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4" 
                    
python run_downstream.py -m train -n epa_hubert_linear_ns_cw_5 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_ns_cw_0 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_ns_cw_1 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_ns_cw_2 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_ns_cw_3 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_ns_cw_4 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4" 
                    
python run_downstream.py -m train -n epa_wavlm_linear_ns_cw_5 -u wavlm \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_ns_cw_0 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_ns_cw_1 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_ns_cw_2 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_ns_cw_3 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_ns_cw_4 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4" 
                    
python run_downstream.py -m train -n epa_wav2vec2_linear_ns_cw_5 -u wav2vec2 \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_ns_cw_0 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_ns_cw_1 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_ns_cw_2 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_ns_cw_3 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_ns_cw_4 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4,,config.optimizer.lr=1e-4" 
                    
python run_downstream.py -m train -n epa_data2vec_large_ll60k_linear_ns_cw_5 -u data2vec_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5,,config.optimizer.lr=1e-4" 
                    