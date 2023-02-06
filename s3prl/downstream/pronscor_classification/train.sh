set -e
python run_downstream.py -m train -n l2arctic_hubert_large_ll60k_linear_ns -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                        -d pronscor_classification -o "config.runner.eval_dataloaders=['dev'],,config.runner.train_dataloader=train,,config.optimizer.lr=3e-5" 
                    
# python run_downstream.py -m train -n l2arctic_hubert_linear_ns -u hubert \
#                         -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
#                         -d pronscor_classification -o "config.runner.eval_dataloaders=['dev'],,config.runner.train_dataloader=train" 
                    
python run_downstream.py -m train -n l2arctic_hubert_large_ll60k_linear_lpp -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev'],,config.runner.train_dataloader=train,,config.optimizer.lr=3e-5" 
                    
# python run_downstream.py -m train -n l2arctic_hubert_linear_lpp -u hubert \
#                         -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
#                         -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev'],,config.runner.train_dataloader=train" 
                    
python run_downstream.py -m train -n l2arctic_hubert_large_ll60k_linear_cw -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev'],,config.runner.train_dataloader=train,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n l2arctic_hubert_linear_cw -u hubert \
                        -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev'],,config.runner.train_dataloader=train" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_ns_0 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_ns_1 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_ns_2 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_ns_3 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_ns_4 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_ns_5 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5,,config.optimizer.lr=3e-5" 
                    
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
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_lpp_0 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_lpp_1 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_lpp_2 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_lpp_3 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_lpp_4 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_lpp_5 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5,,config.optimizer.lr=3e-5" 
                    
# python run_downstream.py -m train -n epa_hubert_linear_lpp_0 -u hubert \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0" 
                    
# python run_downstream.py -m train -n epa_hubert_linear_lpp_1 -u hubert \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1" 
                    
# python run_downstream.py -m train -n epa_hubert_linear_lpp_2 -u hubert \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2" 
                    
# python run_downstream.py -m train -n epa_hubert_linear_lpp_3 -u hubert \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3" 
                    
# python run_downstream.py -m train -n epa_hubert_linear_lpp_4 -u hubert \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4" 
                    
# python run_downstream.py -m train -n epa_hubert_linear_lpp_5 -u hubert \
#                         -c downstream/pronscor_classification/config_epa_linear.yaml \
#                         -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_cw_0 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_cw_1 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_cw_2 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_cw_3 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_cw_4 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_large_ll60k_linear_cw_5 -u hubert_large_ll60k \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5,,config.optimizer.lr=3e-5" 
                    
python run_downstream.py -m train -n epa_hubert_linear_cw_0 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev0'],,config.runner.train_dataloader=train0" 
                    
python run_downstream.py -m train -n epa_hubert_linear_cw_1 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev1'],,config.runner.train_dataloader=train1" 
                    
python run_downstream.py -m train -n epa_hubert_linear_cw_2 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev2'],,config.runner.train_dataloader=train2" 
                    
python run_downstream.py -m train -n epa_hubert_linear_cw_3 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev3'],,config.runner.train_dataloader=train3" 
                    
python run_downstream.py -m train -n epa_hubert_linear_cw_4 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev4'],,config.runner.train_dataloader=train4" 
                    
python run_downstream.py -m train -n epa_hubert_linear_cw_5 -u hubert \
                        -c downstream/pronscor_classification/config_epa_linear.yaml \
                        -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.eval_dataloaders=['dev5'],,config.runner.train_dataloader=train5" 
                    