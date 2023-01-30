set -e
python run_downstream.py -m train -n epa_hubert_linear_ns -u hubert \
                    -c downstream/pronscor_classification/config_epa_linear.yaml \
                    -d pronscor_classification  
                
python run_downstream.py -m train -n epa_hubert_linear_min -u hubert \
                    -c downstream/pronscor_classification/config_epa_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.summarise='min'" 
                
python run_downstream.py -m train -n epa_hubert_linear_softmin -u hubert \
                    -c downstream/pronscor_classification/config_epa_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.summarise='softmin'" 
                
python run_downstream.py -m train -n epa_hubert_linear_lpp -u hubert \
                    -c downstream/pronscor_classification/config_epa_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp'" 
                
python run_downstream.py -m train -n epa_hubert_linear_npc -u hubert \
                    -c downstream/pronscor_classification/config_epa_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.npc=True" 
                
python run_downstream.py -m train -n epa_hubert_linear_cw -u hubert \
                    -c downstream/pronscor_classification/config_epa_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True" 
                
python run_downstream.py -m train -n epa_hubert_linear_ns_ft -u hubert \
                    -c downstream/pronscor_classification/config_epa_linear.yaml \
                    -d pronscor_classification -o "config.runner.total_steps=6000,,config.optimizer.lr=5e-6,,config.runner.gradient_accumulate_steps=4,,config.downstream_expert.datarc.train_batch_size=32" -f -s last_hidden_state
                
python run_downstream.py -m train -n epa_hubert_linear_min_ft -u hubert \
                    -c downstream/pronscor_classification/config_epa_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.summarise='min',,config.runner.total_steps=6000,,config.optimizer.lr=5e-6,,config.runner.gradient_accumulate_steps=4,,config.downstream_expert.datarc.train_batch_size=32" -f -s last_hidden_state
                
python run_downstream.py -m train -n epa_hubert_linear_softmin_ft -u hubert \
                    -c downstream/pronscor_classification/config_epa_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.summarise='softmin',,config.runner.total_steps=6000,,config.optimizer.lr=5e-6,,config.runner.gradient_accumulate_steps=4,,config.downstream_expert.datarc.train_batch_size=32" -f -s last_hidden_state
                
python run_downstream.py -m train -n epa_hubert_linear_lpp_ft -u hubert \
                    -c downstream/pronscor_classification/config_epa_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.total_steps=6000,,config.optimizer.lr=5e-6,,config.runner.gradient_accumulate_steps=4,,config.downstream_expert.datarc.train_batch_size=32" -f -s last_hidden_state
                
python run_downstream.py -m train -n epa_hubert_linear_npc_ft -u hubert \
                    -c downstream/pronscor_classification/config_epa_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.npc=True,,config.runner.total_steps=6000,,config.optimizer.lr=5e-6,,config.runner.gradient_accumulate_steps=4,,config.downstream_expert.datarc.train_batch_size=32" -f -s last_hidden_state
                
python run_downstream.py -m train -n epa_hubert_linear_cw_ft -u hubert \
                    -c downstream/pronscor_classification/config_epa_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.total_steps=6000,,config.optimizer.lr=5e-6,,config.runner.gradient_accumulate_steps=4,,config.downstream_expert.datarc.train_batch_size=32" -f -s last_hidden_state
                
python run_downstream.py -m train -n l2arctic_hubert_linear_ns -u hubert \
                    -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                    -d pronscor_classification  
                
python run_downstream.py -m train -n l2arctic_hubert_linear_min -u hubert \
                    -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.summarise='min'" 
                
python run_downstream.py -m train -n l2arctic_hubert_linear_softmin -u hubert \
                    -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.summarise='softmin'" 
                
python run_downstream.py -m train -n l2arctic_hubert_linear_lpp -u hubert \
                    -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp'" 
                
python run_downstream.py -m train -n l2arctic_hubert_linear_npc -u hubert \
                    -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.npc=True" 
                
python run_downstream.py -m train -n l2arctic_hubert_linear_cw -u hubert \
                    -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True" 
                
python run_downstream.py -m train -n l2arctic_hubert_linear_ns_ft -u hubert \
                    -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                    -d pronscor_classification -o "config.runner.total_steps=6000,,config.optimizer.lr=5e-6,,config.runner.gradient_accumulate_steps=4,,config.downstream_expert.datarc.train_batch_size=32" -f -s last_hidden_state
                
python run_downstream.py -m train -n l2arctic_hubert_linear_min_ft -u hubert \
                    -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.summarise='min',,config.runner.total_steps=6000,,config.optimizer.lr=5e-6,,config.runner.gradient_accumulate_steps=4,,config.downstream_expert.datarc.train_batch_size=32" -f -s last_hidden_state
                
python run_downstream.py -m train -n l2arctic_hubert_linear_softmin_ft -u hubert \
                    -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.summarise='softmin',,config.runner.total_steps=6000,,config.optimizer.lr=5e-6,,config.runner.gradient_accumulate_steps=4,,config.downstream_expert.datarc.train_batch_size=32" -f -s last_hidden_state
                
python run_downstream.py -m train -n l2arctic_hubert_linear_lpp_ft -u hubert \
                    -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.summarise='lpp',,config.runner.total_steps=6000,,config.optimizer.lr=5e-6,,config.runner.gradient_accumulate_steps=4,,config.downstream_expert.datarc.train_batch_size=32" -f -s last_hidden_state
                
python run_downstream.py -m train -n l2arctic_hubert_linear_npc_ft -u hubert \
                    -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.npc=True,,config.runner.total_steps=6000,,config.optimizer.lr=5e-6,,config.runner.gradient_accumulate_steps=4,,config.downstream_expert.datarc.train_batch_size=32" -f -s last_hidden_state
                
python run_downstream.py -m train -n l2arctic_hubert_linear_cw_ft -u hubert \
                    -c downstream/pronscor_classification/config_l2arctic_linear.yaml \
                    -d pronscor_classification -o "config.downstream_expert.datarc.class_weight=True,,config.runner.total_steps=6000,,config.optimizer.lr=5e-6,,config.runner.gradient_accumulate_steps=4,,config.downstream_expert.datarc.train_batch_size=32" -f -s last_hidden_state
                