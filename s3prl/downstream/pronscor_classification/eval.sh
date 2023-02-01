set -e
python run_downstream.py -m evaluate -e result/downstream/epa_hubert_linear_ns/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/epa_hubert_linear_min/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/epa_hubert_linear_softmin/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/epa_hubert_linear_lpp/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/epa_hubert_linear_npc/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/epa_hubert_linear_cw/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/epa_hubert_linear_ns_ft/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/epa_hubert_linear_min_ft/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/epa_hubert_linear_softmin_ft/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/epa_hubert_linear_lpp_ft/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/epa_hubert_linear_npc_ft/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/epa_hubert_linear_cw_ft/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/l2arctic_hubert_linear_ns/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/l2arctic_hubert_linear_min/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/l2arctic_hubert_linear_softmin/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/l2arctic_hubert_linear_lpp/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/l2arctic_hubert_linear_npc/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/l2arctic_hubert_linear_cw/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/l2arctic_hubert_linear_ns_ft/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/l2arctic_hubert_linear_min_ft/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/l2arctic_hubert_linear_softmin_ft/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/l2arctic_hubert_linear_lpp_ft/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/l2arctic_hubert_linear_npc_ft/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -e result/downstream/l2arctic_hubert_linear_cw_ft/best-loss-dev.ckpt
                