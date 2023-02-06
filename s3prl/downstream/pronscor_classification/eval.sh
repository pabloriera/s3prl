set -e
python run_downstream.py -m evaluate -t test -e result/downstream/l2arctic_hubert_large_linear_ns/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -t test -e result/downstream/l2arctic_hubert_linear_ns/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -t test -e result/downstream/l2arctic_hubert_large_linear_lpp/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -t test -e result/downstream/l2arctic_hubert_linear_lpp/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -t test -e result/downstream/l2arctic_hubert_large_linear_cw/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -t test -e result/downstream/l2arctic_hubert_linear_cw/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -t test -e result/downstream/epa_hubert_large_linear_ns/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -t test -e result/downstream/epa_hubert_linear_ns/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -t test -e result/downstream/epa_hubert_large_linear_lpp/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -t test -e result/downstream/epa_hubert_linear_lpp/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -t test -e result/downstream/epa_hubert_large_linear_cw/best-loss-dev.ckpt
                
python run_downstream.py -m evaluate -t test -e result/downstream/epa_hubert_linear_cw/best-loss-dev.ckpt
                