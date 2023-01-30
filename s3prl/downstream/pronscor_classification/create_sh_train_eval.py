from itertools import product
confs = [('ns', ""),
         ('min', "config.downstream_expert.datarc.summarise='min'"),
         ('softmin', "config.downstream_expert.datarc.summarise='softmin'"),
         ('lpp', "config.downstream_expert.datarc.summarise='lpp'"),
         ('npc', "config.downstream_expert.datarc.npc=True"),
         ('cw', "config.downstream_expert.datarc.class_weight=True")
         ]

fts = [('', ""),
       ('_ft', "config.runner.total_steps=6000,,config.optimizer.lr=5e-6,,config.runner.gradient_accumulate_steps=4,,config.downstream_expert.datarc.train_batch_size=32"),
       ]


dbs = ['epa', 'l2arctic']

for mode in ['train', 'eval']:
    cmds = []
    for db, ft, conf in product(dbs, fts, confs):
        conf_name, conf = conf
        ft_name, ft_conf = ft
        ft_flag = ""
        oconfs = []
        if conf != "":
            oconfs.append(conf)
        if ft_conf != "":
            oconfs.append(ft_conf)
            ft_flag = "-f -s last_hidden_state"

        if len(oconfs) > 0:
            oconfs = ',,'.join(oconfs)
            oconf = '-o ' + f'"{oconfs}"'
        else:
            oconf = ""

        runname = f'{db}_hubert_linear_{conf_name}{ft_name}'
        if mode == 'train':
            cmd = f"""python run_downstream.py -m train -n {runname} -u hubert \\
                    -c downstream/pronscor_classification/config_{db}_linear.yaml \\
                    -d pronscor_classification {oconf} {ft_flag}
                """
        elif mode == 'eval':
            cmd = f"""python run_downstream.py -m evaluate -e result/downstream/{runname}/best-loss-dev.ckpt
                """

        cmds.append(cmd)

    if mode == 'train':
        fname = 'train.sh'
    elif mode == 'eval':
        fname = 'eval.sh'

    with open(fname, 'w') as fp:
        fp.write('set -e\n')
        fp.write('\n'.join(cmds))
