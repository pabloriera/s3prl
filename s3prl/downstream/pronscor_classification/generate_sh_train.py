from itertools import product
confs = [('ns', ""),
         ('min', "config.downstream_expert.datarc.summarise='min'"),
         ('softmin', "config.downstream_expert.datarc.summarise='softmin'"),
         ('lpp', "config.downstream_expert.datarc.summarise='lpp'"),
         ('npc', "config.downstream_expert.datarc.npc=True"),
         ('cw', "config.downstream_expert.datarc.class_weight=True")
         ]

confs = [('ns', ""),
         ('lpp', "config.downstream_expert.datarc.summarise='lpp'"),
         ('lpp_cw', "config.downstream_expert.datarc.summarise='lpp',,config.downstream_expert.datarc.class_weight=True"),
         ('ns_cw', "config.downstream_expert.datarc.class_weight=True")
         ]

fts = [("", ""),
       ('_ft', "config.runner.total_steps=6000,,config.optimizer.lr=5e-6,,config.runner.gradient_accumulate_steps=4,,config.downstream_expert.datarc.train_batch_size=32"),
       ]

fts = [("", "")]


dbs = ['l2arctic', 'epa']

upstreams = ['hubert', 'wavlm', 'wav2vec2', 'data2vec_large_ll60k']

# for mode in ['train', 'eval']:
for mode in ['train']:
    cmds = []
    for db, ft, conf, upstream in product(dbs, fts, confs, upstreams):
        conf_name, conf = conf
        ft_name, ft_conf = ft
        ft_flag = ""

        runname = f'{db}_{upstream}_linear_{conf_name}{ft_name}'

        if db == 'epa':
            train_splits = [f'train{i}' for i in range(6)]
            dev_splits = [f'dev{i}' for i in range(6)]
        elif db == 'l2arctic':
            train_splits = ['train']
            dev_splits = ['dev']

        for i, (train_split, dev_split) in enumerate(zip(train_splits, dev_splits)):
            if len(train_splits) > 1:
                runname_ = f"{runname}_{i}"
            else:
                runname_ = runname

            if mode == 'train':

                oconfs = []
                if conf != "":
                    oconfs.append(conf)
                if ft_conf != "":
                    oconfs.append(ft_conf)
                    ft_flag = "-f -s last_hidden_state"

                oconfs.append(
                    f"config.runner.eval_dataloaders=['{dev_split}'],,config.runner.train_dataloader={train_split}")

                if 'large' in upstream and ft_flag == "":
                    oconfs.append("config.optimizer.lr=3e-5")

                if len(oconfs) > 0:
                    oconf = ',,'.join(oconfs)
                    oconf = '-o ' + f'"{oconf}"'
                else:
                    oconf = ""

                cmd = f"""python run_downstream.py -m train -n {runname_} -u {upstream} \\
                        -c downstream/pronscor_classification/config_{db}_linear.yaml \\
                        -d pronscor_classification {oconf} {ft_flag}
                    """

                cmds.append(cmd)

            elif mode == 'eval':
                cmd = f"""python run_downstream.py -m evaluate -t test -e result/downstream/{runname_}/best-loss-dev.ckpt
                """
                cmds.append(cmd)

    if mode == 'train':
        fname = 'train.sh'
    elif mode == 'eval':
        fname = 'eval.sh'

    with open(fname, 'w') as fp:
        fp.write('set -e\n')
        fp.write('\n'.join(cmds))
