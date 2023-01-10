import random
import tempfile

import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from s3prl.downstream.runner import Runner
import torch
import numpy as np

ckpt_path = '/home/priera/s3prl/s3prl/result/downstream/test3/states-150000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
ckpt['Args'].device = 'cpu'
ckpt['Args'].init_ckpt = ckpt_path

runner = Runner(ckpt['Args'], ckpt['Config'])


def evaluate(self, split=None, logger=None, global_step=0):
    """evaluate function will always be called on a single process even during distributed training"""

    # When this member function is called directly by command line
    not_during_training = split is None and logger is None and global_step == 0
    if not_during_training:
        split = self.args.evaluate_split
        tempdir = tempfile.mkdtemp()
        logger = SummaryWriter(tempdir)

    # fix seed to guarantee the same evaluation protocol across steps
    random.seed(self.args.seed)
    np.random.seed(self.args.seed)
    torch.manual_seed(self.args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(self.args.seed)
        with torch.cuda.device(self.args.device):
            torch.cuda.empty_cache()

    # record original train/eval states and set all models to eval
    trainings = []
    for entry in self.all_entries:
        trainings.append(entry.model.training)
        entry.model.eval()

    # prepare data
    dataloader = self.downstream.model.get_dataloader(split)
    evaluate_ratio = float(self.config["runner"].get("evaluate_ratio", 1))
    evaluate_steps = round(len(dataloader) * evaluate_ratio)

    batch_ids = []
    records = defaultdict(list)
    for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=split, total=evaluate_steps)):
        if batch_id > evaluate_steps:
            break

        wavs = [torch.FloatTensor(wav).to(self.args.device)
                for wav in wavs]
        with torch.no_grad():
            features = self.upstream.model(wavs)
            features = self.featurizer.model(wavs, features)
            self.downstream.model(
                split,
                features, *others,
                records=records,
                batch_id=batch_id,
            )
            batch_ids.append(batch_id)
