from pathlib import Path
import fire


def main(path, relpath, device='cpu'):
    cmds = []
    for p in sorted(list(Path(path).rglob('best-loss-dev*.ckpt'))):
        if not Path(p.parent, 'test-best_loss.records').exists():
            cmd = f'python run_downstream.py -m evaluate -t test -e {p.relative_to(relpath)} --device {device}'
            cmds.append(cmd)

    with open('eval.sh', 'w') as fp:
        fp.write('set -e\n')
        fp.write('\n'.join(cmds))


if __name__ == '__main__':
    fire.Fire(main)
