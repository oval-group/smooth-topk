import os
import subprocess
import pprint

dataset = 'imagenet'
xp_dir = '../xp/{}'.format(dataset)

# multiple crops option
if dataset == 'imagenet':
    crops_opt = '--multiple-crops'
elif dataset == 'cifar100':
    crops_opt = ''
else:
    raise ValueError

for _, _, files in os.walk(xp_dir):

    to_analyze = sorted(filter(lambda x: 'best' in x and x.endswith('.pkl'), files))
    n_analyze = len(to_analyze)
    print("Found {} files to evaluate:".format(n_analyze))
    pp = pprint.PrettyPrinter(indent=4)
    msg = pp.pformat(to_analyze)
    print(msg)

    for idx, xp_file in enumerate(to_analyze):

        print('-' * 80)
        print('Evaluating {} ({} out of {})'.format(xp_file, idx + 1, n_analyze))

        # find loss used for training
        if 'svm' in xp_file:
            loss = 'svm'
        elif 'ce' in xp_file:
            loss = 'ce'
        else:
            raise ValueError('Could not parse loss name from filename')

        filename = os.path.join(xp_dir, xp_file)
        cmd = "python main.py --loss {} --load-model {} --dataset {} --eval --parallel-gpu {}"\
            .format(loss, filename, dataset, crops_opt)
        cmd = cmd.split()
        subprocess.call(cmd)
