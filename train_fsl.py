import numpy as np
import torch
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)
from model.FUNIT.utils import get_train_loaders, get_config
# from ipdb import launch_ipdb_on_exception

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    # with launch_ipdb_on_exception():
    config = get_config('./picker.yaml')
    assert config['max_iter'] == args.max_epoch * args.episodes_per_epoch
    assert config['way_size'] == args.way
    config['batch_size'] == args.eval_query + args.eval_shot
    pprint(vars(args))

    from model.trainer.fsl_trainer import FSLTrainer

    set_gpu(args.gpu)
    trainer = FSLTrainer(args, config)
    # trainer.train()
    trainer.evaluate_test()
    # trainer.final_record()
    print(args.save_path)



