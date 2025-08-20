from learner import Learner
from config import get_parser
if __name__ == '__main__':

    parser = get_parser()
    parser.add_argument("--valid_freq", help='frequency to evaluate on valid/test set', default=500, type=int)
    parser.add_argument("--log_freq", help='frequency to log on tensorboard', default=10, type=int)
    parser.add_argument("--save_freq", help='frequency to save model checkpoint', default=1000, type=int)
    parser.add_argument("--num_steps", help="# of iterations", default= 5000, type=int)
    args = parser.parse_args()
    if args.method == 'fit':
        args.local = True
    # init learner
    learner = Learner(args)

    # actual training
    print('Training starts ...')
    
    if args.method in ['vanilla','aug_vanilla']:
        learner.train_vanilla(args)
    elif args.method == 'fit':
        learner.fit_other(args)
    else:
        print('choose one of the two options ...')
        import sys
        sys.exit(0)
