from learner import Learner
from config import get_parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.local = True

    # init learner
    learner = Learner(args)

    # actual training
    print('Test starts ...')

    #learner.test_ours(args)
    learner.test_vanilla(args)
