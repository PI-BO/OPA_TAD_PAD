import os, sys, getopt
sys.path.append(os.path.abspath("./"))

import pickle
import matplotlib.pyplot as plt
import numpy as np

def main(argv):

    fontsize = 18
    inputfile = None
    tile ='Peak usage example'
    
    # Read console parameters
    try:
        opts, args = getopt.getopt(argv,"hi:t:")
    except getopt.GetoptError:
        print (__file__,' -i <input file> -t <title>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print (__file__,'  -i <input file> -t <title>')
            sys.exit()
        elif opt == '-i':
            inputfile = arg        
        elif opt == '-t':
            tile = arg
    if inputfile is None:
        print (__file__,' -i <input file> -t <title>')
        sys.exit(2)

    ## peak time
    with open(inputfile, 'rb') as f:
        anonymity_levels, losses_best, losses_generic, \
        losses_linear, losses_deep, distances_lm, distances_dm, distances_gt = pickle.load(f)


    plt.plot(anonymity_levels, losses_best,'o-', label='Ground truth metric')
    plt.plot(anonymity_levels, losses_generic, 's-',label='Generic metric')
    plt.errorbar(anonymity_levels, np.mean(losses_linear, axis=1), np.std(losses_linear, axis=1),
                fmt='^--', capthick=2,label='Linear metric')
    plt.errorbar(anonymity_levels, np.mean(losses_deep, axis=1), np.std(losses_deep, axis=1),
                fmt='X--',capthick=2,label='Nonlinear metric')
    plt.xlabel('Anonymity level', fontsize=fontsize)
    plt.ylabel('Information loss', fontsize=fontsize)
    plt.title(tile, fontsize=fontsize)
    plt.legend()
    plt.show()


    # peak time (presanitized)
    with open(inputfile[0:-7] + '_presanitized.pickle', 'rb') as f:
        anonymity_levels, losses_best, losses_generic, \
        losses_linear, losses_deep, distances_lm, distances_dm, distances_gt = pickle.load(f)

    plt.plot(anonymity_levels, losses_best,'o-', label='Ground truth metric')
    plt.plot(anonymity_levels, losses_generic, 's-',label='Generic metric')
    plt.errorbar(anonymity_levels, np.mean(losses_linear, axis=1), np.std(losses_linear, axis=1),
                fmt='^--', capthick=2,label='Linear metric')
    plt.errorbar(anonymity_levels, np.mean(losses_deep, axis=1), np.std(losses_deep, axis=1),
                fmt='X--',capthick=2,label='Nonlinear metric')
    plt.xlabel('Anonymity level', fontsize=fontsize)
    plt.ylabel('Information loss', fontsize=fontsize)
    plt.title(tile, fontsize=fontsize)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
