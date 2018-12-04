import os, sys, getopt
sys.path.append('./')


from utilities.helper import Utilities, PerformanceEvaluation
from metric_learning.metric_learning import MetricLearning
from utilities.user_feedback import Similarity
from utilities.subsampling import Subsampling
from scipy.misc import comb

import pandas as pd
import numpy as np
import pickle
import requests
import matplotlib.pyplot as plt
import datetime as dt
import logging
import warnings

# ignore warnings
warnings.filterwarnings("ignore")

# speed_class methode: returns the classId depending on the number of classes
def speed_class(numberClasses, speed):
    # define the classes boundaries depending on the number of classes. numberClasse:[startBoundarie1,endBoundarie1/startBoundarie2,endBoundarie2...]
    classBoundarie = {2:[0,0,300],8:[0,0,40,60,90,120,140,300]}

    for index, boundarie in enumerate(classBoundarie[numberClasses]):
        if index + 1 < len(classBoundarie[numberClasses]):
            if speed == 0:
                return 0
            elif speed > classBoundarie[numberClasses][index] and speed <= classBoundarie[numberClasses][index+1]:
                return index
        else:
            return -1

# Main methode
def main(argv):

    # set default values
    logDir = './log'
    resultDir = './result'
    inputfile = ''  

    # number of classes
    numberClasses = 2

    k_init = 10
    batch_size = 20
    batch_size_unif = 20
    mc_num = 5

    rep_mode = 'mean'
    # desired anonymity level
    anonymity_level = 2
    lam_vec = [1e-3,1e-2,1e-1,1,10]

    # data user specifies his/her interest. In the example, the data user is interested in preserving the
    # information of a segment of entire time series. In this case, he/she would also need to specify the starting and
    # ending time of the time series segment of interest.

    # capturing the user's interest
    interest = 'segment'
    # window specifies the starting and ending time of the period that the data user is interested in
    window = [11,15]

    # create logger with script name
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the console handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    # add the console handler to the logger
    logger.addHandler(ch)
    # create logger dir if it not exists
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            logger.error("Cannot create directory %s"% logDir)
    # create file handler which logs even debug messages
    try:
        fh = logging.FileHandler(logDir + "/" + os.path.basename(__file__[0:-3]) + "_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
    except:
        logger.error("Cannot create file: %s/%s_%s.log"% (logDir, os.path.basename(__file__[0:-3]), dt.datetime.now().strftime("%Y%m%d_%H%M%S")))
    else:
        fh.setLevel(logging.DEBUG)
        # add formatter it to the file handler
        fh.setFormatter(formatter)
        # add the file handler to the logger
        logger.addHandler(fh)

    # create result dir if it not exists
    if not os.path.exists(resultDir):
        try:
            os.makedirs(resultDir)
        except OSError:
            logger.error("Cannot create directory: %s"% resultDir)

    # Initialization of some useful classes
    util = Utilities()
    pe = PerformanceEvaluation()
    mel = MetricLearning()

    # Read console parameters
    try:
        opts, args = getopt.getopt(argv,"hi:a:m:c:")
    except getopt.GetoptError:
        print (__file__,' -i <input file> -a <anonymity level> -m <mc num> -c <number of classes>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print (__file__,' -i <input file> -a <anonymity level> -m <mc num> -c <number of classes>')
            sys.exit()
        elif opt == '-i':
            inputfile = arg
        elif opt == '-a':
            anonymity_level = int(arg)
            if anonymity_level <= 2 and anonymity_level >= 10:
                logger.error("anonymity level must be between 2 and 10")
                sys.exit()
        elif opt == '-m':
            mc_num = int(arg) 
            if mc_num <= 1 and mc_num >= 5:
                logger.error("mc num must be between 1 and 5")
        elif opt == '-c':
            numberClasses = int(arg) 
            if numberClasses <= 2 and numberClasses >= 8:
                logger.error("number of classes must be between 2 and 8")  

    # Log console parameters
    logger.info("input file: %s,  anonymity level: %i, mc num: %i, number of classes: %i"% (inputfile, anonymity_level, mc_num, numberClasses))

    if inputfile[-4:] == '.pkl':
        # get the database to be published
        # ../dataset/dataframe_all_binary.pkl
        day_profile = pd.read_pickle(inputfile)
        day_profile = day_profile.iloc[0::4,0::60]#day_profile.iloc[0::4,0::60]

    elif inputfile[-4:] == '.csv':
        # Read CSV File   
        df = pd.read_csv(inputfile, sep=';', header=None)
        day_profile = pd.DataFrame(columns=df.columns)

        # Create Day Profil with classID's
        for row_index,row in df.iterrows():
            list = [] 
            for index, speed in row.iteritems():  
                list.append(speed_class(numberClasses, speed))
            day_profile.loc[row_index] = list

        day_profile = day_profile.iloc[0::2,0::1] # row, column
    else:
        logger.error("input file not valid with %s"% inputfile[-4:])
        sys.exit()


    logger.debug(day_profile)

    # pre-sanitize the database
    sanitized_profile_baseline = util.sanitize_data(day_profile, distance_metric='euclidean',
                                                    anonymity_level=anonymity_level,rep_mode = rep_mode)
    loss_generic_metric = pe.get_information_loss(data_gt=day_profile,
                                                    data_sanitized=sanitized_profile_baseline.round(), window=window)
    logger.info("information loss with generic metric %s" % loss_generic_metric)
    df_subsampled_from = sanitized_profile_baseline.drop_duplicates().sample(frac=1)
    subsample_size_max = int(comb(len(df_subsampled_from),2))
    logger.info('total number of pairs is %s' % subsample_size_max)

    # # obtain ground truth similarity labels
    sp = Subsampling(data=df_subsampled_from)

    data_pair_all, data_pair_all_index = sp.uniform_sampling(subsample_size=subsample_size_max,seed=0)
    sim = Similarity(data=data_pair_all)
    sim.extract_interested_attribute(interest=interest, window=window)
    similarity_label_all, class_label_all = sim.label_via_silhouette_analysis(range_n_clusters=range(2,8))
    similarity_label_all_series = pd.Series(similarity_label_all)
    similarity_label_all_series.index = data_pair_all_index
    logger.info('similarity balance is %s'% [sum(similarity_label_all),len(similarity_label_all)])

    seed_vec = np.arange(0,mc_num)

    # ##################
    # uniform sampling

    loss_iters_unif = []

    pairdata_all = []
    pairlabel_all = []
    dist_metric = []

    for mc_i in range(mc_num):
        loss_learned_metric_unif_mc = []
        pairdata_each_mc = []
        pairlabel_each_mc = []
        dist_metric_mc = []
        k = k_init
        while k <= subsample_size_max:
            if k == k_init:
                pairdata,pairdata_idx = sp.uniform_sampling(subsample_size=k,seed=seed_vec[mc_i])
                pairdata_label = similarity_label_all_series.loc[pairdata_idx]
            else:
                pairdata, pairdata_idx = sp.uniform_sampling(subsample_size=k, seed=None)
                pairdata_label = similarity_label_all_series.loc[pairdata_idx]

            dist_metric = mel.learn_with_similarity_label_regularization(data=pairdata,
                                                                        label=pairdata_label,
                                                                        lam_vec=lam_vec,
                                                                        train_portion=0.8)
            logger.info("dist_metric")
            logger.info(type(dist_metric))
            logger.debug(dist_metric)
            #dist_metric = mel.learn_with_similarity_label(pairdata, pairdata_label, "diag",lam_vec)
            if dist_metric is None:
                loss_learned_metric_unif = np.nan
            else:
                sanitized_profile_unif = util.sanitize_data(day_profile, distance_metric = 'mahalanobis',
                                                            anonymity_level=anonymity_level, rep_mode=rep_mode, VI=dist_metric)
                loss_learned_metric_unif = pe.get_information_loss(data_gt=day_profile,
                                                                    data_sanitized=sanitized_profile_unif.round(),
                                                                    window=window)
            loss_learned_metric_unif_mc.append(loss_learned_metric_unif)
            pairdata_each_mc.append(pairdata)
            pairlabel_each_mc.append(pairdata_label)
            logger.info("sampled size %s" % k)
            logger.info('k is %s' % k)
            logger.info("information loss with uniform metric %s" % np.mean(loss_learned_metric_unif_mc))
            k += batch_size
        loss_iters_unif.append(loss_learned_metric_unif_mc)
        pairdata_all.append(pairdata_each_mc)
        pairlabel_all.append(pairlabel_each_mc)


    try:
        with open(resultDir + "/" + os.path.basename(__file__[0:-3]) +"_loss_uniform_cv.pickle", "wb") as f:
            pickle.dump([loss_iters_unif,k_init,subsample_size_max,batch_size,loss_generic_metric,
                        pairdata_all,pairlabel_all], f)
    except:
        logger.error("Cannot create file: %s/%s_loss_uniform_cv.pickle"% (resultDir, os.path.basename(__file__[0:-3])))

    ##################
    # active learning

    # actively sample a subset of pre-sanitized database

    pairdata_all_active = []
    pairlabel_all_active = []
    loss_iters_active = []
    for mc_i in range(mc_num):
        pairdata_each_mc_active = []
        pairlabel_each_mc_active = []

        pairdata_active_mc = []
        pairdata_label_active_mc = []
        loss_iters_active_mc = []
        k = k_init
        dist_metric = None
        logger.info(type(dist_metric))
        sp.reset()
        while k <= subsample_size_max:
            pairdata, pairdata_idx = sp.active_sampling(dist_metric=dist_metric,
                                                        k_init=k_init,
                                                        batch_size=1,
                                                        seed=seed_vec[mc_i])
            pairdata_active_mc = pairdata_active_mc + pairdata
            similarity_label = similarity_label_all_series.loc[pairdata_idx].tolist()
            pairdata_label_active_mc = pairdata_label_active_mc + similarity_label
            # _, _, dist_metric = mel.learn_with_similarity_label(pairdata_active_mc, pairdata_label_active_mc, "diag", lam_vec)
            dist_metric = mel.learn_with_similarity_label_regularization(data = pairdata_active_mc,
                                                                        label = pairdata_label_active_mc,
                                                                        lam_vec = lam_vec,
                                                                        train_portion = 0.8)
            sanitized_profile_active = util.sanitize_data(day_profile, distance_metric = 'mahalanobis',
                                                anonymity_level=anonymity_level, rep_mode=rep_mode, VI=dist_metric)

            if (k-k_init) % 1 == 0:
                loss_learned_metric_active = pe.get_information_loss(data_gt=day_profile,
                                                            data_sanitized=sanitized_profile_active.round(),
                                                            window=window)
                loss_iters_active_mc.append(loss_learned_metric_active)
                pairdata_each_mc_active.append(pairdata_active_mc)
                pairlabel_each_mc_active.append(pairdata_label_active_mc)


                logger.info("sampled size %s" % sp.k_already)
                logger.info('k is %s' % k)
                logger.info("information loss with active metric %s" % loss_learned_metric_active)
            k += 1
        loss_iters_active.append(loss_iters_active_mc)
        pairdata_all_active.append(pairdata_each_mc_active)
        pairlabel_all_active.append(pairlabel_each_mc_active)


    try:
        with open(resultDir + "/" + os.path.basename(__file__[0:-3]) + "_loss_active_cv.pickle", "wb") as f:
            pickle.dump([loss_iters_active,k_init,subsample_size_max,batch_size,loss_generic_metric,
                        pairdata_all_active,pairlabel_all_active], f)
    except:
        logger.error("Cannot create file: %s/%s_loss_active_cv.pickle"% (resultDir, os.path.basename(__file__[0:-3])))

    # plot
    try:
        with open(resultDir + "/" + os.path.basename(__file__[0:-3]) + "_loss_active_lam1_diag.pickle", "rb") as f:
            loss_iters_active, k_init, subsample_size_max, batch_size, loss_generic_metric = \
                pickle.load(f)
    except:
        logger.error("Cannot create file: %s/%s_loss_active_lam1_diag.pickle"% (resultDir, os.path.basename(__file__[0:-3])))

    try:
        with open(resultDir + "/" + os.path.basename(__file__[0:-3]) +"_loss_uniform_lam1_diag.pickle", "rb") as f:
            loss_iters_unif, k_init_unif, subsample_size_max_unif, batch_size_unif, loss_generic_metric_unif = \
                pickle.load(f)
    except:
        logger.error("Cannot create file: %s/%s_loss_uniform_lam1_diag.pickle"% (resultDir, os.path.basename(__file__[0:-3])))

    loss_iters_active_format = np.asarray(loss_iters_active)
    loss_active_mean = np.mean(loss_iters_active_format,axis=0)
    loss_iters_unif_format = np.asarray(loss_iters_unif)
    loss_unif_mean = np.mean(loss_iters_unif,axis=0)
    eval_k = np.arange(k_init,subsample_size_max+1,batch_size_unif)

    # check platform
    if sys.platform  == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        # load the cpu time from shell
        cpuTime = os.popen("ps -e | grep " + str(os.getpid()) + " | awk {'print $3'}").read().split()[0]
    else:
        cpuTime = 'n/a'
    
    # Define the label of the plot
    plotSubTitel = os.path.splitext(__file__)[0] + " | cpu time: " + cpuTime + " | input file: " + inputfile
    plotTitel = "data sets: " +  str(len(day_profile)) + " | mc num: " + str(mc_num) + " | anonymity level: " + str(anonymity_level) + " | number of classes: " + str(numberClasses)

    plt.figure()
    plt.errorbar(eval_k,loss_unif_mean,np.std(loss_iters_unif_format,axis=0),label='uniform sampling')
    plt.errorbar(eval_k,loss_active_mean[eval_k-k_init],np.std(loss_iters_active_format,axis=0)[eval_k-k_init],
                label='active sampling')
    # plt.errorbar(np.arange(k_init,subsample_size_max+1),loss_active_mean,np.std(loss_iters_active_format),
    #              label='active sampling')
    # plt.plot(np.arange(k_init,subsample_size_max+1),loss_active_mean, label='active sampling')
    # for i in range(mc_num):
    #     plt.plot(np.arange(k_init,subsample_size_max+1),loss_iters_active_format[i,:], label='active sample',color='red')
    #     plt.plot(eval_k,loss_iters_unif_format[i,:],label='uniform sample', color='blue')
    plt.plot((k_init,subsample_size_max),(loss_generic_metric,loss_generic_metric),'r--',label="generic metric")
    plt.legend()
    plt.xlabel("Number of labeled data pairs")
    plt.ylabel("Information loss")
    plt.title(plotTitel)
    plt.suptitle(plotSubTitel)  # Add a title so we know which it is
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])