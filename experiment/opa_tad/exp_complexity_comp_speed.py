import os, sys, getopt
sys.path.append(os.path.abspath("./"))
from helper import Utilities, PerformanceEvaluation, prepare_data
import pandas as pd
from subsampling import Subsampling
from user_feedback import Similarity
from scipy.misc import comb
from deep_metric_learning import Deep_Metric
from deep_metric_learning_duplicate import Deep_Metric_Duplicate
import numpy as np
import pickle

import datetime as dt
import logging

# Disable SettingWithCopyWarning 
pd.options.mode.chained_assignment = None  # default='warn'

# set platform environ for Mac OS
if sys.platform == "darwin":
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
    inputfile = './dataset/opa_tad/speed.csv'  

    # number of classes
    numberClasses = 8 

    anonymity_level = 4
    mc_num = 5 
  
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
            if anonymity_level <= 2 and anonymity_level >= 20:
                logger.error("anonymity level must be between 2 and 20")
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
        #day_profile_all = pd.read_pickle('dataset/dataframe_all_energy.pkl')
        day_profile_all = pd.read_pickle(inputfile)

    elif inputfile[-4:] == '.csv':
        # Read CSV File   
        df = pd.read_csv(inputfile, sep=';', header=None)
        day_profile_all = pd.DataFrame(columns=df.columns)

        # Create Day Profil with classID's
        for row_index,row in df.iterrows():
            list = [] 
            for index, speed in row.iteritems():  
                #list.append(speed_class(numberClasses, speed))
                list.append(speed)
            day_profile_all.loc[row_index] = list
    else:
        logger.error('input file not valid with %s'% inputfile[-4:])
        sys.exit()    

    logger.info('experiement sample complexity script')
    util = Utilities()
    pe = PerformanceEvaluation()

    # load dataset
    day_profile_all = day_profile_all.fillna(0)
    res = 4

    logger.debug(day_profile_all)

    # define use case
    interest = 'window-usage'
    window = [17, 21]
    rep_mode = 'mean'

    # specify the data set for learning and for sanitization
    n_rows = 50
    pub_size = 80
    day_profile = day_profile_all.iloc[:n_rows,0::res]
    day_profile_learning = day_profile_all.iloc[n_rows:n_rows+pub_size,0::res]

    sanitized_profile_best = util.sanitize_data(day_profile, distance_metric='self-defined',
                                            anonymity_level=anonymity_level, rep_mode=rep_mode,
                                            mode=interest, window=window)
    sanitized_profile_baseline = util.sanitize_data(day_profile, distance_metric='euclidean',
                                                    anonymity_level=anonymity_level, rep_mode=rep_mode)
    loss_best_metric = pe.get_information_loss(data_gt=day_profile, data_sanitized=sanitized_profile_best,
                                            window=window)
    loss_generic_metric = pe.get_information_loss(data_gt=day_profile,
                                            data_sanitized=sanitized_profile_baseline.round(),
                                            window=window)

    df_subsampled_from = day_profile_learning.drop_duplicates()
    subsample_size_max = int(comb(len(df_subsampled_from), 2))
    logger.info('total number of pairs is %s' % subsample_size_max)

    sp = Subsampling(data=df_subsampled_from)
    data_pair_all, data_pair_all_index = sp.uniform_sampling(subsample_size=subsample_size_max, seed=0)

    sim = Similarity(data=data_pair_all)
    sim.extract_interested_attribute(interest='statistics', stat_type=interest, window=window)
    similarity_label_all, class_label_all = sim.label_via_silhouette_analysis(range_n_clusters=range(2, 8))
    similarity_label_all_series = pd.Series(similarity_label_all)
    similarity_label_all_series.index = data_pair_all_index
    logger.info('similarity balance is %s'% [sum(similarity_label_all),len(similarity_label_all)])

    sample_size_vec = np.concatenate(([0.01,0.05],np.arange(0.1,1.1,0.1))) #np.array([1e-3,5e-3,1e-2,5e-2,1e-1,2e-2,3e-1])#

    seed_vec = np.arange(mc_num)
    loss_unif_mc_linear = np.zeros((mc_num,len(sample_size_vec)))
    for mc_i in range(len(seed_vec)):
        for j in range(len(sample_size_vec)):
                sample_size = sample_size_vec[j]*subsample_size_max
                pairdata, pairdata_idx = sp.uniform_sampling(subsample_size=int(sample_size), seed=seed_vec[mc_i])
                pairdata_label = similarity_label_all_series.loc[pairdata_idx]

                x1_train, x2_train, y_train, x1_test, x2_test, y_test = prepare_data(pairdata, pairdata_label, 1)
                # dm = Deep_Metric(mode='relu')
                # dm.train(x1_train, x2_train, y_train, x1_test, x2_test, y_test)
                lm = Deep_Metric_Duplicate(mode='linear')
                lm.train(x1_train, x2_train, y_train, x1_test, x2_test, y_test)

                sanitized_profile_linear = util.sanitize_data(day_profile, distance_metric="deep",
                                                        anonymity_level=anonymity_level,
                                                        rep_mode=rep_mode, deep_model=lm)
                #
                # sanitized_profile_deep = util.sanitize_data(day_profile, distance_metric="deep",
                #                                             anonymity_level=anonymity_level,
                #                                             rep_mode=rep_mode, deep_model=dm)

                loss_learned_metric_linear = pe.get_information_loss(data_gt=day_profile,
                                                                data_sanitized=sanitized_profile_linear.round(),
                                                                window=window)
                #
                # loss_learned_metric_deep = pe.get_information_loss(data_gt=day_profile,
                #                                                    data_sanitized=sanitized_profile_deep.round(),
                #                                                    window=window)

                loss_unif_mc_linear[mc_i,j] = loss_learned_metric_linear

                logger.info('====================')
                logger.info('random state %s ' % mc_i)
                logger.info('sample size %s '% sample_size)
                logger.info("information loss with best metric %s" % loss_best_metric)
                logger.info("information loss with generic metric %s" % loss_generic_metric)
                logger.info("information loss with linear metric deep  %s" % loss_learned_metric_linear)

        try:
            with open(resultDir + '/' + os.path.basename(__file__[0:-3]) +'.pickle', 'wb') as f:
                pickle.dump([loss_best_metric,loss_generic_metric, loss_unif_mc_linear,subsample_size_max], f)
        except:
            logger.error('Cannot create file: %s/%s.pickle'% (resultDir, os.path.basename(__file__[0:-3])))

if __name__ == "__main__":
    main(sys.argv[1:])
