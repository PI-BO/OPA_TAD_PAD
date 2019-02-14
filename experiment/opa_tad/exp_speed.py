import os, sys, getopt
sys.path.append(os.path.abspath("./"))
from helper import Utilities, PerformanceEvaluation,prepare_data,get_ground_truth_distance
import pandas as pd
from user_feedback import Similarity
from scipy.misc import comb
from deep_metric_learning import Deep_Metric
import numpy as np
import pickle
from deep_metric_learning_duplicate import Deep_Metric_Duplicate
from subsampling import Subsampling
import pdb

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
                return float(index)
        else:
            return -1

# Main methode
def main(argv):

    # set default values
    logDir = './log'
    resultDir = './result'
    inputfile = './dataset/opa_tad/speed.csv' 
    numberClasses = 0 
    preSanitization = False 

    mc_num = 5
    res = 4
  
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
        opts, args = getopt.getopt(argv,"hi:m:c:p")
    except getopt.GetoptError:
        print (__file__,' -i <input file> -m <mc num> -p (pre-senitization) -c <number of classes>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print (__file__,' -i <input file> -m <mc num> -p (pre-senitization)  -c <number of classes>')
            sys.exit()
        elif opt == '-i':
            inputfile = arg        
        elif opt == '-m':
            mc_num = int(arg) 
            if mc_num <= 1 and mc_num >= 5:
                logger.error("mc num must be between 1 and 5")
        elif opt == '-c':
            numberClasses = int(arg) 
            if numberClasses <= 0 and numberClasses >= 8:
                logger.error("number of classes must be between 0 and 8")  
        elif opt == '-p':
            preSanitization = True

    # Log console parameters
    logger.info("input file: %s, mc num: %i, pre-senitization: %r, number of classes: %i"% (inputfile, mc_num, preSanitization, numberClasses))

    if inputfile[-4:] == '.pkl': 
        day_profile_all = pd.read_pickle(inputfile)
    elif inputfile[-4:] == '.csv':
        # Read CSV File   
        df = pd.read_csv(inputfile, sep=';', header=None)
        day_profile_all = pd.DataFrame(columns=df.columns)

        # Create Day Profil with classID's
        for row_index,row in df.iterrows():
            list = [] 
            for index, speed in row.iteritems():
                if numberClasses == 0:
                    list.append(speed)
                else:
                    list.append(speed_class(numberClasses, speed))

            day_profile_all.loc[row_index] = list
    else:
        logger.error('input file not valid with %s'% inputfile[-4:])
        sys.exit()
    day_profile_all = day_profile_all.fillna(0)  

    if preSanitization:
        logger.info('speed pre-senitization')
    else:
        logger.info('speed hour')
      
    def evaluate_peak_time(anonymity_level,df_subsampled_from,day_profile,interest,window):
        subsample_size = int(round(subsample_size_max))
        sp = Subsampling(data=df_subsampled_from)
        data_pair, data_pair_all_index = sp.uniform_sampling(subsample_size=subsample_size, seed=None)

        if preSanitization:
            data_train_portion = 0.5

        sim = Similarity(data=data_pair)
        sim.extract_interested_attribute(interest='statistics', stat_type=interest, window=window)
        similarity_label, data_subsample = sim.label_via_silhouette_analysis(range_n_clusters=range(2, 8))

        if preSanitization:
            x1_train, x2_train, y_train, x1_test, x2_test, y_test = prepare_data(data_pair,similarity_label,data_train_portion)
        else:
            x1_train, x2_train, y_train, x1_test, x2_test, y_test = prepare_data(data_pair,similarity_label,0.5)
        
        lm = Deep_Metric_Duplicate(mode='linear')
        lm.train(x1_train,x2_train,y_train,x1_test,x2_test,y_test)

        dm = Deep_Metric_Duplicate(mode='relu')
        dm.train(x1_train,x2_train,y_train,x1_test,x2_test,y_test)

        sanitized_profile_linear = util.sanitize_data(day_profile, distance_metric="deep", anonymity_level=anonymity_level,
                                            rep_mode=rep_mode, deep_model=lm)

        sanitized_profile_deep = util.sanitize_data(day_profile, distance_metric="deep", anonymity_level=anonymity_level,
                                                    rep_mode=rep_mode, deep_model=dm)

        loss_learned_metric_linear = pe.get_statistics_loss(data_gt=day_profile,
                                                        data_sanitized=sanitized_profile_linear.round(),
                                                        mode=interest, window=window)

        loss_learned_metric_deep = pe.get_statistics_loss(data_gt=day_profile,
                                                        data_sanitized=sanitized_profile_deep.round(),
                                                        mode=interest, window=window)

        if preSanitization:
            if data_train_portion == 1:
                distance_gt = np.nan
                distance_lm = np.nan
                distance_dm = np.nan
            else:
                distance_gt = get_ground_truth_distance(x1_test,x2_test,mode=interest,window=window)
                distance_lm = lm.d_test
                distance_dm = dm.d_test
        else:
            distance_gt = get_ground_truth_distance(x1_test,x2_test,mode=interest,window=window)
            distance_lm = lm.d_test
            distance_dm = dm.d_test

        # print('anonymity level %s' % anonymity_level)
        logger.info("sampled size %s" % subsample_size)
        logger.info("information loss with best metric %s" % loss_best_metric)
        logger.info("information loss with generic metric %s" % loss_generic_metric)
        logger.info("information loss with learned metric %s" % loss_learned_metric_linear)
        logger.info("information loss with learned metric deep  %s" % (loss_learned_metric_deep))

        return loss_learned_metric_linear,loss_learned_metric_deep,sanitized_profile_linear,sanitized_profile_deep,\
            distance_lm,distance_dm,distance_gt


    # Initialization of some useful classes
    util = Utilities()
    pe = PerformanceEvaluation()  

    # define use case
    interest = 'window-usage'
    window = [17, 21]
    rep_mode = 'mean'

    # specify the data set for learning and for sanitization
    n_rows = 80
    day_profile = day_profile_all.iloc[:n_rows,0::res]

    if not preSanitization:
        pub_size = n_rows
        day_profile_learning = day_profile_all.iloc[n_rows:n_rows+pub_size,0::res]

    logger.debug(day_profile)
    logger.debug(day_profile_all)

    frac = 0.8
    anonymity_levels = np.arange(2,8)
    losses_best = np.zeros(len(anonymity_levels))
    losses_generic = np.zeros(len(anonymity_levels))
    losses_linear = np.zeros((len(anonymity_levels),mc_num))
    losses_deep = np.zeros((len(anonymity_levels),mc_num))

    distances_dm = {}
    distances_lm = {}
    distances_gt = {}

    for i in range(len(anonymity_levels)):
        anonymity_level = anonymity_levels[i]
        sanitized_profile_best = util.sanitize_data(day_profile, distance_metric='self-defined',
                                                    anonymity_level=anonymity_level, rep_mode=rep_mode,
                                                    mode=interest, window=window)
        sanitized_profile_baseline = util.sanitize_data(day_profile, distance_metric='euclidean',
                                                        anonymity_level=anonymity_level, rep_mode=rep_mode)
        loss_best_metric = pe.get_statistics_loss(data_gt=day_profile, data_sanitized=sanitized_profile_best,
                                                mode=interest, window=window)
        loss_generic_metric = pe.get_statistics_loss(data_gt=day_profile, data_sanitized=sanitized_profile_baseline,
                                                    mode=interest, window=window)
        losses_best[i] = loss_best_metric
        losses_generic[i] = loss_generic_metric

        for mc_i in range(mc_num):
            if preSanitization:
                df_subsampled_from = sanitized_profile_baseline.sample(frac=frac,replace=False, random_state=mc_i)
            else:
                df_subsampled_from = day_profile_learning.sample(frac=frac,replace=False, random_state=mc_i)
            subsample_size_max = int(comb(len(df_subsampled_from), 2))
            logger.info('total number of pairs is %s' % subsample_size_max)
            loss_learned_metric_linear, loss_learned_metric_deep, sanitized_profile_linear, sanitized_profile_deep, \
                distance_lm, distance_dm, distance_gt = evaluate_peak_time(anonymity_level,df_subsampled_from,day_profile,
                                                                        interest,window)
            losses_linear[i,mc_i] = loss_learned_metric_linear
            losses_deep[i,mc_i] = loss_learned_metric_deep
            distances_lm[(i,mc_i)] = distance_lm
            distances_dm[(i,mc_i)] = distance_dm
            distances_gt[(i, mc_i)] = distance_gt

            logger.info('==========================')
            logger.info('anonymity level index %s'% i)
            logger.info('mc iteration %s' % mc_i)     
       
        try:
            if preSanitization:
                with open(resultDir + '/' + os.path.basename(__file__[0:-3]) +'_presanitized.pickle', 'wb') as f:
                    pickle.dump([anonymity_levels,losses_best,losses_generic,losses_linear,losses_deep,distances_lm,distances_dm,distances_gt], f)
            else:
                with open(resultDir + '/' + os.path.basename(__file__[0:-3]) +'.pickle', 'wb') as f:
                    pickle.dump([anonymity_levels,losses_best,losses_generic,losses_linear,losses_deep,distances_lm,distances_dm,distances_gt], f)
        except:
            logger.error('Cannot create file: %s/%s.pickle'% (resultDir, os.path.basename(__file__[0:-3])))

if __name__ == "__main__":
    main(sys.argv[1:])
