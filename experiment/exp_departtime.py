import sys; import os
sys.path.append(os.path.abspath("./"))

from helper import Utilities, PerformanceEvaluation,prepare_data,get_ground_truth_distance
import pandas as pd
from user_feedback import Similarity
from scipy.misc import comb
from deep_metric_learning import Deep_Metric
import numpy as np
import pickle
from deep_metric_learning_duplicate import Linear_Metric
from subsampling import Subsampling
import pdb

print('departure time')
def evaluate_nonlinear(anonymity_level,df_subsampled_from,day_profile,rep_mode,mode):
    subsample_size_max = int(comb(len(df_subsampled_from), 2))
    print('total number of pairs is %s' % subsample_size_max)
    subsample_size = int(round(subsample_size_max))
    sp = Subsampling(data=df_subsampled_from)
    data_pair, data_pair_all_index = sp.uniform_sampling(subsample_size=subsample_size, seed=0)

    sim = Similarity(data=data_pair)
    sim.extract_interested_attribute(interest='statistics',stat_type=mode)
    similarity_label, data_subsample = sim.label_via_silhouette_analysis(range_n_clusters=range(2, 8))

    x1_train, x2_train, y_train, x1_test, x2_test, y_test = prepare_data(data_pair,similarity_label,0.5)
    lm = Deep_Metric(mode='linear')
    lm.train(x1_train,x2_train,y_train,x1_test,x2_test,y_test)

    dm = Deep_Metric(mode='relu')
    dm.train(x1_train,x2_train,y_train,x1_test,x2_test,y_test)

    sanitized_profile_linear = util.sanitize_data(day_profile, distance_metric="deep", anonymity_level=anonymity_level,
                                           rep_mode=rep_mode, deep_model=lm)

    sanitized_profile_deep = util.sanitize_data(day_profile, distance_metric="deep", anonymity_level=anonymity_level,
                                                rep_mode=rep_mode, deep_model=dm)

    loss_learned_metric_linear = pe.get_statistics_loss(data_gt=day_profile,
                                                  data_sanitized=sanitized_profile_linear.round(),
                                                  mode=mode)

    loss_learned_metric_deep = pe.get_statistics_loss(data_gt=day_profile,
                                                  data_sanitized=sanitized_profile_deep.round(),
                                                  mode=mode)

    distance_gt = get_ground_truth_distance(x1_test,x2_test,mode=mode)
    distance_lm = lm.d_test
    distance_dm = dm.d_test

    # print('anonymity level %s' % anonymity_level)
    print("sampled size %s" % subsample_size)
    print("information loss with best metric %s" % loss_best_metric)
    print("information loss with generic metric %s" % loss_generic_metric)
    print("information loss with learned metric %s" % loss_learned_metric_linear)
    print("information loss with learned metric deep  %s" % (loss_learned_metric_deep))
    # pdb.set_trace()
    return loss_learned_metric_linear,loss_learned_metric_deep,sanitized_profile_linear,sanitized_profile_deep,\
           distance_lm,distance_dm,distance_gt


# Initialization of some useful classes
util = Utilities()
pe = PerformanceEvaluation()

# load dataset
day_profile_all = pd.read_pickle('dataset/dataframe_all_binary.pkl')
day_profile_all = day_profile_all.fillna(0)
day_profile_all[day_profile_all > 0] = 1
res = 15

# define use case
interest = 'statistics'
rep_mode = 'mean'
mode='departure'

# specify the data set for learning and for sanitization
n_rows = 80
pub_size = n_rows
day_profile = day_profile_all.iloc[:n_rows,0::res]
day_profile_learning = day_profile_all.iloc[n_rows:n_rows+pub_size,0::res]

mc_num = 5
frac = 0.8
anonymity_levels = np.arange(2,8)
losses_best = np.ones(len(anonymity_levels)) * np.nan
losses_generic = np.ones(len(anonymity_levels))* np.nan
losses_linear = np.ones((len(anonymity_levels),mc_num))* np.nan
losses_deep = np.ones((len(anonymity_levels),mc_num))* np.nan

distances_dm = {}
distances_lm = {}
distances_gt = {}

for i in range(len(anonymity_levels)):
    anonymity_level = anonymity_levels[i]
    sanitized_profile_best = util.sanitize_data(day_profile, distance_metric='self-defined',
                                                anonymity_level=anonymity_level, rep_mode=rep_mode,
                                                mode=mode)

    sanitized_profile_baseline = util.sanitize_data(day_profile, distance_metric='euclidean',
                                                    anonymity_level=anonymity_level, rep_mode=rep_mode)

    loss_best_metric = pe.get_statistics_loss(data_gt=day_profile, data_sanitized=sanitized_profile_best,
                                              mode=mode)

    loss_generic_metric = pe.get_statistics_loss(data_gt=day_profile,
                                                 data_sanitized=sanitized_profile_baseline,
                                                 mode=mode)
    losses_best[i] = loss_best_metric
    losses_generic[i] = loss_generic_metric

    for mc_i in range(mc_num):

        df_subsampled_from = day_profile_learning.sample(frac=frac,replace=False, random_state=mc_i)
        loss_learned_metric_linear, loss_learned_metric_deep, sanitized_profile_linear, sanitized_profile_deep, \
            distance_lm, distance_dm, distance_gt = evaluate_nonlinear(anonymity_level,df_subsampled_from,day_profile,
                                                                       rep_mode,mode)
        losses_linear[i,mc_i] = loss_learned_metric_linear
        losses_deep[i,mc_i] = loss_learned_metric_deep
        distances_lm[(i,mc_i)] = distance_lm
        distances_dm[(i,mc_i)] = distance_dm
        distances_gt[(i, mc_i)] = distance_gt

        print('==========================')
        print('anonymity level index %s'% i)

    with open('./result_nonlinear/departtime.pickle', 'wb') as f:
        pickle.dump([anonymity_levels,losses_best,losses_generic,losses_linear,losses_deep,distances_lm,distances_dm,distances_gt], f)

# with open('./result_deep/lunchtime.pickle', 'rb') as f:
#    anonymity_levels, losses_best, losses_generic, losses_linear, losses_deep, distances_lm, distances_dm,distances_gt = pickle.load(f)
#
#
#


