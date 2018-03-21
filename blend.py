# -*- coding: utf-8 -*-

'''
The core idea behind all blends is "diversity". 
By blending some moderate model results (with some weights), we can create a more "diverse" stable results.
Errors from one model will be covered by others. Same goes for all the models. 
So, we get more stable results after blendings. 
'''
import pandas as pd

biLSTMConv = pd.read_csv("submit/blend/bilstm_conv.csv") # 0.9842
biGRUCONV = pd.read_csv("submit/blend/BiGRU_CONV_FastText_badwords.csv") # 0.9800
LG = pd.read_csv("submit/blend/lg_wors_ngram_prec.csv") # 0.9783
nbsvm = pd.read_csv("submit/blend/NB_SVM.csv") # 0.9772
poolBiGRU = pd.read_csv("submit/blend/pool_BiGRU_emoji.csv") # 0.9831
textCNN = pd.read_csv("submit/blend/textCNN.csv") # 0.9788
tunedLR = pd.read_csv("submit/blend/submission-tuned-LR-01.csv") # 0.9802
gruattention = pd.read_csv("submit/blend/simple_lstm_glove_vectors_0.25_0.25.csv") # 0.9837

xgb = pd.read_csv("submit/blend/xgb_words_ngram.csv") # 0.9778
wordbatch = pd.read_csv("submit/blend/lvl0_wordbatch_clean_sub.csv") # 0.9813
twoRNNcnn = pd.read_csv("submit/blend/two_RNN_cnn.csv") # 0.9843
textCNN2d = pd.read_csv("submit/blend/textCNN2d_prec.csv") # 0.9832
dpcnn = pd.read_csv("submit/blend/dpcnn_test_preds.csv") # 0.9831
lgb = pd.read_csv("submit/blend/lgb_submission.csv") # 0.9798
oof_stacking = pd.read_csv("submit/blend/oof_stacking.csv") # 0.9858
mlp = pd.read_csv("submit/blend/mlp_auc-valid_0.985635.csv") # 0.981

capsule_gru = pd.read_csv("submit/blend/capsule_gru.csv") # 0.9839
ridge = pd.read_csv("submit/blend/5-fold_elast_test.csv") # 0.9810
hillclimb = pd.read_csv("submit/sub/hillclimb.csv") # 0.9843

'''
best = pd.read_csv("submit/blend/blend_it_all.csv") # 0.9867

b1 = best.copy()
col = best.columns

col = col.tolist()
col.remove("id")

for i in col:
    b1[i] = (biLSTMConv[i] * 4 + biGRUCONV[i] * 2 + LG[i] + xgb[i] + nbsvm[i] + poolBiGRU[i] * 4 + textCNN[i] + tunedLR[i] * 2 + gruattention[i] * 2 +
      xgb[i] + wordbatch[i] * 2 + twoRNNcnn[i] * 4 + textCNN2d[i] * 2 + dpcnn[i] * 2 + lgb[i] + oof_stacking[i] * 5 + lvl0_lgbm[i] * 2 + mlp[i] * 2 + textCNN[i] * 2 +
      best[i] * 4) / 33

b1.to_csv('submit/self_blend.csv', index = False)'''


labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

from sklearn.preprocessing import minmax_scale
for label in labels:
    biLSTMConv[label] = minmax_scale(biLSTMConv[label])
    oof_stacking[label] = minmax_scale(oof_stacking[label])
    gruattention[label] = minmax_scale(gruattention[label])
    capsule_gru[label] = minmax_scale(capsule_gru[label])
    poolBiGRU[label] = minmax_scale(poolBiGRU[label])
    dpcnn[label] = minmax_scale(dpcnn[label])
    hillclimb[label] = minmax_scale(hillclimb[label])
    twoRNNcnn[label] = minmax_scale(twoRNNcnn[label])
    textCNN2d[label] = minmax_scale(textCNN2d[label])
    

submission = pd.DataFrame()
submission['id'] = biLSTMConv['id']
'''
submission[labels] = (biLSTMConv[labels]*4 + 
                     biGRUCONV[labels] +
                     LG[labels] + 
                     nbsvm[labels] +
                     tunedLR[labels]+
                     xgb[labels]+
                     lgb[labels]+
                     mlp[labels]+
                     hillclimb[labels] * 4 + 
                     oof_stacking[labels]*5 + 
                     gruattention[labels]*3 + 
                     capsule_gru[labels]*3 + 
                     poolBiGRU[labels]*3 + 
                     twoRNNcnn[labels]*4 + 
                     textCNN2d[labels]*3 +
                     ridge[labels]+
                     wordbatch[labels]+
                     dpcnn[labels]*3) / 41'''

submission[labels] = (biLSTMConv[labels] + 
                     oof_stacking[labels] + 
                     gruattention[labels] + 
                     capsule_gru[labels] + 
                     poolBiGRU[labels] + 
                     twoRNNcnn[labels] + 
                     textCNN2d[labels] +
                     hillclimb[labels]+
                     #ridge[labels]+
                     #wordbatch[labels]+
                     dpcnn[labels]) / 9

submission.to_csv('submit/ensemble/avg.csv', index=False)


