Encoder_prefix_prompt_length = 100
Encoder_inter_prompt_length = 100
Decoder_prefix_prompt_length = 0

#target_data_set = ['BERTS2S',"PtGen","TConvS2S","TranS2S","Gold"]
target_data_set = ['BERTS2S',"PtGen","TConvS2S","TranS2S"]
#target_data_set = ['BERTS2S']
invalid_sample_id = [882,1271,1357,2025,2074,2175]

'''
10 + 10 Sentence Train on 5 Test on 4
Target Data   : BERTS2S
Acc           : 0.6461538461538462
Recall        : 0.8030095759233926
Precision     : 0.5180935569285083
F1 Score      : 0.6298283261802575
Total     Num : 0.3748717948717949
Predicted Num : 0.581025641025641
Target Data   : PtGen
Acc           : 0.6125343092406221
Recall        : 0.8685220729366603
Precision     : 0.5603715170278638
F1 Score      : 0.6812194203989462
Total     Num : 0.47666971637694416
Predicted Num : 0.7387923147301007
Target Data   : TConvS2S
Acc           : 0.6013268998793727
Recall        : 0.8972602739726028
Precision     : 0.5792188651436994
F1 Score      : 0.7039856695029109
Total     Num : 0.5283474065138721
Predicted Num : 0.8184559710494572
Target Data   : TranS2S
Acc           : 0.5763239875389408
Recall        : 0.836340206185567
Precision     : 0.5399334442595674
F1 Score      : 0.6562184024266935
Total     Num : 0.4834890965732087
Predicted Num : 0.7489096573208722
Acc      0.6110285173672118
Recall   0.8545985401459854
Prec     0.5515357075560581
predict  0.71725908906609
F1       0.6704076958314246
634
=== Predict in Corpus Level Threshold ===
[{'data': 'PtGen', 'not_in': 1, 'prob_s': -70.5021460056305, 'label': 1}, {'data': 'TConvS2S', 'not_in': 0, 'prob_s': -67.51661293208599, 'label': 1}, {'data': 'PtGen', 'not_in': 1, 'prob_s': -67.44863486289978, 'label': 1}, {'data': 'BERTS2S', 'not_in': 0, 'prob_s': -67.42245195806026, 'label': 0}, {'data': 'TConvS2S', 'not_in': 0, 'prob_s': -67.01294595003128, 'label': 1}]
Dataset  BERTS2S
Acc      0.583076923076923
Recall   0.8686730506155951
Prec     0.46967455621301774
predict  0.6933333333333334
labeled  0.3748717948717949
F1       0.6096975516082572
Dataset  PtGen
Acc      0.6161939615736505
Recall   0.8675623800383877
Prec     0.5632398753894081
predict  0.7342177493138152
labeled  0.47666971637694416
F1       0.6830374008311296
Dataset  TConvS2S
Acc      0.638118214716526
Recall   0.8378995433789954
Prec     0.6157718120805369
predict  0.7189384800965019
labeled  0.5283474065138721
F1       0.7098646034816247
Dataset  TranS2S
Acc      0.5800623052959502
Recall   0.8118556701030928
Prec     0.5440414507772021
predict  0.7214953271028037
labeled  0.4834890965732087
F1       0.6514994829369184
Dataset Total Corpus
Acc      0.6045411542100284
Recall   0.8475912408759124
Prec     0.5470133785566234
predict  0.71725908906609
labeled  0.46290039194485744
F1       0.6649106733852497

Test On 5
Target Data   : BERTS2S
Acc           : 0.6461538461538462
Recall        : 0.8030095759233926
Precision     : 0.5180935569285083
F1 Score      : 0.6298283261802575
Total     Num : 0.3748717948717949
Predicted Num : 0.581025641025641
Target Data   : PtGen
Acc           : 0.6125343092406221
Recall        : 0.8685220729366603
Precision     : 0.5603715170278638
F1 Score      : 0.6812194203989462
Total     Num : 0.47666971637694416
Predicted Num : 0.7387923147301007
Target Data   : TConvS2S
Acc           : 0.6013268998793727
Recall        : 0.8972602739726028
Precision     : 0.5792188651436994
F1 Score      : 0.7039856695029109
Total     Num : 0.5283474065138721
Predicted Num : 0.8184559710494572
Target Data   : TranS2S
Acc           : 0.5763239875389408
Recall        : 0.836340206185567
Precision     : 0.5399334442595674
F1 Score      : 0.6562184024266935
Total     Num : 0.4834890965732087
Predicted Num : 0.7489096573208722
Target Data   : Gold
Acc           : 0.6447841726618705
Recall        : 0.750332005312085
Precision     : 0.48414738646101113
F1 Score      : 0.5885416666666666
Total     Num : 0.3385791366906475
Predicted Num : 0.5247302158273381
Acc      0.6188298867297101
Recall   0.8358066060315941
Prec     0.5393883225208527
predict  0.672763171568118
F1       0.6556515208411566
810
=== Predict in Corpus Level Threshold ===
[{'data': 'PtGen', 'not_in': 1, 'prob_s': -70.5021460056305, 'label': 1}, {'data': 'Gold', 'not_in': 1, 'prob_s': -69.67279148101807, 'label': 1}, {'data': 'Gold', 'not_in': 1, 'prob_s': -67.9790911078453, 'label': 1}, {'data': 'Gold', 'not_in': 1, 'prob_s': -67.66708274930716, 'label': 1}, {'data': 'TConvS2S', 'not_in': 0, 'prob_s': -67.51661293208599, 'label': 1}]
Dataset  BERTS2S
Acc      0.6230769230769231
Recall   0.8426812585499316
Prec     0.49838187702265374
predict  0.6338461538461538
labeled  0.3748717948717949
F1       0.6263345195729537
Dataset  PtGen
Acc      0.6354071363220494
Recall   0.836852207293666
Prec     0.5817211474316211
predict  0.6857273559011894
labeled  0.47666971637694416
F1       0.6863439590712318
Dataset  TConvS2S
Acc      0.6519903498190591
Recall   0.8070776255707762
Prec     0.6340807174887892
predict  0.672496984318456
labeled  0.5283474065138721
F1       0.7101958814665997
Dataset  TranS2S
Acc      0.5919003115264797
Recall   0.7693298969072165
Prec     0.5563839701770736
predict  0.6685358255451713
labeled  0.4834890965732087
F1       0.6457544618712818
Dataset  Gold
Acc      0.560251798561151
Recall   0.8804780876494024
Prec     0.4274661508704062
predict  0.6973920863309353
labeled  0.3385791366906475
F1       0.5755208333333333
Dataset Total Corpus
Acc      0.6111399771381066
Recall   0.8269506941120153
Prec     0.5336731541550819
predict  0.672763171568118
labeled  0.4341681388340434
F1       0.648704468644386
(shesj_cuda11) shesj@3090ti-1:~/workspace/workspace/Code/NLP-Research/fine_grained/PromptTuing$ 

'''



'''
init with 
4 splt
Target Data   : BERTS2S
Acc           : 0.637948717948718
Recall        : 0.7920656634746922
Precision     : 0.5110326566637247
F1 Score      : 0.621244635193133
Total     Num : 0.3748717948717949
Predicted Num : 0.581025641025641
Target Data   : PtGen
Acc           : 0.6235132662397073
Recall        : 0.8800383877159309
Precision     : 0.5678018575851393
F1 Score      : 0.6902521640948438
Total     Num : 0.47666971637694416
Predicted Num : 0.7387923147301007
Target Data   : TConvS2S
Acc           : 0.6254523522316043
Recall        : 0.9200913242009132
Precision     : 0.5939572586588062
F1 Score      : 0.7218987908643081
Total     Num : 0.5283474065138721
Predicted Num : 0.8184559710494572
Target Data   : TranS2S
Acc           : 0.5688473520249221
Recall        : 0.8286082474226805
Precision     : 0.5349417637271214
F1 Score      : 0.6501516683518705
Total     Num : 0.4834890965732087
Predicted Num : 0.7489096573208722
Acc      0.6158940397350994
Recall   0.8598540145985402
Prec     0.5549274543056341
predict  0.71725908906609
F1       0.6745304626660559
634
=== Predict in Corpus Level Threshold ===
[{'data': 'BERTS2S', 'not_in': 1, 'prob_s': -25.698721289634705, 'label': 0}, {'data': 'PtGen', 'not_in': 0, 'prob_s': -24.098883777856827, 'label': 1}, {'data': 'PtGen', 'not_in': 0, 'prob_s': -22.389554668217897, 'label': 1}, {'data': 'TranS2S', 'not_in': 1, 'prob_s': -22.26242595911026, 'label': 1}, {'data': 'TConvS2S', 'not_in': 0, 'prob_s': -22.143380165100098, 'label': 1}]
Dataset  BERTS2S
Acc      0.5764102564102564
Recall   0.8768809849521204
Prec     0.46550472040668117
predict  0.7061538461538461
labeled  0.3748717948717949
F1       0.6081593927893738
Dataset  PtGen
Acc      0.6248856358645929
Recall   0.8800383877159309
Prec     0.5688585607940446
predict  0.737419945105215
labeled  0.47666971637694416
F1       0.6910324039186134
Dataset  TConvS2S
Acc      0.6706875753920386
Recall   0.8584474885844748
Prec     0.6405451448040886
predict  0.7080820265379976
labeled  0.5283474065138721
F1       0.733658536585366
Dataset  TranS2S
Acc      0.5713395638629284
Recall   0.7938144329896907
Prec     0.5384615384615384
predict  0.712772585669782
labeled  0.4834890965732087
F1       0.6416666666666667
Dataset Total Corpus
Acc      0.6107582105689958
Recall   0.8543065693430657
Prec     0.5513472771810816
predict  0.71725908906609
labeled  0.46290039194485744
F1       0.6701786532295007
(shesj_cuda11) shesj@3090ti-1:~/workspace/workspace/Code/NLP-Research/fine_grained/PromptTuing$ 


(shesj_cuda10.1) shesj@v100-13:~/workspace/Code/NLP-Research/fine_grained/PromptTuing$ CUDA_VISIBLE_DEVICES=1 python3 rebuid.py 
Some weights of BartPromptForConditionalGeneration were not initialized from the model checkpoint at ../../../../Data/PLM/BARTCNN and are newly initialized: ['model.decoder.prefix_decoder.weight', 'model.encoder.prefix_encoder.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
loading from local Model
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:09<00:00, 51.98it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:09<00:00, 54.18it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:08<00:00, 62.35it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:06<00:00, 72.32it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:09<00:00, 55.27it/s]
Target Data   : BERTS2S
Acc           : 0.6430769230769231
Recall        : 0.79890560875513
Precision     : 0.5154457193292145
F1 Score      : 0.6266094420600858
Total     Num : 0.3748717948717949
Predicted Num : 0.581025641025641
Target Data   : PtGen
Acc           : 0.6290027447392498
Recall        : 0.8857965451055663
Precision     : 0.5715170278637771
F1 Score      : 0.6947685359427926
Total     Num : 0.47666971637694416
Predicted Num : 0.7387923147301007
Target Data   : TConvS2S
Acc           : 0.6254523522316043
Recall        : 0.9200913242009132
Precision     : 0.5939572586588062
F1 Score      : 0.7218987908643081
Total     Num : 0.5283474065138721
Predicted Num : 0.8184559710494572
Target Data   : TranS2S
Acc           : 0.5813084112149532
Recall        : 0.8414948453608248
Precision     : 0.543261231281198
F1 Score      : 0.6602628918099089
Total     Num : 0.4834890965732087
Predicted Num : 0.7489096573208722
Target Data   : Gold
Acc           : 0.6510791366906474
Recall        : 0.7596281540504648
Precision     : 0.49014567266495285
F1 Score      : 0.5958333333333333
Total     Num : 0.3385791366906475
Predicted Num : 0.5247302158273381
Acc      0.6283903148706225
Recall   0.8468166586883676
Prec     0.5464936669755946
predict  0.672763171568118
F1       0.6642883965452497
810
498
=== Predict in Corpus Level Threshold ===
[{'data': 'BERTS2S', 'not_in': 0, 'prob_s': -79.81827628612518, 'label': 1}, {'data': 'BERTS2S', 'not_in': 0, 'prob_s': -78.69722040742636, 'label': 1}, {'data': 'TranS2S', 'not_in': 1, 'prob_s': -78.54056099057198, 'label': 1}, {'data': 'TConvS2S', 'not_in': 1, 'prob_s': -76.05288165807724, 'label': 1}, {'data': 'Gold', 'not_in': 1, 'prob_s': -75.60191372036934, 'label': 1}]
Dataset  BERTS2S
Acc      0.6087179487179487
Recall   0.8522571819425444
Prec     0.4874804381846635
predict  0.6553846153846153
labeled  0.3748717948717949
F1       0.6202090592334494
Dataset  PtGen
Acc      0.6582799634034767
Recall   0.8522072936660269
Prec     0.599594868332208
predict  0.6774931381518756
labeled  0.47666971637694416
F1       0.7039239001189062
Dataset  TConvS2S
Acc      0.6731001206272618
Recall   0.8299086757990868
Prec     0.6491071428571429
predict  0.6755126658624849
labeled  0.5283474065138721
F1       0.7284569138276553
Dataset  TranS2S
Acc      0.609968847352025
Recall   0.770618556701031
Prec     0.5717017208413002
predict  0.6517133956386293
labeled  0.4834890965732087
F1       0.6564215148188803
Dataset  Gold
Acc      0.5656474820143885
Recall   0.8871181938911022
Prec     0.4312459651387992
predict  0.6964928057553957
labeled  0.3385791366906475
F1       0.5803649000868809
Dataset Total Corpus
Acc      0.621323911462122
Recall   0.8386787936811871
Prec     0.541241890639481
predict  0.672763171568118
labeled  0.4341681388340434
F1       0.65790461885092
9623
(shesj_cuda10.1) shesj@v100-13:~/workspace/Code/NLP-Research/fine_grained/PromptTuing$ 
'''



'''20+20
(shesj_cuda10.1) shesj@v100-13:~/workspace/Code/NLP-Research/fine_grained/PromptTuing$ CUDA_VISIBLE_DEVICES=3 python3 rebuid.py 
Some weights of BartPromptForConditionalGeneration were not initialized from the model checkpoint at ../../../../Data/PLM/BARTCNN and are newly initialized: ['model.decoder.prefix_decoder.weight', 'model.encoder.prefix_encoder.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
loading from local Model
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:09<00:00, 50.51it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:09<00:00, 53.34it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:08<00:00, 61.52it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:07<00:00, 71.20it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:09<00:00, 54.35it/s]
Target Data   : BERTS2S
Acc           : 0.642051282051282
Recall        : 0.7975376196990424
Precision     : 0.5145631067961165
F1 Score      : 0.6255364806866953
Total     Num : 0.3748717948717949
Predicted Num : 0.581025641025641
Target Data   : PtGen
Acc           : 0.6235132662397073
Recall        : 0.8800383877159309
Precision     : 0.5678018575851393
F1 Score      : 0.6902521640948438
Total     Num : 0.47666971637694416
Predicted Num : 0.7387923147301007
Target Data   : TConvS2S
Acc           : 0.612183353437877
Recall        : 0.9075342465753424
Precision     : 0.5858511422254974
F1 Score      : 0.7120465741155396
Total     Num : 0.5283474065138721
Predicted Num : 0.8184559710494572
Target Data   : TranS2S
Acc           : 0.567601246105919
Recall        : 0.8273195876288659
Precision     : 0.5341098169717138
F1 Score      : 0.6491405460060667
Total     Num : 0.4834890965732087
Predicted Num : 0.7489096573208722
Target Data   : Gold
Acc           : 0.6618705035971223
Recall        : 0.7755644090305445
Precision     : 0.5004284490145673
F1 Score      : 0.6083333333333333
Total     Num : 0.3385791366906475
Predicted Num : 0.5247302158273381
Acc      0.6248571131663723
Recall   0.8427477261847774
Prec     0.5438677788075379
predict  0.672763171568118
F1       0.6610965076980848
810
498
=== Predict in Corpus Level Threshold ===
[{'data': 'TranS2S', 'not_in': 1, 'prob_s': -88.34344723820686, 'label': 1}, {'data': 'TranS2S', 'not_in': 0, 'prob_s': -87.84150386601686, 'label': 1}, {'data': 'BERTS2S', 'not_in': 1, 'prob_s': -86.02703121304512, 'label': 1}, {'data': 'Gold', 'not_in': 1, 'prob_s': -84.68202074989676, 'label': 1}, {'data': 'Gold', 'not_in': 0, 'prob_s': -84.14464677125216, 'label': 0}]
Dataset  BERTS2S
Acc      0.6133333333333333
Recall   0.8508891928864569
Prec     0.4909234411996843
predict  0.6497435897435897
labeled  0.3748717948717949
F1       0.6226226226226226
Dataset  PtGen
Acc      0.646843549862763
Recall   0.8349328214971209
Prec     0.5918367346938775
predict  0.6724611161939615
labeled  0.47666971637694416
F1       0.6926751592356688
Dataset  TConvS2S
Acc      0.6640530759951749
Recall   0.8162100456621004
Prec     0.6435643564356436
predict  0.6700844390832328
labeled  0.5283474065138721
F1       0.719677906391545
Dataset  TranS2S
Acc      0.5869158878504673
Recall   0.7358247422680413
Prec     0.5549076773566569
predict  0.6411214953271028
labeled  0.4834890965732087
F1       0.6326869806094182
Dataset  Gold
Acc      0.5557553956834532
Recall   0.9043824701195219
Prec     0.4264245460237946
predict  0.7180755395683454
labeled  0.3385791366906475
F1       0.5795744680851063
Dataset Total Corpus
Acc      0.6119713187155773
Recall   0.827908089995213
Prec     0.5342910101946247
predict  0.672763171568118
labeled  0.4341681388340434
F1       0.6494555013143072
9623


Some weights of BartPromptForConditionalGeneration were not initialized from the model checkpoint at ../../../../Data/PLM/BARTCNN and are newly initialized: ['model.decoder.prefix_decoder.weight', 'model.encoder.prefix_encoder.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
loading from local Model
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:09<00:00, 51.25it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:09<00:00, 53.30it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:08<00:00, 62.05it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:06<00:00, 71.86it/s]
Target Data   : BERTS2S
Acc           : 0.642051282051282
Recall        : 0.7975376196990424
Precision     : 0.5145631067961165
F1 Score      : 0.6255364806866953
Total     Num : 0.3748717948717949
Predicted Num : 0.581025641025641
Target Data   : PtGen
Acc           : 0.6235132662397073
Recall        : 0.8800383877159309
Precision     : 0.5678018575851393
F1 Score      : 0.6902521640948438
Total     Num : 0.47666971637694416
Predicted Num : 0.7387923147301007
Target Data   : TConvS2S
Acc           : 0.612183353437877
Recall        : 0.9075342465753424
Precision     : 0.5858511422254974
F1 Score      : 0.7120465741155396
Total     Num : 0.5283474065138721
Predicted Num : 0.8184559710494572
Target Data   : TranS2S
Acc           : 0.567601246105919
Recall        : 0.8273195876288659
Precision     : 0.5341098169717138
F1 Score      : 0.6491405460060667
Total     Num : 0.4834890965732087
Predicted Num : 0.7489096573208722
Acc      0.6137315853493716
Recall   0.8575182481751825
Prec     0.5534200113058225
predict  0.71725908906609
F1       0.6726981218506642
634
394
=== Predict in Corpus Level Threshold ===
[{'data': 'TranS2S', 'not_in': 1, 'prob_s': -88.34344723820686, 'label': 1}, {'data': 'TranS2S', 'not_in': 0, 'prob_s': -87.84150386601686, 'label': 1}, {'data': 'BERTS2S', 'not_in': 1, 'prob_s': -86.02703121304512, 'label': 1}, {'data': 'PtGen', 'not_in': 0, 'prob_s': -83.83350957930088, 'label': 1}, {'data': 'BERTS2S', 'not_in': 1, 'prob_s': -83.65787953138351, 'label': 1}]
Dataset  BERTS2S
Acc      0.5897435897435898
Recall   0.9015047879616963
Prec     0.47512617159336695
predict  0.7112820512820512
labeled  0.3748717948717949
F1       0.6222851746931066
Dataset  PtGen
Acc      0.6299176578225069
Recall   0.8752399232245681
Prec     0.5732243871778756
predict  0.7278133577310155
labeled  0.47666971637694416
F1       0.6927459172047096
Dataset  TConvS2S
Acc      0.6477683956574186
Recall   0.8538812785388128
Prec     0.6212624584717608
predict  0.7261761158021713
labeled  0.5283474065138721
F1       0.7192307692307692
Dataset  TranS2S
Acc      0.573208722741433
Recall   0.7835051546391752
Prec     0.5404444444444444
predict  0.7009345794392523
labeled  0.4834890965732087
F1       0.6396633350867964
Dataset Total Corpus
Acc      0.6110285173672118
Recall   0.8545985401459854
Prec     0.5515357075560581
predict  0.71725908906609
labeled  0.46290039194485744
F1       0.6704076958314246
7399
(shesj_cuda10.1) shesj@v100-13:~/workspace/Code/NLP-Research/fine_grained/PromptTuing$ 

'''