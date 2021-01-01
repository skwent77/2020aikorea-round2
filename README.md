### [2020년도 AI 문제해결 경진대회] 본선

<img src="https://user-images.githubusercontent.com/46518769/102458252-51ad3300-4087-11eb-982c-002e5c2248db.png" width="70%"></img>

## (대회내용) 산업·사회를 혁신시킬 수 있는 AI 문제를 발굴하고 참가자들이 AI 알고리즘을 활용해 해결하는 문제해결 경진대회

## (대회기간) 2020년 10월12일 ~ 23일 15시 (2주간 접수), 10월30일 ~ 11월13일 15시 (대회 개최)

# 2020 aikorea-round2

task: detect small crack in tile 타일 이미지 내 미세 Crack 검출 모델
team_id = "1350"
team_name = "머닝러신머닝"
task_no = "153"

최종 3위 기록 F1 score 0.948729

<img src="https://user-images.githubusercontent.com/46518769/103121347-1af69e80-46bf-11eb-8a3a-4b112bb1df2a.png" width="70%"></img>

### <데이터 전처리 및 데이터 Augmentation>

데이터 전처리는 따로 수행하지 않았습니다.

['1149.tif', '8558.tif', '13636.tif', '1459.tif', '13712.tif', '764.tif', '6251.tif', '8199.tif', '6196.tif', '3869.tif', '6988.tif', '8229.tif', '11209.tif', '1369.tif', '4960.tif', '8741.tif', '13410.tif', '13164.tif', '7348.tif', '5850.tif', '12024.tif', '394.tif', '7483.tif', '5885.tif', '8658.tif', '8959.tif', '8765.tif', '11834.tif', '13391.tif', '6089.tif', '3772.tif', 'train.csv', '4712.tif', '13510.tif', '6869.tif', '2387.tif', '3493.tif', '2075.tif', '11664.tif', '6473.tif', '13718.tif', '4314.tif', '3724.tif', '5871.tif', '8054.tif', '14004.tif', '11806.tif', '9251.tif', '13841.tif', '10456.tif', '9424.tif', '3100.tif', '9130.tif', '11352.tif', '7076.tif', '3177.tif', '374.tif', '11639.tif', '6597.tif', '4993.tif', '872.tif', '13443.tif', '5816.tif', '10260.tif', '3424.tif', '11809.tif', '12650.tif', '5690.tif', '5276.tif', '7650.tif', '12688.tif', '5084.tif', '4785.tif', '6495.tif', '11049.tif', '11731.tif', '308.tif', '13775.tif', '2683.tif', '7097.tif', '8349.tif', '2269.tif', '621.tif', '7186.tif', '11508.tif', '7853.tif', '3653.tif', '3977.tif', '1350.tif', '8455.tif', '7269.tif', '7270.tif', '2965.tif', '8613.tif', '8559.tif', '11035.tif', '422.tif', '3060.tif', '171.tif', '9608.tif', '5970.tif', '12512.tif', '168.tif', '1470.tif', '3111.tif', '12311.tif', '2847.tif', '8554.tif', '11417.tif', '5282.tif', '4295.tif', '11118.tif', '12711.tif', '13014.tif', '5977.tif', '6188.tif', '12111.tif', '920.tif', '7430.tif', '12955.tif', '8447.tif', '763.tif', '7059.tif', '9780.tif', '4915.tif', '3715.tif', '6195.tif', '10489.tif', '10719.tif', '2841.tif', '11278.tif', '11269.tif', '4340.tif', '1494.tif', '2834.tif', '13257.tif', '12062.tif', '12184.tif', '3290.tif', '5310.tif', '8081.tif', '10727.tif', '7095.tif', '8164.tif', '13106.tif', '1398.tif', '2735.tif', '6910.tif', '13285.tif', '13196.tif', '448.tif', '6936.tif', '2408.tif', '3542.tif', '819.tif', '9839.tif', '12793.tif', '3934.tif', '11413.tif', '11458.tif', '2074.tif', '11361.tif', '13496.tif', '8634.tif', '4891.tif', '4456.tif', '12619.tif', '9752.tif', '6334.tif', '1621.tif', '1961.tif', '12765.tif', '8392.tif', '1893.tif', '1200.tif', '9592.tif', '13658.tif', '10726.tif', '6255.tif', '11603.tif', '7025.tif', '1738.tif', '9763.tif', '9749.tif', '12825.tif', '6911.tif', '3842.tif', '13842.tif', '10139.tif', '6916.tif', '11561.tif', '12018.tif', '11841.tif', '1608.tif', '10334.tif', '1843.tif', '1032.tif', '7407.tif', '11122.tif', '11247.tif', '10388.tif', '3264.tif', '3732.tif', '12315.tif', '8422.tif', '5992.tif', '9756.tif', '10761.tif', '6536.tif', '6889.tif', '5020.tif', '727.tif', '1068.tif', '1474.tif', '1439.tif', '1500.tif', '2763.tif', '5144.tif', '6050.tif', '2537.tif', '6450.tif', '6125.tif', '7551.tif', '4505.tif', '3571.tif', '598.tif', '11083.tif', '8167.tif', '6203.tif', '6422.tif', '1945.tif', '12868.tif', '7794.tif', '9950.tif', '4238.tif', '10476.tif', '3607.tif', '6894.tif', '13436.tif', '5323.tif', '1127.tif', '12554.tif', '8037.tif', '720.tif', '12096.tif', '3058.tif', '6509.tif', '7219.tif', '5562.tif', '528.tif', '4475.tif', '2264.tif',..]

IMAGE 파일이 tif의 형태로 주어졌다. 

 
Data Augmentation 과정에서 다른 여러 기법들을 적용해보았지만 좋지않았고, 오직 두가지의 기법만 사용했습니다.
- Horizontal Flip
- 이미지의 Pixel-Value를 (-1~1)로 Normalization

Training 과정에서 validation set은 미리 나누어 놓지않고, scikit-learn의 train_test_split을 이용하여 전체 train set에서 train data와 validation data를 트레이닝 시작 시 나누어 사용하였습니다. validation data는 validation_gt.csv로 저장하여 네트워크 학습 후에 확인할 수 있도록 저장하도록 했고 매 epoch마다 validation.csv로 validation set을 prediction한 레이블 값을 저장하도록 구성하였습니다.



### <모델 학습> (Train)
 
- 사용한 모델: Xception, EfficientNet-B5 (ImageNet pretrained)
- 10 epochs
- batch-size: 64
- 초기 learning rate: 1e-3
- StepLearningRateDecay(매 3 epoch마다 lr을 10분의 1로 줄임)
- Adam Optimizer
- CrossEntropyLoss

## 모델 학습 과정 부연설명 
1) ImageNet에서 pretrain된 EficientNetB5 모델과 Xception 모델을 활용하여 전이학습(Transfer-Learning)을 수행하였다.
2) 앙상블 활용 : Xception과 EfficientNetB5 모델을 각각 3개씩 multi-model ensemble(앙상블)을 활용하였다. ResNet 등 다른 Pretrained 모델과 직접 모델을 구축하여 사용해보았지만 성능면에서 큰 이점이 없었다. 

_infer_ensemble메서드와 feed_infer메서드를 활용하였다.

### 모델 검증 (Validation)

베이스라인 코드로 주어진 evaluation.py를 통해 예측값의 f1_score를 검증에 활용하였다.




## <모델 예측>

- 6개의 모델 앙상블(Xception 3개, EfficientNet-B5 3개)
- 단순히 6개의 모델의 결과 값의 합을 사용

### Train_and_Test.ipynb에서 4번을 실행하면 모델예측이 완료됩니다.
