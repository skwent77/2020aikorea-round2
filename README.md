### [2020년도 AI 문제해결 경진대회] 본선

https://github.com/harrywinks/2020aikorea-round2/issues/1#issue-769649097

## (대회내용) 산업·사회를 혁신시킬 수 있는 AI 문제를 발굴하고 참가자들이 AI 알고리즘을 활용해 해결하는 문제해결 경진대회

## (대회기간) 2020년 10월12일 ~ 23일 15시 (2주간 접수), 10월30일 ~ 11월13일 15시 (대회 개최)

# 2020 aikorea-round2

task: detect small crack in tile 타일 이미지 내 미세 Crack 검출 모델
team_id = "1350"
team_name = "머닝러신머닝"
task_no = "153"



### <데이터 전처리 및 데이터 Augmentation>

데이터 전처리는 따로 수행하지 않았습니다.

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
