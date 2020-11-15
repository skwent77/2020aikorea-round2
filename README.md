

# 2020aikorea-round2
task: detect small crack in tile
team_id = "1350"
team_name = "머닝러신머닝"
task_no = "153"



### <데이터 전처리 및 데이터 Augmentation>

데이터 전처리는 따로 수행하지 않았습니다.

Data Augmentation 과정에서 다른 여러 기법들을 적용해보았지만 좋지않았고, 오직 두가지의 기법만 사용했습니다.
- Horizontal Flip
- 이미지의 Pixel-Value를 (-1~1)로 Normalization

Training 과정에서 validation set은 미리 나누어 놓지않고, scikit-learn의 train_test_split을 이용하여 전체 train set에서 train data와 validation data를 트레이닝 시작 시 나누어 사용하였습니다. validation data는 validation_gt.csv로 저장하여 네트워크 학습 후에 확인할 수 있도록 저장하도록 했고 매 epoch마다 validation.csv로 validation set을 prediction한 레이블 값을 저장하도록 구성하였습니다.



### <모델 학습>

- 사용한 모델: Xception, EfficientNet-B5 (ImageNet pretrained)
- 10 epochs
- batch-size: 64
- 초기 learning rate: 1e-3
- StepLearningRateDecay(매 3 epoch마다 lr을 10분의 1로 줄임)
- Adam Optimizer
- CrossEntropyLoss

### Train_and_Test.ipynb에서 1~3번 까지 실행하면 트레이닝이 완료됩니다.



## <모델 예측>

- 6개의 모델 앙상블(Xception 3개, EfficientNet-B5 3개)
- 단순히 6개의 모델의 결과 값의 합을 사용

### Train_and_Test.ipynb에서 4번을 실행하면 모델예측이 완료됩니다.
