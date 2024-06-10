### 최근 6개월 판매량 예측

#### Description

- naver 쇼핑몰 상품의 최근 6개월 판매량을 예측
- review의 daily_review_count와 info의 purchaseCnt(최근 6개월 판매량)을 활용하여 daily_sales(y)를 생성
  - (daily_review_count/특정날짜를 기준으로 180일 동안의 sum(daily_review_count))*purchaseCnt = 하루 판매량(하루 리뷰당 몇 개가 판매되었는가?)
  - daily_sales를 이러한 방법으로 구한 것은 review의 수가 판매량에 큰 영향을 미친다는 가정하에(하루 평균 몇 개의 리뷰가 달리면 몇 개가 팔릴 것이다.) 만들어낸 변수
- 이를 DNN 모델로 학습
- 저장된 모델을 로딩하여 item별 daily_sales를 predict 하고 특정 날짜 기준 180일의 daily_sales를 sum 해줌

### files

- train-test-generator_final : data preprocessing & training dataset generate
- DNN.py : DNN model
- kerastuner.py : DNN hyperparameter optimization
- TIPS_KPI_Predict.ipynb : predict real data

