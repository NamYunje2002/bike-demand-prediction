# Seoul Bike Rental Demand Prediction
날씨 및 시간 정보를 기반으로 서울 자전거 대여 수요를 예측합니다.   
PyTorch 기반 딥러닝 모델(MLP)과 머신러닝 모델(선형 회귀, 결정 트리 회귀)을 비교하여 성능을 분석합니다.


## 프로젝트 개요
- **데이터셋**: UCI Machine Learning Repository - [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
- **예측 대상**: 시간 단위 자전거 대여량 (`Rented_Bike_Count`)
- **사용 모델**:
  - MLP (PyTorch)
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest
  - Gradient Boosting
  - AdaBoost
- **성능 지표**:
  - MSE
  - MAE
  - R2
- **전처리**:
  - 컬럼 제거 (`Date`, `Functioning Day`)
  - `MinMaxScaler`를 통한 입력 정규화