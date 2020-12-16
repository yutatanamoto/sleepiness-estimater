# 機械学習を用いた眠気予測  
## 機械学習手法  
- サポートベクターマシン (Support Vector Machine, 以降SVM)
- パラメータはデフォルト値を使用 ([Scikit-learn official document](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html))

## 使用データ  
- 心拍変動データ (RRI)
- データを[-1.0, 1.0]に標準化
- 全データを訓練データとテストデータに分割
    - 8割を訓練, 2割をテスト

## モデル訓練/テスト  
- 訓練データでSVMモデルを訓練
- テスト精度は95.8%

## プログラム  
- [これ](../src/svc_sample.py)