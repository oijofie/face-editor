# Face Editor
動画内にある任意の顔を、任意の画像で上書きします

## スクリプトの動かし方(ryeで動かします)
1. ryeで環境構築するために、ターミナルで以下を押下
    ```bash
    rye sync
    ```

2. ryeでスクリプトを動かす
    ```bash
    rye run python main.py
    ```

## 動画の上書き方法（スクリプト実行後の作業）

1. ポップアップがでるので、上書きしたい元動画を指定
2. ポップアップがでるので、上書きする画像を指定
3. 検知された顔の上に番号が振り分けられた画像がポップアップされるので、上書きしたい顔の番号を押下
(上書きしない場合は9を押下。ポップアップが出るたびにこれを繰り返す)