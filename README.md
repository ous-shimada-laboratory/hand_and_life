# 高度手認識対応ライフゲーム（Advanced Hand Recognition Conway's Game of Life）

## 使用技術一覧

![Python](https://img.shields.io/badge/-Python-3776AB.svg?logo=python&style=for-the-badge)
![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8.svg?logo=opencv&style=for-the-badge)
![MediaPipe](https://img.shields.io/badge/-MediaPipe-4285F4.svg?logo=google&style=for-the-badge)
![NumPy](https://img.shields.io/badge/-NumPy-013243.svg?logo=numpy&style=for-the-badge)

## 目次

1. [プロジェクトについて](#プロジェクトについて)
2. [環境](#環境)
3. [ディレクトリ構成](#ディレクトリ構成)
4. [開発環境構築](#開発環境構築)
5. [使い方](#使い方)
6. [機能詳細](#機能詳細)
7. [トラブルシューティング](#トラブルシューティング)

MediaPipeによる高精度手認識機能とコンウェイのライフゲームを組み合わせた革新的なインタラクティブアプリケーションです。

## プロジェクトについて

このプロジェクトは、**MediaPipe**の21点ランドマーク追跡技術を使用して、リアルタイムで手の形状を詳細に検出し、その結果を**コンウェイのライフゲーム**の初期パターンとして活用する画期的なアプリケーションです。

Qiita記事『[Mediapipeで手の形状検出を試してみた(Python)](https://qiita.com/bianca26neve/items/...)』の手法をベースに、より高度な分析機能とライフゲームの組み合わせを実現しています。

### 🎯 **主な特徴**

- **🖐️ 高精度手認識**: 21点ランドマークによる詳細な手の形状追跡
- **🎨 美しい可視化**: 指別カラーコーディング、骨格描画、接続線表示
- **📊 詳細分析**: 左右判定、信頼度スコア、手のサイズ計算
- **🎮 ライフゲーム**: 1200x900の高解像度でパターン進化を観察
- **💾 データ保存**: JSON形式での検出データ、スクリーンショット保存
- **📈 リアルタイム統計**: FPS、検出率、信頼度の監視

### 🚀 **革新的な組み合わせ**

1. **手の形状キャプチャ** → カメラで手をかざす
2. **高精度認識** → MediaPipeで21点を追跡
3. **美しい可視化** → 骨格・指先接続線を描画
4. **ライフゲーム変換** → 手の形状を初期パターンに
5. **進化観察** → パターンの生命的進化を楽しむ

<p align="right">(<a href="#top">トップへ</a>)</p>

## 環境

| 言語・ライブラリ | バージョン | 説明 |
| -------------- | ---------- | ---- |
| Python | 3.8以上 | メイン言語 |
| OpenCV | 4.5以上 | 画像処理・カメラ制御 |
| MediaPipe | 0.8以上 | 手認識・ランドマーク追跡 |
| NumPy | 1.20以上 | 数値計算・配列操作 |

### 推奨システム要件

| 項目 | 推奨スペック |
| ---- | ---------- |
| OS | Windows 10/11, macOS 10.15+, Ubuntu 18.04+ |
| CPU | Intel Core i5 4世代以上 / AMD Ryzen 5以上 |
| メモリ | 8GB以上 |
| カメラ | Webカメラ（720p以上推奨） |
| GPU | 不要（CPU処理のみ） |

## ディレクトリ構成

```
hand_recognition_life_game/
├── enhanced_life_game.py          # メインプログラム
├── README.md                      # このファイル
├── requirements.txt               # 必要ライブラリ一覧
├── screenshots/                   # 自動生成：スクリーンショット保存
│   ├── captured_hand_advanced_*.png
│   ├── realtime_capture_*.png
│   └── advanced_life_game_*.png
├── detection_data/                # 自動生成：検出データ保存
│   ├── capture_data_*.json
│   ├── realtime_data_*.json
│   └── hand_detection_data_*.json
└── examples/                      # サンプル画像・動画（オプション）
    ├── sample_hands/
    └── demo_videos/
```

## 開発環境構築

### 1. Pythonの確認とインストール

```bash
# Pythonバージョン確認
python --version
# または
python3 --version

# Python 3.8以上が必要です
```

Python未インストールの場合：
- **Windows**: [Python公式サイト](https://www.python.org/downloads/)からダウンロード
- **Mac**: `brew install python3` または公式サイト
- **Ubuntu**: `sudo apt update && sudo apt install python3 python3-pip`

### 2. プロジェクトのクローン

```bash
git clone <repository-url>
cd hand_recognition_life_game
```

### 3. 仮想環境の作成（推奨）

```bash
# 仮想環境作成
python -m venv venv

# 仮想環境の有効化
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 4. 必要ライブラリのインストール

#### 方法1: requirements.txtを使用（推奨）

```bash
pip install -r requirements.txt
```

#### 方法2: 個別インストール

```bash
# 基本ライブラリ
pip install opencv-python>=4.5.0
pip install mediapipe>=0.8.0
pip install numpy>=1.20.0

# オプション: Jupyter Notebookでの実験用
pip install jupyter matplotlib
```

### 5. カメラの動作確認

```bash
# 簡単なカメラテスト
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.read()[0] else 'Camera Error'); cap.release()"
```

### 6. プログラム実行

```bash
python enhanced_life_game.py
```

## 使い方

### 🎬 基本的な流れ

1. **プログラム起動**
   ```bash
   python enhanced_life_game.py
   ```

2. **カメラ画面での操作**
   - 手をカメラに向ける
   - リアルタイムで骨格・接続線が表示される
   - **SPACE**キーで撮影 → ライフゲーム開始

3. **ライフゲーム画面**
   - 自動でパターンが進化
   - **SPACE**で一時停止/再開
   - **ESC**で終了

### 🎮 詳細操作方法

#### カメラ画面
| キー | 機能 | 説明 |
|------|------|------|
| **SPACE** | 撮影・ライフゲーム開始 | 現在の手の形状でライフゲーム開始 |
| **ESC** | 終了 | アプリケーション終了 |
| **h** | 手認識詳細情報 | 21点座標、信頼度など詳細表示 |
| **s** | 検出統計 | FPS、検出率、パフォーマンス表示 |
| **d** | データ保存 | 現在の検出データをJSON形式で保存 |
| **c** | キャリブレーション情報 | 最適な撮影環境のガイド表示 |
| **f** | フルスクリーン切り替え | 画面表示モード変更 |
| **r** | 統計リセット | 検出統計カウンターをリセット |

#### ライフゲーム画面
| キー | 機能 | 説明 |
|------|------|------|
| **SPACE** | 一時停止/再開 | ゲームの進行を制御 |
| **ESC** | 終了 | カメラ画面に戻る |
| **+/-** | 速度調整 | 進化速度を10ms～1000msで調整 |
| **r** | リセット | 初期パターンに戻す |
| **g** | グリッド表示切り替え | セルの境界線表示ON/OFF |
| **s** | スクリーンショット保存 | 現在の状態を画像保存 |
| **i** | 詳細情報表示 | 世代数、密度、パターン状態など |
| **d** | 検出データ保存 | 手認識データをJSON保存 |

## 機能詳細

### 🖐️ **高精度手認識機能**

#### **MediaPipe 21点ランドマーク追跡**
- 手首、各指の関節（計21ポイント）を正確に追跡
- 左右の手を自動判定（最大2つまで同時検出）
- 信頼度スコア（0.0-1.0）でどの結果精度を数値化

#### **視覚的表現**
- **指別カラーコーディング**: 
  - 🔴 親指（赤）、🟢 人差し指（緑）、🔵 中指（青）
  - 🟡 薬指（黄）、🟣 小指（マゼンタ）、⚪ 手のひら（白）
- **骨格描画**: 関節を線で繋いで手の構造を表示
- **指先強調**: 三重円で指先を美しく強調
- **接続線**: 指先同士を線で繋ぐアート的な表現

#### **詳細分析データ**
```json
{
  "hand_label": "Left",
  "confidence": 0.9999,
  "hand_size": 0.234,
  "palm_center": [0.456, 0.789],
  "fingertip_coords": [
    {"finger": "THUMB_TIP", "x": 0.123, "y": 0.456, "z": -0.002}
  ]
}
```

### 🎮 **高解像度ライフゲーム**

#### **コンウェイのライフゲームルール**
1. **誕生**: 死んだセルの周囲に3つの生きたセルがあると誕生
2. **生存**: 生きたセルの周囲に2または3つの生きたセルがあると生存
3. **死滅**: その他の場合は死滅

#### **高度な機能**
- **解像度**: 1200x900ピクセルの高解像度描画
- **パターン分析**: 安定/振動/進化中の自動判定
- **統計表示**: 世代数、生存セル数、密度
- **履歴追跡**: 過去10世代のパターンを記録

#### **プリセットパターン**
カメラが使用できない場合の代替パターン：
1. **グライダー**: 斜めに移動する基本パターン
2. **振動パターン**: 一定周期で形状が変化
3. **パルサー**: 複雑な15周期振動
4. **ランダム**: カオス的な進化パターン

### 💾 **データ保存・分析機能**

#### **自動保存される情報**
- **スクリーンショット**: PNG形式（タイムスタンプ付き）
- **検出データ**: JSON形式の詳細情報
- **統計情報**: FPS、検出率、信頼度履歴

#### **保存ファイル例**
```
screenshots/
├── captured_hand_advanced_20241201_143022.png
├── realtime_capture_20241201_143156.png
└── advanced_life_game_20241201_143234.png

detection_data/
├── capture_data_20241201_143022.json
├── realtime_data_20241201_143156.json
└── hand_detection_data_20241201_143234.json
```

<p align="right">(<a href="#top">トップへ</a>)</p>

## トラブルシューティング

### 🚨 **よくある問題と解決策**

#### **1. MediaPipeインストールエラー**
```bash
# エラー例: "No module named 'mediapipe'"
# 解決策:
pip install --upgrade pip
pip install mediapipe

# M1/M2 Macの場合:
pip install mediapipe-silicon
```

#### **2. カメラが認識されない**
```python
# 確認コマンド:
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.read()[0]:
        print(f'Camera {i}: OK')
    cap.release()
"
```

**対処法:**
- カメラのプライバシー設定を確認
- 他のアプリケーションがカメラを使用していないか確認
- USBカメラの場合は別のUSBポートを試す

#### **3. 手が認識されない**
**環境を改善:**
- ✅ 明るい照明を確保
- ✅ 背景をシンプルに（単色推奨）
- ✅ カメラから30-60cm程度の距離
- ✅ 手全体がフレーム内に入るよう調整
- ❌ 手袋は外す
- ❌ 暗い場所や複雑な背景は避ける

#### **4. 動作が重い・FPSが低い**
**パフォーマンス改善:**
```python
# カメラ解像度を下げる
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# MediaPipeの設定を調整
hands = mp_hands.Hands(
    min_detection_confidence=0.5,  # 0.7 → 0.5に下げる
    min_tracking_confidence=0.3    # 0.5 → 0.3に下げる
)
```

#### **5. ライフゲームが正しく動作しない**
- 手の形状が単純すぎる場合は複雑なポーズを試す
- プリセットパターンで動作確認
- 解像度を下げて負荷を軽減

### 🐛 **デバッグ情報の取得**

```bash
# 詳細ログ付きで実行
python enhanced_life_game.py --verbose

# システム情報確認
python -c "
import cv2, sys
print(f'Python: {sys.version}')
print(f'OpenCV: {cv2.__version__}')
try:
    import mediapipe as mp
    print(f'MediaPipe: {mp.__version__}')
except:
    print('MediaPipe: Not installed')
"
```

### 📞 **サポート**

問題が解決しない場合：
1. **Issues報告**: GitHubのIssuesで報告
2. **ログ情報**: エラーメッセージと環境情報を添付
3. **再現手順**: 問題の発生手順を詳しく記載

<p align="right">(<a href="#top">トップへ</a>)</p>

## 参考文献・謝辞

このプロジェクトは以下の優れた記事・技術を参考にしています：

- 📝 **[Mediapipeで手の形状検出を試してみた(Python)](https://qiita.com/bianca26neve/items/...)** by @bianca26neve on Qiita
  - 21点ランドマーク追跡の基本実装
  - 左右判定と信頼度スコアの活用方法
  - 検出データの分析手法

- 🛠️ **[MediaPipe公式ドキュメント](https://google.github.io/mediapipe/)**
  - Hand Landmarkの詳細仕様
  - パフォーマンス最適化のガイドライン

- 🎲 **コンウェイのライフゲーム**
  - 1970年にジョン・ホートン・コンウェイが考案
  - セルオートマトンの古典的な例

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 問い合わせ先

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝<br>
株式会社アニメツーリズム<br>
取締役CTO　担当：川上<br>
sota@jam-info.com<br>

株式会社アニメツーリズム・JapanAnimeMaps運営<br>
お問い合わせ<br>
contact@animetourism.co.jp<br>

公式サイト<br>
https://animetourism.co.jp<br>

お問い合わせフォーム<br>
https://animetourism.co.jp/contact.html<br>

アプリはこちら<br>
https://apps.apple.com/jp/app/japananimemaps/id6608967051<br>

〒150-0043<br>
東京都 渋谷区道玄坂1丁目10番8号渋谷道玄坂東急ビル2F-C<br>
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝<br>
※本文に掲載された記事を許可なく転載することを禁じます。<br>
(c)2024 JapanAnimeMaps.All Rights Reserved.<br>

<p align="right">(<a href="#top">トップへ</a>)</p>