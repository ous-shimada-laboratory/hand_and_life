import cv2
import numpy as np
import time
import os
import json
from datetime import datetime

# MediaPipeのインポートをtry-catchで囲む
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipeがインストールされていません。基本的な肌色検出のみ使用します。")
    print("pip install mediapipe でインストールできます。")

# ===== 改良された手と指の検出・分析クラス =====
class AdvancedHandTracker:
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            self.hands = None
            return
            
        # MediaPipe初期化
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 記事を参考にした詳細設定
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,  # 記事より少し高めに設定
            min_tracking_confidence=0.5    # 記事と同じ値
        )
        
        # 21個のランドマークの名前リスト（記事より）
        self.landmark_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
        
        # 指先のランドマークID
        self.fingertip_ids = [4, 8, 12, 16, 20]  # THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP
        
        # 指の関節ID（各指の骨格）
        self.finger_joints = {
            'thumb': [1, 2, 3, 4],      # CMC, MCP, IP, TIP
            'index': [5, 6, 7, 8],      # MCP, PIP, DIP, TIP
            'middle': [9, 10, 11, 12],  # MCP, PIP, DIP, TIP
            'ring': [13, 14, 15, 16],   # MCP, PIP, DIP, TIP
            'pinky': [17, 18, 19, 20]   # MCP, PIP, DIP, TIP
        }
        
        # 手のひらの接続（記事を参考に詳細化）
        self.palm_connections = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
        
        # 指ごとの色（より鮮やかに）
        self.finger_colors = {
            'thumb': (255, 100, 100),    # 明るい赤
            'index': (100, 255, 100),    # 明るい緑
            'middle': (100, 100, 255),   # 明るい青
            'ring': (255, 255, 100),     # 明るい黄
            'pinky': (255, 100, 255),    # 明るいマゼンタ
            'palm': (255, 255, 255)      # 白
        }
        
        # 検出統計用
        self.detection_stats = {
            'total_frames': 0,
            'hands_detected': 0,
            'left_hands': 0,
            'right_hands': 0,
            'confidence_scores': []
        }
    
    def detect_hands(self, image):
        """手の検出を実行し、詳細な結果を返す"""
        if not MEDIAPIPE_AVAILABLE or self.hands is None:
            return None
            
        # 記事と同じ色変換処理
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        # 統計更新
        self.detection_stats['total_frames'] += 1
        if results.multi_hand_landmarks:
            self.detection_stats['hands_detected'] += len(results.multi_hand_landmarks)
            
            # 左右の手の統計
            if results.multi_handedness:
                for handedness in results.multi_handedness:
                    if handedness.classification[0].label == 'Left':
                        self.detection_stats['left_hands'] += 1
                    else:
                        self.detection_stats['right_hands'] += 1
                    
                    # 信頼度スコア記録
                    confidence = handedness.classification[0].score
                    self.detection_stats['confidence_scores'].append(confidence)
        
        return results
    
    def analyze_hand_landmarks(self, hand_landmarks, handedness_info=None):
        """手のランドマークを詳細分析（記事の分析手法を参考）"""
        analysis = {
            'landmark_coords': [],
            'fingertip_coords': [],
            'hand_label': 'Unknown',
            'confidence': 0.0,
            'finger_angles': {},
            'hand_size': 0.0,
            'palm_center': (0, 0)
        }
        
        # ランドマーク座標を取得
        for i, landmark in enumerate(hand_landmarks.landmark):
            coord_info = {
                'id': i,
                'name': self.landmark_names[i],
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            }
            analysis['landmark_coords'].append(coord_info)
        
        # 指先座標を特別に取得
        for tip_id in self.fingertip_ids:
            tip = hand_landmarks.landmark[tip_id]
            analysis['fingertip_coords'].append({
                'finger': self.landmark_names[tip_id],
                'x': tip.x,
                'y': tip.y,
                'z': tip.z
            })
        
        # 左右判定情報
        if handedness_info:
            analysis['hand_label'] = handedness_info.classification[0].label
            analysis['confidence'] = handedness_info.classification[0].score
        
        # 手のサイズ計算（手首から中指先端までの距離）
        wrist = hand_landmarks.landmark[0]
        middle_tip = hand_landmarks.landmark[12]
        analysis['hand_size'] = np.sqrt(
            (middle_tip.x - wrist.x)**2 + 
            (middle_tip.y - wrist.y)**2
        )
        
        # 手のひら中心計算
        palm_landmarks = [0, 5, 9, 13, 17]  # 手首と各指の付け根
        palm_x = sum(hand_landmarks.landmark[i].x for i in palm_landmarks) / len(palm_landmarks)
        palm_y = sum(hand_landmarks.landmark[i].y for i in palm_landmarks) / len(palm_landmarks)
        analysis['palm_center'] = (palm_x, palm_y)
        
        return analysis
    
    def draw_detailed_hand_skeleton(self, image, hand_landmarks, handedness_info=None):
        """記事を参考にした詳細な手の骨格描画"""
        if not MEDIAPIPE_AVAILABLE:
            return [], {}
            
        h, w, _ = image.shape
        
        # ランドマークを画像座標に変換
        landmarks = []
        for lm in hand_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))
        
        # 手の詳細分析
        analysis = self.analyze_hand_landmarks(hand_landmarks, handedness_info)
        
        # 指ごとに骨格を描画（記事の描画方法を改良）
        for finger_name, joints in self.finger_joints.items():
            color = self.finger_colors[finger_name]
            
            # 指の骨格を線で繋ぐ（太さを関節ごとに変化）
            for i in range(len(joints) - 1):
                start_point = landmarks[joints[i]]
                end_point = landmarks[joints[i + 1]]
                
                # 関節の重要度に応じて太さを変更
                thickness = 5 if i == 0 else 4 if i == 1 else 3
                cv2.line(image, start_point, end_point, color, thickness)
                
                # 関節点を円で描画（サイズも変化）
                radius = 7 if i == 0 else 6 if i == 1 else 5
                cv2.circle(image, start_point, radius, color, -1)
                cv2.circle(image, end_point, radius, color, -1)
        
        # 手のひらの接続を描画
        palm_color = self.finger_colors['palm']
        for connection in self.palm_connections:
            start_point = landmarks[connection[0]]
            end_point = landmarks[connection[1]]
            cv2.line(image, start_point, end_point, palm_color, 3)
        
        # 指先を特別に強調（記事より詳細化）
        for i, tip_id in enumerate(self.fingertip_ids):
            tip_point = landmarks[tip_id]
            finger_name = list(self.finger_joints.keys())[i]
            finger_color = self.finger_colors[finger_name]
            
            # 三重円で強調
            cv2.circle(image, tip_point, 15, (0, 255, 255), -1)    # 最外 - 黄色
            cv2.circle(image, tip_point, 12, finger_color, -1)      # 中間 - 指色
            cv2.circle(image, tip_point, 8, (255, 255, 255), -1)    # 内側 - 白
            cv2.circle(image, tip_point, 18, (0, 0, 0), 2)          # 黒い外枠
        
        # 手のラベルと信頼度を描画
        if handedness_info:
            label = handedness_info.classification[0].label
            confidence = handedness_info.classification[0].score
            
            # 手首付近にラベル表示
            wrist_point = landmarks[0]
            label_pos = (wrist_point[0] - 50, wrist_point[1] - 30)
            
            cv2.putText(image, f"{label} Hand", label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Conf: {confidence:.3f}", 
                       (label_pos[0], label_pos[1] + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        
        return landmarks, analysis
    
    def draw_advanced_fingertip_connections(self, image, landmarks_list, analysis_list):
        """複数の手の指先を高度に接続する描画"""
        if len(landmarks_list) == 0:
            return
        
        # 単一の手の場合
        if len(landmarks_list) == 1:
            landmarks = landmarks_list[0]
            self._draw_single_hand_connections(image, landmarks)
        
        # 複数の手の場合（記事では最大2つまで検出）
        elif len(landmarks_list) == 2:
            self._draw_multi_hand_connections(image, landmarks_list, analysis_list)
    
    def _draw_single_hand_connections(self, image, landmarks):
        """単一の手の指先接続パターン"""
        fingertips = [landmarks[i] for i in self.fingertip_ids]
        
        # 基本的な多角形接続
        for i in range(len(fingertips)):
            start_point = fingertips[i]
            end_point = fingertips[(i + 1) % len(fingertips)]
            cv2.line(image, start_point, end_point, (255, 255, 0), 3)
        
        # 星型パターン（中心から放射）
        if len(fingertips) >= 3:
            center_x = sum(p[0] for p in fingertips) // len(fingertips)
            center_y = sum(p[1] for p in fingertips) // len(fingertips)
            center_point = (center_x, center_y)
            
            for i, tip in enumerate(fingertips):
                color = list(self.finger_colors.values())[i]
                cv2.line(image, center_point, tip, color, 2)
            
            cv2.circle(image, center_point, 10, (0, 255, 255), -1)
    
    def _draw_multi_hand_connections(self, image, landmarks_list, analysis_list):
        """複数の手の間の接続パターン"""
        # 各手の指先を取得
        all_fingertips = []
        hand_centers = []
        
        for landmarks, analysis in zip(landmarks_list, analysis_list):
            fingertips = [landmarks[i] for i in self.fingertip_ids]
            all_fingertips.extend(fingertips)
            
            # 手の中心も計算
            center_x = sum(p[0] for p in fingertips) // len(fingertips)
            center_y = sum(p[1] for p in fingertips) // len(fingertips)
            hand_centers.append((center_x, center_y))
        
        # 左右の手の中心を接続
        if len(hand_centers) == 2:
            cv2.line(image, hand_centers[0], hand_centers[1], (255, 0, 255), 4)
            
            # 中点も描画
            mid_x = (hand_centers[0][0] + hand_centers[1][0]) // 2
            mid_y = (hand_centers[0][1] + hand_centers[1][1]) // 2
            cv2.circle(image, (mid_x, mid_y), 8, (255, 0, 255), -1)
        
        # 対応する指先同士を接続（親指同士、人差し指同士など）
        if len(landmarks_list) == 2:
            for i in range(len(self.fingertip_ids)):
                tip1 = landmarks_list[0][self.fingertip_ids[i]]
                tip2 = landmarks_list[1][self.fingertip_ids[i]]
                color = list(self.finger_colors.values())[i]
                cv2.line(image, tip1, tip2, color, 2, cv2.LINE_AA)
    
    def save_detection_data(self, landmarks_list, analysis_list, frame_number):
        """検出データを記事のような形式で保存"""
        detection_data = {
            'frame_number': frame_number,
            'timestamp': datetime.now().isoformat(),
            'hands_count': len(landmarks_list),
            'hands_data': []
        }
        
        for i, (landmarks, analysis) in enumerate(zip(landmarks_list, analysis_list)):
            hand_data = {
                'hand_id': i,
                'hand_label': analysis['hand_label'],
                'confidence': analysis['confidence'],
                'hand_size': analysis['hand_size'],
                'palm_center': analysis['palm_center'],
                'landmarks': []
            }
            
            # ランドマーク情報を保存
            for coord_info in analysis['landmark_coords']:
                hand_data['landmarks'].append(coord_info)
            
            detection_data['hands_data'].append(hand_data)
        
        return detection_data
    
    def get_detection_statistics(self):
        """検出統計情報を取得"""
        if self.detection_stats['total_frames'] == 0:
            return {}
        
        avg_confidence = 0
        if self.detection_stats['confidence_scores']:
            avg_confidence = sum(self.detection_stats['confidence_scores']) / len(self.detection_stats['confidence_scores'])
        
        detection_rate = (self.detection_stats['hands_detected'] / self.detection_stats['total_frames']) * 100
        
        return {
            'total_frames': self.detection_stats['total_frames'],
            'hands_detected': self.detection_stats['hands_detected'],
            'left_hands': self.detection_stats['left_hands'],
            'right_hands': self.detection_stats['right_hands'],
            'detection_rate': detection_rate,
            'average_confidence': avg_confidence,
            'confidence_scores': self.detection_stats['confidence_scores'][-10:]  # 最新10個
        }

# ===== 改良された肌色検出（記事の手法と組み合わせ）=====
def enhanced_skin_mask_with_analysis(img, hand_tracker):
    """記事の分析結果も含めた改良版肌色検出"""
    # 従来の肌色検出（複数の色空間）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # HSVでの肌色範囲（記事を参考に最適化）
    lower1 = np.array([0, 30, 60], dtype=np.uint8)
    upper1 = np.array([20, 150, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    
    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 150, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    
    # YCrCbでの肌色検出
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    mask3 = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # 基本の肌色マスク
    skin_mask = cv2.bitwise_or(mask1, mask2)
    skin_mask = cv2.bitwise_or(skin_mask, mask3)
    
    # 高精度手認識結果
    hand_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    landmarks_list = []
    analysis_list = []
    detection_info = {
        'hands_detected': 0,
        'left_hands': 0,
        'right_hands': 0,
        'total_confidence': 0,
        'hand_analyses': []
    }
    
    if MEDIAPIPE_AVAILABLE and hand_tracker.hands is not None:
        results = hand_tracker.detect_hands(img)
        
        if results.multi_hand_landmarks:
            detection_info['hands_detected'] = len(results.multi_hand_landmarks)
            
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 左右判定情報を取得
                handedness_info = None
                if results.multi_handedness and i < len(results.multi_handedness):
                    handedness_info = results.multi_handedness[i]
                    label = handedness_info.classification[0].label
                    confidence = handedness_info.classification[0].score
                    
                    if label == 'Left':
                        detection_info['left_hands'] += 1
                    else:
                        detection_info['right_hands'] += 1
                    
                    detection_info['total_confidence'] += confidence
                
                # 詳細な手の骨格を描画
                landmarks, analysis = hand_tracker.draw_detailed_hand_skeleton(
                    img, hand_landmarks, handedness_info
                )
                
                landmarks_list.append(landmarks)
                analysis_list.append(analysis)
                detection_info['hand_analyses'].append(analysis)
                
                # より正確な手の形状マスクを作成
                individual_mask = create_precise_hand_mask(img, hand_landmarks)
                hand_mask = cv2.bitwise_or(hand_mask, individual_mask)
        
        # 高度な指先接続を描画
        if landmarks_list:
            hand_tracker.draw_advanced_fingertip_connections(img, landmarks_list, analysis_list)
    
    # マスクの結合と最適化
    if MEDIAPIPE_AVAILABLE:
        # 手認識結果を優先的に使用
        combined_mask = cv2.bitwise_or(skin_mask, hand_mask)
        # 手認識の精度が高い場合は、それを重視
        if detection_info['hands_detected'] > 0:
            avg_confidence = detection_info['total_confidence'] / detection_info['hands_detected']
            if avg_confidence > 0.8:  # 高信頼度の場合
                combined_mask = cv2.addWeighted(hand_mask, 0.7, skin_mask, 0.3, 0)
    else:
        combined_mask = skin_mask
    
    # ノイズ除去（記事を参考に最適化）
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # ガウシアンブラーで滑らかに
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    
    # 詳細情報をオーバーレイ
    display_detection_info(img, detection_info)
    
    binary_img = cv2.merge([combined_mask, combined_mask, combined_mask])
    return binary_img, detection_info, landmarks_list, analysis_list

def create_precise_hand_mask(image, hand_landmarks):
    """記事を参考にした高精度手マスク作成"""
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # ランドマークから手の輪郭を作成
    landmarks = []
    for lm in hand_landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        landmarks.append([x, y])
    
    # より正確な手の輪郭を定義（記事の手法を改良）
    # 手の外側の輪郭点を順番に接続
    hand_outline = [
        landmarks[0],   # 手首
        landmarks[1], landmarks[2], landmarks[3], landmarks[4],  # 親指
        landmarks[8],   # 人差し指先
        landmarks[12],  # 中指先
        landmarks[16],  # 薬指先
        landmarks[20],  # 小指先
        landmarks[19], landmarks[18], landmarks[17],  # 小指側面
        landmarks[5],   # 人差し指付け根
        landmarks[0]    # 手首に戻る
    ]
    
    # より滑らかな輪郭のために点を補間
    pts = np.array(hand_outline, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    # 指の間の隙間も埋めるために、各指の周りに円を描画
    finger_tip_ids = [4, 8, 12, 16, 20]
    for tip_id in finger_tip_ids:
        center = tuple(landmarks[tip_id])
        cv2.circle(mask, center, 20, 255, -1)
    
    # 手のひら部分も確実に塗りつぶし
    palm_center_ids = [0, 5, 9, 13, 17]
    palm_points = [landmarks[i] for i in palm_center_ids]
    palm_pts = np.array(palm_points, dtype=np.int32)
    cv2.fillPoly(mask, [palm_pts], 255)
    
    return mask

def display_detection_info(image, detection_info):
    """検出情報を画像にオーバーレイ表示"""
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # 基本情報
    cv2.putText(image, f"Hands: {detection_info['hands_detected']}", 
               (10, y_offset), font, font_scale, (0, 255, 0), thickness)
    y_offset += 30
    
    if detection_info['hands_detected'] > 0:
        # 左右の手の情報
        cv2.putText(image, f"Left: {detection_info['left_hands']}, Right: {detection_info['right_hands']}", 
                   (10, y_offset), font, 0.6, (0, 255, 255), thickness)
        y_offset += 25
        
        # 平均信頼度
        avg_conf = detection_info['total_confidence'] / detection_info['hands_detected']
        cv2.putText(image, f"Avg Confidence: {avg_conf:.3f}", 
                   (10, y_offset), font, 0.6, (255, 255, 0), thickness)
        y_offset += 25
        
        # 詳細分析結果
        for i, analysis in enumerate(detection_info['hand_analyses']):
            cv2.putText(image, f"Hand {i+1}: {analysis['hand_label']} (Size: {analysis['hand_size']:.3f})", 
                       (10, y_offset), font, 0.5, (200, 200, 200), 1)
            y_offset += 20

# ===== 使用可能なカメラを探す =====
def find_available_camera():
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            cap.release()
            return i
        cap.release()
    return -1

# ===== ライフゲーム関連機能（変更なし、記事の品質向上を適用）=====
def game_of_life_step(board):
    """高速化されたライフゲームの1ステップ計算"""
    rows, cols = board.shape
    padded = np.pad(board, pad_width=1, mode='constant', constant_values=0)
    
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbors = cv2.filter2D(padded, -1, kernel)[1:-1, 1:-1]
    
    new_board = np.zeros_like(board)
    new_board[(board == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
    new_board[(board == 0) & (neighbors == 3)] = 1
    
    return new_board.astype(np.uint8)

def draw_board(board, scale=2, show_grid=False):
    """高解像度ライフゲームの盤面描画"""
    h, w = board.shape
    img = np.zeros((h*scale, w*scale, 3), dtype=np.uint8)
    
    # より美しいセル描画
    for i in range(h):
        for j in range(w):
            if board[i, j] == 1:
                y1, y2 = i*scale, (i+1)*scale
                x1, x2 = j*scale, (j+1)*scale
                
                # グラデーション効果
                center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
                cv2.circle(img, (center_x, center_y), scale//2, [255, 255, 255], -1)
                
                if scale >= 4:
                    cv2.circle(img, (center_x, center_y), max(1, scale//4), [255, 255, 255], -1)
    
    # グリッド線描画
    if show_grid and scale >= 3:
        grid_color = (64, 64, 64)
        for i in range(0, h*scale, scale):
            cv2.line(img, (0, i), (w*scale, i), grid_color, 1)
        for j in range(0, w*scale, scale):
            cv2.line(img, (j, 0), (j, h*scale), grid_color, 1)
    
    return img

def analyze_pattern(board, generation, history=None):
    """パターン分析と統計"""
    alive_cells = np.sum(board)
    total_cells = board.shape[0] * board.shape[1]
    density = alive_cells / total_cells * 100
    
    is_stable = False
    is_oscillating = False
    
    if history is not None and len(history) > 1:
        # 前の世代と比較
        if np.array_equal(board, history[-1]):
            is_stable = True
        
        # 振動パターンチェック
        if len(history) >= 3:
            for i in range(2, min(len(history), 6)):
                if np.array_equal(board, history[-i]):
                    is_oscillating = True
                    break
    
    return {
        'generation': generation,
        'alive_cells': int(alive_cells),
        'total_cells': total_cells,
        'density': density,
        'is_stable': is_stable,
        'is_oscillating': is_oscillating
    }

# ===== 高度なライフゲーム実行（記事品質での分析機能付き）=====
def run_advanced_life_game_from_array(mask, detection_info, landmarks_list, analysis_list, resolution=(1000, 800), speed=60):
    """記事レベルの詳細分析機能付きライフゲーム"""
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, resolution, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(resized, 50, 1, cv2.THRESH_BINARY)
    
    print(f"🎮 高度分析ライフゲーム開始！({resolution[0]}x{resolution[1]})")
    print("=" * 60)
    print("手認識分析結果:")
    if detection_info['hands_detected'] > 0:
        print(f"  検出された手: {detection_info['hands_detected']}個")
        print(f"  左手: {detection_info['left_hands']}個, 右手: {detection_info['right_hands']}個")
        avg_conf = detection_info['total_confidence'] / detection_info['hands_detected']
        print(f"  平均信頼度: {avg_conf:.3f}")
        
        for i, analysis in enumerate(analysis_list):
            print(f"  手{i+1}: {analysis['hand_label']}, サイズ: {analysis['hand_size']:.3f}")
    else:
        print("  手は検出されませんでした（肌色検出のみ使用）")
    
    print("=" * 60)
    print("操作方法:")
    print("  ESC: 終了")
    print("  SPACE: 一時停止/再開")
    print("  +/-: 速度調整")
    print("  r: リセット")
    print("  g: グリッド表示切り替え")
    print("  s: スクリーンショット保存")
    print("  i: 詳細情報表示")
    print("  d: 検出データ保存")
    print("  h: ヘルプ表示")
    print("=" * 60)
    
    original_binary = binary.copy()
    generation = 0
    paused = False
    show_grid = False
    current_speed = speed
    stats_height = 160
    
    # 履歴保存（パターン検出用）
    history = []
    saved_data = []
    
    while True:
        if not paused:
            binary = game_of_life_step(binary)
            generation += 1
            
            # 履歴を保存（最大10世代）
            history.append(binary.copy())
            if len(history) > 10:
                history.pop(0)
        
        # 統計分析
        stats = analyze_pattern(binary, generation, history)
        
        # メイン画面描画
        img = draw_board(binary, scale=2, show_grid=show_grid)
        
        # 詳細統計情報パネル
        stats_img = np.zeros((stats_height, img.shape[1], 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 1
        
        # 基本統計（左列）
        cv2.putText(stats_img, f"Generation: {stats['generation']}", 
                   (10, 25), font, font_scale, color, thickness)
        cv2.putText(stats_img, f"Alive Cells: {stats['alive_cells']}/{stats['total_cells']}", 
                   (10, 50), font, font_scale, color, thickness)
        cv2.putText(stats_img, f"Density: {stats['density']:.2f}%", 
                   (10, 75), font, font_scale, color, thickness)
        
        # 手認識情報
        detection_text = f"Hand Detection: {'ADVANCED' if MEDIAPIPE_AVAILABLE else 'BASIC'}"
        detection_color = (0, 255, 0) if MEDIAPIPE_AVAILABLE else (255, 255, 0)
        cv2.putText(stats_img, detection_text, 
                   (10, 100), font, font_scale, detection_color, thickness)
        
        # 検出された手の詳細情報
        if detection_info['hands_detected'] > 0:
            hands_text = f"Hands: {detection_info['hands_detected']} (L:{detection_info['left_hands']}, R:{detection_info['right_hands']})"
            cv2.putText(stats_img, hands_text, 
                       (10, 125), font, 0.5, (0, 255, 255), thickness)
            
            avg_conf = detection_info['total_confidence'] / detection_info['hands_detected']
            cv2.putText(stats_img, f"Confidence: {avg_conf:.3f}", 
                       (10, 145), font, 0.5, (255, 255, 0), thickness)
        
        # パターン状態（中央列）
        pattern_x = 350
        if stats['is_stable']:
            cv2.putText(stats_img, "Pattern: STABLE", 
                       (pattern_x, 25), font, font_scale, (0, 255, 255), thickness)
        elif stats['is_oscillating']:
            cv2.putText(stats_img, "Pattern: OSCILLATING", 
                       (pattern_x, 25), font, font_scale, (255, 0, 255), thickness)
        else:
            cv2.putText(stats_img, "Pattern: EVOLVING", 
                       (pattern_x, 25), font, font_scale, (255, 255, 0), thickness)
        
        # 右側の操作情報
        cv2.putText(stats_img, f"Speed: {current_speed}ms", 
                   (pattern_x, 50), font, font_scale, color, thickness)
        status_color = (0, 255, 255) if paused else (0, 255, 0)
        cv2.putText(stats_img, f"Status: {'PAUSED' if paused else 'RUNNING'}", 
                   (pattern_x, 75), font, font_scale, status_color, thickness)
        cv2.putText(stats_img, f"Grid: {'ON' if show_grid else 'OFF'}", 
                   (pattern_x, 100), font, font_scale, color, thickness)
        
        resolution_text = f"Resolution: {resolution[0]}x{resolution[1]}"
        cv2.putText(stats_img, resolution_text, 
                   (pattern_x, 125), font, font_scale, color, thickness)
        
        # データ保存情報
        cv2.putText(stats_img, f"Saved Data: {len(saved_data)} frames", 
                   (pattern_x, 145), font, 0.5, (200, 200, 200), thickness)
        
        # 画像を結合
        combined_img = np.vstack((img, stats_img))
        
        # ウィンドウに表示
        cv2.imshow("Advanced Hand-Enhanced Conway's Game of Life", combined_img)
        
        # キー入力処理
        key = cv2.waitKey(current_speed) & 0xFF
        
        if key == 27:  # ESC - 終了
            print("ライフゲーム終了")
            break
        elif key == 32:  # SPACE - 一時停止/再開
            paused = not paused
            print(f"{'一時停止' if paused else '再開'}")
        elif key == ord('r'):  # r - リセット
            binary = original_binary.copy()
            generation = 0
            paused = False
            history.clear()
            saved_data.clear()
            print("ゲームをリセットしました")
        elif key == ord('g'):  # g - グリッド表示切り替え
            show_grid = not show_grid
            print(f"グリッド表示: {'ON' if show_grid else 'OFF'}")
        elif key == ord('+') or key == ord('='):  # + - 速度上げる
            current_speed = max(10, current_speed - 20)
            print(f"速度: {current_speed}ms")
        elif key == ord('-'):  # - - 速度下げる
            current_speed = min(1000, current_speed + 20)
            print(f"速度: {current_speed}ms")
        elif key == ord('s'):  # s - スクリーンショット保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"advanced_life_game_{timestamp}_gen_{generation}.png"
            cv2.imwrite(filename, combined_img)
            print(f"📸 スクリーンショット保存: {filename}")
        elif key == ord('d'):  # d - 検出データ保存
            if MEDIAPIPE_AVAILABLE and landmarks_list:
                hand_tracker = AdvancedHandTracker()  # 一時的なインスタンス
                data = hand_tracker.save_detection_data(landmarks_list, analysis_list, generation)
                saved_data.append(data)
                
                # JSONファイルとして保存
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_filename = f"hand_detection_data_{timestamp}.json"
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(saved_data, f, indent=2, ensure_ascii=False)
                print(f"💾 検出データ保存: {json_filename}")
            else:
                print("⚠️ 保存できる検出データがありません")
        elif key == ord('i'):  # i - 詳細情報
            print(f"\n=== 詳細分析情報 (Generation {generation}) ===")
            print(f"生存セル数: {stats['alive_cells']}")
            print(f"総セル数: {stats['total_cells']}")
            print(f"密度: {stats['density']:.2f}%")
            print(f"パターン状態: {'安定' if stats['is_stable'] else '振動' if stats['is_oscillating'] else '進化中'}")
            print(f"手認識モード: {'高度分析' if MEDIAPIPE_AVAILABLE else '基本モード'}")
            
            if detection_info['hands_detected'] > 0:
                print("\n手認識詳細:")
                for i, analysis in enumerate(analysis_list):
                    print(f"  手 {i+1}:")
                    print(f"    ラベル: {analysis['hand_label']}")
                    print(f"    信頼度: {analysis['confidence']:.3f}")
                    print(f"    手のサイズ: {analysis['hand_size']:.3f}")
                    print(f"    手のひら中心: ({analysis['palm_center'][0]:.3f}, {analysis['palm_center'][1]:.3f})")
                    print(f"    指先座標数: {len(analysis['fingertip_coords'])}")
        elif key == ord('h'):  # h - ヘルプ
            print("\n" + "=" * 70)
            print("🖐️ 高度手認識ライフゲーム - ヘルプ")
            print("=" * 70)
            print("このプログラムは記事『Mediapipeで手の形状検出を試してみた』の手法を")
            print("ベースに、より高度な手認識とライフゲームの組み合わせを実現しています。")
            print()
            print("📊 分析機能:")
            print("  • 21点ランドマーク追跡")
            print("  • 左右の手の自動判定")
            print("  • 信頼度スコア表示")
            print("  • 手のサイズ・中心座標計算")
            print("  • 指先座標の詳細追跡")
            print()
            print("🎮 ライフゲーム機能:")
            print("  • パターン状態の自動判定（安定/振動/進化）")
            print("  • リアルタイム統計表示")
            print("  • 高解像度描画")
            print("  • データ保存機能")
            print()
            print("💾 保存される情報:")
            print("  • スクリーンショット（s キー）")
            print("  • 手認識データ（d キー、JSON形式）")
            print("  • ランドマーク座標")
            print("  • 信頼度スコア")
            print("=" * 70)
    
    cv2.destroyAllWindows()

# ===== プリセットパターン生成 =====
def create_preset_pattern(pattern_name, size=(1000, 800)):
    """高品質プリセットパターン生成"""
    board = np.zeros(size[::-1], dtype=np.uint8)
    h, w = board.shape
    center_y, center_x = h // 2, w // 2
    
    if pattern_name == "glider":
        # 複数のグライダー配置
        glider = np.array([[0, 1, 0],
                          [0, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        # 複数箇所に配置
        positions = [
            (center_y-50, center_x-50),
            (center_y+50, center_x+50),
            (center_y-100, center_x+100)
        ]
        for py, px in positions:
            if py >= 0 and px >= 0 and py+3 <= h and px+3 <= w:
                board[py:py+3, px:px+3] = glider
                
    elif pattern_name == "oscillator":
        # 複数の振動パターン
        patterns = [
            # ビーコン
            [(center_y-1, center_x-1, 2, 2), (center_y+1, center_x+1, 2, 2)],
            # ブリンカー
            [(center_y-50, center_x-50, 1, 3)],
            [(center_y+50, center_x+50, 3, 1)]
        ]
        for pattern in patterns:
            for py, px, h_size, w_size in pattern:
                if py >= 0 and px >= 0 and py+h_size <= h and px+w_size <= w:
                    board[py:py+h_size, px:px+w_size] = 1
                    
    elif pattern_name == "random":
        # より興味深いランダムパターン
        board = np.random.choice([0, 1], size=(h, w), p=[0.75, 0.25]).astype(np.uint8)
        
        # 中央に高密度領域を作成
        center_region = board[center_y-100:center_y+100, center_x-100:center_x+100]
        high_density = np.random.choice([0, 1], size=center_region.shape, p=[0.5, 0.5])
        board[center_y-100:center_y+100, center_x-100:center_x+100] = high_density
        
    elif pattern_name == "pulsar":
        # パルサー（周期15の振動パターン）
        pulsar_pattern = [
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,0,0,0,1,1,1,0,0]
        ]
        pulsar = np.array(pulsar_pattern, dtype=np.uint8)
        py, px = pulsar.shape
        
        # 複数のパルサーを配置
        positions = [
            (center_y-py//2, center_x-px//2),
            (center_y-py//2-80, center_x-px//2-80),
            (center_y-py//2+80, center_x-px//2+80)
        ]
        for pos_y, pos_x in positions:
            if pos_y >= 0 and pos_x >= 0 and pos_y+py <= h and pos_x+px <= w:
                board[pos_y:pos_y+py, pos_x:pos_x+px] = pulsar
    
    return board

# ===== メイン処理（記事品質の完全版）=====
def main():
    print("🖐️ === 高度手認識対応 ライフゲーム ===")
    print("Qiita記事『Mediapipeで手の形状検出を試してみた』ベース")
    print("21点ランドマーク追跡 + 詳細分析機能搭載！")
    print()
    
    if MEDIAPIPE_AVAILABLE:
        print("✅ MediaPipe: 利用可能 - 高精度手認識機能ON")
        print("   • 21点ランドマーク追跡")
        print("   • 左右の手自動判定")
        print("   • 信頼度スコア表示")
        print("   • 指先座標詳細分析")
    else:
        print("⚠️  MediaPipe: 未インストール - 基本肌色検出のみ")
        print("   pip install mediapipe でアップグレード可能")
    
    # 高度手追跡器を初期化
    hand_tracker = AdvancedHandTracker()
    
    cam_index = find_available_camera()
    if cam_index == -1:
        print("\n📷 カメラが見つかりません！")
        print("プリセットパターンでライフゲームを開始しますか？ (y/n)")
        if input().lower() == 'y':
            print("\nパターンを選択してください:")
            print("1. グライダー（移動パターン×3）")
            print("2. 振動パターン（複数の振動子）")
            print("3. パルサー（複雑な周期パターン×3）")
            print("4. ランダム（高密度中央領域付き）")
            choice = input("選択 (1-4): ")
            
            patterns = {
                "1": "glider", 
                "2": "oscillator", 
                "3": "pulsar",
                "4": "random"
            }
            if choice in patterns:
                print(f"\n{patterns[choice]}パターンを生成中...")
                board = create_preset_pattern(patterns[choice])
                mask = np.stack([board*255]*3, axis=-1).astype(np.uint8)
                
                # 空の検出情報を作成
                empty_detection_info = {
                    'hands_detected': 0,
                    'left_hands': 0,
                    'right_hands': 0,
                    'total_confidence': 0,
                    'hand_analyses': []
                }
                
                run_advanced_life_game_from_array(
                    mask, empty_detection_info, [], [], 
                    resolution=(1000, 800)
                )
        return

    # カメラ初期化
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ カメラを起動できません！")
        return

    # カメラ設定（高解像度）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"\n📷 カメラ {cam_index} を起動しました")
    print("🔴 高度リアルタイム手認識中...")
    print("\n操作方法:")
    print("  SPACE: 撮影してライフゲーム開始")
    print("  ESC: 終了")
    print("  h: 手認識詳細情報表示")
    print("  c: キャリブレーション情報")
    print("  s: 検出統計表示")
    print("  d: 現在の検出データ保存")
    print("  f: フルスクリーン切り替え")
    print("=" * 60)

    fullscreen = False
    frame_count = 0
    fps_timer = time.time()
    current_fps = 0
    last_detection_info = {}
    last_landmarks_list = []
    last_analysis_list = []

    # 検出データ保存用ディレクトリ作成
    os.makedirs("detection_data", exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ カメラからフレームを取得できません！")
            break

        frame_count += 1
        
        # FPS計算
        if frame_count % 30 == 0:
            current_time = time.time()
            current_fps = 30 / (current_time - fps_timer)
            fps_timer = current_time

        # フレームを水平反転（鏡像効果）
        frame = cv2.flip(frame, 1)
        
        # 高度な手認識と肌色検出を実行
        display_frame = frame.copy()
        mask, detection_info, landmarks_list, analysis_list = enhanced_skin_mask_with_analysis(
            display_frame, hand_tracker
        )
        
        # 最新の検出結果を保存
        last_detection_info = detection_info
        last_landmarks_list = landmarks_list
        last_analysis_list = analysis_list
        
        # FPS情報表示
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}", 
                   (display_frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # フレーム数表示
        cv2.putText(display_frame, f"Frame: {frame_count}", 
                   (display_frame.shape[1] - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # 検出統計表示
        if MEDIAPIPE_AVAILABLE:
            stats = hand_tracker.get_detection_statistics()
            if stats:
                cv2.putText(display_frame, f"Detection Rate: {stats['detection_rate']:.1f}%", 
                           (display_frame.shape[1] - 200, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 操作ガイド
        guide_y = display_frame.shape[0] - 90
        cv2.putText(display_frame, "SPACE: Capture & Start Advanced Life Game", 
                   (10, guide_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "ESC: Exit | H: Hand Info | S: Statistics | D: Save Data", 
                   (10, guide_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        cv2.putText(display_frame, "C: Calibration | F: Fullscreen", 
                   (10, guide_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        
        # 結果を横並びで表示
        combined = np.hstack((display_frame, mask))
        
        # ウィンドウサイズ調整
        if fullscreen:
            cv2.namedWindow("Advanced Hand Recognition + Life Game", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Advanced Hand Recognition + Life Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Advanced Hand Recognition + Life Game", combined)
        else:
            display_width = 1600
            display_height = int(combined.shape[0] * display_width / combined.shape[1])
            resized_combined = cv2.resize(combined, (display_width, display_height))
            cv2.imshow("Advanced Hand Recognition + Life Game", resized_combined)

        # キー入力処理
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC - 終了
            print("👋 アプリケーション終了")
            break
            
        elif key == 32:  # SPACE - 撮影してライフゲーム開始
            # 高品質な画像とデータを保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 画像保存
            image_filename = f"screenshots/captured_hand_advanced_{timestamp}.png"
            cv2.imwrite(image_filename, mask)
            
            # 検出データ保存
            if MEDIAPIPE_AVAILABLE and landmarks_list:
                data_filename = f"detection_data/capture_data_{timestamp}.json"
                capture_data = hand_tracker.save_detection_data(
                    landmarks_list, analysis_list, frame_count
                )
                with open(data_filename, 'w', encoding='utf-8') as f:
                    json.dump(capture_data, f, indent=2, ensure_ascii=False)
                print(f"💾 検出データ保存: {data_filename}")
            
            print(f"📸 高度手認識画像キャプチャ完了: {image_filename}")
            print("🎮 高度分析ライフゲーム開始！")
            
            time.sleep(0.1)  # 保存完了を待つ
            run_advanced_life_game_from_array(
                mask, detection_info, landmarks_list, analysis_list,
                resolution=(1200, 900), speed=50
            )
            
            print("📷 カメラ画面に戻りました")
            
        elif key == ord('h'):  # h - 手認識詳細情報
            print("\n" + "=" * 80)
            print("🖐️  高度手認識詳細情報")
            print("=" * 80)
            if MEDIAPIPE_AVAILABLE:
                print("✅ MediaPipe手認識: アクティブ")
                print(f"   検出された手の数: {detection_info['hands_detected']}")
                if detection_info['hands_detected'] > 0:
                    print("   機能:")
                    print("     • 21点ランドマーク追跡")
                    print("     • 左右の手自動判定")
                    print("     • 信頼度スコア計算")
                    print("     • 指先座標詳細分析")
                    print("     • 手のサイズ・中心座標計算")
                    print("     • 骨格描画 + 接続線表示")
                    print("     • 色分け表示（指別）")
                    
                    print("\n   検出詳細:")
                    for i, analysis in enumerate(analysis_list):
                        print(f"     手 {i+1}:")
                        print(f"       ラベル: {analysis['hand_label']}")
                        print(f"       信頼度: {analysis['confidence']:.4f}")
                        print(f"       手のサイズ: {analysis['hand_size']:.4f}")
                        print(f"       手のひら中心: ({analysis['palm_center'][0]:.4f}, {analysis['palm_center'][1]:.4f})")
                        print(f"       ランドマーク数: {len(analysis['landmark_coords'])}")
                        
                        # 指先座標を表示
                        print(f"       指先座標:")
                        for fingertip in analysis['fingertip_coords']:
                            print(f"         {fingertip['finger']}: ({fingertip['x']:.4f}, {fingertip['y']:.4f}, {fingertip['z']:.4f})")
                else:
                    print("   状態: 手を検出中...")
                    print("   ヒント: 手をカメラに向けてください")
                    print("   対応: 肌色、標準的な手の形、明るい環境")
            else:
                print("⚠️  基本モード: 肌色検出のみ")
                print("   機能: HSV + YCrCb色空間による肌色検出")
                print("   アップグレード: pip install mediapipe")
            
            # 検出統計
            if MEDIAPIPE_AVAILABLE:
                stats = hand_tracker.get_detection_statistics()
                if stats:
                    print(f"\n📊 検出統計:")
                    print(f"   総フレーム数: {stats['total_frames']}")
                    print(f"   検出された手: {stats['hands_detected']}")
                    print(f"   左手: {stats['left_hands']}, 右手: {stats['right_hands']}")
                    print(f"   検出率: {stats['detection_rate']:.2f}%")
                    print(f"   平均信頼度: {stats['average_confidence']:.4f}")
                    if stats['confidence_scores']:
                        print(f"   最新信頼度: {stats['confidence_scores'][-1]:.4f}")
            
            print("=" * 80)
            
        elif key == ord('c'):  # c - キャリブレーション情報
            print("\n" + "=" * 80)
            print("🔧 高度キャリブレーション情報")
            print("=" * 80)
            print("最適な検出のための詳細ヒント:")
            print()
            print("🌞 環境設定:")
            print("  • 明るく均一な照明を使用してください")
            print("  • 直接光や影を避けてください")
            print("  • 背景をシンプルに保ってください（単色推奨）")
            print("  • カメラから30-60cm程度の距離を保ってください")
            print()
            print("✋ 手の状態:")
            print("  • 手全体がフレーム内に入るようにしてください")
            print("  • 指をできるだけ開いてください")
            print("  • 手袋は外してください")
            print("  • 複数の手を同時に検出可能です（最大2つ）")
            print("  • 左右の手は自動で判定されます")
            print()
            if MEDIAPIPE_AVAILABLE:
                print("🎯 MediaPipe特有の注意点:")
                print("  • 標準的な肌色での検出精度が最も高いです")
                print("  • アニメキャラや非現実的な手は検出されません")
                print("  • 複雑すぎる手のポーズは一部誤検出の可能性があります")
                print("  • 奥行き情報（z座標）も取得されます")
                print("  • 隠れた指も推定・補完されます")
            print("=" * 80)
            
        elif key == ord('s'):  # s - 検出統計表示
            print(f"\n📊 リアルタイム検出統計:")
            print(f"現在のフレーム: {frame_count}")
            print(f"現在のFPS: {current_fps:.2f}")
            
            if MEDIAPIPE_AVAILABLE:
                stats = hand_tracker.get_detection_statistics()
                if stats:
                    print(f"\n検出パフォーマンス:")
                    print(f"  総処理フレーム数: {stats['total_frames']}")
                    print(f"  検出成功数: {stats['hands_detected']}")
                    print(f"  検出率: {stats['detection_rate']:.2f}%")
                    print(f"  左手検出数: {stats['left_hands']}")
                    print(f"  右手検出数: {stats['right_hands']}")
                    print(f"  平均信頼度: {stats['average_confidence']:.4f}")
                    
                    if stats['confidence_scores']:
                        recent_scores = stats['confidence_scores']
                        print(f"  最新信頼度スコア:")
                        for i, score in enumerate(recent_scores[-5:]):  # 最新5個
                            print(f"    {i+1}: {score:.4f}")
            
            print(f"\n現在の検出状況:")
            if detection_info['hands_detected'] > 0:
                print(f"  検出中の手: {detection_info['hands_detected']}個")
                print(f"  左手: {detection_info['left_hands']}個")
                print(f"  右手: {detection_info['right_hands']}個")
                avg_conf = detection_info['total_confidence'] / detection_info['hands_detected']
                print(f"  現在の平均信頼度: {avg_conf:.4f}")
            else:
                print("  現在手は検出されていません")
                
        elif key == ord('d'):  # d - 現在の検出データ保存
            if MEDIAPIPE_AVAILABLE and landmarks_list:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 詳細な検出データを保存
                save_data = {
                    'timestamp': timestamp,
                    'frame_number': frame_count,
                    'fps': current_fps,
                    'detection_info': detection_info,
                    'hand_analyses': analysis_list,
                    'detection_statistics': hand_tracker.get_detection_statistics()
                }
                
                # ランドマーク座標も詳細に保存
                detailed_landmarks = []
                for i, landmarks in enumerate(landmarks_list):
                    landmark_data = {
                        'hand_id': i,
                        'coordinates': landmarks,
                        'analysis': analysis_list[i] if i < len(analysis_list) else {}
                    }
                    detailed_landmarks.append(landmark_data)
                
                save_data['detailed_landmarks'] = detailed_landmarks
                
                filename = f"detection_data/realtime_data_{timestamp}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
                
                print(f"💾 現在の検出データ保存完了: {filename}")
                
                # 画像も同時保存
                image_filename = f"screenshots/realtime_capture_{timestamp}.png"
                cv2.imwrite(image_filename, combined)
                print(f"📸 現在の画面も保存: {image_filename}")
            else:
                print("⚠️ 保存できる検出データがありません")
                
        elif key == ord('f'):  # f - フルスクリーン切り替え
            fullscreen = not fullscreen
            if not fullscreen:
                cv2.namedWindow("Advanced Hand Recognition + Life Game", cv2.WINDOW_AUTOSIZE)
            print(f"フルスクリーン: {'ON' if fullscreen else 'OFF'}")
            
        elif key == ord('q'):  # q - 品質・パフォーマンス情報
            print(f"\n📈 システム品質情報:")
            print(f"カメラ解像度: {frame.shape[1]}x{frame.shape[0]}")
            print(f"カラースペース: BGR → RGB (MediaPipe用)")
            print(f"処理モード: {'高精度MediaPipe' if MEDIAPIPE_AVAILABLE else '基本肌色検出'}")
            print(f"現在のFPS: {current_fps:.2f}")
            print(f"フレーム処理数: {frame_count}")
            
            if MEDIAPIPE_AVAILABLE:
                print(f"MediaPipe設定:")
                print(f"  static_image_mode: False")
                print(f"  max_num_hands: 2")
                print(f"  min_detection_confidence: 0.7")
                print(f"  min_tracking_confidence: 0.5")
                
        elif key == ord('r'):  # r - 統計リセット
            if MEDIAPIPE_AVAILABLE:
                hand_tracker.detection_stats = {
                    'total_frames': 0,
                    'hands_detected': 0,
                    'left_hands': 0,
                    'right_hands': 0,
                    'confidence_scores': []
                }
                print("📊 検出統計をリセットしました")
            frame_count = 0
            fps_timer = time.time()
            print("📈 フレームカウンターをリセットしました")

    # リソース解放
    cap.release()
    cv2.destroyAllWindows()
    
    if MEDIAPIPE_AVAILABLE and hand_tracker.hands:
        hand_tracker.hands.close()
    
    # 最終統計表示
    if MEDIAPIPE_AVAILABLE:
        final_stats = hand_tracker.get_detection_statistics()
        if final_stats:
            print("\n" + "=" * 60)
            print("📊 最終検出統計")
            print("=" * 60)
            print(f"総処理フレーム数: {final_stats['total_frames']}")
            print(f"手検出成功数: {final_stats['hands_detected']}")
            print(f"最終検出率: {final_stats['detection_rate']:.2f}%")
            print(f"左手検出数: {final_stats['left_hands']}")
            print(f"右手検出数: {final_stats['right_hands']}")
            print(f"平均信頼度: {final_stats['average_confidence']:.4f}")
            print("=" * 60)
    
    print("🎯 すべてのリソースを解放しました")
    print("💾 保存されたファイル:")
    print("  - screenshots/ : キャプチャした画像")
    print("  - detection_data/ : 検出データ（JSON形式）")

# ===== エントリーポイント =====
if __name__ == "__main__":
    try:
        # 必要なディレクトリを作成
        for directory in ['screenshots', 'detection_data']:
            os.makedirs(directory, exist_ok=True)
        
        main()
    except KeyboardInterrupt:
        print("\n⚠️  ユーザーによって中断されました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
        if not MEDIAPIPE_AVAILABLE:
            print("\n💡 ヒント: MediaPipeをインストールすると手認識機能が利用できます")
            print("   コマンド: pip install mediapipe")
            print("   記事参考: https://qiita.com/bianca26neve/items/...")
    finally:
        cv2.destroyAllWindows()
        print("👋 プログラム終了")
        print("\n🙏 参考にした記事:")
        print("『Mediapipeで手の形状検出を試してみた(Python)』")
        print("by @bianca26neve on Qiita")
        print("MediaPipeの素晴らしい機能を活用させていただきました！")