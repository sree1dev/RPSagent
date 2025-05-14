import numpy as np
import pickle
import os
import random
import msvcrt
import time
import sys
import tempfile
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque

class ControllerNetwork(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        if x.shape[-1] != self.input_size:
            print(f"Warning: ControllerNetwork input shape {x.shape} does not match expected {self.input_size}")
        return self.network(x)

class HighLevelPolicy(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        if x.shape[-1] != self.input_size:
            print(f"Warning: HighLevelPolicy input shape {x.shape} does not match expected {self.input_size}")
        return self.network(x)

class MetaAINeuralNetwork(nn.Module):
    def __init__(self, input_size=25, hidden_size=16):
        super().__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 6)  # history_length, alpha, epsilon, freq_bias_weight, penalty_scale, reward_scale
        )
    def forward(self, x):
        if x.shape[-1] != self.input_size:
            print(f"Warning: MetaAINeuralNetwork input shape {x.shape} does not match expected {self.input_size}")
        return self.network(x)

class RewardPredictor(nn.Module):
    def __init__(self, input_size=25, hidden_size=16):
        super().__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),  # penalty_scale, reward_scale, loss_scale
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        if x.shape[-1] != self.input_size:
            print(f"Warning: RewardPredictor input shape {x.shape} does not match expected {self.input_size}")
        return self.network(x)

class RPS_AI:
    def __init__(self, history_length=2, alpha=0.1, gamma=0.9, epsilon=0.1, freq_bias_weight=0.15,
                 penalty_scale=2.0, reward_scale=5.0, loss_scale=-5.0):
        self.history_length = history_length
        self.actions = ['R', 'P', 'S']
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.freq_bias_weight = freq_bias_weight
        self.penalty_scale = penalty_scale
        self.reward_scale = reward_scale
        self.loss_scale = loss_scale
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        self.history = []
        self.move_counts = {'R': 0, 'P': 0, 'S': 0}
        self.q_table_file = "rps_q_table.pkl"
        self.backup_q_table_file = "backup_q_table.pkl"
        self.current_round = 0
        self.ai_score = 0
        self.player_score = 0
        self.new_states_visited = 0
        self.load_q_table()

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'rb') as f:
                    loaded_q_table = pickle.load(f)
                    self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), loaded_q_table)
            except Exception as e:
                print(f"Error loading Q-table: {e}. Starting fresh.")

    def save_q_table(self, filename=None):
        filename = filename or self.q_table_file
        try:
            temp_fd, temp_path = tempfile.mkstemp()
            with open(temp_path, 'wb') as f:
                pickle.dump(dict(self.q_table), f)
            os.close(temp_fd)
            shutil.move(temp_path, filename)
            return True
        except Exception as e:
            print(f"Error saving Q-table: {e}")
            return False

    def backup_q_table(self):
        self.save_q_table(self.backup_q_table_file)

    def restore_q_table(self):
        if os.path.exists(self.backup_q_table_file):
            try:
                with open(self.backup_q_table_file, 'rb') as f:
                    loaded_q_table = pickle.load(f)
                    self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), loaded_q_table)
                self.save_q_table()
                os.remove(self.backup_q_table_file)
            except Exception as e:
                print(f"Error restoring Q-table: {e}")

    def validate_q_table(self):
        try:
            q_values = [q for state in self.q_table.values() for q in state]
            if q_values and (max(abs(q) for q in q_values) > 50 or np.std(q_values) > 10):
                print("Warning: Unbalanced Q-values. Restoring Q-table.")
                return False
            return True
        except Exception:
            print("Error validating Q-table. Restoring Q-table.")
            return False

    def get_state(self):
        if len(self.history) < self.history_length * 2:
            return tuple(['-'] * (self.history_length * 2))
        return tuple(self.history[-self.history_length * 2:])

    def is_win_guaranteed(self):
        remaining_rounds = 15 - self.current_round
        score_lead = self.ai_score - self.player_score
        return score_lead > remaining_rounds + 1

    def determine_winner(self, player_move, ai_move):
        if player_move == ai_move:
            return 'Tie', 0.0
        elif (player_move == 'R' and ai_move == 'S') or \
             (player_move == 'P' and ai_move == 'R') or \
             (player_move == 'S' and ai_move == 'P'):
            return 'Player', self.loss_scale
        else:
            return 'AI', self.reward_scale

    def choose_action(self, state, high_level_strategy):
        if high_level_strategy == 'explore' or self.is_win_guaranteed():
            temp_epsilon = min(self.epsilon + 0.5, 0.8)
            temp_freq_bias_weight = self.freq_bias_weight if high_level_strategy == 'explore' else 0.0
            if self.is_win_guaranteed():
                print(f"Safe Learning Mode at Round {self.current_round + 1}: AI {self.ai_score}, Player {self.player_score}")
        else:
            temp_epsilon = self.epsilon
            temp_freq_bias_weight = self.freq_bias_weight
        if random.random() < temp_epsilon:
            return random.choice(self.actions)
        q_values = self.q_table[state].copy()
        total_moves = sum(self.move_counts.values()) + 1e-10
        freq_bias = {
            'R': self.move_counts['S'] / total_moves,
            'P': self.move_counts['R'] / total_moves,
            'S': self.move_counts['P'] / total_moves
        }
        for i, action in enumerate(self.actions):
            q_values[i] += temp_freq_bias_weight * freq_bias[action] * 2.0  # Increased from 1.5
        return self.actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.new_states_visited += 1
        action_idx = self.actions.index(action)
        current_q = self.q_table[state][action_idx]
        next_q = np.max(self.q_table[next_state])
        self.q_table[state][action_idx] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

    def play_round(self, player_move, high_level_strategy):
        self.current_round += 1
        state = self.get_state()
        self.move_counts[player_move] += 1
        ai_move = self.choose_action(state, high_level_strategy)
        winner, reward = self.determine_winner(player_move, ai_move)
        next_state = self.get_state() + (player_move, ai_move)
        self.history.extend([player_move, ai_move])
        is_safe_mode = self.is_win_guaranteed()
        if is_safe_mode and winner != 'AI':
            reward = -self.penalty_scale
        elif high_level_strategy == 'explore' and is_safe_mode:
            reward += 0.1 * self.new_states_visited
        self.update_q_table(state, action=ai_move, reward=reward, next_state=next_state)
        if winner == 'Player':
            self.player_score += 1
        elif winner == 'AI':
            self.ai_score += 1
        return ai_move, winner, is_safe_mode

    def get_valid_keypress(self):
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').upper()
                if key in ['R', 'P', 'S', 'Q']:
                    return key
                print(f"\nInvalid key '{key}'! Press R, P, S, or Q: ", end='', flush=True)
            time.sleep(0.01)

    def play_game(self, high_level_strategy='exploit'):
        self.player_score = 0
        self.ai_score = 0
        self.current_round = 0
        self.history = []
        self.new_states_visited = 0
        self.backup_q_table()
        print("Press R (Rock), P (Paper), S (Scissors), or Q to quit.")
        player_moves = []
        ai_moves = []
        for round_num in range(1, 16):
            print(f"\nRound {round_num}/15 - Your move (R/P/S): ", end='', flush=True)
            key = self.get_valid_keypress()
            if key == 'Q':
                self.restore_q_table()
                return None, None, None
            player_move = key
            player_moves.append(player_move)
            print(f"You played: {player_move}")
            ai_move, winner, is_safe_mode = self.play_round(player_move, high_level_strategy)
            ai_moves.append(ai_move)
            if not self.validate_q_table():
                self.restore_q_table()
                return None, None, None
            print(f"AI plays: {ai_move}")
            print(f"Result: {winner}")
            print(f"Score - You: {self.player_score}, AI: {self.ai_score}")
            if is_safe_mode:
                print(f"Note: AI in Safe Learning Mode.")
        if os.path.exists(self.backup_q_table_file):
            os.remove(self.backup_q_table_file)
        if len(player_moves) == 15:  # Only for completed games
            human_pattern = ' '.join(player_moves)
            ai_pattern = ' '.join(ai_moves)
            print(f"\nHuman Pattern: {human_pattern}")
            print(f"AI Pattern: {ai_pattern}")
        return ('Player' if self.player_score > self.ai_score else 'AI' if self.ai_score > self.ai_score else 'Tie',
                (self.player_score, self.ai_score), player_moves, ai_moves)

class MetaAISupervisor:
    def __init__(self, input_size=25, hidden_size=16, learning_rate=0.0005, buffer_size=200):  # Reduced buffer_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = MetaAINeuralNetwork(input_size, hidden_size).to(self.device)
        self.reward_predictor = RewardPredictor(input_size, hidden_size).to(self.device)
        self.controller = ControllerNetwork(input_size).to(self.device)
        self.high_level_policy = HighLevelPolicy(input_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.reward_optimizer = optim.Adam(self.reward_predictor.parameters(), lr=0.001)
        self.controller_optimizer = optim.Adam(self.controller.parameters(), lr=0.001)
        self.high_level_optimizer = optim.Adam(self.high_level_policy.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.replay_buffer = deque(maxlen=buffer_size)
        self.input_size = input_size
        self.min_history_length = 2
        self.max_history_length = 6
        self.min_alpha = 0.05
        self.max_alpha = 0.5
        self.min_epsilon = 0.05
        self.max_epsilon = 0.3
        self.min_freq_bias = 0.15
        self.max_freq_bias = 0.3
        self.min_penalty_scale = 2.0
        self.max_penalty_scale = 10.0
        self.min_reward_scale = 1.0
        self.max_reward_scale = 8.0
        self.min_loss_scale = -8.0
        self.max_loss_scale = -1.0
        self.model_file = "meta_ai_model.pth"
        self.reward_model_file = "reward_predictor_model.pth"
        self.load_model()
        self.dynamic_features = [0.0] * 4  # Initialize with 4 placeholder features
        self.last_pattern = None
        self.pattern_losses = 0
        self.loss_history = deque(maxlen=50)
        self.win_history = deque(maxlen=20)  # Track recent wins for autonomy

    def load_model(self):
        if os.path.exists(self.model_file):
            try:
                state_dict = torch.load(self.model_file, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                print("Loaded Meta-AI model.")
            except Exception as e:
                print(f"Error loading Meta-AI model: {e}. Starting fresh.")
        if os.path.exists(self.reward_model_file):
            try:
                state_dict = torch.load(self.reward_model_file, map_location=self.device, weights_only=True)
                self.reward_predictor.load_state_dict(state_dict)
                print("Loaded Reward Predictor model.")
            except Exception as e:
                print(f"Error loading Reward Predictor model: {e}. Starting fresh.")

    def save_model(self):
        try:
            torch.save(self.model.state_dict(), self.model_file)
            torch.save(self.reward_predictor.state_dict(), self.reward_model_file)
        except Exception as e:
            print(f"Error saving models: {e}")

    def reset_model_if_stuck(self, game_count):
        if game_count >= 20 and len(self.win_history) == 20:  # Earlier reset
            win_rate = sum(1 for w in self.win_history if w == 'AI') / 20
            if win_rate < 0.5:
                print(f"Low win rate ({win_rate:.2f}) after {game_count} games. Resetting neural networks.")
                self.model = MetaAINeuralNetwork(self.input_size, 16).to(self.device)
                self.reward_predictor = RewardPredictor(self.input_size, 16).to(self.device)
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
                self.reward_optimizer = optim.Adam(self.reward_predictor.parameters(), lr=0.001)
                self.replay_buffer.clear()
                self.dynamic_features = [0.0] * 4
                self.save_model()

    def adjust_learning_rate(self):
        if len(self.loss_history) >= 10:
            avg_loss = sum(self.loss_history) / len(self.loss_history)
            if avg_loss > 2.0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = min(param_group['lr'] * 1.5, 0.001)
                for param_group in self.reward_optimizer.param_groups:
                    param_group['lr'] = min(param_group['lr'] * 1.5, 0.002)
            elif avg_loss < 0.5:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.5, 0.0001)
                for param_group in self.reward_optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.5, 0.0002)

    def get_ngram_counts(self, moves, n=3):
        if len(moves) < n:
            return 0
        count = 0
        for i in range(len(moves) - n + 1):
            if all(moves[i + j] == moves[i] for j in range(n)):
                count += 1
            elif n == 3 and moves[i:i+3] in [['S', 'R', 'P'], ['R', 'P', 'S'], ['P', 'S', 'R']]:
                count += 2  # Increased weight for cyclic patterns
        return count / (len(moves) - n + 1) if len(moves) >= n else 0.0

    def detect_pattern_failure(self, player_moves, ai, winner):
        if len(player_moves) < 3:
            return 0.0
        recent_wins = sum(1 for i in range(max(0, len(player_moves)-3), len(player_moves))
                         if ai.determine_winner(player_moves[i], ai.history[2*i+1])[0] == 'Player')
        pattern = tuple(player_moves[-3:])
        if recent_wins >= 2:  # Trigger on recent player wins
            self.pattern_losses += 1
        else:
            self.pattern_losses = max(0, self.pattern_losses - 1)
        self.last_pattern = pattern
        return min(self.pattern_losses / 3.0, 1.0)

    def get_features(self, ai, player_score, ai_score, player_moves, safe_mode_rounds):
        global game_count
        win_rate = ai_score / 15.0
        tie_rate = (15 - player_score - ai_score) / 15.0
        loss_rate = player_score / 15.0
        move_counts = {'R': 0, 'P': 0, 'S': 0}
        for move in player_moves:
            move_counts[move] += 1
        move_freq = [move_counts[m] / 15.0 for m in ['R', 'P', 'S']]
        repeats = sum(1 for i in range(len(player_moves)-1) if player_moves[i] == player_moves[i+1])
        transition_rate = (14 - repeats) / 14.0 if repeats < 14 else 0.0
        q_values = [q for state in ai.q_table.values() for q in state]
        q_mean = np.mean(q_values) if q_values else 0.0
        q_std = np.std(q_values) if q_values else 0.0
        sss_ngram = self.get_ngram_counts(player_moves, n=3)
        cycle_ngram_4 = self.get_ngram_counts(player_moves, n=4)
        recent_wins = sum(1 for i in range(max(0, len(player_moves)-5), len(player_moves))
                         if ai.determine_winner(player_moves[i], ai.history[2*i+1])[0] == 'AI')
        recent_win_rate = recent_wins / 5.0 if len(player_moves) >= 5 else ai_score / 15.0
        move_probs = [move_counts[m] / 15.0 for m in ['R', 'P', 'S']]
        move_entropy = -sum(p * np.log2(p + 1e-10) for p in move_probs if p > 0)
        recent_repeats = sum(1 for i in range(max(0, len(player_moves)-5), len(player_moves)-1)
                            if player_moves[i] == player_moves[i+1]) / 4.0 if len(player_moves) >= 5 else 0.0
        pattern_failure = self.detect_pattern_failure(player_moves, ai, 'Player' if player_score > ai_score else 'AI')
        current_params = [ai.history_length, ai.alpha, ai.epsilon, ai.freq_bias_weight]
        safe_mode_ratio = safe_mode_rounds / 15.0
        base_features = [
            win_rate, tie_rate, loss_rate,  # 3
            *move_freq,  # 3
            repeats / 14.0 if repeats > 0 else 0.0, transition_rate,  # 2
            q_mean, q_std,  # 2
            sss_ngram, cycle_ngram_4,  # 2
            recent_win_rate, move_entropy,  # 2
            recent_repeats, pattern_failure,  # 2
            *current_params,  # 4
            safe_mode_ratio  # 1
        ]  # Total: 21
        remaining_slots = self.input_size - len(base_features)
        selected_dynamic = self.dynamic_features[:remaining_slots]
        features = base_features + selected_dynamic
        if len(features) < self.input_size:
            print(f"Warning: Feature count {len(features)} less than input_size {self.input_size}. Padding with zeros.")
            with open("rps_game_log.txt", "a") as f:
                f.write(f"Game {game_count}, Feature Padding: {self.input_size - len(features)} zeros added\n")
            features.extend([0.0] * (self.input_size - len(features)))
        elif len(features) > self.input_size:
            print(f"Warning: Feature count {len(features)} exceeds input_size {self.input_size}. Truncating.")
            with open("rps_game_log.txt", "a") as f:
                f.write(f"Game {game_count}, Truncated Features: {features[self.input_size:]}\n")
            features = features[:self.input_size]
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def propose_feature(self, ai, player_score, ai_score, current_round):
        feature_options = [
            (ai_score - player_score) / 15.0,
            current_round / 15.0,
            (ai_score + player_score) / 15.0,
            ai.new_states_visited / 15.0,
            (ai.move_counts['R'] - ai.move_counts['S']) / 15.0,
            (ai.q_table[tuple(ai.history[-ai.history_length * 2:])].max() if ai.history else 0.0)
        ]
        # Prioritize features based on recent performance
        if len(self.win_history) >= 5:
            win_rate = sum(1 for w in self.win_history if w == 'AI') / len(self.win_history)
            if win_rate < 0.5:
                feature_options.append((ai_score - player_score) / 15.0)  # Emphasize score difference
        return [random.choice(feature_options)] if len(self.dynamic_features) < 6 and random.random() < 0.5 else []  # Increased to 6, prob 0.5

    def get_target_params(self, ai_score, player_score, tie_count, player_moves, safe_mode_rounds):
        reward = ai_score - player_score
        norm_reward = (reward + 15) / 30
        player_win_rate = player_score / 15.0
        move_counts = {'R': 0, 'P': 0, 'S': 0}
        for move in player_moves:
            move_counts[move] += 1
        move_freq = [move_counts[m] / 15.0 for m in ['R', 'P', 'S']]
        freq_imbalance = max(move_freq) - min(move_freq)
        pattern_strength = sum(1 for i in range(len(player_moves)-2) if player_moves[i:i+3] == player_moves[i:i+3]) / 12.0 if len(player_moves) >= 3 else 0.0
        target_history_length = self.min_history_length + (self.max_history_length - self.min_history_length) * norm_reward
        target_alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * norm_reward
        target_epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * (player_win_rate + freq_imbalance + pattern_strength * 2.0) / 3  # Increased weight
        target_freq_bias = self.min_freq_bias + (self.max_freq_bias - self.min_freq_bias) * (freq_imbalance + pattern_strength * 2.0) / 2
        target_penalty_scale = self.min_penalty_scale + (self.max_penalty_scale - self.min_penalty_scale) * (player_win_rate + safe_mode_rounds / 15.0) / 2
        target_reward_scale = self.min_reward_scale + (self.max_reward_scale - self.min_reward_scale) * norm_reward
        target_loss_scale = self.max_loss_scale - (self.max_loss_scale - self.min_loss_scale) * (player_win_rate + freq_imbalance) / 2
        return torch.tensor([
            target_history_length,
            target_alpha,
            target_epsilon,
            target_freq_bias,
            target_penalty_scale,
            target_reward_scale,
            target_loss_scale
        ], dtype=torch.float32).to(self.device)

    def store_experience(self, features, target_params, is_safe_mode=False):
        self.replay_buffer.append((features, target_params, 2.0 if is_safe_mode else 1.0))

    def train(self, batch_size=8):  # Reduced batch_size
        if len(self.replay_buffer) < 4:  # Minimum threshold
            print(f"Skipping training: replay_buffer size {len(self.replay_buffer)} < 4")
            return None
        batch_size = min(batch_size, len(self.replay_buffer))  # Use available experiences
        batch = random.sample(self.replay_buffer, batch_size)
        features, targets, weights = zip(*batch)
        features = torch.stack(features).to(self.device)
        targets = torch.stack(targets).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(features)
        loss = self.criterion(outputs, targets[:, :6])
        weighted_loss = (loss * weights).mean()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.reward_optimizer.zero_grad()
        reward_outputs = self.reward_predictor(features)
        reward_loss = self.criterion(reward_outputs, targets[:, 4:7])
        reward_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_predictor.parameters(), max_norm=0.5)
        self.reward_optimizer.step()

        self.high_level_optimizer.zero_grad()
        strategy_probs = self.high_level_policy(features)
        strategy_loss = -torch.log(strategy_probs[:, 1]) * weights
        strategy_loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.high_level_policy.parameters(), max_norm=0.5)
        self.high_level_optimizer.step()

        self.controller_optimizer.zero_grad()
        controller_probs = self.controller(features)
        controller_loss = -torch.log(controller_probs[:, 0]) * weights
        controller_loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), max_norm=0.5)
        self.controller_optimizer.step()

        self.adjust_learning_rate()  # Adaptive learning rate
        self.save_model()
        print(f"Meta-AI Training Loss: {weighted_loss.item():.4f}")
        self.loss_history.append(weighted_loss.item())
        return weighted_loss.item()

    def suggest_parameters(self, ai, player_score, ai_score, player_moves, safe_mode_rounds):
        features = self.get_features(ai, player_score, ai_score, player_moves, safe_mode_rounds)
        with torch.no_grad():
            suggested_params = self.model(features).cpu().numpy()
            reward_params = self.reward_predictor(features).cpu().numpy()
        pattern_failure = self.detect_pattern_failure(player_moves, ai, 'Player' if player_score > ai_score else 'AI')
        history_length = max(self.min_history_length, min(self.max_history_length, int(round(suggested_params[0] + pattern_failure))))
        alpha = max(self.min_alpha, min(self.max_alpha, suggested_params[1]))
        epsilon = max(self.min_epsilon, min(self.max_epsilon, suggested_params[2] + pattern_failure * 0.1))
        freq_bias_weight = max(self.min_freq_bias, min(self.max_freq_bias, suggested_params[3] + pattern_failure * 0.05))
        penalty_scale = max(self.min_penalty_scale, min(self.max_penalty_scale, reward_params[0] * (self.max_penalty_scale - self.min_penalty_scale) + self.min_penalty_scale))
        reward_scale = max(self.min_reward_scale, min(self.max_reward_scale, reward_params[1] * (self.max_reward_scale - self.min_reward_scale) + self.min_reward_scale))
        loss_scale = max(self.min_loss_scale, min(self.max_loss_scale, reward_params[2] * (self.max_loss_scale - self.min_loss_scale) + self.min_loss_scale))
        return {
            'history_length': history_length,
            'alpha': alpha,
            'epsilon': epsilon,
            'freq_bias_weight': freq_bias_weight,
            'penalty_scale': penalty_scale,
            'reward_scale': reward_scale,
            'loss_scale': loss_scale
        }

def main():
    global game_count
    ai = RPS_AI()
    meta_ai = MetaAISupervisor(input_size=25, hidden_size=16, buffer_size=200)
    game_count = 0
    baseline_ai_score = 7.5
    max_games = 1000
    save_interval = 10

    try:
        while game_count < max_games:
            game_count += 1
            print(f"\n=== Game {game_count} ===")
            features = meta_ai.get_features(ai, ai.player_score, ai.ai_score, [], 0)
            features = features.unsqueeze(0)
            high_level_strategy = 'explore' if meta_ai.high_level_policy(features)[0, 1] > 0.5 else 'exploit'
            winner, scores, player_moves, ai_moves = ai.play_game(high_level_strategy=high_level_strategy)
            if winner is None:
                print("Game ended. Thanks for playing!")
                human_pattern = ' '.join(player_moves) if player_moves else "None"
                ai_pattern = ' '.join(ai_moves) if ai_moves else "None"
                with open("rps_game_log.txt", "a") as f:
                    f.write(f"Game {game_count}, Early Exit, Human Pattern: {human_pattern}, AI Pattern: {ai_pattern}\n")
                break
            player_score, ai_score = scores
            tie_count = 15 - player_score - ai_score
            safe_mode_rounds = sum(1 for i in range(len(player_moves)) if ai.is_win_guaranteed() and i >= ai.current_round - len(player_moves))
            try:
                features = meta_ai.get_features(ai, player_score, ai_score, player_moves, safe_mode_rounds)
                move_counts = {'R': 0, 'P': 0, 'S': 0}
                for move in player_moves:
                    move_counts[move] += 1
                move_sequence = ''.join(player_moves[-3:]) if len(player_moves) >= 3 else ''
                human_pattern = ' '.join(player_moves) if player_moves else "None"
                ai_pattern = ' '.join(ai_moves) if ai_moves else "None"
                features = features.unsqueeze(0)
                high_level_strategy = 'explore' if meta_ai.high_level_policy(features)[0, 1] > 0.5 else 'exploit'
                target_params = meta_ai.get_target_params(ai_score, player_score, tie_count, player_moves, safe_mode_rounds)
                meta_reward = 0.1 * ai.new_states_visited + 0.2 * (features[0, 6].item()) + 0.5 * (ai_score - baseline_ai_score)
                meta_ai.store_experience(features.squeeze(0), target_params, is_safe_mode=safe_mode_rounds > 0)
                if random.random() < 0.5:  # Increased probability
                    meta_ai.dynamic_features.extend(meta_ai.propose_feature(ai, player_score, ai_score, ai.current_round))
                loss = meta_ai.train(batch_size=8)
                meta_ai.reset_model_if_stuck(game_count)
                meta_ai.win_history.append(winner)  # Track wins
                new_params = meta_ai.suggest_parameters(ai, player_score, ai_score, player_moves, safe_mode_rounds)
                print(f"Meta-AI Suggested Params: {new_params}, Strategy: {high_level_strategy}, Meta-Reward: {meta_reward:.3f}")
                ai.history_length = new_params['history_length']
                ai.alpha = new_params['alpha']
                ai.epsilon = new_params['epsilon']
                ai.freq_bias_weight = new_params['freq_bias_weight']
                ai.penalty_scale = new_params['penalty_scale']
                ai.reward_scale = new_params['reward_scale']
                ai.loss_scale = new_params['loss_scale']
                ai.move_counts = {'R': 0, 'P': 0, 'S': 0}
                ai.new_states_visited = 0
                if game_count % save_interval == 0:
                    ai.save_q_table()
                q_values = [q for state in ai.q_table.values() for q in state]
                q_std = np.std(q_values) if q_values else 0.0
                q_max = max(abs(q) for q in q_values) if q_values else 0.0
                loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                with open("rps_game_log.txt", "a") as f:
                    f.write(f"Game {game_count}, AI Wins: {ai_score}, Ties: {tie_count}, Player Wins: {player_score}, "
                            f"Params: history_length={ai.history_length}, alpha={ai.alpha:.3f}, "
                            f"epsilon={ai.epsilon:.3f}, freq_bias={ai.freq_bias_weight:.3f}, "
                            f"penalty_scale={ai.penalty_scale:.3f}, reward_scale={ai.reward_scale:.3f}, "
                            f"loss_scale={ai.loss_scale:.3f}, "
                            f"Loss: {loss_str}, Q-Std: {q_std:.3f}, Q-Max: {q_max:.3f}, "
                            f"Feature Count: {len(features.squeeze(0))}, "
                            f"Safe Mode Rounds: {safe_mode_rounds}, Strategy: {high_level_strategy}, "
                            f"Meta-Reward: {meta_reward:.3f}, Dynamic Features: {meta_ai.dynamic_features}, "
                            f"Player Moves: R={move_counts['R']}, P={move_counts['P']}, S={move_counts['S']}, "
                            f"Last Sequence: {move_sequence}, Pattern Failure: {meta_ai.detect_pattern_failure(player_moves, ai, winner):.3f}, "
                            f"Human Pattern: {human_pattern}, AI Pattern: {ai_pattern}\n")
            except Exception as e:
                print(f"Error during Meta-AI processing: {e}. Continuing with current parameters.")
                with open("rps_game_log.txt", "a") as f:
                    f.write(f"Game {game_count}, Error: {str(e)}, Human Pattern: {human_pattern}, AI Pattern: {ai_pattern}\n")

            print(f"\nGame Over! Final Score - You: {player_score}, AI: {ai_score}")
            print(f"Winner: {winner}")

    except KeyboardInterrupt:
        print("\nProgram interrupted. Saving Q-table...")
        ai.save_q_table()
        print("Thanks for playing!")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}. Saving Q-table...")
        ai.save_q_table()
        print("Thanks for playing!")
        sys.exit(1)

if __name__ == "__main__":
    main()