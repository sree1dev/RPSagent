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

class MetaAINeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(MetaAINeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)  # Outputs: history_length, alpha, epsilon, freq_bias_weight
        )
    
    def forward(self, x):
        return self.network(x)

class RPS_AI:
    def __init__(self, history_length=2, alpha=0.1, gamma=0.9, epsilon=0.1, freq_bias_weight=0.1):
        self.history_length = history_length
        self.actions = ['R', 'P', 'S']
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.freq_bias_weight = freq_bias_weight
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        self.history = []
        self.move_counts = {'R': 0, 'P': 0, 'S': 0}
        self.q_table_file = "rps_q_table.pkl"
        self.backup_q_table_file = "backup_q_table.pkl"
        self.load_q_table()
        self.current_round = 0
        self.ai_score = 0
        self.player_score = 0

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'rb') as f:
                    loaded_q_table = pickle.load(f)
                    self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), loaded_q_table)
            except Exception as e:
                print(f"Error loading Q-table: {e}. Starting with fresh Q-table.")

    def save_q_table(self, filename=None, retries=3, delay=0.1):
        if filename is None:
            filename = self.q_table_file
        for attempt in range(retries):
            try:
                temp_fd, temp_path = tempfile.mkstemp()
                try:
                    with open(temp_path, 'wb') as f:
                        pickle.dump(dict(self.q_table), f)
                    os.close(temp_fd)
                    shutil.move(temp_path, filename)
                    return True
                except:
                    os.close(temp_fd)
                    os.remove(temp_path) if os.path.exists(temp_path) else None
                    raise
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                print(f"Error saving Q-table to {filename}: {e}")
                return False
        return False

    def backup_q_table(self):
        if not self.save_q_table(self.backup_q_table_file):
            print("Warning: Failed to create Q-table backup.")

    def restore_q_table(self):
        if os.path.exists(self.backup_q_table_file):
            try:
                with open(self.backup_q_table_file, 'rb') as f:
                    loaded_q_table = pickle.load(f)
                    self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), loaded_q_table)
                self.save_q_table()
                os.remove(self.backup_q_table_file)
            except Exception as e:
                print(f"Error restoring Q-table: {e}. Keeping current Q-table.")

    def validate_q_table(self):
        try:
            q_values = [q for state in self.q_table.values() for q in state]
            if q_values:
                max_q = max(abs(q) for q in q_values)
                std_q = np.std(q_values)
                if max_q > 50 or std_q > 10:
                    print("Warning: Unbalanced Q-values detected. Restoring Q-table.")
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
        return score_lead > remaining_rounds

    def choose_action(self, state):
        if self.is_win_guaranteed():
            # Safe learning mode: high exploration, no frequency bias
            print(f"Safe Learning Mode activated at Round {self.current_round + 1}: AI {self.ai_score}, Player {self.player_score}")
            temp_epsilon = 0.8  # High exploration
            temp_freq_bias_weight = 0.0  # Disable frequency bias
            temp_history_length = min(self.history_length + 1, 4)  # Try longer history
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
                q_values[i] += temp_freq_bias_weight * freq_bias[action]
            return self.actions[np.argmax(q_values)]
        # Normal mode
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = self.q_table[state].copy()
        total_moves = sum(self.move_counts.values()) + 1e-10
        freq_bias = {
            'R': self.move_counts['S'] / total_moves,
            'P': self.move_counts['R'] / total_moves,
            'S': self.move_counts['P'] / total_moves
        }
        for i, action in enumerate(self.actions):
            q_values[i] += self.freq_bias_weight * freq_bias[action]
        return self.actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state):
        action_idx = self.actions.index(action)
        current_q = self.q_table[state][action_idx]
        next_q = np.max(self.q_table[next_state])
        self.q_table[state][action_idx] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

    def determine_winner(self, player_move, ai_move):
        if player_move == ai_move:
            return 'Tie', -0.1  # Penalize ties
        elif (player_move == 'R' and ai_move == 'S') or \
             (player_move == 'P' and ai_move == 'R') or \
             (player_move == 'S' and ai_move == 'P'):
            return 'Player', -1
        else:
            return 'AI', 1

    def play_round(self, player_move):
        self.current_round += 1
        state = self.get_state()
        self.move_counts[player_move] += 1
        ai_move = self.choose_action(state)
        winner, reward = self.determine_winner(player_move, ai_move)
        next_state = self.get_state() + (player_move, ai_move)
        self.history.extend([player_move, ai_move])
        self.update_q_table(state, ai_move, reward, next_state)
        if winner == 'Player':
            self.player_score += 1
        elif winner == 'AI':
            self.ai_score += 1
        return ai_move, winner, self.is_win_guaranteed()

    def get_valid_keypress(self):
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').upper()
                if key in ['R', 'P', 'S', 'Q']:
                    return key
                print(f"\nInvalid key '{key}'! Press R, P, S, or Q: ", end='', flush=True)
            time.sleep(0.01)

    def play_game(self, scripted_moves=None):
        self.player_score = 0
        self.ai_score = 0
        self.current_round = 0
        self.history = []
        player_moves = []
        self.backup_q_table()
        print("Press R (Rock), P (Paper), S (Scissors), or Q to quit.")

        for round_num in range(1, 16):
            print(f"\nRound {round_num}/15 - Your move (R/P/S): ", end='', flush=True)
            if scripted_moves and round_num - 1 < len(scripted_moves):
                key = scripted_moves[round_num - 1]
                print(f"{key}")
            else:
                key = self.get_valid_keypress()
            if key == 'Q':
                self.restore_q_table()
                return None, None, None
            player_move = key
            player_moves.append(player_move)
            ai_move, winner, is_safe_mode = self.play_round(player_move)
            if not self.validate_q_table():
                self.restore_q_table()
                return None, None, None
            self.save_q_table()
            print(f"AI plays: {ai_move}")
            print(f"Result: {winner}")
            print(f"Score - You: {self.player_score}, AI: {self.ai_score}")
            if is_safe_mode:
                print(f"Note: AI in Safe Learning Mode for pattern study.")

        if os.path.exists(self.backup_q_table_file):
            os.remove(self.backup_q_table_file)
        return ('Player' if self.player_score > self.ai_score else 'AI' if self.ai_score > self.player_score else 'Tie',
                (self.player_score, self.ai_score), player_moves)

class MetaAISupervisor:
    def __init__(self, input_size=20, hidden_size=32, learning_rate=0.001, buffer_size=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MetaAINeuralNetwork(input_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.replay_buffer = deque(maxlen=buffer_size)
        self.input_size = input_size
        self.min_history_length = 2
        self.max_history_length = 5
        self.min_alpha = 0.05
        self.max_alpha = 0.5
        self.min_epsilon = 0.005
        self.max_epsilon = 0.3
        self.min_freq_bias = 0.0
        self.max_freq_bias = 0.3
        self.model_file = "meta_ai_model.pth"
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_file):
            try:
                state_dict = torch.load(self.model_file, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print("Loaded Meta-AI model from disk.")
            except Exception as e:
                print(f"Error loading Meta-AI model: {e}. Starting with fresh model.")
                if os.path.exists(self.model_file):
                    backup_model = f"meta_ai_model_backup_{int(time.time())}.pth"
                    shutil.move(self.model_file, backup_model)
                    print(f"Backed up incompatible model to {backup_model}.")

    def save_model(self):
        try:
            torch.save(self.model.state_dict(), self.model_file)
        except Exception as e:
            print(f"Error saving Meta-AI model: {e}")

    def get_ngram_counts(self, moves, n=3):
        if len(moves) < n:
            return 0
        count = 0
        for i in range(len(moves) - n + 1):
            if all(moves[i + j] == moves[i] for j in range(n)):
                count += 1
            elif n == 3 and moves[i:i+3] in [['S', 'R', 'P'], ['R', 'P', 'S'], ['P', 'S', 'R']]:
                count += 1
        return count / (len(moves) - n + 1) if len(moves) >= n else 0.0

    def get_features(self, ai, player_score, ai_score, player_moves, safe_mode_rounds):
        win_rate = ai_score / 15.0
        tie_rate = (15 - player_score - ai_score) / 15.0
        loss_rate = player_score / 15.0
        move_counts = {'R': 0, 'P': 0, 'S': 0}
        for move in player_moves:
            move_counts[move] += 1
        move_freq = [move_counts[m] / 15.0 for m in ['R', 'P', 'S']]
        repeats = sum(1 for i in range(len(player_moves)-1) if player_moves[i] == player_moves[i+1])
        transitions = sum(1 for i in range(len(player_moves)-1) if player_moves[i] != player_moves[i+1])
        repeat_rate = repeats / 14.0 if repeats > 0 else 0.0
        transition_rate = transitions / 14.0 if transitions > 0 else 0.0
        q_values = [q for state in ai.q_table.values() for q in state]
        q_mean = np.mean(q_values) if q_values else 0.0
        q_std = np.std(q_values) if q_values else 0.0
        sss_ngram = self.get_ngram_counts(player_moves, n=3)
        cycle_ngram_3 = self.get_ngram_counts(player_moves, n=3)
        cycle_ngram_4 = self.get_ngram_counts(player_moves, n=4)
        recent_wins = sum(1 for i in range(max(0, len(player_moves)-5), len(player_moves))
                         if ai.determine_winner(player_moves[i], ai.history[2*i+1])[0] == 'AI')
        recent_win_rate = recent_wins / 5.0 if len(player_moves) >= 5 else ai_score / 15.0
        move_probs = [move_counts[m] / 15.0 for m in ['R', 'P', 'S']]
        move_entropy = -sum(p * np.log2(p + 1e-10) for p in move_probs if p > 0)
        current_params = [ai.history_length, ai.alpha, ai.epsilon, ai.freq_bias_weight]
        safe_mode_ratio = safe_mode_rounds / 15.0
        features = [
            win_rate, tie_rate, loss_rate,           # 3
            *move_freq,                              # 3
            repeat_rate, transition_rate,            # 2
            q_mean, q_std,                           # 2
            sss_ngram, cycle_ngram_3, cycle_ngram_4, # 3
            recent_win_rate, move_entropy,           # 2
            *current_params,                         # 4
            safe_mode_ratio                          # 1 (new: tracks safe mode usage)
        ]  # Total: 3 + 3 + 2 + 2 + 3 + 2 + 4 + 1 = 20
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def get_target_params(self, ai_score, player_score, tie_count, player_moves, safe_mode_rounds):
        reward = ai_score - player_score
        norm_reward = (reward + 15) / 30
        player_win_rate = player_score / 15.0
        move_counts = {'R': 0, 'P': 0, 'S': 0}
        for move in player_moves:
            move_counts[move] += 1
        move_freq = [move_counts[m] / 15.0 for m in ['R', 'P', 'S']]
        freq_imbalance = max(move_freq) - min(move_freq)
        safe_mode_factor = 1.0 + safe_mode_rounds / 15.0  # Boost history_length for safe mode
        target_history_length = self.min_history_length + (self.max_history_length - self.min_history_length) * norm_reward * safe_mode_factor
        target_alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * norm_reward
        target_epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * (1 - norm_reward + player_win_rate) / 2
        target_freq_bias = self.min_freq_bias + (self.max_freq_bias - self.min_freq_bias) * (norm_reward + freq_imbalance) / 2
        return torch.tensor([
            target_history_length,
            target_alpha,
            target_epsilon,
            target_freq_bias
        ], dtype=torch.float32).to(self.device)

    def store_experience(self, features, target_params, is_safe_mode=False):
        # Store with higher weight for safe mode experiences
        self.replay_buffer.append((features, target_params, 2.0 if is_safe_mode else 1.0))

    def train(self, batch_size=16):
        if len(self.replay_buffer) < batch_size:
            return None
        batch = random.sample(self.replay_buffer, batch_size)
        features, targets, weights = zip(*batch)
        features = torch.stack(features).to(self.device)
        targets = torch.stack(targets).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        total_loss = 0.0
        for _ in range(3):
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            weighted_loss = (loss * weights).mean()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += weighted_loss.item()
        self.save_model()
        print(f"Meta-AI Training Loss: {total_loss / 3:.4f}")
        return total_loss / 3

    def suggest_parameters(self, ai, player_score, ai_score, player_moves, safe_mode_rounds):
        features = self.get_features(ai, player_score, ai_score, player_moves, safe_mode_rounds)
        with torch.no_grad():
            suggested_params = self.model(features).cpu().numpy()
        history_length = max(self.min_history_length, min(self.max_history_length, int(round(suggested_params[0] * (1 + features[-1].item())))))
        alpha = max(self.min_alpha, min(self.max_alpha, suggested_params[1]))
        epsilon = max(self.min_epsilon, min(self.max_epsilon, suggested_params[2]))
        freq_bias_weight = max(self.min_freq_bias, min(self.max_freq_bias, suggested_params[3]))
        return {
            'history_length': history_length,
            'alpha': alpha,
            'epsilon': epsilon,
            'freq_bias_weight': freq_bias_weight
        }

def main():
    ai = RPS_AI()
    meta_ai = MetaAISupervisor(input_size=20, hidden_size=32, learning_rate=0.001, buffer_size=1000)
    game_count = 0
    
    try:
        while True:
            game_count += 1
            print(f"\n=== Game {game_count} ===")
            winner, scores, player_moves = ai.play_game()
            if winner is None:
                print("Game ended. Thanks for playing!")
                break
            player_score, ai_score = scores
            tie_count = 15 - player_score - ai_score
            safe_mode_rounds = sum(1 for i in range(len(player_moves)) if ai.is_win_guaranteed() and i >= ai.current_round - len(player_moves))
            try:
                features = meta_ai.get_features(ai, player_score, ai_score, player_moves, safe_mode_rounds)
                target_params = meta_ai.get_target_params(ai_score, player_score, tie_count, player_moves, safe_mode_rounds)
                meta_ai.store_experience(features, target_params, is_safe_mode=safe_mode_rounds > 0)
                loss = meta_ai.train(batch_size=16)
                new_params = meta_ai.suggest_parameters(ai, player_score, ai_score, player_moves, safe_mode_rounds)
                print(f"Meta-AI Suggested Params: {new_params}")
                ai.history_length = new_params['history_length']
                ai.alpha = new_params['alpha']
                ai.epsilon = new_params['epsilon']
                ai.freq_bias_weight = new_params['freq_bias_weight']
                ai.move_counts = {'R': 0, 'P': 0, 'S': 0}
                loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                with open("rps_game_log.txt", "a") as f:
                    f.write(f"Game {game_count}, AI Wins: {ai_score}, Ties: {tie_count}, Player Wins: {player_score}, "
                            f"Params: history_length={ai.history_length}, alpha={ai.alpha:.3f}, "
                            f"epsilon={ai.epsilon:.3f}, freq_bias={ai.freq_bias_weight:.3f}, "
                            f"Loss: {loss_str}, Safe Mode Rounds: {safe_mode_rounds}, "
                            f"Features: {features.cpu().numpy().tolist()}\n")
            except Exception as e:
                print(f"Error during Meta-AI processing: {e}. Continuing with current parameters.")
            
            print(f"\nGame Over! Final Score - You: {player_score}, AI: {ai_score}")
            print(f"Winner: {winner}")
            
    except KeyboardInterrupt:
        print("\nProgram interrupted. Restoring Q-table...")
        ai.restore_q_table()
        print("Thanks for playing!")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}. Restoring Q-table...")
        ai.restore_q_table()
        print("Thanks for playing!")
        sys.exit(1)

if __name__ == "__main__":
    main()