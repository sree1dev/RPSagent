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
from torch.utils.data import DataLoader, TensorDataset

class MetaAINeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=32):  # Reduced hidden size for RTX 2050
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

    def choose_action(self, state):
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
            return 'Tie', 0
        elif (player_move == 'R' and ai_move == 'S') or \
             (player_move == 'P' and ai_move == 'R') or \
             (player_move == 'S' and ai_move == 'P'):
            return 'Player', -1
        else:
            return 'AI', 1

    def play_round(self, player_move):
        state = self.get_state()
        self.move_counts[player_move] += 1
        ai_move = self.choose_action(state)
        winner, reward = self.determine_winner(player_move, ai_move)
        next_state = self.get_state() + (player_move, ai_move)
        self.history.extend([player_move, ai_move])
        self.update_q_table(state, ai_move, reward, next_state)
        return ai_move, winner

    def get_valid_keypress(self):
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').upper()
                if key in ['R', 'P', 'S', 'Q']:
                    return key
                print(f"\nInvalid key '{key}'! Press R, P, S, or Q: ", end='', flush=True)
            time.sleep(0.01)

    def get_test_keypress(self, pattern, move_idx):
        return pattern[move_idx % len(pattern)]

    def play_game(self, first_move=None, test_pattern=None):
        player_score = 0
        ai_score = 0
        self.history = []
        player_moves = []
        self.backup_q_table()
        print("Press R (Rock), P (Paper), S (Scissors), or Q to quit.")

        start_round = 1
        if first_move in ['R', 'P', 'S']:
            print(f"\nRound 1/15 - Your move: {first_move}")
            player_moves.append(first_move)
            ai_move, winner = self.play_round(first_move)
            if not self.validate_q_table():
                self.restore_q_table()
                return None, None, None
            self.save_q_table()
            print(f"AI plays: {ai_move}")
            print(f"Result: {winner}")
            if winner == 'Player':
                player_score += 1
            elif winner == 'AI':
                ai_score += 1
            print(f"Score - You: {player_score}, AI: {ai_score}")
            start_round = 2

        for round_num in range(start_round, 16):
            print(f"\nRound {round_num}/15 - Your move (R/P/S): ", end='', flush=True)
            if test_pattern:
                key = self.get_test_keypress(test_pattern, round_num - start_round)
            else:
                key = self.get_valid_keypress()
            if key == 'Q':
                self.restore_q_table()
                return None, None, None
            player_move = key
            player_moves.append(player_move)
            print(f"{player_move}")
            ai_move, winner = self.play_round(player_move)
            if not self.validate_q_table():
                self.restore_q_table()
                return None, None, None
            self.save_q_table()
            print(f"\nAI plays: {ai_move}")
            print(f"Result: {winner}")
            if winner == 'Player':
                player_score += 1
            elif winner == 'AI':
                ai_score += 1
            print(f"Score - You: {player_score}, AI: {ai_score}")

        if os.path.exists(self.backup_q_table_file):
            os.remove(self.backup_q_table_file)
        return ('Player' if player_score > ai_score else 'AI' if ai_score > player_score else 'Tie',
                (player_score, ai_score), player_moves)

class MetaAISupervisor:
    def __init__(self, input_size=14, hidden_size=32, learning_rate=0.001, buffer_size=500):  # Reduced buffer size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MetaAINeuralNetwork(input_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.replay_buffer = deque(maxlen=buffer_size)
        self.input_size = input_size
        self.min_history_length = 2
        self.max_history_length = 4
        self.min_alpha = 0.05
        self.max_alpha = 0.3
        self.min_epsilon = 0.01
        self.max_epsilon = 0.1
        self.min_freq_bias = 0.0
        self.max_freq_bias = 0.2
        self.model_file = "meta_ai_model.pth"
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_file):
            try:
                self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
                print("Loaded Meta-AI model from disk.")
            except Exception as e:
                print(f"Error loading Meta-AI model: {e}. Starting with fresh model.")

    def save_model(self):
        try:
            torch.save(self.model.state_dict(), self.model_file)
        except Exception as e:
            print(f"Error saving Meta-AI model: {e}")

    def get_features(self, ai, player_score, ai_score, player_moves):
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
        current_params = [ai.history_length, ai.alpha, ai.epsilon, ai.freq_bias_weight]
        features = [
            win_rate, tie_rate, loss_rate,
            *move_freq,
            repeat_rate, transition_rate,
            q_mean, q_std,
            *current_params
        ]
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def get_target_params(self, ai_score, player_score):
        reward = ai_score - player_score
        target_history_length = self.min_history_length + (self.max_history_length - self.min_history_length) * (reward + 15) / 30
        target_alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (reward + 15) / 30
        target_epsilon = self.max_epsilon - (self.max_epsilon - self.min_epsilon) * (reward + 15) / 30
        target_freq_bias = self.min_freq_bias + (self.max_freq_bias - self.min_freq_bias) * (reward + 15) / 30
        return torch.tensor([target_history_length, target_alpha, target_epsilon, target_freq_bias], dtype=torch.float32).to(self.device)

    def store_experience(self, features, target_params):
        self.replay_buffer.append((features, target_params))

    def train(self, batch_size=16):  # Smaller batch size for RTX 2050
        if len(self.replay_buffer) < batch_size:
            return None
        batch = random.sample(self.replay_buffer, batch_size)
        features, targets = zip(*batch)
        features = torch.stack(features).to(self.device)
        targets = torch.stack(targets).to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(features)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        self.save_model()
        return loss.item()

    def suggest_parameters(self, ai, player_score, ai_score, player_moves):
        features = self.get_features(ai, player_score, ai_score, player_moves)
        with torch.no_grad():
            suggested_params = self.model(features).cpu().numpy()
        
        history_length = max(self.min_history_length, min(self.max_history_length, int(round(suggested_params[0]))))
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
    meta_ai = MetaAISupervisor(input_size=14, hidden_size=32, learning_rate=0.001, buffer_size=500)
    patterns = [
        ['P', 'S', 'R', 'R', 'S', 'P', 'P', 'R', 'S', 'S', 'P', 'R', 'P', 'S', 'R'],  # Mirror Trap
        ['R', 'P', 'S', 'R', 'P', 'S', 'R', 'R', 'P', 'P', 'S', 'S', 'R', 'P', 'S'],  # Cycle Breaker
        ['S', 'S', 'S', 'P', 'P', 'R', 'R', 'S', 'S', 'P', 'P', 'R', 'S', 'P', 'R'],  # Repetition Lure
        ['P', 'S', 'R', 'P', 'S', 'R', 'S', 'P', 'R', 'S', 'P', 'R', 'P', 'S', 'R'],  # False Cycle
        ['R', 'P', 'R', 'S', 'P', 'S', 'R', 'P', 'S', 'R', 'P', 'S', 'R', 'P', 'S'],  # Switchback Trap
        ['P', 'P', 'S', 'S', 'R', 'R', 'P', 'S', 'R', 'P', 'S', 'R', 'P', 'S', 'R'],  # Double Deception
        ['S', 'R', 'P', 'S', 'R', 'P', 'R', 'S', 'P', 'R', 'S', 'P', 'S', 'R', 'P'],  # Random Facade
        ['R', 'S', 'P', 'R', 'S', 'P', 'S', 'R', 'P', 'S', 'R', 'P', 'R', 'S', 'P'],  # Memory Overload
        ['P', 'R', 'S', 'P', 'P', 'R', 'S', 'S', 'P', 'R', 'R', 'S', 'P', 'S', 'R'],  # Bluffing Sequence
        ['S', 'S', 'P', 'R', 'R', 'S', 'P', 'P', 'R', 'S', 'S', 'P', 'R', 'P', 'S'],  # Frequency Disruptor
        ['R', 'S', 'P', 'P', 'S', 'R', 'R', 'P', 'S', 'S', 'R', 'P', 'P', 'S', 'R'],  # Reverse Mirror
        ['P', 'R', 'S', 'R', 'P', 'S', 'P', 'R', 'S', 'P', 'R', 'S', 'R', 'P', 'S'],  # Pattern Feint
        ['R', 'R', 'P', 'P', 'S', 'S', 'R', 'R', 'P', 'P', 'S', 'S', 'R', 'P', 'S'],  # Tie Inducer
        ['S', 'P', 'R', 'S', 'R', 'P', 'S', 'P', 'R', 'S', 'R', 'P', 'S', 'P', 'R'],  # Chaos Switch
        ['R', 'P', 'S', 'R', 'P', 'S', 'P', 'P', 'S', 'S', 'R', 'R', 'P', 'S', 'R'],  # Delayed Repeat
    ]
    try:
        for pattern_idx, pattern in enumerate(patterns, 1):
            print(f"\nTesting Pattern {pattern_idx}: {pattern}")
            for game_num in range(10):
                print(f"\nGame {game_num + 1}")
                winner, scores, player_moves = ai.play_game(test_pattern=pattern)
                if winner is None:
                    print("Game ended. Thanks for playing!")
                    break
                player_score, ai_score = scores
                with open("test_log_optimized.txt", "a") as f:
                    f.write(f"Pattern {pattern_idx}: {pattern}, Game {game_num + 1}, AI Wins: {ai_score}, Ties: {15 - player_score - ai_score}, Player Wins: {player_score}, Params: history_length={ai.history_length}, alpha={ai.alpha:.3f}, epsilon={ai.epsilon:.3f}, freq_bias={ai.freq_bias_weight:.3f}\n")
                print(f"\nGame Over! Final Score - You: {player_score}, AI: {ai_score}")
                print(f"OVERALL Winner: {winner}")
                
                features = meta_ai.get_features(ai, player_score, ai_score, player_moves)
                target_params = meta_ai.get_target_params(ai_score, player_score)
                meta_ai.store_experience(features, target_params)
                loss = meta_ai.train(batch_size=16)
                if loss is not None:
                    print(f"Meta-AI Training Loss: {loss:.4f}")
                
                new_params = meta_ai.suggest_parameters(ai, player_score, ai_score, player_moves)
                print(f"Meta-AI Suggested Params: {new_params}")
                ai.history_length = new_params['history_length']
                ai.alpha = new_params['alpha']
                ai.epsilon = new_params['epsilon']
                ai.freq_bias_weight = new_params['freq_bias_weight']
                
                ai.move_counts = {'R': 0, 'P': 0, 'S': 0}
            
            ai.q_table = defaultdict(lambda: np.zeros(len(ai.actions)))
            ai.save_q_table()
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
