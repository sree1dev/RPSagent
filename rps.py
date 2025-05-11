import numpy as np
import pickle
import os
from collections import defaultdict
import random
import msvcrt
import time
import sys
import tempfile
import shutil
import errno

class RPS_AI:
    def __init__(self, history_length=2, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.history_length = history_length
        self.actions = ['R', 'P', 'S']
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        self.history = []
        self.q_table_file = "rps_q_table.pkl"
        self.backup_q_table_file = "backup_q_table.pkl"
        self.load_q_table()

    def load_q_table(self):
        """Load Q-table from file if it exists."""
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'rb') as f:
                    loaded_q_table = pickle.load(f)
                    self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), loaded_q_table)
            except Exception as e:
                print(f"Error loading Q-table: {e}. Starting with fresh Q-table.")

    def save_q_table(self, filename=None, retries=3, delay=0.1):
        """Save Q-table to file with atomic write and retries to prevent corruption."""
        if filename is None:
            filename = self.q_table_file
        for attempt in range(retries):
            try:
                temp_fd, temp_path = tempfile.mkstemp()
                try:
                    with open(temp_path, 'wb') as f:
                        pickle.dump(dict(self.q_table), f)
                    os.close(temp_fd)  # Explicitly close file descriptor
                    shutil.move(temp_path, filename)
                    return True
                except:
                    os.close(temp_fd)
                    os.remove(temp_path) if os.path.exists(temp_path) else None
                    raise
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay)  # Wait before retrying
                    continue
                print(f"Error saving Q-table to {filename}: {e}")
                return False
        return False

    def backup_q_table(self):
        """Backup current Q-table before starting a game."""
        if not self.save_q_table(self.backup_q_table_file):
            print("Warning: Failed to create Q-table backup. Proceeding without backup.")

    def restore_q_table(self):
        """Restore Q-table from backup if it exists."""
        if os.path.exists(self.backup_q_table_file):
            try:
                with open(self.backup_q_table_file, 'rb') as f:
                    loaded_q_table = pickle.load(f)
                    self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), loaded_q_table)
                self.save_q_table()  # Save restored Q-table to main file
                os.remove(self.backup_q_table_file)  # Delete backup
            except Exception as e:
                print(f"Error restoring Q-table: {e}. Keeping current Q-table.")

    def validate_q_table(self):
        """Validate Q-table to detect potential corruption."""
        try:
            q_values = [q for state in self.q_table.values() for q in state]
            if q_values:
                max_q = max(abs(q) for q in q_values)
                if max_q > 100:  # Arbitrary threshold for extreme values
                    print("Warning: Extreme Q-values detected. Restoring Q-table.")
                    return False
            return True
        except Exception:
            print("Error validating Q-table. Restoring Q-table.")
            return False

    def get_state(self):
        """Get current state based on move history."""
        if len(self.history) < self.history_length * 2:
            return tuple(['-'] * (self.history_length * 2))
        return tuple(self.history[-self.history_length * 2:])

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = self.q_table[state]
        return self.actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning formula."""
        action_idx = self.actions.index(action)
        current_q = self.q_table[state][action_idx]
        next_q = np.max(self.q_table[next_state])
        self.q_table[state][action_idx] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

    def determine_winner(self, human_move, ai_move):
        """Determine round winner and return reward."""
        if human_move == ai_move:
            return 'Tie', 0
        elif (human_move == 'R' and ai_move == 'S') or \
             (human_move == 'P' and ai_move == 'R') or \
             (human_move == 'S' and ai_move == 'P'):
            return 'human', -1
        else:
            return 'AI', 1

    def play_round(self, human_move):
        """Play a single round and update Q-table."""
        state = self.get_state()
        ai_move = self.choose_action(state)
        winner, reward = self.determine_winner(human_move, ai_move)
        next_state = self.get_state() + (human_move, ai_move)
        self.history.extend([human_move, ai_move])
        self.update_q_table(state, ai_move, reward, next_state)
        return ai_move, winner

    def get_valid_keypress(self):
        """Get a valid keypress (R, P, S, or Q), and display it."""
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').upper()
                if key in ['R', 'P', 'S', 'Q']:
                    print(key)  # Echo the key so the user sees their input
                    return key
                print(f"\nInvalid key '{key}'! Press R, P, S, or Q: ", end='', flush=True)
            time.sleep(0.01)


    def play_game(self, first_move=None):
        """Play a 15-round game with per-round Q-table updates."""
        human_score = 0
        ai_score = 0
        self.history = []  # Reset history for new game
        self.backup_q_table()  # Backup Q-table before game
        print("Press R (Rock), P (Paper), S (Scissors), or Q to quit.")

        start_round = 1
        if first_move in ['R', 'P', 'S']:
            # Play first round with provided move
            print(f"\nRound 1/15 - Your move: {first_move}")
            ai_move, winner = self.play_round(first_move)
            if not self.validate_q_table():
                self.restore_q_table()
                return None, None
            self.save_q_table()
            print(f"AI plays: {ai_move}")
            print(f"Result: {winner}")
            if winner == 'human':
                human_score += 1
            elif winner == 'AI':
                ai_score += 1
            print(f"Score - You: {human_score}, AI: {ai_score}")
            start_round = 2

        for round_num in range(start_round, 16):
            print(f"\nRound {round_num}/15 - Your move (R/P/S): ", end='', flush=True)
            key = self.get_valid_keypress()
            if key == 'Q':
                self.restore_q_table()  # Revert to backup on quit
                return None, None

            human_move = key
            ai_move, winner = self.play_round(human_move)
            if not self.validate_q_table():
                self.restore_q_table()
                return None, None
            self.save_q_table()
            print(f"\nAI plays: {ai_move}")
            print(f"Result: {winner}")

            if winner == 'human':
                human_score += 1
            elif winner == 'AI':
                ai_score += 1

            print(f"Score - You: {human_score}, AI: {ai_score}")

        if os.path.exists(self.backup_q_table_file):
            os.remove(self.backup_q_table_file)
        return ('human' if human_score > ai_score else 'AI' if ai_score > human_score else 'Tie',
                (human_score, ai_score))

def main():
    ai = RPS_AI()
    try:
        while True:
            print("\n=== Rock Paper Scissors - 15 Rounds ===")
            winner, scores = ai.play_game()
            if winner is None:
                print("Game ended")
                break
            human_score, ai_score = scores
            print(f"\nGame Over! Final Score - You: {human_score}, AI: {ai_score}")
            print(f"OVERALL Winner: {winner}")
            print("\nPress Y, R, P, or S to play again, any other key to exit: ", end='', flush=True)
            while not msvcrt.kbhit():
                time.sleep(0.01)
            key = msvcrt.getch().decode('utf-8').upper()
            if key not in ['Y', 'R', 'P', 'S']:
                print("\nThanks for playing!")
                break
            # If R, P, or S, pass as first move; Y starts normally
            first_move = key if key in ['R', 'P', 'S'] else None
            winner, scores = ai.play_game(first_move=first_move)
            if winner is None:
                print("Game ended. Thanks for playing!")
                break
            human_score, ai_score = scores
            print(f"\nGame Over! Final Score - You: {human_score}, AI: {ai_score}")
            print(f"OVERALL Winner: {winner}")
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