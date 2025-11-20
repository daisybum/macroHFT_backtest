"""
Low-level Agents Fine-tuning for Bitcoin (with Resume Support)
기존 ETHUSDT 모델을 BTCUSDT 데이터로 fine-tuning
"""

import pathlib
import sys
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import signal

ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from MacroHFT.model.net import subagent
from MacroHFT.env.low_level_env import Testing_Env, Training_Env
from MacroHFT.RL.util.replay_buffer import ReplayBuffer
from MacroHFT.RL.util.utili import LinearDecaySchedule

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size", type=int, default=500000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--epsilon_start", type=float, default=0.3)
parser.add_argument("--epsilon_end", type=float, default=0.05)
parser.add_argument("--decay_length", type=int, default=3)
parser.add_argument("--update_times", type=int, default=5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--transcation_cost", type=float, default=2.0 / 10000)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--n_step", type=int, default=1)
parser.add_argument("--epoch_number", type=int, default=3)
parser.add_argument("--label", type=str, default="label_1")
parser.add_argument("--clf", type=str, default="slope")
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--pretrained_model", type=str, required=True, help="Path to ETHUSDT pretrained model")

args = parser.parse_args()

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class FineTuneDQN:
    def __init__(self, args):
        print("\n" + "="*70)
        print(f"Fine-tuning Low-level Agent: {args.clf} - {args.label}")
        print("="*70)
        
        self.seed = args.seed
        seed_torch(self.seed)
        
        self.device = torch.device(args.device)
        print(f"Device: {self.device}")
        
        # Paths
        self.dataset = "BTCUSDT"
        self.clf = args.clf
        self.label = int(args.label.split('_')[1])
        self.result_path = os.path.join("./MacroHFT/result/low_level", 
                                        self.dataset, self.clf, str(int(args.alpha)), args.label)
        self.model_path = os.path.join(self.result_path, f"seed_{self.seed}")
        os.makedirs(self.model_path, exist_ok=True)
        self.checkpoint_path = os.path.join(self.model_path, "checkpoint.pth")
        
        # Data paths
        self.train_data_path = os.path.join(ROOT, "MacroHFT", "data", self.dataset, "train")
        self.val_data_path = os.path.join(ROOT, "MacroHFT", "data", self.dataset, "val")
        
        # Load labels
        label_file = f'{self.clf}_labels.pkl'
        with open(os.path.join(self.train_data_path, label_file), 'rb') as f:
            self.train_index = pickle.load(f)
        with open(os.path.join(self.val_data_path, label_file), 'rb') as f:
            self.val_index = pickle.load(f)
        
        print(f"\nTrain chunks for label {self.label}: {len(self.train_index[self.label])}")
        print(f"Val chunks for label {self.label}: {len(self.val_index[self.label])}")
        
        # Model parameters
        self.max_holding_number = 0.01  # BTC
        self.transcation_cost = args.transcation_cost
        self.epoch_number = args.epoch_number
        
        # Feature lists
        feature_path = os.path.join(ROOT, "MacroHFT", "data", "feature_list")
        self.tech_indicator_list = np.load(os.path.join(feature_path, 'single_features.npy'), 
                                          allow_pickle=True).tolist()
        self.tech_indicator_list_trend = np.load(os.path.join(feature_path, 'trend_features.npy'), 
                                                 allow_pickle=True).tolist()
        
        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        
        print(f"\nState dimensions: {self.n_state_1} (single) + {self.n_state_2} (trend)")
        
        # Initialize networks
        self.eval_net = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.target_net = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        
        # Training parameters
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.loss_func = nn.MSELoss()
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.buffer_size = args.buffer_size
        self.update_times = args.update_times
        
        # Epsilon schedule
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_scheduler = LinearDecaySchedule(
            start_epsilon=self.epsilon_start, 
            end_epsilon=self.epsilon_end, 
            decay_length=args.decay_length
        )
        self.epsilon = self.epsilon_start
        self.alpha = args.alpha
        
        # Resume state
        self.start_epoch = 0
        self.start_chunk_idx = 0
        self.best_return_rate = -float('inf')
        self.random_list = None

        # Attempt to load checkpoint
        self.load_checkpoint()
        
        if self.start_epoch == 0 and self.start_chunk_idx == 0:
             # Load pretrained model only if not resuming
            print(f"\n[Loading pretrained model from: {args.pretrained_model}]")
            try:
                pretrained_state = torch.load(args.pretrained_model, map_location=self.device)
                self.eval_net.load_state_dict(pretrained_state)
                self.target_net.load_state_dict(pretrained_state)
                print("[OK] Pretrained model loaded successfully!")
            except Exception as e:
                print(f"[WARNING] Could not load pretrained model: {e}")
                print("[INFO] Starting with random initialization...")

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_signal)
        self.stop_requested = False
        
    def handle_signal(self, signum, frame):
        print("\n[INFO] Interrupt received! Stopping after current chunk...")
        self.stop_requested = True

    def save_checkpoint(self, epoch, chunk_idx, random_list):
        checkpoint = {
            'epoch': epoch,
            'chunk_idx': chunk_idx,
            'eval_net': self.eval_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_return_rate': self.best_return_rate,
            'random_list': random_list
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            print(f"\n[INFO] Found checkpoint: {self.checkpoint_path}")
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.eval_net.load_state_dict(checkpoint['eval_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epsilon = checkpoint['epsilon']
                self.start_epoch = checkpoint['epoch']
                self.start_chunk_idx = checkpoint['chunk_idx']
                self.best_return_rate = checkpoint['best_return_rate']
                self.random_list = checkpoint['random_list']
                print(f"[RESUMING] From Epoch {self.start_epoch}, Chunk {self.start_chunk_idx}")
            except Exception as e:
                print(f"[ERROR] Failed to load checkpoint: {e}")
                print("[WARNING] Starting from scratch...")
        else:
            print("[INFO] No checkpoint found. Starting fresh.")

    def ensure_features(self, df):
        """Fill missing features with 0.0"""
        for feat in self.tech_indicator_list:
            if feat not in df.columns:
                df[feat] = 0.0
        for feat in self.tech_indicator_list_trend:
            if feat not in df.columns:
                df[feat] = 0.0
        return df

    def act(self, state, state_trend, info):
        """Select action with epsilon-greedy"""
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device), 0
        )
        
        if np.random.uniform() < (1 - self.epsilon):
            with torch.no_grad():
                actions_value = self.eval_net(x1, x2, previous_action)
                action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:
            action = random.choice([0, 1])
        
        return action
    
    def act_test(self, state, state_trend, info):
        """Select action greedily (no exploration)"""
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long(), 0
        ).to(self.device)
        
        with torch.no_grad():
            actions_value = self.eval_net(x1, x2, previous_action)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        
        return action
    
    def update(self, replay_buffer):
        """Update network parameters"""
        self.eval_net.train()
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Double DQN
        a_argmax = self.eval_net(
            batch['next_state'], 
            batch['next_state_trend'], 
            batch['next_previous_action']
        ).argmax(dim=-1, keepdim=True)
        
        q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(
            batch['next_state'], 
            batch['next_state_trend'], 
            batch['next_previous_action']
        ).gather(-1, a_argmax).squeeze(-1)
        
        q_distribution = self.eval_net(batch['state'], batch['state_trend'], batch['previous_action'])
        q_current = q_distribution.gather(-1, batch['action']).squeeze(-1)
        
        # TD loss
        td_error = self.loss_func(q_current, q_target)
        
        # KL divergence loss (imitation learning)
        demonstration = batch['demo_action']
        KL_loss = F.kl_div(
            (q_distribution.softmax(dim=-1) + 1e-8).log(),
            (demonstration.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )
        
        # Combined loss
        loss = td_error + self.alpha * KL_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1)
        self.optimizer.step()
        
        # Soft update target network
        for param, target_param in zip(self.eval_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.eval_net.eval()
        return td_error.item(), KL_loss.item(), q_current.mean().item(), q_target.mean().item()
    
    def train(self):
        """Fine-tuning loop"""
        print("\n" + "="*70)
        print("Starting Fine-tuning (Resume-enabled)")
        print("="*70)
        
        self.replay_buffer = ReplayBuffer(args, self.n_state_1, self.n_state_2, self.n_action)
        
        df_list = self.train_index[self.label]
        
        for epoch in range(self.start_epoch, self.epoch_number):
            if self.stop_requested:
                break
                
            print(f"\n[Epoch {epoch + 1}/{self.epoch_number}]")
            
            # Shuffle training data if starting a new epoch or if random_list wasn't saved
            if self.random_list is None:
                random_list = self.train_index[self.label].copy()
                random.shuffle(random_list)
                self.random_list = random_list
            else:
                random_list = self.random_list
            
            epoch_rewards = []
            epoch_returns = []
            
            # Training loop
            pbar = tqdm(random_list, desc="Training")
            for i, df_index in enumerate(pbar):
                # Skip already processed chunks if resuming within an epoch
                if i < self.start_chunk_idx:
                    continue
                
                if self.stop_requested:
                    print("\n[INFO] Stop requested by user. Saving and exiting...")
                    self.save_checkpoint(epoch, i, random_list)
                    return self.best_return_rate

                df = pd.read_feather(os.path.join(self.train_data_path, f"df_{df_index}.feather"))
                df = self.ensure_features(df)
                
                train_env = Training_Env(
                    df=df,
                    tech_indicator_list=self.tech_indicator_list,
                    tech_indicator_list_trend=self.tech_indicator_list_trend,
                    transcation_cost=self.transcation_cost,
                    max_holding_number=self.max_holding_number,
                    alpha=self.alpha
                )
                
                state, state_trend, info = train_env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action = self.act(state, state_trend, info)
                    next_state, next_state_trend, reward, done, next_info = train_env.step(action)
                    
                    self.replay_buffer.add(
                        state, state_trend, action, reward, next_state, next_state_trend, 
                        done, info, next_info
                    )
                    
                    episode_reward += reward
                    state, state_trend, info = next_state, next_state_trend, next_info
                    
                    # Update network
                    if len(self.replay_buffer) >= self.batch_size:
                        for _ in range(self.update_times):
                            self.update(self.replay_buffer)
                
                epoch_rewards.append(episode_reward)
                epoch_returns.append(train_env.portfolio_return)
                
                # Save checkpoint periodically (every chunk to be safe on CPU)
                self.save_checkpoint(epoch, i + 1, random_list)
            
            # End of epoch
            self.start_chunk_idx = 0 # Reset chunk index for next epoch
            self.random_list = None # Reset random list for next epoch
            
            # Update epsilon
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch)
            
            # Print epoch stats
            if epoch_rewards:
                avg_reward = np.mean(epoch_rewards)
                avg_return = np.mean(epoch_returns)
                print(f"  Avg Reward: {avg_reward:.4f}")
                print(f"  Avg Return: {avg_return:.4f}%")
                print(f"  Epsilon: {self.epsilon:.4f}")
            
            # Validation
            val_return = self.validate()
            print(f"  Val Return: {val_return:.4f}%")
            
            # Save best model
            if val_return > self.best_return_rate:
                self.best_return_rate = val_return
                save_path = os.path.join(self.model_path, "best_model.pkl")
                torch.save(self.eval_net.state_dict(), save_path)
                print(f"  [NEW BEST] Model saved: {save_path}")
            
            # Checkpoint at end of epoch
            self.save_checkpoint(epoch + 1, 0, None)
        
        print("\n" + "="*70)
        print("Fine-tuning Complete!")
        print(f"Best validation return: {self.best_return_rate:.4f}%")
        print("="*70)
        
        return self.best_return_rate
    
    def validate(self):
        """Validate on validation set"""
        self.eval_net.eval()
        
        val_returns = []
        val_df_list = self.val_index[self.label]
        
        for df_index in val_df_list[:min(5, len(val_df_list))]:  # Validate on subset
            df = pd.read_feather(os.path.join(self.val_data_path, f"df_{df_index}.feather"))
            df = self.ensure_features(df)
            
            test_env = Testing_Env(
                df=df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                transcation_cost=self.transcation_cost,
                max_holding_number=self.max_holding_number
            )
            
            state, state_trend, info = test_env.reset()
            done = False
            
            while not done:
                action = self.act_test(state, state_trend, info)
                state, state_trend, _, done, info = test_env.step(action)
            
            val_returns.append(test_env.portfolio_return)
        
        return np.mean(val_returns) if val_returns else 0.0

if __name__ == "__main__":
    agent = FineTuneDQN(args)
    best_return = agent.train()
    
    print(f"\nFinal best return rate: {best_return:.4f}%")
