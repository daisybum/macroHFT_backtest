"""
High-level Agent Fine-tuning for Bitcoin (with Resume Support)
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
import joblib

ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from MacroHFT.model.net import subagent, hyperagent
from MacroHFT.env.high_level_env import Testing_Env, Training_Env
from MacroHFT.RL.util.replay_buffer import ReplayBuffer_High
from MacroHFT.RL.util.memory import episodicmemory
from MacroHFT.RL.util.utili import LinearDecaySchedule

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size", type=int, default=100000) # Reduced for faster save/load
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epsilon_start", type=float, default=0.5)
parser.add_argument("--epsilon_end", type=float, default=0.1)
parser.add_argument("--decay_length", type=int, default=3)
parser.add_argument("--update_times", type=int, default=5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--transcation_cost", type=float, default=2.0 / 10000)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--n_step", type=int, default=1)
parser.add_argument("--epoch_number", type=int, default=3)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--eval_update_freq", type=int, default=512)
parser.add_argument("--q_value_memorize_freq", type=int, default=10)

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

class FineTuneHighLevel:
    def __init__(self, args):
        print("\n" + "="*70)
        print(f"Fine-tuning High-level Agent (Hyperagent)")
        print("="*70)
        
        self.seed = args.seed
        seed_torch(self.seed)
        self.device = torch.device(args.device)
        print(f"Device: {self.device}")
        
        self.dataset = "BTCUSDT"
        self.result_path = os.path.join("./MacroHFT/result/high_level", self.dataset)
        self.model_path = os.path.join(self.result_path, f"seed_{self.seed}")
        os.makedirs(self.model_path, exist_ok=True)
        self.checkpoint_path = os.path.join(self.model_path, "checkpoint.pth")
        
        # Data paths
        self.train_data_path = os.path.join(ROOT, "MacroHFT", "data", self.dataset, "train")
        self.val_data_path = os.path.join(ROOT, "MacroHFT", "data", self.dataset, "val")
        
        # Feature lists
        feature_path = os.path.join(ROOT, "MacroHFT", "data", "feature_list")
        self.tech_indicator_list = np.load(os.path.join(feature_path, 'single_features.npy'), allow_pickle=True).tolist()
        self.tech_indicator_list_trend = np.load(os.path.join(feature_path, 'trend_features.npy'), allow_pickle=True).tolist()
        self.clf_list = ['slope_360', 'vol_360']
        
        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        
        # Load Sub-agents
        self.slope_agents = {}
        self.vol_agents = {}
        self.load_subagents()
        
        # Initialize Hyperagent
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target.load_state_dict(self.hyperagent.state_dict())
        
        self.optimizer = torch.optim.Adam(self.hyperagent.parameters(), lr=args.lr)
        self.loss_func = nn.MSELoss()
        
        # Memory & Buffer
        self.memory = episodicmemory(4320, 5, self.n_state_1, self.n_state_2, 64, self.device)
        self.replay_buffer = ReplayBuffer_High(args, self.n_state_1, self.n_state_2, self.n_action)
        
        # Params
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.beta = args.beta
        self.update_times = args.update_times
        self.epoch_number = args.epoch_number
        self.eval_update_freq = args.eval_update_freq
        
        self.epsilon_scheduler = LinearDecaySchedule(
            start_epsilon=args.epsilon_start, 
            end_epsilon=args.epsilon_end, 
            decay_length=args.decay_length
        )
        self.epsilon = args.epsilon_start
        
        # Resume state
        self.start_epoch = 0
        self.start_chunk_idx = 0
        self.best_return_rate = -float('inf')
        self.total_step_counter = 0
        
        # Load Checkpoint
        self.load_checkpoint()
        
        # Signal handler
        signal.signal(signal.SIGINT, self.handle_signal)
        self.stop_requested = False

    def ensure_features(self, df):
        """Fill missing features with 0.0"""
        for feat in self.tech_indicator_list:
            if feat not in df.columns:
                df[feat] = 0.0
        for feat in self.tech_indicator_list_trend:
            if feat not in df.columns:
                df[feat] = 0.0
        return df

    def load_subagents(self):
        print("\n[Loading Sub-agents]")
        base_path = "./MacroHFT/result/low_level/BTCUSDT"
        
        subagent_configs = [
            ('slope', 1, 'label_1', 0),
            ('slope', 4, 'label_2', 1),
            ('slope', 0, 'label_3', 2),
            ('vol', 4, 'label_1', 0),
            ('vol', 1, 'label_2', 1),
            ('vol', 1, 'label_3', 2)
        ]
        
        for clf, alpha, label, idx in subagent_configs:
            path = os.path.join(base_path, clf, str(alpha), label, f"seed_{self.seed}", "best_model.pkl")
            if not os.path.exists(path):
                print(f"[WARNING] Model not found: {path}. Using random init (BAD for performance).")
                model = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
            else:
                print(f"  Loading {clf} agent {idx+1} from {path}")
                model = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device))
            
            model.eval()
            if clf == 'slope':
                self.slope_agents[idx] = model
            else:
                self.vol_agents[idx] = model

    def handle_signal(self, signum, frame):
        print("\n[INFO] Interrupt received! Stopping after current chunk...")
        self.stop_requested = True

    def save_checkpoint(self, epoch, chunk_idx):
        print(f"  [Saving checkpoint] Epoch {epoch}, Chunk {chunk_idx}...")
        checkpoint = {
            'epoch': epoch,
            'chunk_idx': chunk_idx,
            'hyperagent': self.hyperagent.state_dict(),
            'hyperagent_target': self.hyperagent_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_return_rate': self.best_return_rate,
            'total_step_counter': self.total_step_counter,
            'memory_buffer': self.memory.buffer, # Save numpy buffer
            'memory_count': self.memory.count,
            'memory_current_size': self.memory.current_size
        }
        
        # Save Replay Buffer separately (can be large)
        buffer_path = os.path.join(self.model_path, "replay_buffer.pkl")
        with open(buffer_path, 'wb') as f:
            pickle.dump(self.replay_buffer, f)
            
        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            print(f"\n[INFO] Found checkpoint: {self.checkpoint_path}")
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.hyperagent.load_state_dict(checkpoint['hyperagent'])
                self.hyperagent_target.load_state_dict(checkpoint['hyperagent_target'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epsilon = checkpoint['epsilon']
                self.start_epoch = checkpoint['epoch']
                self.start_chunk_idx = checkpoint['chunk_idx']
                self.best_return_rate = checkpoint['best_return_rate']
                self.total_step_counter = checkpoint.get('total_step_counter', 0)
                
                # Restore Memory
                self.memory.buffer = checkpoint['memory_buffer']
                self.memory.count = checkpoint['memory_count']
                self.memory.current_size = checkpoint['memory_current_size']
                
                # Restore Replay Buffer
                buffer_path = os.path.join(self.model_path, "replay_buffer.pkl")
                if os.path.exists(buffer_path):
                    with open(buffer_path, 'rb') as f:
                        self.replay_buffer = pickle.load(f)
                    print(f"  Replay Buffer loaded: {self.replay_buffer.size} items")
                
                print(f"[RESUMING] From Epoch {self.start_epoch}, Chunk {self.start_chunk_idx}")
            except Exception as e:
                print(f"[ERROR] Failed to load checkpoint: {e}")
                print("[WARNING] Starting from scratch...")
        else:
            print("[INFO] No checkpoint found. Starting fresh.")

    def calculate_q(self, w, qs):
        q_tensor = torch.stack(qs)
        q_tensor = q_tensor.permute(1, 0, 2)
        weights_reshaped = w.view(-1, 1, 6)
        combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)
        return combined_q

    def act(self, state, state_trend, state_clf, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.device), 0).to(self.device)
        
        if np.random.uniform() < (1 - self.epsilon):
            with torch.no_grad():
                qs = [self.slope_agents[i](x1, x2, previous_action) for i in range(3)] + \
                     [self.vol_agents[i](x1, x2, previous_action) for i in range(3)]
                w = self.hyperagent(x1, x2, x3, previous_action)
                actions_value = self.calculate_q(w, qs)
                action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:
            action = random.choice([0, 1])
        return action
    
    def act_test(self, state, state_trend, state_clf, info):
        with torch.no_grad():
            x1 = torch.FloatTensor(state).to(self.device)
            x2 = torch.FloatTensor(state_trend).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
            previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.device), 0).to(self.device)
            
            qs = [self.slope_agents[i](x1, x2, previous_action) for i in range(3)] + \
                 [self.vol_agents[i](x1, x2, previous_action) for i in range(3)]
            w = self.hyperagent(x1, x2, x3, previous_action)
            actions_value = self.calculate_q(w, qs)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
            return action

    def q_estimate(self, state, state_trend, state_clf, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.device), 0).to(self.device)
        
        with torch.no_grad():
            qs = [self.slope_agents[i](x1, x2, previous_action) for i in range(3)] + \
                 [self.vol_agents[i](x1, x2, previous_action) for i in range(3)]
            w = self.hyperagent(x1, x2, x3, previous_action)
            actions_value = self.calculate_q(w, qs)
            q = torch.max(actions_value, 1)[0].detach().cpu().numpy()
        return q

    def calculate_hidden(self, state, state_trend, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.device), 0).to(self.device)
        with torch.no_grad():
            hs = self.hyperagent.encode(x1, x2, previous_action).cpu().numpy()
        return hs

    def update(self, replay_buffer):
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        w_current = self.hyperagent(batch['state'], batch['state_trend'], batch['state_clf'], batch['previous_action'])
        w_next = self.hyperagent_target(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])
        w_next_ = self.hyperagent(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])

        qs_current = [self.slope_agents[i](batch['state'], batch['state_trend'], batch['previous_action']) for i in range(3)] + \
                     [self.vol_agents[i](batch['state'], batch['state_trend'], batch['previous_action']) for i in range(3)]
        qs_next = [self.slope_agents[i](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']) for i in range(3)] + \
                  [self.vol_agents[i](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']) for i in range(3)]

        q_distribution = self.calculate_q(w_current, qs_current)
        q_current = q_distribution.gather(-1, batch['action']).squeeze(-1)
        a_argmax = self.calculate_q(w_next_, qs_next).argmax(dim=-1, keepdim=True)
        q_nexts = self.calculate_q(w_next, qs_next)
        q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * q_nexts.gather(-1, a_argmax).squeeze(-1)

        td_error = self.loss_func(q_current, q_target)
        memory_error = self.loss_func(q_current, batch['q_memory'])

        demonstration = batch['demo_action']
        KL_loss = F.kl_div(
            (q_distribution.softmax(dim=-1) + 1e-8).log(),
            (demonstration.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )

        loss = td_error + self.alpha * memory_error + self.beta * KL_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.hyperagent.parameters(), 1)
        self.optimizer.step()
        
        for param, target_param in zip(self.hyperagent.parameters(), self.hyperagent_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return td_error.item()

    def train(self):
        print("\n" + "="*70)
        print("Starting High-level Fine-tuning")
        print("="*70)
        
        # Determine number of chunks
        num_chunks = len([f for f in os.listdir(self.train_data_path) if f.startswith("df_") and f.endswith(".feather")])
        print(f"Total chunks: {num_chunks}")
        
        for epoch in range(self.start_epoch, self.epoch_number):
            if self.stop_requested:
                break
            print(f"\n[Epoch {epoch + 1}/{self.epoch_number}]")
            
            chunk_indices = list(range(num_chunks))
            # Random shuffle chunks? Original used whole dataset sequentially.
            # But low-level shuffled. Let's shuffle for better generalization.
            random.shuffle(chunk_indices)
            
            pbar = tqdm(chunk_indices, desc="Training")
            for i, df_index in enumerate(pbar):
                if i < self.start_chunk_idx:
                    continue
                
                if self.stop_requested:
                    print("\n[INFO] Stop requested. Saving...")
                    self.save_checkpoint(epoch, i)
                    return self.best_return_rate

                df = pd.read_feather(os.path.join(self.train_data_path, f"df_{df_index}.feather"))
                df = self.ensure_features(df)
                
                train_env = Training_Env(
                    df=df,
                    tech_indicator_list=self.tech_indicator_list,
                    tech_indicator_list_trend=self.tech_indicator_list_trend,
                    clf_list=self.clf_list,
                    transcation_cost=args.transcation_cost,
                    back_time_length=1,
                    max_holding_number=0.01, # BTC
                    initial_action=random.choice([0, 1]),
                    alpha=0
                )
                
                s, s2, s3, info = train_env.reset()
                done = False
                
                while not done:
                    a = self.act(s, s2, s3, info)
                    s_, s2_, s3_, r, done, info_ = train_env.step(a)
                    
                    hs = self.calculate_hidden(s, s2, info)
                    
                    # Q-memory query
                    q_memory = self.memory.query(hs, a)
                    if np.isnan(q_memory):
                        # If not found, approximate with current Q-estimate
                        q = r + self.gamma * (1 - done) * self.q_estimate(s_, s2_, s3_, info_)[0]
                        q_memory = q
                    
                    self.replay_buffer.store_transition(
                        s, s2, s3, info['previous_action'], info['q_value'], 
                        a, r, s_, s2_, s3_, info_['previous_action'], info_['q_value'], 
                        done, q_memory
                    )
                    
                    # Add to memory
                    # Need q_value for memory. Use estimated Q? 
                    # Original code: q = r + gamma * q_next.
                    # Re-estimate Q for current state to store in memory?
                    # Original code lines 361: q = r + ...
                    q_target = r + self.gamma * (1 - done) * self.q_estimate(s_, s2_, s3_, info_)[0]
                    self.memory.add(hs, a, q_target, s, s2, info['previous_action'])
                    
                    s, s2, s3, info = s_, s2_, s3_, info_
                    self.total_step_counter += 1
                    
                    # Update
                    if self.total_step_counter % self.eval_update_freq == 0 and len(self.replay_buffer) > self.batch_size:
                        for _ in range(self.update_times):
                            self.update(self.replay_buffer)
                    
                    # Re-encode memory
                    if self.total_step_counter % 4320 == 0 and self.total_step_counter > 0:
                         self.memory.re_encode(self.hyperagent)
                
                # Checkpoint every chunk
                self.save_checkpoint(epoch, i + 1)
            
            self.start_chunk_idx = 0
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch + 1)
            
            # Validate
            val_return = self.validate()
            print(f"  Val Return: {val_return:.4f}%")
            if val_return > self.best_return_rate:
                self.best_return_rate = val_return
                torch.save(self.hyperagent.state_dict(), os.path.join(self.model_path, "best_model.pkl"))
                print(f"  [NEW BEST] Saved best model")

        print("Fine-tuning Complete!")
        return self.best_return_rate

    def validate(self):
        self.hyperagent.eval()
        val_returns = []
        # Validate on first 5 chunks of validation set
        for i in range(5):
            path = os.path.join(self.val_data_path, f"df_{i}.feather")
            if not os.path.exists(path): break
            df = pd.read_feather(path)
            df = self.ensure_features(df)
            
            val_env = Testing_Env(
                df=df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=args.transcation_cost,
                back_time_length=1,
                max_holding_number=0.01
            )
            
            s, s2, s3, info = val_env.reset()
            done = False
            while not done:
                a = self.act_test(s, s2, s3, info)
                s, s2, s3, _, done, info = val_env.step(a)
            val_returns.append(val_env.portfolio_return)
            
        self.hyperagent.train()
        return np.mean(val_returns) if val_returns else 0.0

if __name__ == "__main__":
    agent = FineTuneHighLevel(args)
    agent.train()
