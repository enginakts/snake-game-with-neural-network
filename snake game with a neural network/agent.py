import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGame, Direction, Point
from model import QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    """
    Yılan oyununu oynayan yapay zeka ajanı
    """
    def __init__(self, load_existing=False):
        """
        Ajan başlatıcı fonksiyonu
        
        Args:
            load_existing: Önceden eğitilmiş modeli yükle
        """
        self.n_games = 400 if load_existing else 0  # Önceki eğitimden devam et
        self.epsilon = max(0, 80 - self.n_games)  # Epsilon değerini ayarla
        self.gamma = 0.9  # İndirim faktörü
        self.memory = deque(maxlen=MAX_MEMORY)  # Deneyim belleği
        
        # Model, trainer ve oyun durumu boyutları
        self.model = QNet(11, 256, 3)  # 11 giriş durumu, 256 gizli nöron, 3 çıkış (hareket yönü)
        
        # Eğer önceki model varsa yükle
        if load_existing:
            try:
                self.model.load_state_dict(torch.load('model.pth'))
                self.model.eval()
                print("Önceki model başarıyla yüklendi! Eğitim kaldığı yerden devam edecek.")
            except:
                print("Önceki model yüklenemedi! Yeni model ile başlanıyor.")
                self.n_games = 0
                self.epsilon = 80
        
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def get_state(self, game):
        """
        Oyunun mevcut durumunu vektör olarak alır
        
        Args:
            game: SnakeGame nesnesi
            
        Returns:
            state: Durum vektörü
        """
        head = game.snake[0]
        
        # Yılanın etrafındaki noktalar
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # Mevcut yön
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        state = [
            # Tehlike düz
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            
            # Tehlike sağ
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            
            # Tehlike sol
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            
            # Hareket yönü
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Yemek konumu
            game.food.x < game.head.x,  # yemek sol tarafta
            game.food.x > game.head.x,  # yemek sağ tarafta
            game.food.y < game.head.y,  # yemek yukarıda
            game.food.y > game.head.y   # yemek aşağıda
        ]
        
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Deneyimi belleğe kaydeder
        
        Args:
            state: Mevcut durum
            action: Seçilen aksiyon
            reward: Alınan ödül
            next_state: Sonraki durum
            done: Oyun bitti mi
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        """Deneyim belleğinden örneklem alarak eğitim yapar"""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Tek bir adım için eğitim yapar
        
        Args:
            state: Mevcut durum
            action: Seçilen aksiyon
            reward: Alınan ödül
            next_state: Sonraki durum
            done: Oyun bitti mi
        """
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        """
        Epsilon-greedy stratejisi ile aksiyon seçer
        
        Args:
            state: Mevcut durum
            
        Returns:
            final_move: Seçilen aksiyon
        """
        # Rastgele hareket olasılığı
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            # Rastgele hareket
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Model tahmini
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move